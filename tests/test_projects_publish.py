"""Tests for POST /api/v1/projects/{id}/publish — the production release gate.

Publishing is a real release (a production Cloudflare Pages deploy plus a custom
subdomain), so unlike build-preview it must refuse when Cloudflare is not
configured or the project is not a web app.
"""
import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from httpx import AsyncClient

from app.config import get_settings
from app.main import app


def _cf_settings():
    """Real settings with Cloudflare + publish domain switched on.

    A blanket MagicMock would break the dispatcher's lazy broker client
    (CELERY_BROKER_URL feeds aioredis.from_url), so copy the real object.
    """
    return get_settings().model_copy(
        update={
            "CLOUDFLARE_ACCOUNT_ID": "acct",
            "CLOUDFLARE_API_TOKEN": "tok",
            "PUBLISH_BASE_DOMAIN": "forgefy.dev",
        }
    )


def _unconfigured_cf_settings():
    return get_settings().model_copy(
        update={"CLOUDFLARE_ACCOUNT_ID": "", "CLOUDFLARE_API_TOKEN": ""}
    )


def _project_doc(owner_id: uuid.UUID, **overrides) -> MagicMock:
    from tests.conftest import make_doc_snapshot

    now = datetime.now(UTC)
    data = {
        "owner_id": str(owner_id),
        "app_name": "stockflow-manager",
        "template_key": "next",
        "repo_full_name": "acme/stockflow-manager",
        "github_url": "https://github.com/acme/stockflow-manager",
        "session_id": str(uuid.uuid4()),
        "is_updating": False,
        "created_at": now,
        "updated_at": now,
    }
    data.update(overrides)
    return make_doc_snapshot(data, doc_id=str(uuid.uuid4()))


class TestPublishEndpoint:
    async def test_no_auth_returns_401(self, client: AsyncClient) -> None:
        resp = await client.post(f"/api/v1/projects/{uuid.uuid4()}/publish")
        assert resp.status_code == 401

    async def test_not_owner_returns_403(self, auth_client: AsyncClient, mock_db: MagicMock) -> None:
        mock_db.collection.return_value.document.return_value.get.return_value = _project_doc(
            uuid.uuid4()
        )
        resp = await auth_client.post(f"/api/v1/projects/{uuid.uuid4()}/publish")
        assert resp.status_code == 403

    async def test_build_in_progress_returns_422(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        app.dependency_overrides[get_settings] = _cf_settings
        mock_db.collection.return_value.document.return_value.get.return_value = _project_doc(
            test_user.id, is_updating=True
        )
        resp = await auth_client.post(f"/api/v1/projects/{uuid.uuid4()}/publish")
        assert resp.status_code == 422

    async def test_missing_repo_returns_422(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        app.dependency_overrides[get_settings] = _cf_settings
        mock_db.collection.return_value.document.return_value.get.return_value = _project_doc(
            test_user.id, github_url=""
        )
        resp = await auth_client.post(f"/api/v1/projects/{uuid.uuid4()}/publish")
        assert resp.status_code == 422

    async def test_non_web_project_returns_422(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        app.dependency_overrides[get_settings] = _cf_settings
        mock_db.collection.return_value.document.return_value.get.return_value = _project_doc(
            test_user.id, template_key="flutter"
        )
        resp = await auth_client.post(f"/api/v1/projects/{uuid.uuid4()}/publish")
        assert resp.status_code == 422

    async def test_unconfigured_cloudflare_returns_422(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        app.dependency_overrides[get_settings] = _unconfigured_cf_settings
        mock_db.collection.return_value.document.return_value.get.return_value = _project_doc(
            test_user.id
        )
        resp = await auth_client.post(f"/api/v1/projects/{uuid.uuid4()}/publish")
        assert resp.status_code == 422

    async def test_success_queues_the_publish_task(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        app.dependency_overrides[get_settings] = _cf_settings
        mock_db.collection.return_value.document.return_value.get.return_value = _project_doc(
            test_user.id
        )

        with patch("app.workers.build_worker.publish_project.apply_async") as mock_dispatch:
            resp = await auth_client.post(f"/api/v1/projects/{uuid.uuid4()}/publish")

        assert resp.status_code == 200
        assert resp.json() == {"status": "queued"}
        mock_dispatch.assert_called_once()
        assert mock_dispatch.call_args.kwargs["queue"] == "build"
