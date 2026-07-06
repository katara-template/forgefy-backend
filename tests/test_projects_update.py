"""Tests for POST /api/v1/projects/{id}/update — previously had zero coverage."""
import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from httpx import AsyncClient

from tests.conftest import make_doc_snapshot


def _project_doc(owner_id: uuid.UUID, **overrides) -> MagicMock:
    now = datetime.now(UTC)
    data = {
        "owner_id": str(owner_id),
        "app_name": "stockflow-manager",
        "template_key": "next",
        "repo_full_name": "acme/stockflow-manager",
        "github_url": "https://github.com/acme/stockflow-manager",
        "is_updating": False,
        "created_at": now,
        "updated_at": now,
    }
    data.update(overrides)
    return make_doc_snapshot(data, doc_id=str(uuid.uuid4()))


class TestUpdateProject:
    async def test_no_auth_returns_401(self, client: AsyncClient) -> None:
        resp = await client.post(
            f"/api/v1/projects/{uuid.uuid4()}/update", json={"prompt": "add dark mode"}
        )
        assert resp.status_code == 401

    async def test_not_owner_returns_403(
        self, auth_client: AsyncClient, mock_db: MagicMock
    ) -> None:
        other_owner = uuid.uuid4()
        mock_db.collection.return_value.document.return_value.get.return_value = (
            _project_doc(other_owner)
        )
        resp = await auth_client.post(
            f"/api/v1/projects/{uuid.uuid4()}/update", json={"prompt": "add dark mode"}
        )
        assert resp.status_code == 403

    async def test_already_updating_returns_422(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        mock_db.collection.return_value.document.return_value.get.return_value = (
            _project_doc(test_user.id, is_updating=True)
        )
        resp = await auth_client.post(
            f"/api/v1/projects/{uuid.uuid4()}/update", json={"prompt": "add dark mode"}
        )
        assert resp.status_code == 422

    async def test_success_dispatches_update(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        mock_db.collection.return_value.document.return_value.get.return_value = (
            _project_doc(test_user.id)
        )
        with patch("app.workers.update_worker.apply_update.apply_async") as mock_dispatch:
            resp = await auth_client.post(
                f"/api/v1/projects/{uuid.uuid4()}/update", json={"prompt": "add dark mode"}
            )

        assert resp.status_code == 200
        assert resp.json() == {"status": "queued"}
        mock_dispatch.assert_called_once()
        prompt = mock_dispatch.call_args.kwargs["args"][1]
        assert prompt == "add dark mode"
