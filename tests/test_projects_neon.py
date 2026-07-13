"""Project-level Neon provisioning endpoint tests."""
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import AsyncClient

from app.config import get_settings
from app.core.crypto import decrypt
from app.main import app


def _project_doc(owner_id: uuid.UUID, **overrides) -> MagicMock:
    from tests.conftest import make_doc_snapshot

    now = datetime.now(UTC)
    data = {
        "owner_id": str(owner_id),
        "app_name": "stockflow-manager",
        "template_key": "next",
        "repo_full_name": "acme/stockflow-manager",
        "github_url": "https://github.com/acme/stockflow-manager",
        "created_at": now,
        "updated_at": now,
    }
    data.update(overrides)
    return make_doc_snapshot(data, doc_id=str(uuid.uuid4()))


def _settings(**overrides) -> MagicMock:
    base = dict(NEON_API_KEY="test-neon-api-key")
    base.update(overrides)
    return MagicMock(**base)


class TestConnectNeon:
    async def test_no_auth_returns_401(self, client: AsyncClient) -> None:
        resp = await client.post(f"/api/v1/projects/{uuid.uuid4()}/neon/connect")
        assert resp.status_code == 401

    async def test_not_owner_returns_403(
        self, auth_client: AsyncClient, mock_db: MagicMock
    ) -> None:
        app.dependency_overrides[get_settings] = lambda: _settings()
        try:
            other_owner = uuid.uuid4()
            mock_db.collection.return_value.document.return_value.get.return_value = (
                _project_doc(other_owner)
            )
            resp = await auth_client.post(f"/api/v1/projects/{uuid.uuid4()}/neon/connect")
            assert resp.status_code == 403
        finally:
            del app.dependency_overrides[get_settings]

    async def test_not_configured_returns_422(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        app.dependency_overrides[get_settings] = lambda: _settings(NEON_API_KEY="")
        try:
            mock_db.collection.return_value.document.return_value.get.return_value = (
                _project_doc(test_user.id)
            )
            resp = await auth_client.post(f"/api/v1/projects/{uuid.uuid4()}/neon/connect")
            assert resp.status_code == 422
        finally:
            del app.dependency_overrides[get_settings]

    async def test_success_provisions_and_stores_fields(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        app.dependency_overrides[get_settings] = lambda: _settings()
        mock_db.collection.return_value.document.return_value.get.return_value = (
            _project_doc(test_user.id)
        )
        try:
            with (
                patch(
                    "app.integrations.neon_management.create_project",
                    new=AsyncMock(return_value={
                        "project": {"id": "neon-proj-1"},
                        "branch": {"id": "br-1"},
                        "databases": [{"name": "neondb"}],
                        "connection_uris": [{"connection_uri": "postgres://user:pass@host/db"}],
                    }),
                ),
                patch(
                    "app.integrations.neon_management.enable_data_api",
                    new=AsyncMock(return_value={"url": "https://neon-proj-1.dataapi.neon.tech"}),
                ),
            ):
                resp = await auth_client.post(f"/api/v1/projects/{uuid.uuid4()}/neon/connect")

            assert resp.status_code == 200
            # Already-built project — connecting only provisions; wiring the code
            # into the app needs a separate explicit confirmation.
            assert resp.json() == {
                "status": "connected", "project_id": "neon-proj-1", "prompt_wire_in": True,
            }

            payload = mock_db.collection.return_value.document.return_value.update.call_args[0][0]
            assert payload["neon_project_id"] == "neon-proj-1"
            assert payload["neon_data_api_url"] == "https://neon-proj-1.dataapi.neon.tech"
            # The raw Postgres connection string is a real secret and must be encrypted.
            assert "postgres://user:pass@host/db" not in payload.values()
            assert decrypt(payload["neon_connection_uri"]) == "postgres://user:pass@host/db"
        finally:
            del app.dependency_overrides[get_settings]

    async def test_pending_db_decision_dispatches_withheld_build(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        app.dependency_overrides[get_settings] = lambda: _settings()
        mock_db.collection.return_value.document.return_value.get.return_value = (
            _project_doc(
                test_user.id,
                repo_full_name="",
                github_url="",
                db_decision_pending=True,
                db_decision_reason="Tracks inventory.",
                session_id=str(uuid.uuid4()),
            )
        )
        try:
            with (
                patch(
                    "app.integrations.neon_management.create_project",
                    new=AsyncMock(return_value={
                        "project": {"id": "neon-proj-1"},
                        "branch": {"id": "br-1"},
                        "databases": [{"name": "neondb"}],
                        "connection_uris": [{"connection_uri": "postgres://user:pass@host/db"}],
                    }),
                ),
                patch(
                    "app.integrations.neon_management.enable_data_api",
                    new=AsyncMock(return_value={"url": "https://neon-proj-1.dataapi.neon.tech"}),
                ),
                patch("app.workers.build_worker.run_build.apply_async") as mock_dispatch,
            ):
                resp = await auth_client.post(f"/api/v1/projects/{uuid.uuid4()}/neon/connect")

            assert resp.status_code == 200
            assert resp.json()["build_queued"] is True
            mock_dispatch.assert_called_once()
        finally:
            del app.dependency_overrides[get_settings]
