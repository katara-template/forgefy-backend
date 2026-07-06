"""Project-level Firebase provisioning endpoint tests."""
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import AsyncClient


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


class TestConnectFirebase:
    async def test_no_auth_returns_401(self, client: AsyncClient) -> None:
        resp = await client.post(f"/api/v1/projects/{uuid.uuid4()}/firebase/connect")
        assert resp.status_code == 401

    async def test_not_owner_returns_403(
        self, auth_client: AsyncClient, mock_db: MagicMock
    ) -> None:
        other_owner = uuid.uuid4()
        mock_db.collection.return_value.document.return_value.get.return_value = (
            _project_doc(other_owner)
        )
        resp = await auth_client.post(f"/api/v1/projects/{uuid.uuid4()}/firebase/connect")
        assert resp.status_code == 403

    async def test_no_linked_google_account_returns_422(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        mock_db.collection.return_value.document.return_value.get.return_value = (
            _project_doc(test_user.id)
        )
        with patch(
            "app.build.firebase_token.get_valid_firebase_token",
            new=AsyncMock(return_value=None),
        ):
            resp = await auth_client.post(f"/api/v1/projects/{uuid.uuid4()}/firebase/connect")
        assert resp.status_code == 422

    async def test_success_provisions_and_stores_fields(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        mock_db.collection.return_value.document.return_value.get.return_value = (
            _project_doc(test_user.id)
        )
        fake_config = {
            "projectId": "fg-stockflow-manager-abc123",
            "apiKey": "fake-api-key",
            "authDomain": "fg-stockflow-manager-abc123.firebaseapp.com",
            "storageBucket": "fg-stockflow-manager-abc123.appspot.com",
            "messagingSenderId": "123456789",
            "appId": "1:123456789:web:abcdef",
        }
        with (
            patch(
                "app.build.firebase_token.get_valid_firebase_token",
                new=AsyncMock(return_value="gcp-access-token"),
            ),
            patch(
                "app.integrations.firebase_management.provision_project",
                new=AsyncMock(return_value=fake_config),
            ),
        ):
            resp = await auth_client.post(f"/api/v1/projects/{uuid.uuid4()}/firebase/connect")

        assert resp.status_code == 200
        # Already-built project — connecting only provisions; wiring the code
        # into the app needs a separate explicit confirmation.
        assert resp.json() == {
            "status": "connected",
            "project_id": "fg-stockflow-manager-abc123",
            "prompt_wire_in": True,
        }

        payload = mock_db.collection.return_value.document.return_value.update.call_args[0][0]
        assert payload["firebase_project_id"] == "fg-stockflow-manager-abc123"
        assert payload["firebase_api_key"] == "fake-api-key"
        assert payload["firebase_auth_domain"] == fake_config["authDomain"]
        assert payload["firebase_storage_bucket"] == fake_config["storageBucket"]
        assert payload["firebase_messaging_sender_id"] == fake_config["messagingSenderId"]
        assert payload["firebase_app_id"] == fake_config["appId"]

    async def test_pending_db_decision_dispatches_withheld_build(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
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
        fake_config = {
            "projectId": "fg-stockflow-manager-abc123",
            "apiKey": "fake-api-key",
            "authDomain": "fg-stockflow-manager-abc123.firebaseapp.com",
            "storageBucket": "fg-stockflow-manager-abc123.appspot.com",
            "messagingSenderId": "123456789",
            "appId": "1:123456789:web:abcdef",
        }
        with (
            patch(
                "app.build.firebase_token.get_valid_firebase_token",
                new=AsyncMock(return_value="gcp-access-token"),
            ),
            patch(
                "app.integrations.firebase_management.provision_project",
                new=AsyncMock(return_value=fake_config),
            ),
            patch("app.workers.build_worker.run_build.apply_async") as mock_dispatch,
        ):
            resp = await auth_client.post(f"/api/v1/projects/{uuid.uuid4()}/firebase/connect")

        assert resp.status_code == 200
        assert resp.json()["build_queued"] is True
        mock_dispatch.assert_called_once()
