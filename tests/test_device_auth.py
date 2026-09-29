"""Tests for the CLI's browser device-code login (app/api/v1/device_auth.py)."""
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import AsyncClient

from app.config import get_settings
from app.db.models.user import User
from app.main import app
from tests.conftest import make_doc_snapshot


def _settings(**overrides) -> MagicMock:
    base = dict(FRONTEND_URL="https://forgefy.app")
    base.update(overrides)
    return MagicMock(**base)


def _device_doc(*, status="pending", user_code="ABCD-EFGH", api_key=None, expired=False) -> MagicMock:
    now = datetime.now(UTC)
    snap = make_doc_snapshot(
        {
            "user_code": user_code,
            "status": status,
            "owner_user_id": None,
            "api_key": api_key,
            "key_id": None,
            "created_at": now,
            "expires_at": now - timedelta(seconds=1) if expired else now + timedelta(minutes=10),
        },
        doc_id="device-code-abc",
    )
    snap.reference.set = AsyncMock()
    snap.reference.delete = AsyncMock()
    return snap


class TestStartDeviceLogin:
    async def test_returns_codes_and_frontend_urls(self, client: AsyncClient, mock_db: MagicMock) -> None:
        app.dependency_overrides[get_settings] = lambda: _settings()
        resp = await client.post("/api/v1/auth/device/start")
        assert resp.status_code == 200
        body = resp.json()
        assert body["verification_uri"] == "https://forgefy.app/cli-auth"
        assert body["verification_uri_complete"] == f"https://forgefy.app/cli-auth?user_code={body['user_code']}"
        assert len(body["device_code"]) > 20
        assert "-" in body["user_code"]
        assert body["expires_in"] > 0
        assert body["interval"] > 0
        mock_db.collection.return_value.document.return_value.set.assert_awaited_once()


class TestPollDeviceLogin:
    async def test_unknown_code_returns_expired(self, client: AsyncClient, mock_db: MagicMock) -> None:
        mock_db.collection.return_value.document.return_value.get.return_value = make_doc_snapshot(None)
        resp = await client.post("/api/v1/auth/device/poll", json={"device_code": "nope"})
        assert resp.status_code == 200
        assert resp.json() == {"status": "expired", "api_key": None}

    async def test_expired_code_deletes_and_returns_expired(
        self, client: AsyncClient, mock_db: MagicMock
    ) -> None:
        doc = _device_doc(expired=True)
        mock_db.collection.return_value.document.return_value.get.return_value = doc
        resp = await client.post("/api/v1/auth/device/poll", json={"device_code": "device-code-abc"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "expired"
        mock_db.collection.return_value.document.return_value.delete.assert_awaited_once()

    async def test_pending_code_returns_pending(self, client: AsyncClient, mock_db: MagicMock) -> None:
        doc = _device_doc(status="pending")
        mock_db.collection.return_value.document.return_value.get.return_value = doc
        resp = await client.post("/api/v1/auth/device/poll", json={"device_code": "device-code-abc"})
        assert resp.json() == {"status": "pending", "api_key": None}

    async def test_denied_code_deletes_and_returns_denied(
        self, client: AsyncClient, mock_db: MagicMock
    ) -> None:
        doc = _device_doc(status="denied")
        mock_db.collection.return_value.document.return_value.get.return_value = doc
        resp = await client.post("/api/v1/auth/device/poll", json={"device_code": "device-code-abc"})
        assert resp.json()["status"] == "denied"
        mock_db.collection.return_value.document.return_value.delete.assert_awaited_once()

    async def test_approved_code_delivers_key_exactly_once(
        self, client: AsyncClient, mock_db: MagicMock
    ) -> None:
        doc = _device_doc(status="approved", api_key="fgy_live_abc123")
        mock_db.collection.return_value.document.return_value.get.return_value = doc
        resp = await client.post("/api/v1/auth/device/poll", json={"device_code": "device-code-abc"})
        assert resp.json() == {"status": "approved", "api_key": "fgy_live_abc123"}
        mock_db.collection.return_value.document.return_value.delete.assert_awaited_once()

    async def test_approved_without_stored_key_returns_expired(
        self, client: AsyncClient, mock_db: MagicMock
    ) -> None:
        # Defensive: shouldn't happen (approve always sets api_key), but a poll
        # racing a second delivery of the same doc must never hand out None.
        doc = _device_doc(status="approved", api_key=None)
        mock_db.collection.return_value.document.return_value.get.return_value = doc
        resp = await client.post("/api/v1/auth/device/poll", json={"device_code": "device-code-abc"})
        assert resp.json()["status"] == "expired"


class TestApproveDeviceLogin:
    async def test_requires_auth(self, client: AsyncClient) -> None:
        resp = await client.post("/api/v1/auth/device/approve", json={"user_code": "ABCD-EFGH"})
        assert resp.status_code == 401

    async def test_unknown_code_returns_404(self, auth_client: AsyncClient, mock_db: MagicMock) -> None:
        mock_db.collection.return_value.where.return_value.limit.return_value.get.return_value = []
        resp = await auth_client.post("/api/v1/auth/device/approve", json={"user_code": "ABCD-EFGH"})
        assert resp.status_code == 404

    async def test_already_approved_code_returns_404(
        self, auth_client: AsyncClient, mock_db: MagicMock
    ) -> None:
        doc = _device_doc(status="approved", api_key="fgy_live_already")
        mock_db.collection.return_value.where.return_value.limit.return_value.get.return_value = [doc]
        resp = await auth_client.post("/api/v1/auth/device/approve", json={"user_code": "ABCD-EFGH"})
        assert resp.status_code == 404

    async def test_at_key_cap_returns_422(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user: User
    ) -> None:
        doc = _device_doc(status="pending")
        mock_db.collection.return_value.where.return_value.limit.return_value.get.return_value = [doc]
        with patch("app.api.v1.device_auth.count_active_api_keys", AsyncMock(return_value=10)):
            resp = await auth_client.post("/api/v1/auth/device/approve", json={"user_code": "ABCD-EFGH"})
        assert resp.status_code == 422

    async def test_success_mints_key_and_marks_doc_approved(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user: User
    ) -> None:
        doc = _device_doc(status="pending")
        mock_db.collection.return_value.where.return_value.limit.return_value.get.return_value = [doc]
        with patch("app.api.v1.device_auth.count_active_api_keys", AsyncMock(return_value=0)):
            resp = await auth_client.post("/api/v1/auth/device/approve", json={"user_code": "abcd-efgh"})
        assert resp.status_code == 204

        create_call = mock_db.collection.return_value.document.return_value.set.await_args
        created = create_call.args[0]
        assert created["owner_user_id"] == str(test_user.id)
        assert created["name"] == "CLI login"
        assert created["prefix"].startswith("fgy_live_")

        doc.reference.set.assert_awaited_once()
        updated = doc.reference.set.await_args.args[0]
        assert updated["status"] == "approved"
        assert updated["owner_user_id"] == str(test_user.id)
        assert updated["api_key"].startswith("fgy_live_")
