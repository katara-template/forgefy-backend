"""Firebase (Google) OAuth account-linking tests."""
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs, urlparse

from httpx import AsyncClient

from app.config import get_settings
from app.core.crypto import decrypt, encrypt
from app.main import app
from tests.conftest import make_doc_snapshot


def _settings(**overrides) -> MagicMock:
    base = dict(
        FIREBASE_OAUTH_CLIENT_ID="test-client-id",
        FIREBASE_OAUTH_CLIENT_SECRET="test-client-secret",
        SECRET_KEY="test-secret-key",
        PUBLIC_API_BASE_URL="https://api.example.com",
        FRONTEND_URL="https://app.example.com",
    )
    base.update(overrides)
    return MagicMock(**base)


class TestFirebaseAuthorize:
    async def test_requires_auth(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/auth/firebase/authorize")
        assert resp.status_code == 401

    async def test_not_configured_returns_error(
        self, auth_client: AsyncClient
    ) -> None:
        app.dependency_overrides[get_settings] = lambda: _settings(FIREBASE_OAUTH_CLIENT_ID="")
        try:
            resp = await auth_client.get("/api/v1/auth/firebase/authorize")
            assert resp.status_code == 200
            assert resp.json() == {"error": "Firebase OAuth not configured"}
        finally:
            del app.dependency_overrides[get_settings]

    async def test_returns_authorize_url_and_stores_verifier(
        self, auth_client: AsyncClient, mock_redis: AsyncMock
    ) -> None:
        app.dependency_overrides[get_settings] = lambda: _settings()
        try:
            resp = await auth_client.get("/api/v1/auth/firebase/authorize")
            assert resp.status_code == 200
            url = resp.json()["url"]
            parsed = urlparse(url)
            qs = parse_qs(parsed.query)

            assert parsed.netloc == "accounts.google.com"
            assert qs["client_id"] == ["test-client-id"]
            assert qs["scope"] == ["https://www.googleapis.com/auth/cloud-platform"]
            assert qs["code_challenge_method"] == ["S256"]
            assert qs["access_type"] == ["offline"]
            assert qs["prompt"] == ["consent"]
            assert "state" in qs and "code_challenge" in qs

            mock_redis.set.assert_called_once()
            redis_key = mock_redis.set.call_args[0][0]
            assert redis_key.startswith("firebase_oauth_verifier:")
        finally:
            del app.dependency_overrides[get_settings]


class TestFirebaseCallback:
    async def test_invalid_state_redirects_with_error(self, client: AsyncClient) -> None:
        app.dependency_overrides[get_settings] = lambda: _settings()
        try:
            resp = await client.get(
                "/api/v1/auth/firebase/callback",
                params={"code": "abc", "state": "garbage"},
                follow_redirects=False,
            )
            assert resp.status_code in (302, 307)
            assert "firebase_error=invalid_state" in resp.headers["location"]
        finally:
            del app.dependency_overrides[get_settings]

    async def test_expired_verifier_redirects_with_error(
        self, client: AsyncClient, mock_redis: AsyncMock, test_user
    ) -> None:
        from app.api.v1.auth import _make_state

        settings = _settings()
        app.dependency_overrides[get_settings] = lambda: settings
        mock_redis.get.return_value = None
        try:
            state = _make_state(str(test_user.id), settings.SECRET_KEY)
            resp = await client.get(
                "/api/v1/auth/firebase/callback",
                params={"code": "abc", "state": state},
                follow_redirects=False,
            )
            assert resp.status_code in (302, 307)
            assert "firebase_error=expired_state" in resp.headers["location"]
        finally:
            del app.dependency_overrides[get_settings]

    async def test_success_stores_encrypted_tokens(
        self, client: AsyncClient, mock_db: MagicMock, mock_redis: AsyncMock, test_user
    ) -> None:
        from app.api.v1.auth import _make_state

        settings = _settings()
        app.dependency_overrides[get_settings] = lambda: settings
        mock_redis.get.return_value = "the-code-verifier"
        try:
            state = _make_state(str(test_user.id), settings.SECRET_KEY)
            with patch(
                "app.integrations.firebase_management.exchange_code_for_token",
                new=AsyncMock(return_value={
                    "access_token": "gcp-access-token",
                    "refresh_token": "gcp-refresh-token",
                    "expires_in": 3600,
                }),
            ):
                resp = await client.get(
                    "/api/v1/auth/firebase/callback",
                    params={"code": "abc", "state": state},
                    follow_redirects=False,
                )
            assert resp.status_code in (302, 307)
            assert "firebase=connected" in resp.headers["location"]

            mock_redis.delete.assert_called_once()
            update_payload = mock_db.collection.return_value.document.return_value.update.call_args[0][0]
            assert decrypt(update_payload["firebase_access_token"]) == "gcp-access-token"
            assert decrypt(update_payload["firebase_refresh_token"]) == "gcp-refresh-token"
        finally:
            del app.dependency_overrides[get_settings]

    async def test_exchange_failure_redirects_with_server_error(
        self, client: AsyncClient, mock_redis: AsyncMock, test_user
    ) -> None:
        from app.api.v1.auth import _make_state

        settings = _settings()
        app.dependency_overrides[get_settings] = lambda: settings
        mock_redis.get.return_value = "the-code-verifier"
        try:
            state = _make_state(str(test_user.id), settings.SECRET_KEY)
            with patch(
                "app.integrations.firebase_management.exchange_code_for_token",
                new=AsyncMock(side_effect=RuntimeError("boom")),
            ):
                resp = await client.get(
                    "/api/v1/auth/firebase/callback",
                    params={"code": "abc", "state": state},
                    follow_redirects=False,
                )
            assert resp.status_code in (302, 307)
            assert "firebase_error=server_error" in resp.headers["location"]
        finally:
            del app.dependency_overrides[get_settings]


class TestFirebaseStatus:
    async def test_linked_true_when_token_present(
        self, auth_client: AsyncClient, mock_db: MagicMock
    ) -> None:
        mock_db.collection.return_value.document.return_value.get.return_value = (
            make_doc_snapshot({"firebase_access_token": encrypt("x")})
        )
        resp = await auth_client.get("/api/v1/auth/firebase/status")
        assert resp.status_code == 200
        assert resp.json() == {"linked": True}

    async def test_linked_false_when_no_token(
        self, auth_client: AsyncClient, mock_db: MagicMock
    ) -> None:
        mock_db.collection.return_value.document.return_value.get.return_value = (
            make_doc_snapshot({})
        )
        resp = await auth_client.get("/api/v1/auth/firebase/status")
        assert resp.status_code == 200
        assert resp.json() == {"linked": False}
