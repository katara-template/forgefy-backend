"""Supabase OAuth account-linking tests."""
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs, urlparse

from httpx import AsyncClient

from app.config import get_settings
from app.core.crypto import decrypt, encrypt
from app.main import app
from tests.conftest import make_doc_snapshot


def _settings(**overrides) -> MagicMock:
    base = dict(
        SUPABASE_CLIENT_ID="test-client-id",
        SUPABASE_CLIENT_SECRET="test-client-secret",
        SECRET_KEY="test-secret-key",
        PUBLIC_API_BASE_URL="https://api.example.com",
        FRONTEND_URL="https://app.example.com",
    )
    base.update(overrides)
    return MagicMock(**base)


class TestSupabaseAuthorize:
    async def test_requires_auth(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/auth/supabase/authorize")
        assert resp.status_code == 401

    async def test_not_configured_returns_error(
        self, auth_client: AsyncClient
    ) -> None:
        app.dependency_overrides[get_settings] = lambda: _settings(SUPABASE_CLIENT_ID="")
        try:
            resp = await auth_client.get("/api/v1/auth/supabase/authorize")
            assert resp.status_code == 200
            assert resp.json() == {"error": "Supabase OAuth not configured"}
        finally:
            del app.dependency_overrides[get_settings]

    async def test_returns_authorize_url_and_stores_verifier(
        self, auth_client: AsyncClient, mock_redis: AsyncMock
    ) -> None:
        app.dependency_overrides[get_settings] = lambda: _settings()
        try:
            resp = await auth_client.get("/api/v1/auth/supabase/authorize")
            assert resp.status_code == 200
            url = resp.json()["url"]
            parsed = urlparse(url)
            qs = parse_qs(parsed.query)

            assert parsed.netloc == "api.supabase.com"
            assert qs["client_id"] == ["test-client-id"]
            assert qs["scope"] == ["all"]
            assert qs["code_challenge_method"] == ["S256"]
            assert "state" in qs and "code_challenge" in qs

            mock_redis.set.assert_called_once()
            redis_key = mock_redis.set.call_args[0][0]
            assert redis_key.startswith("supabase_oauth_verifier:")
        finally:
            del app.dependency_overrides[get_settings]


class TestSupabaseCallback:
    async def test_invalid_state_redirects_with_error(self, client: AsyncClient) -> None:
        app.dependency_overrides[get_settings] = lambda: _settings()
        try:
            resp = await client.get(
                "/api/v1/auth/supabase/callback",
                params={"code": "abc", "state": "garbage"},
                follow_redirects=False,
            )
            assert resp.status_code in (302, 307)
            assert "supabase_error=invalid_state" in resp.headers["location"]
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
                "/api/v1/auth/supabase/callback",
                params={"code": "abc", "state": state},
                follow_redirects=False,
            )
            assert resp.status_code in (302, 307)
            assert "supabase_error=expired_state" in resp.headers["location"]
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
                "app.integrations.supabase_management.exchange_code_for_token",
                new=AsyncMock(return_value={
                    "access_token": "sb-access-token",
                    "refresh_token": "sb-refresh-token",
                    "expires_in": 3600,
                }),
            ):
                resp = await client.get(
                    "/api/v1/auth/supabase/callback",
                    params={"code": "abc", "state": state},
                    follow_redirects=False,
                )
            assert resp.status_code in (302, 307)
            assert "supabase=connected" in resp.headers["location"]

            mock_redis.delete.assert_called_once()
            update_payload = mock_db.collection.return_value.document.return_value.update.call_args[0][0]
            assert decrypt(update_payload["supabase_access_token"]) == "sb-access-token"
            assert decrypt(update_payload["supabase_refresh_token"]) == "sb-refresh-token"
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
                "app.integrations.supabase_management.exchange_code_for_token",
                new=AsyncMock(side_effect=RuntimeError("boom")),
            ):
                resp = await client.get(
                    "/api/v1/auth/supabase/callback",
                    params={"code": "abc", "state": state},
                    follow_redirects=False,
                )
            assert resp.status_code in (302, 307)
            assert "supabase_error=server_error" in resp.headers["location"]
        finally:
            del app.dependency_overrides[get_settings]


class TestSupabaseStatus:
    async def test_linked_true_when_token_present(
        self, auth_client: AsyncClient, mock_db: MagicMock
    ) -> None:
        mock_db.collection.return_value.document.return_value.get.return_value = (
            make_doc_snapshot({"supabase_access_token": encrypt("x")})
        )
        resp = await auth_client.get("/api/v1/auth/supabase/status")
        assert resp.status_code == 200
        assert resp.json() == {"linked": True}

    async def test_linked_false_when_no_token(
        self, auth_client: AsyncClient, mock_db: MagicMock
    ) -> None:
        mock_db.collection.return_value.document.return_value.get.return_value = (
            make_doc_snapshot({})
        )
        resp = await auth_client.get("/api/v1/auth/supabase/status")
        assert resp.status_code == 200
        assert resp.json() == {"linked": False}


class TestSupabaseOrganizations:
    async def test_no_linked_account_returns_401(
        self, auth_client: AsyncClient, mock_db: MagicMock
    ) -> None:
        mock_db.collection.return_value.document.return_value.get.return_value = (
            make_doc_snapshot({})
        )
        resp = await auth_client.get("/api/v1/auth/supabase/organizations")
        assert resp.status_code == 401

    async def test_linked_account_returns_org_list(
        self, auth_client: AsyncClient, mock_db: MagicMock
    ) -> None:
        import time

        mock_db.collection.return_value.document.return_value.get.return_value = (
            make_doc_snapshot({
                "supabase_access_token": encrypt("sb-access-token"),
                "supabase_refresh_token": encrypt("sb-refresh-token"),
                "supabase_token_expires_at": time.time() + 3600,
            })
        )
        with patch(
            "app.integrations.supabase_management.list_organizations",
            new=AsyncMock(return_value=[{"id": "org-1", "name": "Acme"}]),
        ):
            resp = await auth_client.get("/api/v1/auth/supabase/organizations")
        assert resp.status_code == 200
        assert resp.json() == [{"id": "org-1", "name": "Acme"}]
