"""CORS configuration tests.

create_app() previously hardcoded allow_origins=["*"] with allow_credentials=True,
which ignored the documented CORS_ORIGINS setting entirely and — because Starlette
echoes the caller's Origin back when wildcard and credentials are combined — let
any site issue credentialed cross-origin requests. These tests pin the wiring.
"""
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.core.exceptions import _cors_headers
from app.main import create_app


def _app_with_origins(origins: list[str], env: str = "development"):
    """Build a fresh app whose settings carry the given CORS origins."""
    settings = get_settings().model_copy(update={"CORS_ORIGINS": origins, "APP_ENV": env})
    with patch("app.main.get_settings", return_value=settings):
        return create_app()


async def _preflight(app, origin: str):
    """Send a CORS preflight. CORSMiddleware answers it without touching a route."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        return await ac.options(
            "/api/v1/keys",
            headers={"Origin": origin, "Access-Control-Request-Method": "GET"},
        )


class TestCorsAllowlist:
    async def test_configured_origin_is_allowed(self):
        app = _app_with_origins(["https://app.forgefy.dev"])
        res = await _preflight(app, "https://app.forgefy.dev")

        assert res.headers["access-control-allow-origin"] == "https://app.forgefy.dev"

    async def test_unconfigured_origin_is_rejected(self):
        app = _app_with_origins(["https://app.forgefy.dev"])
        res = await _preflight(app, "https://evil.example.com")

        # Starlette answers a disallowed preflight with 400 and no allow-origin header.
        assert res.status_code == 400
        assert "access-control-allow-origin" not in res.headers

    async def test_credentials_allowed_for_an_explicit_allowlist(self):
        app = _app_with_origins(["https://app.forgefy.dev"])
        res = await _preflight(app, "https://app.forgefy.dev")

        assert res.headers.get("access-control-allow-credentials") == "true"

    @pytest.mark.parametrize("env", ["development", "production"])
    async def test_wildcard_never_grants_credentials(self, env: str):
        """A literal "*" must not be combined with credentials in any environment.

        This is the specific footgun the old hardcoded config had: with both set,
        Starlette reflects the caller's Origin, so every site becomes trusted.
        """
        app = _app_with_origins(["*"], env=env)
        res = await _preflight(app, "https://evil.example.com")

        assert "access-control-allow-credentials" not in res.headers

    async def test_multiple_configured_origins_are_each_allowed(self):
        origins = ["http://localhost:8080", "https://forgefy-meeting-to-app-ashy.vercel.app"]
        app = _app_with_origins(origins)

        for origin in origins:
            res = await _preflight(app, origin)
            assert res.headers["access-control-allow-origin"] == origin


def _request_with_cors(origin: str, **cors_kwargs) -> MagicMock:
    """A Request whose app exposes a real CORSMiddleware for _cors_headers to find."""
    request = MagicMock()
    request.headers = {"origin": origin}
    request.app.middleware_stack = CORSMiddleware(app=MagicMock(), **cors_kwargs)
    return request


class TestErrorResponseCorsHeaders:
    """Error responses build CORS headers by hand (app/core/exceptions.py).

    They must mirror the middleware — an error path that advertises credentials
    the middleware refuses reintroduces the wildcard footgun through the back door.
    """

    def test_allowed_origin_gets_credentials_under_an_explicit_allowlist(self):
        request = _request_with_cors(
            "https://app.forgefy.dev",
            allow_origins=["https://app.forgefy.dev"],
            allow_credentials=True,
        )

        headers = _cors_headers(request)

        assert headers["Access-Control-Allow-Origin"] == "https://app.forgefy.dev"
        assert headers["Access-Control-Allow-Credentials"] == "true"

    def test_wildcard_without_credentials_does_not_advertise_credentials(self):
        request = _request_with_cors(
            "https://evil.example.com", allow_origins=["*"], allow_credentials=False
        )

        headers = _cors_headers(request)

        assert headers["Access-Control-Allow-Origin"] == "*"
        assert "Access-Control-Allow-Credentials" not in headers

    def test_disallowed_origin_gets_no_cors_headers(self):
        request = _request_with_cors(
            "https://evil.example.com",
            allow_origins=["https://app.forgefy.dev"],
            allow_credentials=True,
        )

        assert _cors_headers(request) == {}
