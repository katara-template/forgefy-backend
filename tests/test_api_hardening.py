"""Tests for developer-API hardening: per-key rate identity, SSRF guard, load shedding."""
import hashlib
from unittest.mock import MagicMock, patch

import pytest
from httpx import AsyncClient

from app.core.rate_limit import api_key_ident

# ── Per-key rate-limit identity ───────────────────────────────────────────────


def _request_with(auth: str | None, ip: str = "203.0.113.9") -> MagicMock:
    request = MagicMock()
    request.headers = {"authorization": auth} if auth else {}
    request.client.host = ip
    return request


class TestApiKeyIdent:
    def test_api_key_header_hashes_the_key(self) -> None:
        header = "Bearer fgy_live_abc123"
        ident = api_key_ident(_request_with(header))
        assert ident == hashlib.sha256(header.encode()).hexdigest()

    def test_same_key_from_different_ips_shares_one_bucket(self) -> None:
        header = "Bearer fgy_live_abc123"
        a = api_key_ident(_request_with(header, ip="198.51.100.1"))
        b = api_key_ident(_request_with(header, ip="203.0.113.9"))
        assert a == b

    def test_jwt_falls_back_to_ip(self) -> None:
        ident = api_key_ident(_request_with("Bearer eyJhbGciOi.jwt.token"))
        assert ident == "203.0.113.9"

    def test_no_header_falls_back_to_ip(self) -> None:
        assert api_key_ident(_request_with(None)) == "203.0.113.9"


# ── SSRF guard on webhook delivery ────────────────────────────────────────────


def _addrinfo(ip: str) -> list:
    return [(2, 1, 6, "", (ip, 0))]


class TestWebhookSsrfGuard:
    def _production_settings(self):
        return MagicMock(APP_ENV="production")

    def test_private_address_refused(self) -> None:
        from app.workers import extract_api_worker as mod

        with (
            patch.object(mod, "get_settings", return_value=self._production_settings()),
            patch.object(mod.socket, "getaddrinfo", return_value=_addrinfo("10.0.0.5")),
            pytest.raises(mod.WebhookHostError),
        ):
            mod._assert_public_webhook_host("https://internal.example.com/hook")

    def test_loopback_refused(self) -> None:
        from app.workers import extract_api_worker as mod

        with (
            patch.object(mod, "get_settings", return_value=self._production_settings()),
            patch.object(mod.socket, "getaddrinfo", return_value=_addrinfo("127.0.0.1")),
            pytest.raises(mod.WebhookHostError),
        ):
            mod._assert_public_webhook_host("https://sneaky.example.com/hook")

    def test_metadata_service_refused(self) -> None:
        from app.workers import extract_api_worker as mod

        with (
            patch.object(mod, "get_settings", return_value=self._production_settings()),
            patch.object(mod.socket, "getaddrinfo", return_value=_addrinfo("169.254.169.254")),
            pytest.raises(mod.WebhookHostError),
        ):
            mod._assert_public_webhook_host("https://metadata.example.com/hook")

    def test_public_address_allowed(self) -> None:
        from app.workers import extract_api_worker as mod

        with (
            patch.object(mod, "get_settings", return_value=self._production_settings()),
            patch.object(mod.socket, "getaddrinfo", return_value=_addrinfo("93.184.216.34")),
        ):
            mod._assert_public_webhook_host("https://example.com/hook")  # no raise

    def test_development_bypasses_guard(self) -> None:
        from app.workers import extract_api_worker as mod

        with patch.object(
            mod, "get_settings", return_value=MagicMock(APP_ENV="development")
        ):
            mod._assert_public_webhook_host("http://localhost:9000/hook")  # no raise

    def test_delivery_dropped_not_retried_on_ssrf(self) -> None:
        from app.workers import extract_api_worker as mod

        with (
            patch.object(mod, "_assert_public_webhook_host", side_effect=mod.WebhookHostError("nope")),
            patch.object(mod.httpx, "post") as post,
        ):
            mod.deliver_extract_webhook("https://evil/hook", "sec", {"job_id": "j1"})

        post.assert_not_called()


# ── Sync-extract load shedding ────────────────────────────────────────────────


class TestSyncConcurrencyCeiling:
    async def test_at_capacity_returns_429(self, api_client: AsyncClient) -> None:
        with patch("app.api.v1.extract._sync_in_flight", 8):
            resp = await api_client.post(
                "/api/v1/extract", json={"transcript": "we need oauth login"}
            )
        assert resp.status_code == 429

    async def test_counter_released_after_request(self, api_client: AsyncClient) -> None:
        from app.api.v1 import extract as mod

        result = {"events": [], "usage": {"input_tokens": 1, "output_tokens": 1}, "errors": []}
        with patch("app.api.v1.extract.run_extraction", return_value=result):
            resp = await api_client.post(
                "/api/v1/extract", json={"transcript": "we need oauth login"}
            )
        assert resp.status_code == 200
        assert mod._sync_in_flight == 0

    async def test_counter_released_on_failure(self, api_client: AsyncClient) -> None:
        from app.api.v1 import extract as mod

        result = {"events": [], "usage": {"input_tokens": 0, "output_tokens": 0},
                  "errors": ["feature_extractor: boom"]}
        with patch("app.api.v1.extract.run_extraction", return_value=result):
            resp = await api_client.post(
                "/api/v1/extract", json={"transcript": "we need oauth login"}
            )
        assert resp.status_code == 502
        assert mod._sync_in_flight == 0
