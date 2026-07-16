"""Tests for GET /api/v1/usage — the machine-authed quota endpoint."""
from unittest.mock import AsyncMock, patch

from httpx import AsyncClient


class TestUsageEndpoint:
    async def test_returns_owner_usage(self, api_client: AsyncClient) -> None:
        with (
            patch("app.core.usage.get_user_tier", new=AsyncMock(return_value="starter")),
            patch("app.core.usage.get_monthly_tokens", new=AsyncMock(return_value=1_200_000)),
        ):
            resp = await api_client.get("/api/v1/usage")

        assert resp.status_code == 200
        body = resp.json()
        assert body["tier"] == "starter"
        assert body["tier_name"] == "Starter"
        assert body["monthly_tokens"] == 5_000_000
        assert body["tokens_used"] == 1_200_000
        assert body["tokens_remaining"] == 3_800_000
        assert body["resets_at"]  # ISO datetime of the next month boundary

    async def test_remaining_never_negative(self, api_client: AsyncClient) -> None:
        with (
            patch("app.core.usage.get_user_tier", new=AsyncMock(return_value="free")),
            patch("app.core.usage.get_monthly_tokens", new=AsyncMock(return_value=999_999_999)),
        ):
            resp = await api_client.get("/api/v1/usage")

        assert resp.status_code == 200
        assert resp.json()["tokens_remaining"] == 0

    async def test_defaults_to_free_tier(self, api_client: AsyncClient) -> None:
        # conftest's autouse fixture pins tier="free", tokens=0
        resp = await api_client.get("/api/v1/usage")
        assert resp.status_code == 200
        body = resp.json()
        assert body["tier"] == "free"
        assert body["tokens_used"] == 0

    async def test_requires_api_key(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/usage")
        assert resp.status_code == 401
