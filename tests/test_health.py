"""Health endpoint tests — dependency readiness, not just liveness."""
from unittest.mock import MagicMock

from httpx import AsyncClient


class TestHealth:
    async def test_healthy_when_redis_and_firestore_ok(self, client: AsyncClient) -> None:
        resp = await client.get("/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["checks"] == {"redis": "ok", "firestore": "ok"}

    async def test_503_when_redis_unreachable(
        self, client: AsyncClient, mock_redis: MagicMock
    ) -> None:
        mock_redis.ping.side_effect = ConnectionError("connection refused")

        resp = await client.get("/health")

        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["checks"]["redis"].startswith("error:")
        assert body["checks"]["firestore"] == "ok"

    async def test_503_when_firestore_unreachable(
        self, client: AsyncClient, mock_db: MagicMock
    ) -> None:
        mock_db.collection.return_value.document.return_value.get.side_effect = (
            RuntimeError("unavailable")
        )

        resp = await client.get("/health")

        assert resp.status_code == 503
        body = resp.json()
        assert body["checks"]["firestore"].startswith("error:")
        assert body["checks"]["redis"] == "ok"
