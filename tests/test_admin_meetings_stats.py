"""Tests for GET /api/v1/admin/meetings/stats — aggregate-only meeting stats.

Deliberately has NO endpoint/test that exposes individual meeting records
(titles, dates, session ids) — the admin dashboard's Meetings page only ever
shows counts/breakdowns, never a specific user's meeting content.
"""
from unittest.mock import MagicMock

from httpx import AsyncClient

from app.deps import get_current_user
from app.main import app


class TestGetMeetingsStats:
    async def test_no_auth_returns_401(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/admin/meetings/stats")
        assert resp.status_code == 401

    async def test_non_admin_returns_403(self, auth_client: AsyncClient) -> None:
        resp = await auth_client.get("/api/v1/admin/meetings/stats")
        assert resp.status_code == 403

    async def test_returns_aggregate_counts_only(
        self, client: AsyncClient, mock_db: MagicMock, admin_user
    ) -> None:
        app.dependency_overrides[get_current_user] = lambda: admin_user
        try:
            mock_db.collection.return_value.count.return_value.get.return_value = [
                [MagicMock(value=500)]
            ]
            mock_db.collection.return_value.where.return_value.count.return_value.get.return_value = [
                [MagicMock(value=12)]
            ]
            # per_day uses a double where() (created_at >= start AND < end), a
            # separate mock path from the single-where() calls above.
            (
                mock_db.collection.return_value.where.return_value.where.return_value
                .count.return_value.get
            ).return_value = [[MagicMock(value=12)]]

            resp = await client.get("/api/v1/admin/meetings/stats")
        finally:
            del app.dependency_overrides[get_current_user]

        assert resp.status_code == 200
        body = resp.json()

        # No per-meeting fields anywhere in the response shape.
        assert "sessions" not in body
        assert "meetings" not in body
        assert "id" not in body
        assert "title" not in body

        assert body["total"] == 500
        assert body["this_month"] == 12
        assert body["by_status"] == {"processed": 12, "pending": 12, "failed": 12}
        assert body["by_platform"] == {"meet": 12, "zoom": 12, "teams": 12, "physical": 12}
        assert len(body["per_day"]) == 30
        assert all(day["count"] == 12 for day in body["per_day"])
        assert all(set(day.keys()) == {"date", "count"} for day in body["per_day"])
