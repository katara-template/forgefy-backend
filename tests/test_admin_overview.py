"""Tests for GET /api/v1/admin/overview — stats + recent activity for the
dashboard's Overview page.
"""
import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock

from httpx import AsyncClient

from app.deps import get_current_user
from app.main import app


def _project_doc(**overrides) -> MagicMock:
    from tests.conftest import make_doc_snapshot

    now = datetime.now(UTC)
    data = {
        "app_name": "stockflow-manager",
        "template_key": "next",
        "is_updating": False,
        "build_error": None,
        "preview_url": None,
        "updated_at": now,
    }
    data.update(overrides)
    return make_doc_snapshot(data, doc_id=str(uuid.uuid4()))


class TestGetOverview:
    async def test_no_auth_returns_401(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/admin/overview")
        assert resp.status_code == 401

    async def test_non_admin_returns_403(self, auth_client: AsyncClient) -> None:
        resp = await auth_client.get("/api/v1/admin/overview")
        assert resp.status_code == 403

    async def test_returns_stats_and_activity(
        self, client: AsyncClient, mock_db: MagicMock, admin_user
    ) -> None:
        app.dependency_overrides[get_current_user] = lambda: admin_user
        try:
            # Unfiltered counts (bare collection.count()) — sessions/projects/users totals.
            mock_db.collection.return_value.count.return_value.get.return_value = [
                [MagicMock(value=100)]
            ]
            # Every filtered count (where(...).count()) shares this same mock path
            # regardless of which field/value was filtered on.
            mock_db.collection.return_value.where.return_value.count.return_value.get.return_value = [
                [MagicMock(value=7)]
            ]
            # Recent activity — one project per derived action/status.
            mock_db.collection.return_value.order_by.return_value.limit.return_value.get.return_value = [
                _project_doc(app_name="failed-app", build_error="boom"),
                _project_doc(app_name="building-app", is_updating=True),
                _project_doc(app_name="shipped-app", preview_url="https://x.pages.dev"),
                _project_doc(app_name="queued-app"),
            ]

            resp = await client.get("/api/v1/admin/overview")
        finally:
            del app.dependency_overrides[get_current_user]

        assert resp.status_code == 200
        body = resp.json()

        stats = body["stats"]
        assert stats["meetings_all_time"] == 100
        assert stats["apps_total"] == 100
        assert stats["active_users"] == 100
        assert stats["meetings_this_month"] == 7
        assert stats["new_signups_week"] == 7
        assert stats["apps_by_framework"] == {"Flutter": 7, "React Native": 7, "Next.js": 7}
        assert stats["pipeline"] == {
            "meetings_in_progress": 7,
            "apps_building": 7,
            "apps_shipped": 7,
        }

        activity = {a["title"]: a for a in body["recent_activity"]}
        assert activity["failed-app"]["action"] == "Build failed"
        assert activity["failed-app"]["status"] == "failed"
        assert activity["building-app"]["action"] == "Build started"
        assert activity["building-app"]["status"] == "building"
        assert activity["shipped-app"]["action"] == "App shipped"
        assert activity["shipped-app"]["status"] == "success"
        assert activity["queued-app"]["action"] == "Build queued"
        assert activity["queued-app"]["status"] == "queued"
        assert activity["shipped-app"]["framework"] == "Next.js"
