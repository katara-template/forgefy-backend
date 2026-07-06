"""Tests for GET /api/v1/admin/analytics — trend, framework performance,
subscriptions, project performance, and conversion funnel. Every figure is an
aggregate; no individual meeting/project/user record is ever returned.
"""
import uuid
from unittest.mock import MagicMock

from httpx import AsyncClient

from app.deps import get_current_user
from app.main import app
from tests.conftest import make_doc_snapshot, wire_firestore_chain


def _project_doc(**overrides) -> MagicMock:
    data = {"template_key": "next", "preview_url": None, "build_error": None}
    data.update(overrides)
    return make_doc_snapshot(data, doc_id=str(uuid.uuid4()))


class TestGetAnalytics:
    async def test_no_auth_returns_401(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/admin/analytics")
        assert resp.status_code == 401

    async def test_non_admin_returns_403(self, auth_client: AsyncClient) -> None:
        resp = await auth_client.get("/api/v1/admin/analytics")
        assert resp.status_code == 403

    async def test_returns_all_sections_with_correct_math(
        self, client: AsyncClient, mock_db: MagicMock, admin_user
    ) -> None:
        app.dependency_overrides[get_current_user] = lambda: admin_user
        try:
            sessions_col = MagicMock()
            wire_firestore_chain(sessions_col)
            sessions_col.count.return_value.get.return_value = [[MagicMock(value=50)]]
            sessions_col.where.return_value.where.return_value.count.return_value.get.return_value = [
                [MagicMock(value=2)]
            ]

            projects_col = MagicMock()
            wire_firestore_chain(projects_col)
            projects_col.count.return_value.get.return_value = [[MagicMock(value=20)]]
            projects_col.where.return_value.where.return_value.count.return_value.get.return_value = [
                [MagicMock(value=3)]
            ]
            projects_col.where.return_value.get.return_value = [
                _project_doc(preview_url="https://x.pages.dev"),
                _project_doc(preview_url="https://y.pages.dev"),
                _project_doc(build_error="boom"),
            ]
            projects_col.where.return_value.count.return_value.get.return_value = [
                [MagicMock(value=7)]
            ]

            users_col = MagicMock()
            wire_firestore_chain(users_col)
            users_col.where.return_value.count.return_value.get.return_value = [
                [MagicMock(value=10)]
            ]

            blueprints_col = MagicMock()
            wire_firestore_chain(blueprints_col)
            blueprints_col.count.return_value.get.return_value = [[MagicMock(value=15)]]
            blueprints_col.where.return_value.count.return_value.get.return_value = [
                [MagicMock(value=8)]
            ]

            mock_db.collection.side_effect = lambda name: {
                "sessions": sessions_col,
                "projects": projects_col,
                "users": users_col,
                "blueprints": blueprints_col,
            }[name]

            resp = await client.get("/api/v1/admin/analytics")
        finally:
            del app.dependency_overrides[get_current_user]

        assert resp.status_code == 200
        body = resp.json()

        # Trend: 30 days, each with the mocked daily counts.
        assert len(body["trend"]) == 30
        assert all(p["meetings"] == 2 and p["apps_built"] == 3 for p in body["trend"])

        # Framework performance: 3 frameworks, each tallied from the same
        # shared doc fetch (2 shipped, 1 failed, out of 3 total).
        assert len(body["frameworks"]) == 3
        for fw in body["frameworks"]:
            assert fw["total"] == 3
            assert fw["shipped"] == 2
            assert fw["failed"] == 1
            assert fw["success_rate"] == 66.7

        # Subscriptions: 4 tiers, all sharing the mocked count of 10.
        sub = body["subscriptions"]
        assert len(sub["by_tier"]) == 4
        assert sub["free_users"] == 10
        assert sub["paid_users"] == 30
        assert sub["conversion_rate"] == 75.0
        assert sub["mrr_usd"] == 10 * (0 + 19 + 49 + 149)

        # Project performance: total=20 (unfiltered), building/shipped/failed=7 (shared where path).
        perf = body["project_performance"]
        assert perf["total"] == 20
        assert perf["building"] == 7
        assert perf["shipped"] == 7
        assert perf["failed"] == 7
        assert perf["success_rate"] == 35.0

        # Funnel: meetings=50, blueprints=15, approved=8, shipped=7 (shared where path).
        funnel = {f["stage"]: f["count"] for f in body["funnel"]}
        assert funnel["Meetings"] == 50
        assert funnel["Blueprints created"] == 15
        assert funnel["Blueprints approved"] == 8
        assert funnel["Apps shipped"] == 7

        # No per-record fields anywhere.
        assert "session_id" not in str(body.keys())
        assert "email" not in body
        assert "title" not in body
