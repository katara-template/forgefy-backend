"""Tests for GET /api/v1/admin/users — account/billing info + aggregate
counts only. No meeting content of any kind is exposed here.
"""
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import AsyncClient

from app.deps import get_current_user
from app.main import app
from tests.conftest import make_doc_snapshot, wire_firestore_chain


def _user_doc(uid: str, omit: tuple[str, ...] = (), **overrides) -> MagicMock:
    data = {
        "email": "founder@example.com",
        "hashed_password": "irrelevant",
        "tier": "pro",
        "created_at": datetime(2025, 3, 1, tzinfo=UTC),
    }
    data.update(overrides)
    for key in omit:
        data.pop(key, None)
    return make_doc_snapshot(data, doc_id=uid)


class TestListUsers:
    async def test_no_auth_returns_401(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/admin/users")
        assert resp.status_code == 401

    async def test_non_admin_returns_403(self, auth_client: AsyncClient) -> None:
        resp = await auth_client.get("/api/v1/admin/users")
        assert resp.status_code == 403

    async def test_returns_email_tier_joined_and_counts(
        self, client: AsyncClient, mock_db: MagicMock, admin_user
    ) -> None:
        app.dependency_overrides[get_current_user] = lambda: admin_user
        try:
            uid = str(uuid.uuid4())
            users_col = MagicMock()
            wire_firestore_chain(users_col)
            users_col.order_by.return_value.limit.return_value.get.return_value = [
                _user_doc(uid, email="founder@example.com", tier="pro")
            ]

            sessions_col = MagicMock()
            wire_firestore_chain(sessions_col)
            sessions_col.where.return_value.count.return_value.get.return_value = [
                [MagicMock(value=5)]
            ]

            projects_col = MagicMock()
            wire_firestore_chain(projects_col)
            projects_col.where.return_value.count.return_value.get.return_value = [
                [MagicMock(value=2)]
            ]

            mock_db.collection.side_effect = lambda name: {
                "users": users_col,
                "sessions": sessions_col,
                "projects": projects_col,
            }[name]

            resp = await client.get("/api/v1/admin/users")
        finally:
            del app.dependency_overrides[get_current_user]

        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        assert body[0] == {
            "email": "founder@example.com",
            "tier": "pro",
            "joined": "2025-03-01T00:00:00Z",
            "meetings_count": 5,
            "apps_count": 2,
            "tokens_used_this_month": 0,
            "monthly_token_budget": 20_000_000,
        }
        # No per-meeting or per-session identifying fields.
        assert "session_id" not in body[0]
        assert "name" not in body[0]

    async def test_defaults_to_free_tier_when_missing(
        self, client: AsyncClient, mock_db: MagicMock, admin_user
    ) -> None:
        app.dependency_overrides[get_current_user] = lambda: admin_user
        try:
            uid = str(uuid.uuid4())
            users_col = MagicMock()
            wire_firestore_chain(users_col)
            users_col.order_by.return_value.limit.return_value.get.return_value = [
                _user_doc(uid, omit=("tier",))
            ]

            empty_col = MagicMock()
            wire_firestore_chain(empty_col)
            empty_col.where.return_value.count.return_value.get.return_value = [
                [MagicMock(value=0)]
            ]

            mock_db.collection.side_effect = lambda name: {
                "users": users_col,
                "sessions": empty_col,
                "projects": empty_col,
            }[name]

            resp = await client.get("/api/v1/admin/users")
        finally:
            del app.dependency_overrides[get_current_user]

        assert resp.json()[0]["tier"] == "free"

    async def test_surfaces_real_monthly_token_usage(
        self, client: AsyncClient, mock_db: MagicMock, admin_user
    ) -> None:
        """Verify tokens_used_this_month reflects actual usage, not just the
        conftest default of 0 — proves the wiring is real, not coincidental."""
        app.dependency_overrides[get_current_user] = lambda: admin_user
        try:
            uid = str(uuid.uuid4())
            users_col = MagicMock()
            wire_firestore_chain(users_col)
            users_col.order_by.return_value.limit.return_value.get.return_value = [
                _user_doc(uid, tier="free")
            ]

            empty_col = MagicMock()
            wire_firestore_chain(empty_col)
            empty_col.where.return_value.count.return_value.get.return_value = [
                [MagicMock(value=0)]
            ]

            mock_db.collection.side_effect = lambda name: {
                "users": users_col,
                "sessions": empty_col,
                "projects": empty_col,
            }[name]

            with patch(
                "app.core.usage.get_monthly_tokens", new=AsyncMock(return_value=432_000)
            ):
                resp = await client.get("/api/v1/admin/users")
        finally:
            del app.dependency_overrides[get_current_user]

        body = resp.json()[0]
        assert body["tokens_used_this_month"] == 432_000
        assert body["monthly_token_budget"] == 500_000
