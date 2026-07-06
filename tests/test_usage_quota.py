"""Tests for the synchronous quota gate (check_not_over_limit) — added so a
user already out of tokens gets an immediate rejection from the API instead
of a "queued" response that silently does nothing once a worker picks it up.
"""
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient

from app.core.exceptions import QuotaExceededError
from app.core.usage import check_not_over_limit
from tests.conftest import make_doc_snapshot


class TestCheckNotOverLimit:
    async def test_raises_when_over_limit(self) -> None:
        db = MagicMock()
        with (
            patch("app.core.usage.get_user_tier", new=AsyncMock(return_value="free")),
            patch("app.core.usage.get_monthly_tokens", new=AsyncMock(return_value=600_000)),
            pytest.raises(QuotaExceededError) as exc_info,
        ):
            await check_not_over_limit(db, "user-1")

        assert "Free" in exc_info.value.detail
        assert exc_info.value.status_code == 402

    async def test_no_op_when_under_limit(self) -> None:
        db = MagicMock()
        with (
            patch("app.core.usage.get_user_tier", new=AsyncMock(return_value="free")),
            patch("app.core.usage.get_monthly_tokens", new=AsyncMock(return_value=100)),
        ):
            await check_not_over_limit(db, "user-1")  # must not raise


def _project_doc(owner_id: uuid.UUID, **overrides) -> MagicMock:
    now = datetime.now(UTC)
    data = {
        "owner_id": str(owner_id),
        "app_name": "stockflow-manager",
        "template_key": "next",
        "repo_full_name": "acme/stockflow-manager",
        "github_url": "https://github.com/acme/stockflow-manager",
        "is_updating": False,
        "created_at": now,
        "updated_at": now,
    }
    data.update(overrides)
    return make_doc_snapshot(data, doc_id=str(uuid.uuid4()))


class TestEndpointsRejectOverQuota:
    async def test_chat_with_project_returns_402(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        mock_db.collection.return_value.document.return_value.get.return_value = (
            _project_doc(test_user.id)
        )
        with (
            patch("app.core.usage.get_user_tier", new=AsyncMock(return_value="free")),
            patch("app.core.usage.get_monthly_tokens", new=AsyncMock(return_value=600_000)),
        ):
            resp = await auth_client.post(
                f"/api/v1/projects/{uuid.uuid4()}/chat",
                json={"message": "add dark mode"},
            )
        assert resp.status_code == 402

    async def test_update_project_returns_402(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        mock_db.collection.return_value.document.return_value.get.return_value = (
            _project_doc(test_user.id)
        )
        with (
            patch("app.core.usage.get_user_tier", new=AsyncMock(return_value="free")),
            patch("app.core.usage.get_monthly_tokens", new=AsyncMock(return_value=600_000)),
        ):
            resp = await auth_client.post(
                f"/api/v1/projects/{uuid.uuid4()}/update",
                json={"prompt": "add dark mode"},
            )
        assert resp.status_code == 402

    async def test_approve_blueprint_returns_402(self, auth_client: AsyncClient) -> None:
        from app.db.models.blueprint import Blueprint

        bp = Blueprint(
            id=uuid.uuid4(),
            session_id=uuid.uuid4(),
            json_output={"app_name": "x", "features": []},
            approved=False,
            created_at=datetime.now(UTC),
        )
        with (
            patch("app.api.v1.blueprints._get_owned_blueprint", return_value=bp),
            patch("app.core.usage.get_user_tier", new=AsyncMock(return_value="free")),
            patch("app.core.usage.get_monthly_tokens", new=AsyncMock(return_value=600_000)),
        ):
            resp = await auth_client.post(f"/api/v1/voxa/blueprint/{bp.id}/approve")
        assert resp.status_code == 402
