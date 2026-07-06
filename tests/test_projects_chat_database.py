"""Tests for chat_with_project's needs_database consent gate.

The chat classifier (app/api/v1/projects.py::chat_with_project) must never queue
an update that would need a database without asking first — see the "DATABASE
CONSENT" block added to its system prompt.
"""
import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from httpx import AsyncClient


def _project_doc(**overrides) -> MagicMock:
    from tests.conftest import make_doc_snapshot

    now = datetime.now(UTC)
    data = {
        "owner_id": "",  # patched to test_user.id by callers via _for_user
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


def _fake_settings(**overrides) -> MagicMock:
    base = dict(
        BUILD_MODEL="claude",
        ANTHROPIC_API_KEY="fake-api-key",
        ANTHROPIC_MODEL="claude-test",
        REDIS_URL="redis://localhost:6379/99",
    )
    base.update(overrides)
    return MagicMock(**base)


def _anthropic_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.content = [MagicMock(text=text)]
    return resp


class TestChatNeedsDatabase:
    async def test_needs_database_true_stays_clarify_no_update_queued(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        mock_db.collection.return_value.document.return_value.get.return_value = (
            _project_doc(owner_id=str(test_user.id))
        )
        ai_json = (
            '{"type": "clarify", "response": "This needs somewhere to save orders. '
            'Want to connect a database first?", "needs_database": true}'
        )
        with (
            patch("app.config.get_settings", return_value=_fake_settings()),
            patch("anthropic.Anthropic") as mock_cls,
            patch("app.workers.update_worker.apply_update.apply_async") as mock_dispatch,
        ):
            mock_cls.return_value.messages.create.return_value = _anthropic_response(ai_json)
            resp = await auth_client.post(
                f"/api/v1/projects/{uuid.uuid4()}/chat",
                json={"message": "let users save their orders"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["needs_database"] is True
        assert data["type"] == "clarify"
        assert data["update_queued"] is False
        mock_dispatch.assert_not_called()

    async def test_needs_database_false_for_normal_update(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        mock_db.collection.return_value.document.return_value.get.return_value = (
            _project_doc(owner_id=str(test_user.id))
        )
        ai_json = (
            '{"type": "update", "response": "Adding dark mode now!", '
            '"update_prompt": "Add a dark mode toggle to settings", "needs_database": false}'
        )
        with (
            patch("app.config.get_settings", return_value=_fake_settings()),
            patch("anthropic.Anthropic") as mock_cls,
            patch("app.workers.update_worker.apply_update.apply_async") as mock_dispatch,
        ):
            mock_cls.return_value.messages.create.return_value = _anthropic_response(ai_json)
            resp = await auth_client.post(
                f"/api/v1/projects/{uuid.uuid4()}/chat",
                json={"message": "add dark mode"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["needs_database"] is False
        assert data["update_queued"] is True
        mock_dispatch.assert_called_once()

    async def test_needs_database_true_defensively_overrides_update_intent(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        """Even if the classifier mistakenly pairs needs_database with type=update,
        the endpoint must never queue an update without consent."""
        mock_db.collection.return_value.document.return_value.get.return_value = (
            _project_doc(owner_id=str(test_user.id))
        )
        ai_json = (
            '{"type": "update", "response": "Sure, adding it now!", '
            '"update_prompt": "Add saved orders", "needs_database": true}'
        )
        with (
            patch("app.config.get_settings", return_value=_fake_settings()),
            patch("anthropic.Anthropic") as mock_cls,
            patch("app.workers.update_worker.apply_update.apply_async") as mock_dispatch,
        ):
            mock_cls.return_value.messages.create.return_value = _anthropic_response(ai_json)
            resp = await auth_client.post(
                f"/api/v1/projects/{uuid.uuid4()}/chat",
                json={"message": "let users save their orders"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["needs_database"] is True
        assert data["type"] == "clarify"
        assert data["update_queued"] is False
        mock_dispatch.assert_not_called()
