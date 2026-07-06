"""Tests for chat_with_project's structured clarify_options contract.

Every clarifying question during an update must be a real choice the user taps
(yes/no or exactly three options) — never free text. See the "ASKING CLARIFYING
QUESTIONS" block added to chat_with_project's system prompt.
"""
import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from httpx import AsyncClient


def _project_doc(**overrides) -> MagicMock:
    from tests.conftest import make_doc_snapshot

    now = datetime.now(UTC)
    data = {
        "owner_id": "",
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


class TestChatClarifyOptions:
    async def test_three_option_clarify_passed_through(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        mock_db.collection.return_value.document.return_value.get.return_value = (
            _project_doc(owner_id=str(test_user.id))
        )
        ai_json = (
            '{"type": "clarify", "response": "How should the list be laid out?", '
            '"clarify_options": ["Grid", "List", "Cards"]}'
        )
        with (
            patch("app.config.get_settings", return_value=_fake_settings()),
            patch("anthropic.Anthropic") as mock_cls,
            patch("app.workers.update_worker.apply_update.apply_async") as mock_dispatch,
        ):
            mock_cls.return_value.messages.create.return_value = _anthropic_response(ai_json)
            resp = await auth_client.post(
                f"/api/v1/projects/{uuid.uuid4()}/chat",
                json={"message": "make it better"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "clarify"
        assert data["clarify_options"] == ["Grid", "List", "Cards"]
        assert data["update_queued"] is False
        mock_dispatch.assert_not_called()

    async def test_yes_no_clarify_passed_through(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        mock_db.collection.return_value.document.return_value.get.return_value = (
            _project_doc(owner_id=str(test_user.id))
        )
        ai_json = (
            '{"type": "clarify", "response": "Should this apply to all screens?", '
            '"clarify_options": ["Yes", "No"]}'
        )
        with (
            patch("app.config.get_settings", return_value=_fake_settings()),
            patch("anthropic.Anthropic") as mock_cls,
        ):
            mock_cls.return_value.messages.create.return_value = _anthropic_response(ai_json)
            resp = await auth_client.post(
                f"/api/v1/projects/{uuid.uuid4()}/chat",
                json={"message": "make it better"},
            )

        assert resp.status_code == 200
        assert resp.json()["clarify_options"] == ["Yes", "No"]

    async def test_malformed_options_are_dropped_not_broken(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        """If the classifier doesn't comply (wrong count, non-strings), fall back
        to a plain clarify message rather than rendering a broken button row."""
        mock_db.collection.return_value.document.return_value.get.return_value = (
            _project_doc(owner_id=str(test_user.id))
        )
        ai_json = (
            '{"type": "clarify", "response": "What do you mean exactly?", '
            '"clarify_options": ["Only one option"]}'
        )
        with (
            patch("app.config.get_settings", return_value=_fake_settings()),
            patch("anthropic.Anthropic") as mock_cls,
        ):
            mock_cls.return_value.messages.create.return_value = _anthropic_response(ai_json)
            resp = await auth_client.post(
                f"/api/v1/projects/{uuid.uuid4()}/chat",
                json={"message": "make it better"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "clarify"
        assert data["clarify_options"] is None

    async def test_update_intent_never_carries_clarify_options(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        mock_db.collection.return_value.document.return_value.get.return_value = (
            _project_doc(owner_id=str(test_user.id))
        )
        ai_json = (
            '{"type": "update", "response": "Adding it now!", '
            '"update_prompt": "Add a settings page", '
            '"clarify_options": ["Should", "Not", "Appear"]}'
        )
        with (
            patch("app.config.get_settings", return_value=_fake_settings()),
            patch("anthropic.Anthropic") as mock_cls,
            patch("app.workers.update_worker.apply_update.apply_async") as mock_dispatch,
        ):
            mock_cls.return_value.messages.create.return_value = _anthropic_response(ai_json)
            resp = await auth_client.post(
                f"/api/v1/projects/{uuid.uuid4()}/chat",
                json={"message": "add a settings page"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "update"
        assert data["update_queued"] is True
        mock_dispatch.assert_called_once()

    async def test_needs_database_response_has_no_clarify_options(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        mock_db.collection.return_value.document.return_value.get.return_value = (
            _project_doc(owner_id=str(test_user.id))
        )
        ai_json = (
            '{"type": "clarify", "response": "Want to connect a database first?", '
            '"needs_database": true, "clarify_options": ["Yes", "No"]}'
        )
        with (
            patch("app.config.get_settings", return_value=_fake_settings()),
            patch("anthropic.Anthropic") as mock_cls,
        ):
            mock_cls.return_value.messages.create.return_value = _anthropic_response(ai_json)
            resp = await auth_client.post(
                f"/api/v1/projects/{uuid.uuid4()}/chat",
                json={"message": "let users save their orders"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["needs_database"] is True
        # The DB-specific flow owns its own fixed Yes/No UI — clarify_options must
        # stay unset so the frontend doesn't render a second, redundant button row.
        assert data["clarify_options"] is None
