"""Tests for the global dashboard help assistant (app/api/v1/assistant.py).

Covers both audiences and the multi-thread model: signed-in users get named
conversation threads (each its own context) plus shared learned memory and the
start_session action; anonymous visitors get stateless advice with any
privileged request gated behind an auth_required prompt.
"""
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from httpx import AsyncClient

from app.ai.openrouter import OpenRouterError
from app.deps import get_optional_user
from app.main import app


@pytest.fixture
async def signed_in(client: AsyncClient, test_user) -> AsyncGenerator[AsyncClient, None]:
    """Client whose optional-auth dependency resolves to test_user.

    The endpoint uses OptionalUser (get_optional_user), so — unlike auth_client,
    which overrides get_current_user — we override that dependency directly.
    """
    app.dependency_overrides[get_optional_user] = lambda: test_user
    yield client
    app.dependency_overrides.pop(get_optional_user, None)


def _doc(mock_db: MagicMock, data: dict | None, doc_id: str = "c1") -> None:
    from tests.conftest import make_doc_snapshot

    mock_db.collection.return_value.document.return_value.get.return_value = make_doc_snapshot(
        data, doc_id=doc_id
    )


def _query_result(mock_db: MagicMock, snapshots: list) -> None:
    mock_db.collection.return_value.where.return_value.get.return_value = snapshots


def _set_call(mock_db: MagicMock):
    return mock_db.collection.return_value.document.return_value.set


def _delete_call(mock_db: MagicMock):
    return mock_db.collection.return_value.document.return_value.delete


class TestSignedInChat:
    async def test_reply_and_valid_links_pass_through(
        self, signed_in: AsyncClient, mock_db: MagicMock
    ) -> None:
        _doc(mock_db, None)
        ai = {
            "response": "You can upgrade your plan from Billing.",
            "links": [{"label": "Open Billing", "to": "/billing"}],
        }
        with patch("app.api.v1.assistant.call_openrouter", return_value=ai) as mock_call:
            resp = await signed_in.post(
                "/api/v1/assistant/chat",
                json={"message": "how do I upgrade?", "page": "/dashboard"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] == "You can upgrade your plan from Billing."
        assert data["links"] == [{"label": "Open Billing", "to": "/billing"}]
        assert data["authenticated"] is True
        mock_call.assert_called_once()

    async def test_new_thread_is_created_and_id_returned(
        self, signed_in: AsyncClient, mock_db: MagicMock
    ) -> None:
        _doc(mock_db, None)
        with patch("app.api.v1.assistant.call_openrouter", return_value={"response": "Hi!"}):
            resp = await signed_in.post("/api/v1/assistant/chat", json={"message": "hello"})

        assert resp.status_code == 200
        assert resp.json()["conversation_id"]  # a fresh thread id was minted
        _set_call(mock_db).assert_called()

    async def test_continue_existing_thread(
        self, signed_in: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        _doc(
            mock_db,
            {
                "owner_id": str(test_user.id),
                "title": "Billing help",
                "messages": [{"role": "user", "text": "earlier question"}],
            },
        )
        with patch("app.api.v1.assistant.call_openrouter", return_value={"response": "Sure."}):
            resp = await signed_in.post(
                "/api/v1/assistant/chat",
                json={"message": "follow up", "conversation_id": "c1"},
            )

        assert resp.status_code == 200
        assert resp.json()["conversation_id"] == "c1"

    async def test_thread_ownership_enforced(
        self, signed_in: AsyncClient, mock_db: MagicMock
    ) -> None:
        _doc(mock_db, {"owner_id": "someone-else", "messages": []})
        with patch("app.api.v1.assistant.call_openrouter", return_value={"response": "x"}):
            resp = await signed_in.post(
                "/api/v1/assistant/chat",
                json={"message": "hi", "conversation_id": "c1"},
            )

        assert resp.status_code == 403

    async def test_offsite_and_unknown_links_are_stripped(
        self, signed_in: AsyncClient, mock_db: MagicMock
    ) -> None:
        _doc(mock_db, None)
        ai = {
            "response": "Here you go.",
            "links": [
                {"label": "Phish", "to": "https://evil.example.com"},
                {"label": "Nope", "to": "/does-not-exist"},
                {"label": "Sessions", "to": "/sessions"},
            ],
        }
        with patch("app.api.v1.assistant.call_openrouter", return_value=ai):
            resp = await signed_in.post(
                "/api/v1/assistant/chat", json={"message": "show my sessions"}
            )

        assert resp.status_code == 200
        assert resp.json()["links"] == [{"label": "Sessions", "to": "/sessions"}]

    async def test_start_session_action_passes_through(
        self, signed_in: AsyncClient, mock_db: MagicMock
    ) -> None:
        _doc(mock_db, None)
        ai = {
            "response": "Starting a Zoom session for you.",
            "action": {
                "type": "start_session",
                "platform": "zoom",
                "meeting_url": "https://zoom.us/j/123",
            },
        }
        with patch("app.api.v1.assistant.call_openrouter", return_value=ai):
            resp = await signed_in.post(
                "/api/v1/assistant/chat",
                json={"message": "start a zoom session https://zoom.us/j/123"},
            )

        assert resp.status_code == 200
        action = resp.json()["action"]
        assert action["type"] == "start_session"
        assert action["platform"] == "zoom"
        assert action["meeting_url"] == "https://zoom.us/j/123"

    async def test_physical_session_starts_without_url(
        self, signed_in: AsyncClient, mock_db: MagicMock
    ) -> None:
        _doc(mock_db, None)
        ai = {
            "response": "Starting a physical session.",
            "action": {"type": "start_session", "platform": "physical"},
        }
        with patch("app.api.v1.assistant.call_openrouter", return_value=ai):
            resp = await signed_in.post(
                "/api/v1/assistant/chat", json={"message": "start an in-person session"}
            )

        assert resp.status_code == 200
        assert resp.json()["action"]["platform"] == "physical"

    async def test_online_session_without_url_is_not_started(
        self, signed_in: AsyncClient, mock_db: MagicMock
    ) -> None:
        _doc(mock_db, None)
        ai = {
            "response": "What's the meeting link?",
            "action": {"type": "start_session", "platform": "meet"},
        }
        with patch("app.api.v1.assistant.call_openrouter", return_value=ai):
            resp = await signed_in.post(
                "/api/v1/assistant/chat", json={"message": "start a meet session"}
            )

        assert resp.status_code == 200
        assert resp.json()["action"] is None

    async def test_learned_fact_is_persisted_to_memory(
        self, signed_in: AsyncClient, mock_db: MagicMock
    ) -> None:
        _doc(mock_db, None)
        ai = {"response": "Nice!", "remember": ["Building a Flutter fitness app"]}
        with patch("app.api.v1.assistant.call_openrouter", return_value=ai):
            resp = await signed_in.post(
                "/api/v1/assistant/chat",
                json={"message": "I'm building a fitness app in Flutter"},
            )

        assert resp.status_code == 200
        # Memory is the last write (after the thread), so call_args is the memory doc.
        saved = _set_call(mock_db).call_args.args[0]
        assert "Building a Flutter fitness app" in saved["memory"]

    async def test_model_failure_degrades_gracefully(
        self, signed_in: AsyncClient, mock_db: MagicMock
    ) -> None:
        _doc(mock_db, None)
        with patch(
            "app.api.v1.assistant.call_openrouter",
            side_effect=OpenRouterError("all models down"),
        ):
            resp = await signed_in.post("/api/v1/assistant/chat", json={"message": "hello?"})

        assert resp.status_code == 200
        assert "trouble" in resp.json()["response"].lower()

    async def test_build_app_action_passes_through_with_description(
        self, signed_in: AsyncClient, mock_db: MagicMock
    ) -> None:
        _doc(mock_db, None)
        ai = {
            "response": "Building your fitness app now.",
            "action": {
                "type": "build_app",
                "description": "A fitness tracking app for logging workouts and progress.",
            },
        }
        with patch("app.api.v1.assistant.call_openrouter", return_value=ai):
            resp = await signed_in.post(
                "/api/v1/assistant/chat", json={"message": "build me a fitness app"}
            )

        assert resp.status_code == 200
        action = resp.json()["action"]
        assert action["type"] == "build_app"
        assert "fitness" in action["description"]

    async def test_build_app_without_description_is_dropped(
        self, signed_in: AsyncClient, mock_db: MagicMock
    ) -> None:
        _doc(mock_db, None)
        ai = {"response": "What should the app do?", "action": {"type": "build_app"}}
        with patch("app.api.v1.assistant.call_openrouter", return_value=ai):
            resp = await signed_in.post(
                "/api/v1/assistant/chat", json={"message": "build me an app"}
            )

        assert resp.status_code == 200
        assert resp.json()["action"] is None


class TestBuildEndpoint:
    async def test_build_creates_session_project_and_dispatches(
        self, signed_in: AsyncClient, mock_db: MagicMock
    ) -> None:
        with patch("app.api.v1.assistant.dispatch") as mock_dispatch:
            resp = await signed_in.post(
                "/api/v1/assistant/build",
                json={"description": "A recipe box app to save and organize recipes."},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["project_id"]
        assert data["session_id"]
        # Session + project stubs written, worker dispatched.
        _set_call(mock_db).assert_called()
        mock_dispatch.assert_awaited_once()
        # Dispatched with (session_id, project_id, description, user_id).
        args = mock_dispatch.await_args.kwargs.get("args") or mock_dispatch.await_args.args[1]
        assert data["session_id"] in args and data["project_id"] in args

    async def test_build_rejects_too_short_description(
        self, signed_in: AsyncClient, mock_db: MagicMock
    ) -> None:
        with patch("app.api.v1.assistant.dispatch") as mock_dispatch:
            resp = await signed_in.post("/api/v1/assistant/build", json={"description": "hi"})

        assert resp.status_code == 422
        mock_dispatch.assert_not_called()

    async def test_build_requires_auth(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/assistant/build",
            json={"description": "A note-taking app with tags and search."},
        )
        assert resp.status_code == 401


class TestConversations:
    async def test_list_returns_user_threads(
        self, signed_in: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        from tests.conftest import make_doc_snapshot

        now = datetime.now(UTC)
        snap = make_doc_snapshot(
            {"owner_id": str(test_user.id), "title": "Billing help", "updated_at": now},
            doc_id="c1",
        )
        _query_result(mock_db, [snap])
        resp = await signed_in.get("/api/v1/assistant/conversations")

        assert resp.status_code == 200
        convs = resp.json()["conversations"]
        assert len(convs) == 1
        assert convs[0]["id"] == "c1"
        assert convs[0]["title"] == "Billing help"

    async def test_get_conversation_returns_messages(
        self, signed_in: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        msgs = [{"role": "user", "text": "hi"}, {"role": "assistant", "text": "hey!"}]
        _doc(mock_db, {"owner_id": str(test_user.id), "title": "Chat", "messages": msgs})
        resp = await signed_in.get("/api/v1/assistant/conversations/c1")

        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "c1"
        assert data["messages"] == msgs

    async def test_get_conversation_ownership_enforced(
        self, signed_in: AsyncClient, mock_db: MagicMock
    ) -> None:
        _doc(mock_db, {"owner_id": "someone-else", "messages": []})
        resp = await signed_in.get("/api/v1/assistant/conversations/c1")
        assert resp.status_code == 403

    async def test_delete_conversation(
        self, signed_in: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        _doc(mock_db, {"owner_id": str(test_user.id), "messages": []})
        resp = await signed_in.delete("/api/v1/assistant/conversations/c1")

        assert resp.status_code == 200
        assert resp.json() == {"deleted": True}
        _delete_call(mock_db).assert_called()

    async def test_anonymous_conversation_list_is_empty(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/assistant/conversations")
        assert resp.status_code == 200
        assert resp.json()["conversations"] == []


class TestAnonymousChat:
    async def test_informational_reply_works_without_auth(
        self, client: AsyncClient, mock_db: MagicMock
    ) -> None:
        ai = {"response": "Forgefy turns your meetings into apps.", "links": []}
        with patch("app.api.v1.assistant.call_openrouter", return_value=ai) as mock_call:
            resp = await client.post(
                "/api/v1/assistant/chat", json={"message": "what is forgefy?"}
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["authenticated"] is False
        assert data["response"].startswith("Forgefy")
        mock_call.assert_called_once()
        _set_call(mock_db).assert_not_called()  # never persisted

    async def test_privileged_action_is_gated_to_auth_required(
        self, client: AsyncClient, mock_db: MagicMock
    ) -> None:
        ai = {
            "response": "Sign in and I'll start that session.",
            "action": {"type": "start_session", "platform": "meet"},
        }
        with patch("app.api.v1.assistant.call_openrouter", return_value=ai):
            resp = await client.post(
                "/api/v1/assistant/chat", json={"message": "start a session"}
            )

        assert resp.status_code == 200
        assert resp.json()["action"]["type"] == "auth_required"

    async def test_client_history_is_used_as_context(
        self, client: AsyncClient, mock_db: MagicMock
    ) -> None:
        ai = {"response": "Sure."}
        with patch("app.api.v1.assistant.call_openrouter", return_value=ai) as mock_call:
            resp = await client.post(
                "/api/v1/assistant/chat",
                json={
                    "message": "and after that?",
                    "history": [{"role": "user", "text": "how do I start?"}],
                },
            )

        assert resp.status_code == 200
        system_prompt = mock_call.call_args.args[0]
        assert "how do I start?" in system_prompt


class TestShared:
    async def test_empty_message_short_circuits_without_model_call(
        self, client: AsyncClient
    ) -> None:
        with patch("app.api.v1.assistant.call_openrouter") as mock_call:
            resp = await client.post("/api/v1/assistant/chat", json={"message": "   "})

        assert resp.status_code == 200
        assert resp.json()["response"]
        mock_call.assert_not_called()
