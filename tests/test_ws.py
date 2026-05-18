"""WebSocket gateway tests — JWT auth, event routing, Redis bridge."""
from __future__ import annotations

import json
import uuid
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import get_settings
from app.core.security import create_access_token
from app.db.models.user import User
from app.deps import get_current_user, get_db
from app.main import app


# ── Helpers ───────────────────────────────────────────────────────────────────


def _valid_token(user_id: uuid.UUID | None = None) -> str:
    settings = get_settings()
    return create_access_token(str(user_id or uuid.uuid4()), settings)


def _mock_app():
    """Context manager that patches lifespan DB/Redis init for WS tests."""
    mock_db = AsyncMock()
    mock_db.add = MagicMock()

    async def override_get_db() -> AsyncGenerator:
        yield mock_db

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = lambda: User(
        id=uuid.uuid4(), email="t@t.com", hashed_password="x"
    )
    return mock_db


# ── Auth tests ────────────────────────────────────────────────────────────────


class TestWsAuth:
    def test_ws_route_registered(self) -> None:
        """The /ws/voxa route must exist on the application."""
        paths = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/ws/voxa" in paths

    def test_valid_token_decodes_to_user_id(self) -> None:
        """A token minted with create_access_token round-trips through decode."""
        from app.core.security import decode_access_token

        user_id = uuid.uuid4()
        token = _valid_token(user_id)
        settings = get_settings()
        assert decode_access_token(token, settings) == str(user_id)

    def test_invalid_token_raises_unauthorized(self) -> None:
        """decode_access_token raises UnauthorizedError on a bogus token."""
        from app.core.exceptions import UnauthorizedError
        from app.core.security import decode_access_token

        with pytest.raises(UnauthorizedError):
            decode_access_token("not-a-real-jwt", get_settings())

    async def test_ws_plain_http_get_returns_404(self) -> None:
        """Plain HTTP GET to a WebSocket-only path returns 404.

        Starlette WebSocket routes only match scope["type"] == "websocket";
        HTTP requests to the same path are not routed and return 404.
        """
        _mock_app()
        with (
            patch("app.main.build_engine", return_value=AsyncMock()),
            patch("app.main.build_session_factory", return_value=MagicMock()),
            patch("app.main.aioredis.from_url", return_value=AsyncMock()),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/ws/voxa?token=anything")
                assert resp.status_code == 404
        app.dependency_overrides.clear()


# ── Connection manager unit tests ─────────────────────────────────────────────


class TestConnectionManager:
    async def test_register_and_unregister(self) -> None:
        from app.api.ws.connection_manager import ConnectionManager

        cm = ConnectionManager()
        sid = uuid.uuid4()
        ws = MagicMock()

        cm.register(sid, ws)
        assert ws in cm._connections[sid]

        cm.unregister(sid, ws)
        assert sid not in cm._connections

    async def test_broadcast_sends_to_all(self) -> None:
        from app.api.ws.connection_manager import ConnectionManager

        cm = ConnectionManager()
        sid = uuid.uuid4()
        ws1, ws2 = AsyncMock(), AsyncMock()

        cm.register(sid, ws1)
        cm.register(sid, ws2)
        await cm.broadcast(sid, {"type": "pong"})

        ws1.send_text.assert_called_once_with('{"type": "pong"}')
        ws2.send_text.assert_called_once_with('{"type": "pong"}')

    async def test_broadcast_removes_dead_connections(self) -> None:
        from app.api.ws.connection_manager import ConnectionManager

        cm = ConnectionManager()
        sid = uuid.uuid4()
        dead_ws = AsyncMock()
        dead_ws.send_text.side_effect = RuntimeError("closed")

        cm.register(sid, dead_ws)
        await cm.broadcast(sid, {"type": "ping"})

        assert sid not in cm._connections

    async def test_broadcast_noop_for_unknown_session(self) -> None:
        from app.api.ws.connection_manager import ConnectionManager

        cm = ConnectionManager()
        # Should not raise even with zero subscribers
        await cm.broadcast(uuid.uuid4(), {"type": "pong"})

    async def test_send_to(self) -> None:
        from app.api.ws.connection_manager import ConnectionManager

        cm = ConnectionManager()
        ws = AsyncMock()
        await cm.send_to(ws, {"type": "pong"})
        ws.send_text.assert_called_once_with('{"type": "pong"}')


# ── Schema validation tests ───────────────────────────────────────────────────


class TestWsEventSchemas:
    def test_join_session_parses(self) -> None:
        from pydantic import TypeAdapter

        from app.schemas.ws_events import ClientEvent

        sid = uuid.uuid4()
        adapter = TypeAdapter(ClientEvent)
        event = adapter.validate_json(json.dumps({"type": "joinSession", "session_id": str(sid)}))
        assert event.session_id == sid  # type: ignore[union-attr]

    def test_stream_audio_parses(self) -> None:
        from pydantic import TypeAdapter

        from app.schemas.ws_events import ClientEvent

        adapter = TypeAdapter(ClientEvent)
        raw = json.dumps({"type": "streamAudio", "session_id": str(uuid.uuid4()), "chunk": "abc123"})
        event = adapter.validate_json(raw)
        assert event.chunk == "abc123"  # type: ignore[union-attr]

    def test_ping_parses(self) -> None:
        from pydantic import TypeAdapter

        from app.schemas.ws_events import ClientEvent

        adapter = TypeAdapter(ClientEvent)
        event = adapter.validate_json('{"type": "ping"}')
        assert event.type == "ping"

    def test_unknown_type_raises(self) -> None:
        from pydantic import TypeAdapter, ValidationError

        from app.schemas.ws_events import ClientEvent

        adapter = TypeAdapter(ClientEvent)
        with pytest.raises(ValidationError):
            adapter.validate_json('{"type": "unknown"}')

    def test_server_events_serialize(self) -> None:
        from app.schemas.ws_events import (
            BlueprintReadyEvent,
            FeatureDetectedEvent,
            MeetingStatusEvent,
            PongEvent,
            TranscriptEvent,
        )

        sid = uuid.uuid4()
        assert TranscriptEvent(session_id=sid, text="hello").type == "transcript"
        assert FeatureDetectedEvent(session_id=sid, sub_state="FEATURE_FOUND").type == "featureDetected"
        assert BlueprintReadyEvent(session_id=sid, blueprint_id=uuid.uuid4()).type == "blueprintReady"
        assert MeetingStatusEvent(session_id=sid, status="LISTENING").type == "meetingStatus"
        assert PongEvent().type == "pong"
