"""Session endpoint tests.

Endpoint tests mock VoxaService at the module level so SQLAlchemy column
defaults (UUID PKs) don't interfere with the mock DB.  State machine and
service logic are covered by test_state_machine.py and service unit tests.
"""
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

from app.db.models.enums import Platform, SessionStatus
from app.db.models.meeting_event import MeetingEvent
from app.db.models.meeting_session import MeetingSession


# ── Helpers ───────────────────────────────────────────────────────────────────


def _session(
    user_id: uuid.UUID | None = None,
    status: SessionStatus = SessionStatus.WAITING,
) -> MeetingSession:
    return MeetingSession(
        id=uuid.uuid4(),
        user_id=user_id or uuid.uuid4(),
        status=status,
        platform=Platform.MEET,
        meeting_url="https://meet.google.com/abc",
        created_at=datetime.now(timezone.utc),
    )


def _event(session_id: uuid.UUID) -> MeetingEvent:
    return MeetingEvent(
        id=uuid.uuid4(),
        session_id=session_id,
        event_type="state_transition",
        payload={"from": "WAITING", "to": "JOINING"},
        timestamp=datetime.now(timezone.utc),
    )


# ── /voxa/session/create ──────────────────────────────────────────────────────


class TestCreateSession:
    async def test_success(self, auth_client: AsyncClient, test_user) -> None:
        sess = _session(user_id=test_user.id)

        with patch("app.api.v1.sessions.VoxaService") as MockSvc:
            MockSvc.return_value.create_session = AsyncMock(return_value=sess)
            resp = await auth_client.post(
                "/api/v1/voxa/session/create",
                json={"platform": "meet", "meeting_url": "https://meet.google.com/abc"},
            )

        assert resp.status_code == 201
        body = resp.json()
        assert body["status"] == "WAITING"
        assert body["platform"] == "meet"
        assert body["id"] == str(sess.id)

    async def test_no_auth_returns_401(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/voxa/session/create", json={"platform": "meet"}
        )
        assert resp.status_code == 401

    async def test_invalid_platform_returns_422(self, auth_client: AsyncClient) -> None:
        resp = await auth_client.post(
            "/api/v1/voxa/session/create", json={"platform": "discord"}
        )
        assert resp.status_code == 422


# ── /voxa/session/join ────────────────────────────────────────────────────────


class TestJoinSession:
    async def test_success(self, auth_client: AsyncClient, test_user) -> None:
        sess = _session(user_id=test_user.id, status=SessionStatus.JOINING)

        with patch("app.api.v1.sessions.VoxaService") as MockSvc:
            MockSvc.return_value.join_session = AsyncMock(return_value=sess)
            resp = await auth_client.post(
                "/api/v1/voxa/session/join",
                json={"session_id": str(sess.id)},
            )

        assert resp.status_code == 200
        assert resp.json()["status"] == "JOINING"

    async def test_not_found_returns_404(self, auth_client: AsyncClient) -> None:
        from app.core.exceptions import NotFoundError

        with patch("app.api.v1.sessions.VoxaService") as MockSvc:
            MockSvc.return_value.join_session = AsyncMock(
                side_effect=NotFoundError("not found")
            )
            resp = await auth_client.post(
                "/api/v1/voxa/session/join",
                json={"session_id": str(uuid.uuid4())},
            )

        assert resp.status_code == 404


# ── /voxa/session/end ─────────────────────────────────────────────────────────


class TestEndSession:
    async def test_success_transitions_to_processing(
        self, auth_client: AsyncClient, test_user
    ) -> None:
        sess = _session(user_id=test_user.id, status=SessionStatus.PROCESSING)

        with patch("app.api.v1.sessions.VoxaService") as MockSvc:
            MockSvc.return_value.end_session = AsyncMock(return_value=sess)
            resp = await auth_client.post(
                "/api/v1/voxa/session/end",
                json={"session_id": str(sess.id)},
            )

        assert resp.status_code == 200
        assert resp.json()["status"] == "PROCESSING"

    async def test_invalid_state_returns_422(self, auth_client: AsyncClient) -> None:
        from app.core.exceptions import ValidationError

        with patch("app.api.v1.sessions.VoxaService") as MockSvc:
            MockSvc.return_value.end_session = AsyncMock(
                side_effect=ValidationError("cannot end")
            )
            resp = await auth_client.post(
                "/api/v1/voxa/session/end",
                json={"session_id": str(uuid.uuid4())},
            )

        assert resp.status_code == 422


# ── GET /voxa/session/{id} ────────────────────────────────────────────────────


class TestGetSession:
    async def test_success_with_events(
        self, auth_client: AsyncClient, test_user
    ) -> None:
        sess = _session(user_id=test_user.id)
        events = [_event(sess.id)]

        with patch("app.api.v1.sessions.VoxaService") as MockSvc:
            MockSvc.return_value.get_session = AsyncMock(return_value=(sess, events))
            resp = await auth_client.get(f"/api/v1/voxa/session/{sess.id}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == str(sess.id)
        assert len(body["recent_events"]) == 1
        assert body["recent_events"][0]["event_type"] == "state_transition"

    async def test_not_found_returns_404(self, auth_client: AsyncClient) -> None:
        from app.core.exceptions import NotFoundError

        with patch("app.api.v1.sessions.VoxaService") as MockSvc:
            MockSvc.return_value.get_session = AsyncMock(
                side_effect=NotFoundError("not found")
            )
            resp = await auth_client.get(f"/api/v1/voxa/session/{uuid.uuid4()}")

        assert resp.status_code == 404

    async def test_no_auth_returns_401(self, client: AsyncClient) -> None:
        resp = await client.get(f"/api/v1/voxa/session/{uuid.uuid4()}")
        assert resp.status_code == 401
