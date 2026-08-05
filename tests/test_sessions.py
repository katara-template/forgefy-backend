"""Session endpoint tests.

Endpoint tests mock VoxaService at the module level, so they don't need a
Firestore mock at all. State machine and service logic are covered by
test_state_machine.py and service unit tests.
"""
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

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
        start_time=None,
        end_time=None,
        created_at=datetime.now(UTC),
    )


def _event(session_id: uuid.UUID) -> MeetingEvent:
    return MeetingEvent(
        id=uuid.uuid4(),
        session_id=session_id,
        event_type="state_transition",
        payload={"from": "WAITING", "to": "JOINING"},
        timestamp=datetime.now(UTC),
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


# ── VoxaService end-of-meeting unit tests ─────────────────────────────────────
# The bot-detected meeting end (Recall webhook) and the frontend end button
# must run the same pipeline, and whichever fires first wins without errors.


def _session_doc(session: MeetingSession) -> MagicMock:
    doc = MagicMock()
    doc.exists = True
    doc.id = str(session.id)
    doc.to_dict.return_value = {
        "user_id": str(session.user_id),
        "status": session.status.value,
        "platform": session.platform.value,
        "meeting_url": session.meeting_url,
        "start_time": session.start_time,
        "end_time": session.end_time,
        "created_at": session.created_at,
    }
    return doc


def _mock_db(doc: MagicMock) -> MagicMock:
    db = MagicMock()
    document = db.collection.return_value.document.return_value
    document.get = AsyncMock(return_value=doc)
    document.update = AsyncMock()
    return db


def _service_with_transition(db: MagicMock, session: MeetingSession):
    """VoxaService whose state machine transition returns session as PROCESSING."""
    from app.modules.voxa.service import VoxaService

    service = VoxaService(db)
    ended = MeetingSession(
        id=session.id,
        user_id=session.user_id,
        status=SessionStatus.PROCESSING,
        platform=session.platform,
        meeting_url=session.meeting_url,
        start_time=session.start_time,
        end_time=None,
        created_at=session.created_at,
    )
    service._sm = MagicMock()
    service._sm.transition = AsyncMock(return_value=ended)
    return service


class TestEndSessionFromBot:
    async def test_endable_session_runs_end_pipeline(self) -> None:
        sess = _session(status=SessionStatus.LISTENING)
        service = _service_with_transition(_mock_db(_session_doc(sess)), sess)

        with (
            patch("app.workers.blueprint_worker.generate_blueprint") as mock_bp,
            patch("app.workers.zoom_bot_worker.remove_bot_for_session") as mock_remove,
        ):
            result = await service.end_session_from_bot(sess.id)

        assert result is not None
        assert result.status == SessionStatus.PROCESSING
        assert result.end_time is not None
        mock_bp.apply_async.assert_called_once_with(
            args=[str(sess.id)], queue="meeting.extract"
        )
        # The bot already left the call — no removal needed
        mock_remove.apply_async.assert_not_called()

    async def test_already_ended_session_returns_none(self) -> None:
        sess = _session(status=SessionStatus.PROCESSING)
        service = _service_with_transition(_mock_db(_session_doc(sess)), sess)

        with patch("app.workers.blueprint_worker.generate_blueprint") as mock_bp:
            result = await service.end_session_from_bot(sess.id)

        assert result is None
        mock_bp.apply_async.assert_not_called()

    async def test_missing_session_returns_none(self) -> None:
        from app.modules.voxa.service import VoxaService

        doc = MagicMock()
        doc.exists = False
        service = VoxaService(_mock_db(doc))

        assert await service.end_session_from_bot(uuid.uuid4()) is None


class TestEndSessionService:
    async def test_endable_session_removes_bot_and_enqueues_blueprint(self) -> None:
        sess = _session(status=SessionStatus.LISTENING)
        service = _service_with_transition(_mock_db(_session_doc(sess)), sess)

        with (
            patch("app.workers.blueprint_worker.generate_blueprint") as mock_bp,
            patch("app.workers.zoom_bot_worker.remove_bot_for_session") as mock_remove,
        ):
            result = await service.end_session(sess.id, sess.user_id)

        assert result.status == SessionStatus.PROCESSING
        mock_bp.apply_async.assert_called_once_with(
            args=[str(sess.id)], queue="meeting.extract"
        )
        mock_remove.apply_async.assert_called_once_with(
            args=[str(sess.id)], queue="meeting.audio"
        )

    async def test_already_ended_is_idempotent(self) -> None:
        """If the bot auto-ended the session first, the button is a no-op."""
        sess = _session(status=SessionStatus.PROCESSING)
        service = _service_with_transition(_mock_db(_session_doc(sess)), sess)

        with patch("app.workers.blueprint_worker.generate_blueprint") as mock_bp:
            result = await service.end_session(sess.id, sess.user_id)

        assert result.status == SessionStatus.PROCESSING
        service._sm.transition.assert_not_awaited()
        mock_bp.apply_async.assert_not_called()

    async def test_waiting_session_still_rejected(self) -> None:
        from app.core.exceptions import ValidationError

        sess = _session(status=SessionStatus.WAITING)
        service = _service_with_transition(_mock_db(_session_doc(sess)), sess)

        with pytest.raises(ValidationError):
            await service.end_session(sess.id, sess.user_id)


class TestRegenerateBlueprint:
    """Retrying a failed blueprint reruns aggregation only — the meeting is
    not re-ended, so end_time and the bot lifecycle stay untouched."""

    async def test_failed_session_is_requeued(self) -> None:
        sess = _session(status=SessionStatus.FAILED)
        db = _mock_db(_session_doc(sess))
        service = _service_with_transition(db, sess)

        with patch("app.workers.blueprint_worker.generate_blueprint") as mock_bp:
            result = await service.regenerate_blueprint(sess.id, sess.user_id)

        assert result.status == SessionStatus.PROCESSING
        service._sm.transition.assert_awaited_once_with(sess.id, SessionStatus.PROCESSING)
        mock_bp.apply_async.assert_called_once_with(
            args=[str(sess.id)], queue="meeting.extract"
        )

    async def test_end_time_is_not_touched(self) -> None:
        """The distinction from /end: a retry must not rewrite when the
        meeting actually finished."""
        sess = _session(status=SessionStatus.FAILED)
        db = _mock_db(_session_doc(sess))
        service = _service_with_transition(db, sess)

        with patch("app.workers.blueprint_worker.generate_blueprint"):
            await service.regenerate_blueprint(sess.id, sess.user_id)

        db.collection.return_value.document.return_value.update.assert_not_awaited()

    async def test_no_bot_removal_is_dispatched(self) -> None:
        sess = _session(status=SessionStatus.FAILED)
        service = _service_with_transition(_mock_db(_session_doc(sess)), sess)

        with (
            patch("app.workers.blueprint_worker.generate_blueprint"),
            patch("app.workers.zoom_bot_worker.remove_bot_for_session") as mock_remove,
        ):
            await service.regenerate_blueprint(sess.id, sess.user_id)

        mock_remove.apply_async.assert_not_called()

    @pytest.mark.parametrize(
        "status",
        [
            SessionStatus.WAITING,
            SessionStatus.LISTENING,
            SessionStatus.PROCESSING,
            SessionStatus.BLUEPRINT_READY,
            SessionStatus.BUILDING,
        ],
    )
    async def test_rejects_sessions_that_have_not_failed(
        self, status: SessionStatus
    ) -> None:
        from app.core.exceptions import ValidationError

        sess = _session(status=status)
        service = _service_with_transition(_mock_db(_session_doc(sess)), sess)

        with pytest.raises(ValidationError):
            await service.regenerate_blueprint(sess.id, sess.user_id)

    async def test_another_users_session_is_not_found(self) -> None:
        """Ownership is enforced before status, so a stranger cannot even
        learn whether the session failed."""
        from app.core.exceptions import NotFoundError

        sess = _session(status=SessionStatus.FAILED)
        service = _service_with_transition(_mock_db(_session_doc(sess)), sess)

        with pytest.raises(NotFoundError):
            await service.regenerate_blueprint(sess.id, uuid.uuid4())

    async def test_missing_session_is_not_found(self) -> None:
        from app.core.exceptions import NotFoundError

        doc = MagicMock()
        doc.exists = False
        service = _service_with_transition(_mock_db(doc), _session())

        with pytest.raises(NotFoundError):
            await service.regenerate_blueprint(uuid.uuid4(), uuid.uuid4())
