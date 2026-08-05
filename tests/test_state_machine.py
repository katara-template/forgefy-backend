"""MeetingStateMachine unit tests — pure logic, no HTTP."""
import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from app.core.exceptions import InvalidStateTransition, NotFoundError
from app.db.models.enums import Platform, SessionStatus
from app.modules.voxa.state_machine import MeetingStateMachine
from tests.conftest import make_doc_snapshot, wire_firestore_chain

# ── Helpers ───────────────────────────────────────────────────────────────────


def _fresh_db() -> MagicMock:
    """A standalone Firestore mock, for tests that need several in one function."""
    db = MagicMock()
    wire_firestore_chain(db)
    return db


def _db_with(status: SessionStatus | None, session_id: uuid.UUID, user_id: uuid.UUID) -> MagicMock:
    """A Firestore mock whose sessions/{id} doc reflects the given status (or missing, if None)."""
    db = _fresh_db()
    data = (
        None
        if status is None
        else {
            "user_id": str(user_id),
            "status": status.value,
            "platform": Platform.MEET.value,
            "meeting_url": None,
            "start_time": None,
            "end_time": None,
            "created_at": datetime.now(UTC),
        }
    )
    doc = make_doc_snapshot(data, doc_id=str(session_id))
    db.collection.return_value.document.return_value.get.return_value = doc
    return db


def _events_set_mock(db: MagicMock):
    """The mock for sessions/{id}/events/{event_id}.set(...)."""
    return db.collection.return_value.document.return_value.collection.return_value.document.return_value.set


# ── Transition tests ──────────────────────────────────────────────────────────


class TestTransition:
    async def test_waiting_to_joining(self) -> None:
        session_id, user_id = uuid.uuid4(), uuid.uuid4()
        db = _db_with(SessionStatus.WAITING, session_id, user_id)

        result = await MeetingStateMachine(db).transition(session_id, SessionStatus.JOINING)

        assert result.status == SessionStatus.JOINING
        _events_set_mock(db).assert_called_once()  # event logged

    async def test_full_happy_path(self) -> None:
        """Every step in the main progression must succeed."""
        path = [
            (SessionStatus.WAITING, SessionStatus.JOINING),
            (SessionStatus.JOINING, SessionStatus.LISTENING),
            (SessionStatus.LISTENING, SessionStatus.PROCESSING),
            (SessionStatus.PROCESSING, SessionStatus.BLUEPRINT_READY),
            (SessionStatus.BLUEPRINT_READY, SessionStatus.APPROVED),
            (SessionStatus.APPROVED, SessionStatus.BUILDING),
        ]
        for current, nxt in path:
            session_id, user_id = uuid.uuid4(), uuid.uuid4()
            db = _db_with(current, session_id, user_id)
            result = await MeetingStateMachine(db).transition(session_id, nxt)
            assert result.status == nxt, f"Failed {current} → {nxt}"

    async def test_cancellation_joining_to_processing(self) -> None:
        """JOINING → PROCESSING is the cancellation path."""
        session_id, user_id = uuid.uuid4(), uuid.uuid4()
        db = _db_with(SessionStatus.JOINING, session_id, user_id)

        result = await MeetingStateMachine(db).transition(session_id, SessionStatus.PROCESSING)

        assert result.status == SessionStatus.PROCESSING

    async def test_invalid_transition_raises(self) -> None:
        session_id, user_id = uuid.uuid4(), uuid.uuid4()
        db = _db_with(SessionStatus.WAITING, session_id, user_id)

        with pytest.raises(InvalidStateTransition):
            await MeetingStateMachine(db).transition(session_id, SessionStatus.BUILDING)

    async def test_backward_transition_raises(self) -> None:
        session_id, user_id = uuid.uuid4(), uuid.uuid4()
        db = _db_with(SessionStatus.PROCESSING, session_id, user_id)

        with pytest.raises(InvalidStateTransition):
            await MeetingStateMachine(db).transition(session_id, SessionStatus.WAITING)

    async def test_terminal_state_raises(self) -> None:
        session_id, user_id = uuid.uuid4(), uuid.uuid4()
        db = _db_with(SessionStatus.BUILDING, session_id, user_id)

        with pytest.raises(InvalidStateTransition):
            await MeetingStateMachine(db).transition(session_id, SessionStatus.APPROVED)

    async def test_failed_can_retry_into_processing(self) -> None:
        """FAILED is recoverable: the user can rerun blueprint generation
        without starting a new session."""
        session_id, user_id = uuid.uuid4(), uuid.uuid4()
        db = _db_with(SessionStatus.FAILED, session_id, user_id)

        result = await MeetingStateMachine(db).transition(session_id, SessionStatus.PROCESSING)

        assert result.status == SessionStatus.PROCESSING

    async def test_failed_cannot_skip_straight_to_blueprint_ready(self) -> None:
        """Retrying must re-enter PROCESSING so the work actually reruns."""
        session_id, user_id = uuid.uuid4(), uuid.uuid4()
        db = _db_with(SessionStatus.FAILED, session_id, user_id)

        with pytest.raises(InvalidStateTransition):
            await MeetingStateMachine(db).transition(
                session_id, SessionStatus.BLUEPRINT_READY
            )

    async def test_processing_can_fail_again_after_a_retry(self) -> None:
        """A retry that fails must land back in FAILED, so the button returns."""
        session_id, user_id = uuid.uuid4(), uuid.uuid4()
        db = _db_with(SessionStatus.PROCESSING, session_id, user_id)

        result = await MeetingStateMachine(db).transition(session_id, SessionStatus.FAILED)

        assert result.status == SessionStatus.FAILED

    async def test_not_found_raises(self) -> None:
        session_id, user_id = uuid.uuid4(), uuid.uuid4()
        db = _db_with(None, session_id, user_id)

        with pytest.raises(NotFoundError):
            await MeetingStateMachine(db).transition(uuid.uuid4(), SessionStatus.JOINING)

    async def test_extra_payload_included_in_event(self) -> None:
        session_id, user_id = uuid.uuid4(), uuid.uuid4()
        db = _db_with(SessionStatus.WAITING, session_id, user_id)

        await MeetingStateMachine(db).transition(
            session_id, SessionStatus.JOINING, extra_payload={"initiated_by": "user-123"}
        )

        event_data = _events_set_mock(db).call_args[0][0]
        assert event_data["payload"]["initiated_by"] == "user-123"


# ── Sub-state event tests ─────────────────────────────────────────────────────


class TestSegmentEvents:
    async def test_valid_sub_states_log_event(self) -> None:
        for sub in ("FEATURE_FOUND", "QUESTION_FOUND", "CONFLICT_FOUND", "ACTION_ITEM_FOUND"):
            db = _fresh_db()
            await MeetingStateMachine(db).log_segment_event(
                uuid.uuid4(), sub, {"text": "some content"}
            )
            _events_set_mock(db).assert_called_once()

    async def test_unknown_sub_state_raises(self) -> None:
        with pytest.raises(ValueError):
            await MeetingStateMachine(_fresh_db()).log_segment_event(
                uuid.uuid4(), "MADE_UP_STATE"
            )
