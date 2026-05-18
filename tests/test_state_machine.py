"""MeetingStateMachine unit tests — pure logic, no HTTP."""
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.exceptions import InvalidStateTransition, NotFoundError
from app.db.models.enums import Platform, SessionStatus
from app.db.models.meeting_session import MeetingSession
from app.modules.voxa.state_machine import MeetingStateMachine


# ── Helpers ───────────────────────────────────────────────────────────────────


def _session(status: SessionStatus) -> MeetingSession:
    return MeetingSession(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        status=status,
        platform=Platform.MEET,
    )


def _db_with(session: MeetingSession | None) -> AsyncMock:
    """Mock AsyncSession whose execute() returns a result wrapping session."""
    db = AsyncMock()
    db.add = MagicMock()  # Session.add is synchronous
    result = MagicMock()
    result.scalar_one_or_none.return_value = session
    db.execute.return_value = result
    return db


# ── Transition tests ──────────────────────────────────────────────────────────


class TestTransition:
    async def test_waiting_to_joining(self) -> None:
        s = _session(SessionStatus.WAITING)
        sm = MeetingStateMachine(_db_with(s))

        result = await sm.transition(s.id, SessionStatus.JOINING)

        assert result.status == SessionStatus.JOINING
        sm._db.add.assert_called_once()  # event logged

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
            s = _session(current)
            result = await MeetingStateMachine(_db_with(s)).transition(s.id, nxt)
            assert result.status == nxt, f"Failed {current} → {nxt}"

    async def test_cancellation_joining_to_processing(self) -> None:
        """JOINING → PROCESSING is the cancellation path."""
        s = _session(SessionStatus.JOINING)
        result = await MeetingStateMachine(_db_with(s)).transition(
            s.id, SessionStatus.PROCESSING
        )
        assert result.status == SessionStatus.PROCESSING

    async def test_invalid_transition_raises(self) -> None:
        s = _session(SessionStatus.WAITING)
        with pytest.raises(InvalidStateTransition):
            await MeetingStateMachine(_db_with(s)).transition(
                s.id, SessionStatus.BUILDING
            )

    async def test_backward_transition_raises(self) -> None:
        s = _session(SessionStatus.PROCESSING)
        with pytest.raises(InvalidStateTransition):
            await MeetingStateMachine(_db_with(s)).transition(
                s.id, SessionStatus.WAITING
            )

    async def test_terminal_state_raises(self) -> None:
        s = _session(SessionStatus.BUILDING)
        with pytest.raises(InvalidStateTransition):
            await MeetingStateMachine(_db_with(s)).transition(
                s.id, SessionStatus.APPROVED
            )

    async def test_not_found_raises(self) -> None:
        with pytest.raises(NotFoundError):
            await MeetingStateMachine(_db_with(None)).transition(
                uuid.uuid4(), SessionStatus.JOINING
            )

    async def test_extra_payload_included_in_event(self) -> None:
        s = _session(SessionStatus.WAITING)
        db = _db_with(s)
        await MeetingStateMachine(db).transition(
            s.id, SessionStatus.JOINING, extra_payload={"initiated_by": "user-123"}
        )
        # The MeetingEvent passed to db.add should contain the extra payload
        call_args = db.add.call_args[0][0]
        assert call_args.payload["initiated_by"] == "user-123"


# ── Sub-state event tests ─────────────────────────────────────────────────────


def _async_db() -> AsyncMock:
    """AsyncMock with synchronous add() — mirrors real AsyncSession behaviour."""
    db = AsyncMock()
    db.add = MagicMock()
    return db


class TestSegmentEvents:
    async def test_valid_sub_states_log_event(self) -> None:
        for sub in ("FEATURE_FOUND", "QUESTION_FOUND", "CONFLICT_FOUND", "ACTION_ITEM_FOUND"):
            db = _async_db()
            await MeetingStateMachine(db).log_segment_event(
                uuid.uuid4(), sub, {"text": "some content"}
            )
            db.add.assert_called_once()

    async def test_unknown_sub_state_raises(self) -> None:
        with pytest.raises(ValueError):
            await MeetingStateMachine(_async_db()).log_segment_event(
                uuid.uuid4(), "MADE_UP_STATE"
            )
