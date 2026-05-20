"""Meeting session explicit state machine.

Valid transition graph:
  WAITING → JOINING → LISTENING → PROCESSING → BLUEPRINT_READY → APPROVED → BUILDING

JOINING → PROCESSING is also allowed as a cancellation path when the bot
joins but the meeting never reaches the LISTENING phase.

LISTENING sub-states (FEATURE_FOUND, QUESTION_FOUND, CONFLICT_FOUND,
ACTION_ITEM_FOUND) are logged as meeting_events but do not change
session.status.
"""
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import InvalidStateTransition, NotFoundError
from app.db.models.enums import SessionStatus
from app.db.models.meeting_event import MeetingEvent
from app.db.models.meeting_session import MeetingSession

_TRANSITIONS: dict[SessionStatus, frozenset[SessionStatus]] = {
    SessionStatus.WAITING: frozenset({SessionStatus.JOINING, SessionStatus.LISTENING, SessionStatus.PROCESSING}),
    SessionStatus.JOINING: frozenset({SessionStatus.LISTENING, SessionStatus.PROCESSING}),
    SessionStatus.LISTENING: frozenset({SessionStatus.PROCESSING}),
    SessionStatus.PROCESSING: frozenset({SessionStatus.BLUEPRINT_READY}),
    SessionStatus.BLUEPRINT_READY: frozenset({SessionStatus.APPROVED}),
    SessionStatus.APPROVED: frozenset({SessionStatus.BUILDING}),
    SessionStatus.BUILDING: frozenset(),
}

LISTENING_SUB_STATES: frozenset[str] = frozenset(
    {"FEATURE_FOUND", "QUESTION_FOUND", "CONFLICT_FOUND", "ACTION_ITEM_FOUND"}
)


class MeetingStateMachine:
    """Explicit state machine for meeting session lifecycle.

    The only public state mutator is ``transition()``. Every call writes a
    ``state_transition`` row to ``meeting_events`` before returning.
    """

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def transition(
        self,
        session_id: uuid.UUID,
        new_status: SessionStatus,
        extra_payload: dict | None = None,
    ) -> MeetingSession:
        """Transition *session_id* to *new_status* and log the event.

        Raises:
            NotFoundError: session does not exist.
            InvalidStateTransition: transition is not in the allowed graph.
        """
        result = await self._db.execute(
            select(MeetingSession).where(MeetingSession.id == session_id)
        )
        session = result.scalar_one_or_none()
        if session is None:
            raise NotFoundError(f"Session {session_id} not found")

        current = session.status
        if new_status not in _TRANSITIONS.get(current, frozenset()):
            raise InvalidStateTransition(
                f"Cannot transition from '{current.value}' to '{new_status.value}'"
            )

        session.status = new_status

        payload: dict = {"from": current.value, "to": new_status.value}
        if extra_payload:
            payload.update(extra_payload)

        self._db.add(
            MeetingEvent(
                session_id=session_id,
                event_type="state_transition",
                payload=payload,
            )
        )
        return session

    async def log_segment_event(
        self,
        session_id: uuid.UUID,
        sub_state: str,
        payload: dict | None = None,
    ) -> None:
        """Log a LISTENING-phase sub-state event without changing session status.

        *sub_state* must be one of FEATURE_FOUND, QUESTION_FOUND,
        CONFLICT_FOUND, or ACTION_ITEM_FOUND.
        """
        if sub_state not in LISTENING_SUB_STATES:
            raise ValueError(
                f"Unknown sub-state {sub_state!r}. Must be one of {LISTENING_SUB_STATES}"
            )
        self._db.add(
            MeetingEvent(
                session_id=session_id,
                event_type=f"segment.{sub_state.lower()}",
                payload=payload or {},
            )
        )
