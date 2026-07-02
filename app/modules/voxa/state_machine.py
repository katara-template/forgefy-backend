"""Meeting session explicit state machine.

Valid transition graph:
  WAITING → JOINING → LISTENING → PROCESSING → BLUEPRINT_READY → APPROVED → BUILDING

JOINING → PROCESSING is also allowed as a cancellation path when the bot
joins but the meeting never reaches the LISTENING phase.

LISTENING sub-states (FEATURE_FOUND, QUESTION_FOUND, CONFLICT_FOUND,
ACTION_ITEM_FOUND) are logged as events but do not change session.status.
"""
import uuid
from datetime import UTC, datetime

from google.cloud.firestore import AsyncClient

from app.core.exceptions import InvalidStateTransition, NotFoundError
from app.db.models.enums import Platform, SessionStatus
from app.db.models.meeting_session import MeetingSession

_TRANSITIONS: dict[SessionStatus, frozenset[SessionStatus]] = {
    SessionStatus.WAITING: frozenset({SessionStatus.JOINING, SessionStatus.LISTENING, SessionStatus.PROCESSING}),
    SessionStatus.JOINING: frozenset({SessionStatus.LISTENING, SessionStatus.PROCESSING}),
    SessionStatus.LISTENING: frozenset({SessionStatus.PROCESSING}),
    SessionStatus.PROCESSING: frozenset({SessionStatus.BLUEPRINT_READY, SessionStatus.FAILED}),
    SessionStatus.BLUEPRINT_READY: frozenset({SessionStatus.APPROVED}),
    SessionStatus.APPROVED: frozenset({SessionStatus.BUILDING}),
    SessionStatus.BUILDING: frozenset(),
    SessionStatus.FAILED: frozenset(),
}

LISTENING_SUB_STATES: frozenset[str] = frozenset(
    {"FEATURE_FOUND", "QUESTION_FOUND", "CONFLICT_FOUND", "ACTION_ITEM_FOUND"}
)


class MeetingStateMachine:
    """Explicit state machine for meeting session lifecycle."""

    def __init__(self, db: AsyncClient) -> None:
        self._db = db

    async def transition(
        self,
        session_id: uuid.UUID,
        new_status: SessionStatus,
        extra_payload: dict | None = None,
    ) -> MeetingSession:
        """Transition session to new_status and log the event.

        Raises:
            NotFoundError: session does not exist.
            InvalidStateTransition: transition is not in the allowed graph.
        """
        doc = await self._db.collection("sessions").document(str(session_id)).get()
        if not doc.exists:
            raise NotFoundError(f"Session {session_id} not found")

        data = doc.to_dict()
        current = SessionStatus(data["status"])

        if new_status not in _TRANSITIONS.get(current, frozenset()):
            raise InvalidStateTransition(
                f"Cannot transition from '{current.value}' to '{new_status.value}'"
            )

        await self._db.collection("sessions").document(str(session_id)).update(
            {"status": new_status.value}
        )

        payload: dict = {"from": current.value, "to": new_status.value}
        if extra_payload:
            payload.update(extra_payload)

        now = datetime.now(UTC)
        event_id = str(uuid.uuid4())
        await (
            self._db.collection("sessions")
            .document(str(session_id))
            .collection("events")
            .document(event_id)
            .set({
                "session_id": str(session_id),
                "event_type": "state_transition",
                "payload": payload,
                "timestamp": now,
            })
        )

        return MeetingSession(
            id=session_id,
            user_id=uuid.UUID(data["user_id"]),
            status=new_status,
            platform=Platform(data["platform"]),
            meeting_url=data.get("meeting_url"),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            created_at=data["created_at"],
        )

    async def log_segment_event(
        self,
        session_id: uuid.UUID,
        sub_state: str,
        payload: dict | None = None,
    ) -> None:
        """Log a LISTENING-phase sub-state event without changing session status."""
        if sub_state not in LISTENING_SUB_STATES:
            raise ValueError(
                f"Unknown sub-state {sub_state!r}. Must be one of {LISTENING_SUB_STATES}"
            )
        event_id = str(uuid.uuid4())
        await (
            self._db.collection("sessions")
            .document(str(session_id))
            .collection("events")
            .document(event_id)
            .set({
                "session_id": str(session_id),
                "event_type": f"segment.{sub_state.lower()}",
                "payload": payload or {},
                "timestamp": datetime.now(UTC),
            })
        )
