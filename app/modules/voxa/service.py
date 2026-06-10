"""VoxaService — meeting session lifecycle orchestration."""
import logging
import uuid
from datetime import datetime, timezone

from google.cloud.firestore import AsyncClient

from app.core.exceptions import NotFoundError, ValidationError
from app.db.models.enums import Platform, SessionStatus
from app.db.models.meeting_event import MeetingEvent
from app.db.models.meeting_session import MeetingSession
from app.modules.voxa.state_machine import MeetingStateMachine

logger = logging.getLogger(__name__)


def _doc_to_session(doc) -> MeetingSession:
    data = doc.to_dict()
    return MeetingSession(
        id=uuid.UUID(doc.id),
        user_id=uuid.UUID(data["user_id"]),
        status=SessionStatus(data["status"]),
        platform=Platform(data["platform"]),
        meeting_url=data.get("meeting_url"),
        start_time=data.get("start_time"),
        end_time=data.get("end_time"),
        created_at=data["created_at"],
    )


def _doc_to_event(doc) -> MeetingEvent:
    data = doc.to_dict()
    return MeetingEvent(
        id=uuid.UUID(doc.id),
        session_id=uuid.UUID(data["session_id"]),
        event_type=data["event_type"],
        payload=data.get("payload"),
        timestamp=data["timestamp"],
    )


class VoxaService:
    """Orchestrates session lifecycle for the Voxa meeting-mode module."""

    def __init__(self, db: AsyncClient) -> None:
        self._db = db
        self._sm = MeetingStateMachine(db)

    # ── Public interface ──────────────────────────────────────────────────────

    async def create_session(
        self,
        user_id: uuid.UUID,
        platform: Platform,
        meeting_url: str | None = None,
    ) -> MeetingSession:
        """Create a session in WAITING state and log the creation event."""
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        session_data = {
            "user_id": str(user_id),
            "status": SessionStatus.WAITING.value,
            "platform": platform.value,
            "meeting_url": meeting_url,
            "start_time": None,
            "end_time": None,
            "created_at": now,
        }
        await self._db.collection("sessions").document(session_id).set(session_data)

        event_id = str(uuid.uuid4())
        await (
            self._db.collection("sessions")
            .document(session_id)
            .collection("events")
            .document(event_id)
            .set({
                "session_id": session_id,
                "event_type": "session.created",
                "payload": {"platform": platform.value, "meeting_url": meeting_url},
                "timestamp": now,
            })
        )

        logger.info("Session created: id=%s platform=%s", session_id, platform.value)
        return MeetingSession(
            id=uuid.UUID(session_id),
            user_id=user_id,
            status=SessionStatus.WAITING,
            platform=platform,
            meeting_url=meeting_url,
            start_time=None,
            end_time=None,
            created_at=now,
        )

    async def join_session(
        self,
        session_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> MeetingSession:
        """Transition to JOINING and schedule the platform connector bot."""
        await self._require_owned(session_id, user_id)

        session = await self._sm.transition(
            session_id,
            SessionStatus.JOINING,
            extra_payload={"initiated_by": str(user_id)},
        )
        start_time = datetime.now(timezone.utc)
        await self._db.collection("sessions").document(str(session_id)).update(
            {"start_time": start_time}
        )
        session.start_time = start_time

        event_id = str(uuid.uuid4())
        await (
            self._db.collection("sessions")
            .document(str(session_id))
            .collection("events")
            .document(event_id)
            .set({
                "session_id": str(session_id),
                "event_type": "access",
                "payload": {"action": "join", "user_id": str(user_id)},
                "timestamp": datetime.now(timezone.utc),
            })
        )

        if session.platform != Platform.PHYSICAL:
            from app.workers.connector_worker import dispatch_connector
            dispatch_connector.apply_async(
                args=[str(session_id), session.platform.value, session.meeting_url],
                queue="meeting.audio",
            )
        return session

    async def start_live_transcription(
        self,
        session_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> MeetingSession:
        """Transition a physical session to LISTENING."""
        session = await self._require_owned(session_id, user_id)

        if session.platform != Platform.PHYSICAL:
            raise ValidationError("Live transcription is only available for physical sessions.")
        if session.status != SessionStatus.WAITING:
            raise ValidationError("Session is not in WAITING state.")

        session = await self._sm.transition(
            session_id,
            SessionStatus.LISTENING,
            extra_payload={"initiated_by": str(user_id)},
        )
        start_time = datetime.now(timezone.utc)
        await self._db.collection("sessions").document(str(session_id)).update(
            {"start_time": start_time}
        )
        session.start_time = start_time
        return session

    async def end_session(
        self,
        session_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> MeetingSession:
        """End the meeting: transition to PROCESSING and record end_time."""
        session = await self._require_owned(session_id, user_id)

        endable = {SessionStatus.JOINING, SessionStatus.LISTENING}
        if session.status not in endable:
            raise ValidationError(
                f"Cannot end a session in '{session.status.value}' state"
            )

        session = await self._sm.transition(session_id, SessionStatus.PROCESSING)
        end_time = datetime.now(timezone.utc)
        await self._db.collection("sessions").document(str(session_id)).update(
            {"end_time": end_time}
        )
        session.end_time = end_time

        if session.platform != Platform.PHYSICAL:
            from app.workers.connector_worker import recall_remove_bot
            recall_remove_bot.apply_async(args=[str(session_id)], queue="meeting.audio")

        from app.workers.blueprint_worker import generate_blueprint
        generate_blueprint.apply_async(
            args=[str(session_id)],
            queue="meeting.extract",
            countdown=20,
        )
        return session

    async def get_session(
        self,
        session_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> tuple[MeetingSession, list[MeetingEvent]]:
        """Return session + 20 most-recent events; log the access."""
        session = await self._require_owned(session_id, user_id)

        event_docs = (
            await self._db.collection("sessions")
            .document(str(session_id))
            .collection("events")
            .order_by("timestamp", direction="DESCENDING")
            .limit(20)
            .get()
        )
        events = [_doc_to_event(d) for d in event_docs]

        event_id = str(uuid.uuid4())
        await (
            self._db.collection("sessions")
            .document(str(session_id))
            .collection("events")
            .document(event_id)
            .set({
                "session_id": str(session_id),
                "event_type": "access",
                "payload": {"action": "read", "user_id": str(user_id)},
                "timestamp": datetime.now(timezone.utc),
            })
        )
        return session, events

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _require_owned(
        self,
        session_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> MeetingSession:
        """Return the session only if it exists and is owned by user_id."""
        doc = await self._db.collection("sessions").document(str(session_id)).get()
        if not doc.exists:
            raise NotFoundError(f"Session {session_id} not found")

        data = doc.to_dict()
        if uuid.UUID(data["user_id"]) != user_id:
            raise NotFoundError(f"Session {session_id} not found")

        return _doc_to_session(doc)
