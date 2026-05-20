"""VoxaService — meeting session lifecycle orchestration."""
import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError, ValidationError
from app.db.models.enums import Platform, SessionStatus
from app.db.models.meeting_event import MeetingEvent
from app.db.models.meeting_session import MeetingSession
from app.modules.voxa.state_machine import MeetingStateMachine

logger = logging.getLogger(__name__)


class VoxaService:
    """Orchestrates session lifecycle for the Voxa meeting-mode module."""

    def __init__(self, db: AsyncSession) -> None:
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
        session = MeetingSession(
            user_id=user_id,
            platform=platform,
            meeting_url=meeting_url,
            status=SessionStatus.WAITING,
        )
        self._db.add(session)
        await self._db.flush()  # populate session.id

        self._db.add(
            MeetingEvent(
                session_id=session.id,
                event_type="session.created",
                payload={"platform": platform.value, "meeting_url": meeting_url},
            )
        )
        logger.info("Session created: id=%s platform=%s", session.id, platform.value)
        return session

    async def join_session(
        self,
        session_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> MeetingSession:
        """Transition to JOINING and schedule the platform connector bot.

        For physical sessions the bot is not dispatched — the browser streams
        audio directly via WebSocket instead.
        """
        await self._require_owned(session_id, user_id)

        session = await self._sm.transition(
            session_id,
            SessionStatus.JOINING,
            extra_payload={"initiated_by": str(user_id)},
        )
        session.start_time = datetime.now(timezone.utc)

        self._db.add(
            MeetingEvent(
                session_id=session_id,
                event_type="access",
                payload={"action": "join", "user_id": str(user_id)},
            )
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
        """Transition a physical session to LISTENING so the browser can stream audio.

        Only valid for physical sessions in WAITING state.
        """
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
        session.start_time = datetime.now(timezone.utc)
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
        session.end_time = datetime.now(timezone.utc)

        # Remove Recall.ai bot if one exists for this session (non-blocking task)
        if session.platform != Platform.PHYSICAL:
            from app.workers.connector_worker import recall_remove_bot
            recall_remove_bot.apply_async(args=[str(session_id)], queue="meeting.audio")

        from app.workers.blueprint_worker import generate_blueprint
        generate_blueprint.apply_async(
            args=[str(session_id)],
            queue="meeting.extract",
        )
        return session

    async def get_session(
        self,
        session_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> tuple[MeetingSession, list[MeetingEvent]]:
        """Return session + 20 most-recent events; log the access."""
        session = await self._require_owned(session_id, user_id)

        events_result = await self._db.execute(
            select(MeetingEvent)
            .where(MeetingEvent.session_id == session_id)
            .order_by(MeetingEvent.timestamp.desc())
            .limit(20)
        )
        events = list(events_result.scalars())

        # Security requirement: log every read access
        self._db.add(
            MeetingEvent(
                session_id=session_id,
                event_type="access",
                payload={"action": "read", "user_id": str(user_id)},
            )
        )
        return session, events

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _require_owned(
        self,
        session_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> MeetingSession:
        """Return the session only if it exists and is owned by user_id.

        Returns the same 404 for missing *or* unowned sessions to avoid
        leaking existence of other users' sessions.
        """
        result = await self._db.execute(
            select(MeetingSession).where(
                MeetingSession.id == session_id,
                MeetingSession.user_id == user_id,
            )
        )
        session = result.scalar_one_or_none()
        if session is None:
            raise NotFoundError(f"Session {session_id} not found")
        return session
