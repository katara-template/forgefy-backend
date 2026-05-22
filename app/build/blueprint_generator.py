"""BlueprintAggregator — assembles extraction events into a structured blueprint.

Reads APP_DESCRIPTION / FEATURE_FOUND / QUESTION_FOUND / CONFLICT_FOUND /
ACTION_ITEM_FOUND events from meeting_events, structures them into a JSON
document, saves a Blueprint row, and transitions the session to BLUEPRINT_READY.

If no extraction events exist (extraction pipeline failed or produced nothing),
falls back to synthesising from raw transcript.segment events stored by the
extraction worker.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.blueprint import Blueprint
from app.db.models.meeting_event import MeetingEvent
from app.db.models.meeting_session import MeetingSession
from app.db.models.enums import SessionStatus
from app.modules.voxa.state_machine import MeetingStateMachine

logger = logging.getLogger(__name__)

_EXTRACTION_TYPES = frozenset(
    {"APP_DESCRIPTION", "FEATURE_FOUND", "QUESTION_FOUND", "CONFLICT_FOUND", "ACTION_ITEM_FOUND"}
)


class BlueprintAggregator:
    """Queries extraction events and produces a structured blueprint."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db
        self._sm = MeetingStateMachine(db)

    async def generate(self, session_id: uuid.UUID) -> Blueprint:
        """Aggregate extraction events, write a Blueprint row, transition session.

        Returns the created Blueprint object.
        """
        session_result = await self._db.execute(
            select(MeetingSession).where(MeetingSession.id == session_id)
        )
        session = session_result.scalar_one_or_none()
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        # Fetch all extraction events for this session.
        events_result = await self._db.execute(
            select(MeetingEvent)
            .where(
                MeetingEvent.session_id == session_id,
                MeetingEvent.event_type.in_(_EXTRACTION_TYPES),
            )
            .order_by(MeetingEvent.timestamp.asc())
        )
        events = list(events_result.scalars())

        if not events:
            json_output = await self._synthesize_from_segments(session_id, session)
        else:
            json_output = _build_blueprint_json(session, events)

        blueprint = Blueprint(
            session_id=session_id,
            json_output=json_output,
            approved=False,
        )
        self._db.add(blueprint)
        await self._db.flush()  # populate blueprint.id

        await self._sm.transition(
            session_id,
            SessionStatus.BLUEPRINT_READY,
            extra_payload={"blueprint_id": str(blueprint.id)},
        )

        logger.info(
            "Blueprint generated session=%s blueprint=%s features=%d",
            session_id,
            blueprint.id,
            len(json_output.get("features", [])),
        )
        return blueprint

    async def _synthesize_from_segments(
        self,
        session_id: uuid.UUID,
        session: MeetingSession,
    ) -> dict[str, Any]:
        """Fallback: concatenate stored transcript segments and run single-call synthesis."""
        from app.config import get_settings

        segs_result = await self._db.execute(
            select(MeetingEvent)
            .where(
                MeetingEvent.session_id == session_id,
                MeetingEvent.event_type == "transcript.segment",
            )
            .order_by(MeetingEvent.timestamp.asc())
        )
        segments = list(segs_result.scalars())

        if not segments:
            logger.warning("No transcript segments found for session=%s — blueprint will be empty", session_id)
            return _build_blueprint_json(session, [])

        transcript = " ".join(
            s.payload.get("text", "") for s in segments if s.payload
        ).strip()

        if not transcript:
            return _build_blueprint_json(session, [])

        logger.info(
            "Synthesising blueprint from %d segments (%d chars) session=%s",
            len(segments), len(transcript), session_id,
        )
        settings = get_settings()
        try:
            if settings.BP_MODEL == "Qwen3":
                from app.ai.agents.ollama_synthesizer import run as synthesize
                events = synthesize(transcript, settings.OLLAMA_URL, settings.OLLAMA_MODEL, timeout=settings.OLLAMA_TIMEOUT)
            elif settings.BP_MODEL == "gemini":
                from app.ai.agents.gemini_synthesizer import run as synthesize
                events = synthesize(transcript, settings.GEMINI_API_KEY, settings.GEMINI_MODEL)
            else:
                from app.ai.agents.synthesizer import run as synthesize
                events = synthesize(transcript, settings.ANTHROPIC_API_KEY, settings.ANTHROPIC_MODEL)
        except Exception as exc:
            logger.error("Synthesis failed for session=%s: %s", session_id, exc, exc_info=True)
            return _build_blueprint_json(session, [])

        app_desc_event = next((e for e in events if e["sub_state"] == "APP_DESCRIPTION"), None)
        app_description = app_desc_event["payload"].get("text", "") if app_desc_event else ""
        features = [e["payload"] for e in events if e["sub_state"] == "FEATURE_FOUND"]
        questions = [e["payload"] for e in events if e["sub_state"] == "QUESTION_FOUND"]
        conflicts = [e["payload"] for e in events if e["sub_state"] == "CONFLICT_FOUND"]
        action_items = [e["payload"] for e in events if e["sub_state"] == "ACTION_ITEM_FOUND"]

        return {
            "version": "1.0",
            "session_id": str(session_id),
            "session_platform": session.platform.value,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "app_description": app_description,
            "features": features,
            "open_questions": questions,
            "conflicts": conflicts,
            "action_items": action_items,
            "requirements_count": len(features) + len(questions) + len(conflicts) + len(action_items),
        }


def _build_blueprint_json(
    session: MeetingSession, events: list[MeetingEvent]
) -> dict[str, Any]:
    app_desc_event = next(
        (e for e in events if e.event_type == "APP_DESCRIPTION" and e.payload), None
    )
    app_description = app_desc_event.payload.get("text", "") if app_desc_event else ""
    features = [e.payload for e in events if e.event_type == "FEATURE_FOUND" and e.payload]
    questions = [e.payload for e in events if e.event_type == "QUESTION_FOUND" and e.payload]
    conflicts = [e.payload for e in events if e.event_type == "CONFLICT_FOUND" and e.payload]
    action_items = [e.payload for e in events if e.event_type == "ACTION_ITEM_FOUND" and e.payload]

    return {
        "version": "1.0",
        "session_id": str(session.id),
        "session_platform": session.platform.value,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "app_description": app_description,
        "features": features,
        "open_questions": questions,
        "conflicts": conflicts,
        "action_items": action_items,
        "requirements_count": len(features) + len(questions) + len(conflicts) + len(action_items),
    }


