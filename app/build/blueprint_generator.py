"""BlueprintAggregator — assembles extraction events into a structured blueprint.

Reads FEATURE_FOUND / QUESTION_FOUND / CONFLICT_FOUND / ACTION_ITEM_FOUND
events from meeting_events, structures them into a JSON document, saves a
Blueprint row, and transitions the session to BLUEPRINT_READY.
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
    {"FEATURE_FOUND", "QUESTION_FOUND", "CONFLICT_FOUND", "ACTION_ITEM_FOUND"}
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


def _build_blueprint_json(
    session: MeetingSession, events: list[MeetingEvent]
) -> dict[str, Any]:
    features = [e.payload for e in events if e.event_type == "FEATURE_FOUND" and e.payload]
    questions = [e.payload for e in events if e.event_type == "QUESTION_FOUND" and e.payload]
    conflicts = [e.payload for e in events if e.event_type == "CONFLICT_FOUND" and e.payload]
    action_items = [e.payload for e in events if e.event_type == "ACTION_ITEM_FOUND" and e.payload]

    return {
        "version": "1.0",
        "session_id": str(session.id),
        "session_platform": session.platform.value,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "features": features,
        "open_questions": questions,
        "conflicts": conflicts,
        "action_items": action_items,
        "requirements_count": len(features) + len(questions) + len(conflicts) + len(action_items),
    }
