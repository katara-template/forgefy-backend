"""BlueprintAggregator — assembles extraction events into a structured blueprint.

Reads FEATURE_FOUND / QUESTION_FOUND / CONFLICT_FOUND / ACTION_ITEM_FOUND /
APP_DESCRIPTION events from the session's events subcollection, structures them
into a JSON document, saves a Blueprint document, and transitions the session
to BLUEPRINT_READY.

If no extraction events exist, falls back to synthesising from raw
transcript.segment events stored by the extraction worker.

Firestore collections used:
  sessions/{session_id}/events/  — event documents
  blueprints/{blueprint_id}      — blueprint documents (top-level)
"""
from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from google.cloud.firestore import AsyncClient

from app.db.models.blueprint import Blueprint
from app.db.models.enums import SessionStatus, Platform
from app.db.models.meeting_session import MeetingSession
from app.modules.voxa.state_machine import MeetingStateMachine

logger = logging.getLogger(__name__)


def _name_from_description(description: str) -> str:
    """Extract the app name from the first sentence when it follows 'AppName is/helps/...' pattern."""
    match = re.match(
        r'^([A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){0,3})'
        r'\s+(?:is\b|helps\b|allows\b|provides\b|enables\b|aims\b|was\b|will\b|can\b)',
        description.strip(),
    )
    if match:
        candidate = match.group(1).strip()
        words = candidate.split()
        if 1 <= len(words) <= 4:
            return candidate
    return ""


_EXTRACTION_TYPES = frozenset(
    {"APP_DESCRIPTION", "FEATURE_FOUND", "QUESTION_FOUND", "CONFLICT_FOUND", "ACTION_ITEM_FOUND"}
)


class BlueprintAggregator:
    """Queries extraction events and produces a structured blueprint."""

    def __init__(self, db: AsyncClient) -> None:
        self._db = db
        self._sm = MeetingStateMachine(db)

    async def generate(self, session_id: uuid.UUID) -> Blueprint:
        """Aggregate extraction events, write a Blueprint document, transition session."""
        sess_doc = await self._db.collection("sessions").document(str(session_id)).get()
        if not sess_doc.exists:
            raise ValueError(f"Session {session_id} not found")

        sess_data = sess_doc.to_dict()
        session = MeetingSession(
            id=session_id,
            user_id=uuid.UUID(sess_data["user_id"]),
            status=SessionStatus(sess_data["status"]),
            platform=Platform(sess_data["platform"]),
            meeting_url=sess_data.get("meeting_url"),
            start_time=sess_data.get("start_time"),
            end_time=sess_data.get("end_time"),
            created_at=sess_data["created_at"],
        )

        event_docs = (
            await self._db.collection("sessions")
            .document(str(session_id))
            .collection("events")
            .order_by("timestamp", direction="ASCENDING")
            .get()
        )
        extraction_events = [
            d.to_dict() for d in event_docs
            if d.to_dict().get("event_type") in _EXTRACTION_TYPES
        ]

        if not extraction_events:
            json_output = await self._synthesize_from_segments(session_id, session, event_docs)
        else:
            json_output = _build_blueprint_json(session, extraction_events)

        # Reject completely empty blueprints — better to surface the error than save dead data
        if not json_output.get("app_description") and not json_output.get("features"):
            raise ValueError(
                "No requirements could be extracted from this session. "
                "The recording may be too short, silent, or not contain product discussion."
            )

        # If the meeting produced a description but no features, derive them from the description
        if json_output.get("app_description") and not json_output.get("features"):
            logger.info("No features extracted — deriving from description for session=%s", session_id)
            json_output["features"] = await self._derive_features(json_output["app_description"])
            json_output["requirements_count"] = len(json_output["features"])

        # Extract app name: try description first (fast), fall back to AI call
        if not json_output.get("app_name"):
            json_output["app_name"] = (
                _name_from_description(json_output.get("app_description", ""))
                or await self._infer_app_name(json_output)
            )

        blueprint_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        await self._db.collection("blueprints").document(blueprint_id).set({
            "session_id": str(session_id),
            "json_output": json_output,
            "approved": False,
            "created_at": now,
        })

        await self._sm.transition(
            session_id,
            SessionStatus.BLUEPRINT_READY,
            extra_payload={"blueprint_id": blueprint_id},
        )

        logger.info(
            "Blueprint generated session=%s blueprint=%s features=%d",
            session_id,
            blueprint_id,
            len(json_output.get("features", [])),
        )
        return Blueprint(
            id=uuid.UUID(blueprint_id),
            session_id=session_id,
            json_output=json_output,
            approved=False,
            created_at=now,
        )

    async def _derive_features(self, description: str) -> list[dict]:
        """Ask the AI to infer likely features from the app description when none were extracted."""
        from app.config import get_settings
        import anthropic, json as _json

        settings = get_settings()
        try:
            client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
            msg = client.messages.create(
                model=settings.ANTHROPIC_MODEL,
                max_tokens=1024,
                system=(
                    "You are a product analyst. Given an app description, infer the most likely "
                    "features the app needs. Return ONLY a valid JSON array — no markdown, no explanation.\n"
                    'Each item: {"title": "2-6 word name", "description": "1-2 sentence detail", "priority": "high|med|low"}'
                ),
                messages=[{"role": "user", "content": f"App description:\n{description}"}],
            )
            raw = msg.content[0].text.strip()
            features = _json.loads(raw)
            if isinstance(features, list):
                logger.info("Derived %d features from description", len(features))
                return features
        except Exception as exc:
            logger.warning("Feature derivation failed: %s", exc)
        return []

    async def _infer_app_name(self, blueprint: dict[str, Any]) -> str:
        """Ask the AI for a concise, memorable app name based on the blueprint content."""
        from app.config import get_settings
        import anthropic

        description = (blueprint.get("app_description") or "").strip()
        features = blueprint.get("features") or []
        if not description and not features:
            return ""

        feature_titles = ", ".join(f.get("title", "") for f in features[:6] if f.get("title"))
        content = f"Description: {description}"
        if feature_titles:
            content += f"\nKey features: {feature_titles}"

        settings = get_settings()
        try:
            client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
            msg = client.messages.create(
                model=settings.ANTHROPIC_MODEL,
                max_tokens=20,
                system=(
                    "Generate a concise, memorable product name (2–4 words) for the app described. "
                    "Use the name mentioned in the context if one appears; otherwise invent a fitting name. "
                    "Examples: TaskFlow, MeetMind, ShipTrack, PocketCoach. "
                    "Return ONLY the name — no quotes, no punctuation, no explanation."
                ),
                messages=[{"role": "user", "content": content}],
            )
            name = msg.content[0].text.strip().strip('"').strip("'")
            logger.info("Inferred app name: %r", name)
            return name
        except Exception as exc:
            logger.warning("App name inference failed: %s", exc)
            return ""

    async def _synthesize_from_segments(
        self,
        session_id: uuid.UUID,
        session: MeetingSession,
        all_event_docs,
    ) -> dict[str, Any]:
        """Fallback: concatenate stored transcript segments and run single-call synthesis."""
        from app.config import get_settings

        segments = [
            d.to_dict() for d in all_event_docs
            if d.to_dict().get("event_type") == "transcript.segment"
        ]

        if not segments:
            raise ValueError(f"No transcript data found for session {session_id}. Nothing was recorded or stored.")

        transcript = " ".join(
            s.get("payload", {}).get("text", "") for s in segments
        ).strip()

        if not transcript:
            raise ValueError(f"Transcript segments exist for session {session_id} but all are empty.")

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
            raise ValueError(f"AI synthesis failed: {exc}") from exc

        app_name_event = next((e for e in events if e["sub_state"] == "APP_NAME"), None)
        app_name = app_name_event["payload"].get("text", "") if app_name_event else ""
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
            "app_name": app_name,
            "app_description": app_description,
            "features": features,
            "open_questions": questions,
            "conflicts": conflicts,
            "action_items": action_items,
            "requirements_count": len(features) + len(questions) + len(conflicts) + len(action_items),
        }


def _build_blueprint_json(
    session: MeetingSession, events: list[dict]
) -> dict[str, Any]:
    app_name_event = next(
        (e for e in events if e.get("event_type") == "APP_NAME" and e.get("payload")), None
    )
    app_name = app_name_event["payload"].get("text", "") if app_name_event else ""
    app_desc_event = next(
        (e for e in events if e.get("event_type") == "APP_DESCRIPTION" and e.get("payload")), None
    )
    app_description = app_desc_event["payload"].get("text", "") if app_desc_event else ""
    features = [e["payload"] for e in events if e.get("event_type") == "FEATURE_FOUND" and e.get("payload")]
    questions = [e["payload"] for e in events if e.get("event_type") == "QUESTION_FOUND" and e.get("payload")]
    conflicts = [e["payload"] for e in events if e.get("event_type") == "CONFLICT_FOUND" and e.get("payload")]
    action_items = [e["payload"] for e in events if e.get("event_type") == "ACTION_ITEM_FOUND" and e.get("payload")]

    return {
        "version": "1.0",
        "session_id": str(session.id),
        "session_platform": session.platform.value,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "app_name": app_name,
        "app_description": app_description,
        "features": features,
        "open_questions": questions,
        "conflicts": conflicts,
        "action_items": action_items,
        "requirements_count": len(features) + len(questions) + len(conflicts) + len(action_items),
    }
