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
from datetime import UTC, datetime
from typing import Any

from google.cloud.firestore import AsyncClient

from app.db.models.blueprint import Blueprint
from app.db.models.enums import Platform, SessionStatus
from app.db.models.meeting_session import MeetingSession
from app.modules.voxa.state_machine import MeetingStateMachine

logger = logging.getLogger(__name__)


def _compact_name(name: str) -> str:
    """Normalise any raw name to a 4-8 character alphanumeric string.

    • Single words already in range → returned as-is (case preserved for slugify).
    • Multi-word names → compounded CamelCase, then truncated to 8 if still long.
    • Names < 4 chars after compaction → return "" so the AI fallback fires.
    """
    clean = re.sub(r"[^a-zA-Z0-9 ]", "", name).strip()
    words = clean.split()
    if not words:
        return ""

    # Compound multi-word names: "Task Flow" → "TaskFlow"
    compound = "".join(w.capitalize() for w in words) if len(words) > 1 else words[0]

    if len(compound) > 8:
        compound = compound[:8]

    return compound if len(compound) >= 4 else ""


def _name_from_description(description: str) -> str:
    """Extract the user-specified app name from the first sentence.

    Matches patterns like 'Voxa is…', 'Task Flow helps…', etc.
    Returns a compacted 4-8 char name, or "" if nothing found.
    """
    match = re.match(
        r'^([A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){0,3})'
        r'\s+(?:is\b|helps\b|allows\b|provides\b|enables\b|aims\b|was\b|will\b|can\b)',
        description.strip(),
    )
    if match:
        candidate = match.group(1).strip()
        words = candidate.split()
        if 1 <= len(words) <= 4:
            return _compact_name(candidate)
    return ""


_EXTRACTION_TYPES = frozenset(
    {"APP_DESCRIPTION", "FEATURE_FOUND", "QUESTION_FOUND", "CONFLICT_FOUND", "ACTION_ITEM_FOUND"}
)

# Keyword sets for platform inference.  Multi-word phrases rank higher
# because they carry stronger intent signal than lone words like "app".
_WEB_PHRASES = frozenset({
    "website", "web app", "web application", "web platform", "web portal",
    "web interface", "web-based", "web based", "dashboard", "admin panel",
    "admin dashboard", "landing page", "saas", "browser-based", "next.js",
    "nextjs", "react app",
})

_MOBILE_PHRASES = frozenset({
    "mobile app", "mobile application", "ios app", "android app",
    "mobile platform", "phone app", "flutter app", "react native",
    "app store", "play store", "push notification", "native app", "flutter",
    "ios", "android",
})


def _detect_platform(json_output: dict) -> str:
    """Infer 'web', 'mobile', or 'both' from blueprint content keywords."""
    text = " ".join([
        json_output.get("app_description", "") or "",
        *[f.get("title", "") or "" for f in json_output.get("features", [])],
        *[f.get("description", "") or "" for f in json_output.get("features", [])],
    ]).lower()

    web_hits = sum(1 for phrase in _WEB_PHRASES if phrase in text)
    mobile_hits = sum(1 for phrase in _MOBILE_PHRASES if phrase in text)

    if web_hits > 0 and mobile_hits > 0:
        return "both"
    if web_hits > 0:
        return "web"
    return "mobile"  # default — Flutter when ambiguous


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

        # Extract and normalise app name to 4-8 char brand word.
        # Priority: user-specified from events → pattern extraction → AI generation.
        raw_name = json_output.get("app_name") or ""
        if not raw_name:
            raw_name = (
                _name_from_description(json_output.get("app_description", ""))
                or ""
            )
        compacted = _compact_name(raw_name) if raw_name else ""
        if not compacted:
            # AI generation — already returns a compacted 4-8 char name
            compacted = await self._infer_app_name(json_output)
        json_output["app_name"] = compacted or "Forge"

        # Generate design system for the build agent to use as visual source of truth
        if not json_output.get("design_system"):
            json_output["design_system"] = await self._generate_design_system(json_output)

        platform = _detect_platform(json_output)
        now = datetime.now(UTC)
        feature_count = len(json_output.get("features", []))

        if platform == "both":
            # Save one Flutter blueprint and one Next.js blueprint; let the user
            # decide which to build first.
            bp_mobile_id = str(uuid.uuid4())
            bp_web_id = str(uuid.uuid4())
            mobile_output = {**json_output, "template": "flutter", "platform_variant": "mobile"}
            web_output = {**json_output, "template": "next", "platform_variant": "web"}

            await self._db.collection("blueprints").document(bp_mobile_id).set({
                "session_id": str(session_id),
                "json_output": mobile_output,
                "approved": False,
                "created_at": now,
            })
            await self._db.collection("blueprints").document(bp_web_id).set({
                "session_id": str(session_id),
                "json_output": web_output,
                "approved": False,
                "created_at": now,
            })

            await self._sm.transition(
                session_id,
                SessionStatus.BLUEPRINT_READY,
                extra_payload={"blueprint_id": bp_mobile_id, "blueprint_id_web": bp_web_id},
            )
            logger.info(
                "Dual blueprint generated session=%s mobile=%s web=%s features=%d",
                session_id, bp_mobile_id, bp_web_id, feature_count,
            )
            return Blueprint(
                id=uuid.UUID(bp_mobile_id),
                session_id=session_id,
                json_output=mobile_output,
                approved=False,
                created_at=now,
            )

        # Single-platform path
        json_output["template"] = "next" if platform == "web" else "flutter"
        blueprint_id = str(uuid.uuid4())
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
            "Blueprint generated session=%s blueprint=%s template=%s features=%d",
            session_id, blueprint_id, json_output["template"], feature_count,
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
        import json as _json

        from app.config import get_settings

        settings = get_settings()
        try:
            _SYSTEM = (
                "You are a product analyst. Given an app description, infer the most likely "
                "features the app needs. Return a JSON object with a 'features' array.\n"
                'Each item: {"title": "2-6 word name", "description": "1-2 sentence detail", "priority": "high|med|low"}'
            )
            if settings.BP_MODEL == "Qwen3":
                from app.ai.agents.ollama_synthesizer import call_ollama
                result = call_ollama(
                    system_prompt=_SYSTEM,
                    user_content=f"App description:\n{description}",
                    base_url=settings.OLLAMA_URL,
                    model=settings.OLLAMA_MODEL,
                    timeout=settings.OLLAMA_TIMEOUT,
                )
                features = result.get("features", [])
            elif settings.BP_MODEL == "gemini":
                from app.ai.agents.gemini_synthesizer import call_gemini
                result = call_gemini(
                    system_prompt=_SYSTEM,
                    user_content=f"App description:\n{description}",
                    api_key=settings.GEMINI_API_KEY,
                    model=settings.GEMINI_MODEL,
                    max_tokens=1024,
                )
                features = result.get("features", [])
            else:
                import anthropic
                client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY, timeout=120.0)
                msg = client.messages.create(
                    model=settings.ANTHROPIC_MODEL,
                    max_tokens=1024,
                    system=_SYSTEM,
                    messages=[{"role": "user", "content": f"App description:\n{description}"}],
                )
                features = _json.loads(msg.content[0].text.strip())
            if isinstance(features, list):
                logger.info("Derived %d features from description", len(features))
                return features
        except Exception as exc:
            logger.warning("Feature derivation failed: %s", exc)
        return []

    async def _infer_app_name(self, blueprint: dict[str, Any]) -> str:
        """Ask the AI to generate a branded 4-8 character app name."""
        from app.config import get_settings

        description = (blueprint.get("app_description") or "").strip()
        features = blueprint.get("features") or []
        if not description and not features:
            return ""

        feature_titles = ", ".join(f.get("title", "") for f in features[:6] if f.get("title"))
        content = f"Description: {description}"
        if feature_titles:
            content += f"\nKey features: {feature_titles}"

        _NAME_SYSTEM = (
            "You generate branded app names that are 4–8 characters long.\n\n"
            "Rules (ALL must be satisfied):\n"
            "• Exactly 4 to 8 characters total — count every character\n"
            "• A single word — no spaces, hyphens, or dots\n"
            "• Pronounceable in English (natural to say aloud)\n"
            "• Easy to remember and type\n"
            "• Evocative of the app's domain — not generic\n\n"
            "Examples by length:\n"
            "  4 chars: Loom, Zoom, Hive, Flow, Fuse\n"
            "  5 chars: Slack, Figma, Asana, Pulse, Orbit\n"
            "  6 chars: Notion, Linear, Stripe, Vercel\n"
            "  7 chars: Dropbox, Postman\n"
            "  8 chars: Intercom, Obsidian\n\n"
            "Avoid: generic words (App, Tool, Helper), hard-to-pronounce combos, "
            "multi-word names, numbers, or special characters.\n\n"
            'Return ONLY a JSON object: {"app_name": "<4-8 char name>"}\n'
            "No other text."
        )

        settings = get_settings()
        try:
            if settings.BP_MODEL == "Qwen3":
                from app.ai.agents.ollama_synthesizer import call_ollama
                result = call_ollama(
                    system_prompt=_NAME_SYSTEM,
                    user_content=content,
                    base_url=settings.OLLAMA_URL,
                    model=settings.OLLAMA_MODEL,
                    timeout=settings.OLLAMA_TIMEOUT,
                )
                raw = result.get("app_name", "").strip().strip('"').strip("'")
            elif settings.BP_MODEL == "gemini":
                from app.ai.agents.gemini_synthesizer import call_gemini
                result = call_gemini(
                    system_prompt=_NAME_SYSTEM,
                    user_content=content,
                    api_key=settings.GEMINI_API_KEY,
                    model=settings.GEMINI_MODEL,
                    max_tokens=32,
                )
                raw = result.get("app_name", "").strip().strip('"').strip("'")
            else:
                import json as _json

                import anthropic
                client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY, timeout=120.0)
                msg = client.messages.create(
                    model=settings.ANTHROPIC_MODEL,
                    max_tokens=32,
                    system=_NAME_SYSTEM,
                    messages=[{"role": "user", "content": content}],
                )
                text = msg.content[0].text.strip()
                try:
                    raw = _json.loads(text).get("app_name", "").strip().strip('"').strip("'")
                except Exception:
                    raw = text.strip('"').strip("'")

            # Validate and compact — AI may still return something slightly off-spec
            name = _compact_name(raw) if raw else ""
            logger.info("Inferred app name: %r → %r", raw, name)
            return name
        except Exception as exc:
            logger.warning("App name inference failed: %s", exc)
            return ""

    async def _generate_design_system(self, blueprint: dict[str, Any]) -> dict:
        """Ask the AI to generate a design_system block tailored to this app's domain."""
        import json as _json

        from app.config import get_settings

        settings = get_settings()

        app_name = (blueprint.get("app_name") or "").strip()
        description = (blueprint.get("app_description") or "").strip()
        features = blueprint.get("features") or []
        feature_titles = ", ".join(f.get("title", "") for f in features[:8] if f.get("title"))

        _SYSTEM = (
            "You are a UI/UX design system architect. Given an app's details, produce a "
            "design_system JSON block that drives all visual decisions. Make every choice "
            "specific to the app's domain and audience — do NOT produce generic defaults.\n\n"
            "Return ONLY valid JSON with this exact structure (no other text):\n"
            "{\n"
            '  "personality": "<one sentence: visual personality of this app>",\n'
            '  "palette": {\n'
            '    "primary": "#hex", "primary_dark": "#hex", "secondary": "#hex",\n'
            '    "accent": "#hex", "background": "#hex", "surface": "#hex",\n'
            '    "surface_variant": "#hex", "error": "#hex", "on_primary": "#hex",\n'
            '    "on_background": "#hex", "on_surface": "#hex", "on_surface_muted": "#hex"\n'
            "  },\n"
            '  "typography": {\n'
            '    "display_font": "<Google Font name>",\n'
            '    "body_font": "<different Google Font name>"\n'
            "  },\n"
            '  "signature_element": "<ONE domain-specific design detail that makes this app memorable>",\n'
            '  "icon_style": "outlined",\n'
            '  "image_treatment": "<how images are cropped/styled>"\n'
            "}\n\n"
            "Rules:\n"
            "- palette must have CONTRAST RATIO >= 4.5:1 between on_* and their backgrounds\n"
            "- display_font must be different from body_font (both must be real Google Fonts)\n"
            "- signature_element must be domain-specific, NOT a generic gradient card\n"
            "- personality must reflect the actual app domain, NOT generic SaaS language\n"
            "- JSON only — zero prose outside the JSON object"
        )

        content = f"App name: {app_name}\nDescription: {description}"
        if feature_titles:
            content += f"\nKey features: {feature_titles}"

        try:
            if settings.BP_MODEL == "Qwen3":
                from app.ai.agents.ollama_synthesizer import call_ollama
                result = call_ollama(
                    system_prompt=_SYSTEM,
                    user_content=content,
                    base_url=settings.OLLAMA_URL,
                    model=settings.OLLAMA_MODEL,
                    timeout=settings.OLLAMA_TIMEOUT,
                )
                if isinstance(result, dict) and "palette" in result:
                    return result
            elif settings.BP_MODEL == "gemini":
                from app.ai.agents.gemini_synthesizer import call_gemini
                result = call_gemini(
                    system_prompt=_SYSTEM,
                    user_content=content,
                    api_key=settings.GEMINI_API_KEY,
                    model=settings.GEMINI_MODEL,
                    max_tokens=1024,
                )
                if isinstance(result, dict) and "palette" in result:
                    return result
            else:
                import anthropic
                client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY, timeout=120.0)
                msg = client.messages.create(
                    model=settings.ANTHROPIC_MODEL,
                    max_tokens=1024,
                    system=_SYSTEM,
                    messages=[{"role": "user", "content": content}],
                )
                raw = (msg.content[0].text or "").strip()
                # Strip markdown fences if present
                import re as _re
                raw = _re.sub(r"```[a-z]*\n?", "", raw).strip().strip("`").strip()
                result = _json.loads(raw)
                if isinstance(result, dict) and "palette" in result:
                    return result
        except Exception as exc:
            logger.warning("Design system generation failed: %s", exc)

        return {}

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
            "generated_at": datetime.now(UTC).isoformat(),
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
        "generated_at": datetime.now(UTC).isoformat(),
        "app_name": app_name,
        "app_description": app_description,
        "features": features,
        "open_questions": questions,
        "conflicts": conflicts,
        "action_items": action_items,
        "requirements_count": len(features) + len(questions) + len(conflicts) + len(action_items),
    }


_DB_NEED_SYSTEM = (
    "You determine whether an app idea requires a real, persistent database.\n\n"
    "Answer yes ONLY if the app needs to durably store structured data across "
    "sessions/devices/users — e.g. user accounts, saved records, inventory, posts, "
    "orders, chat history, bookmarks.\n"
    "Answer no for apps that are purely local/ephemeral — a calculator, a stopwatch, "
    "a static info screen, a game with only in-memory/on-device state, a single-user "
    "offline note that doesn't need to sync.\n\n"
    'Reply ONLY with JSON: {"needs_database": true|false, "reason": "<one short sentence, '
    'or empty string if false>"}'
)


async def classify_needs_database(description: str, features: list[dict]) -> tuple[bool, str]:
    """Decide whether an app idea needs a real database, so the build pipeline can ask
    for consent before ever provisioning or wiring one in (see app/api/v1/blueprints.py's
    approve_blueprint and app/api/v1/projects.py's chat_with_project).

    Defaults to (False, "") on any classifier failure — never blocks a build on an AI
    call succeeding, mirroring _derive_features/_infer_app_name above.
    """
    import json as _json

    from app.config import get_settings

    feature_lines = "\n".join(
        f"- {f.get('title', '')}: {f.get('description', '')}" for f in (features or [])
    )
    content = f"App description: {description}\n\nFeatures:\n{feature_lines}"

    settings = get_settings()
    try:
        if settings.BP_MODEL == "Qwen3":
            from app.ai.agents.ollama_synthesizer import call_ollama
            result = call_ollama(
                system_prompt=_DB_NEED_SYSTEM,
                user_content=content,
                base_url=settings.OLLAMA_URL,
                model=settings.OLLAMA_MODEL,
                timeout=settings.OLLAMA_TIMEOUT,
            )
        elif settings.BP_MODEL == "gemini":
            from app.ai.agents.gemini_synthesizer import call_gemini
            result = call_gemini(
                system_prompt=_DB_NEED_SYSTEM,
                user_content=content,
                api_key=settings.GEMINI_API_KEY,
                model=settings.GEMINI_MODEL,
                max_tokens=256,
            )
        else:
            import anthropic
            client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY, timeout=120.0)
            msg = client.messages.create(
                model=settings.ANTHROPIC_MODEL,
                max_tokens=256,
                system=_DB_NEED_SYSTEM,
                messages=[{"role": "user", "content": content}],
            )
            result = _json.loads(msg.content[0].text.strip())

        return bool(result.get("needs_database")), str(result.get("reason") or "")
    except Exception as exc:
        logger.warning("Database-need classification failed, defaulting to no: %s", exc)
        return False, ""
