"""Prompt-to-app worker.

Turns a plain-language app idea (from the assistant's "build_app" action) into a
running build, without a meeting: it synthesises a blueprint from the prompt,
approves it, and hands off to the normal build pipeline.

The session created for it is walked through the same state graph a meeting uses
(PROCESSING → BLUEPRINT_READY → APPROVED → BUILDING), so everything downstream —
the build worker, the project page, the database-consent gate — behaves exactly
as it does for a meeting-originated build.
"""
from __future__ import annotations

import asyncio
import logging
import re
import uuid
from contextlib import suppress
from datetime import UTC, datetime

from app.config import get_settings
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

# Ways a user spells out a name in a build prompt: "called X", "named X",
# "call it X", 'app called "X"'. Captures a single 3-24 char alphanumeric token.
_EXPLICIT_NAME_PATTERNS = (
    re.compile(r'\b(?:called|named)\s+["\']?([A-Za-z][A-Za-z0-9]{2,23})', re.IGNORECASE),
    re.compile(r'\b(?:call|name)\s+it\s+["\']?([A-Za-z][A-Za-z0-9]{2,23})', re.IGNORECASE),
)


def _explicit_name(description: str) -> str:
    """Return a name the user spelled out in the prompt, or "" if none."""
    for pattern in _EXPLICIT_NAME_PATTERNS:
        match = pattern.search(description)
        if match:
            return match.group(1)
    return ""


async def _generate_blueprint(db, session_id: str, description: str, log_fn) -> dict:
    """Build a blueprint JSON from a prompt, reusing the meeting-path helpers.

    Emits granular progress via log_fn so the assistant can stream each phase to
    the user over the build-logs WebSocket.
    """
    from app.build.blueprint_generator import (
        BlueprintAggregator,
        _detect_platform,
        _name_from_description,
    )

    agg = BlueprintAggregator(db)
    json_output: dict = {
        "version": "1.0",
        "session_id": session_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "app_description": description,
        "features": [],
        "open_questions": [],
        "conflicts": [],
        "action_items": [],
        "origin": "prompt",
    }

    log_fn("info", "Outlining the core features…")
    json_output["features"] = await agg._derive_features(description)
    json_output["requirements_count"] = len(json_output["features"])

    log_fn("info", "Naming your app…")
    # Honour a name the user spelled out in the prompt ("…called FitTrack"), then
    # a name stated in the description ("FitTrack helps…"); otherwise let the AI
    # name generator invent a short brand name.
    name = _explicit_name(description) or _name_from_description(description)
    json_output["app_name"] = name[:30] or await agg._infer_app_name(json_output) or "Forge"

    log_fn("info", "Designing the look & feel…")
    json_output["design_system"] = await agg._generate_design_system(json_output)

    platform = _detect_platform(json_output)
    json_output["template"] = "next" if platform == "web" else "flutter"
    log_fn("info", f"Building as a {'web' if platform == 'web' else 'mobile'} app.")
    return json_output


async def _build_from_prompt(
    session_id: str, project_id: str, description: str, user_id: str
) -> dict:
    from app.build.blueprint_generator import classify_needs_database
    from app.build.build_logger import make_log_publisher
    from app.db.firebase import refresh_async_firestore_client
    from app.db.models.enums import SessionStatus
    from app.modules.voxa.state_machine import MeetingStateMachine

    settings = get_settings()
    db = refresh_async_firestore_client()  # bind fresh gRPC channel to this loop
    log_fn = make_log_publisher(project_id, settings.REDIS_URL)

    try:
        log_fn("started", "Designing your app…")
        json_output = await _generate_blueprint(db, session_id, description, log_fn)

        log_fn("info", "Assembling the blueprint…")
        now = datetime.now(UTC)
        blueprint_id = str(uuid.uuid4())
        await db.collection("blueprints").document(blueprint_id).set({
            "session_id": session_id,
            "json_output": json_output,
            "approved": True,
            "project_id": project_id,
            "created_at": now,
        })

        # Walk the session through the normal graph so downstream code is happy.
        sm = MeetingStateMachine(db)
        await sm.transition(
            uuid.UUID(session_id),
            SessionStatus.BLUEPRINT_READY,
            extra_payload={"blueprint_id": blueprint_id},
        )
        await sm.transition(
            uuid.UUID(session_id),
            SessionStatus.APPROVED,
            extra_payload={"blueprint_id": blueprint_id, "approved_by": user_id},
        )

        # Same "never touch a database without consent" gate as approve_blueprint.
        needs_db, db_reason = await classify_needs_database(
            description, json_output.get("features", [])
        )

        app_name = (
            re.sub(r"[^a-zA-Z0-9._-]", "-", json_output["app_name"]).strip("-").lower()[:100]
            or "forgefy-app"
        )
        await db.collection("projects").document(project_id).update({
            "app_name": app_name,
            "template_key": json_output["template"],
            "blueprint_id": blueprint_id,
            "blueprint_context": json_output,
            "is_updating": not needs_db,
            "db_decision_pending": needs_db,
            "db_decision_reason": db_reason if needs_db else None,
            "updated_at": now,
        })

        # Navigation signal for the assistant widget — the blueprint now exists.
        log_fn("blueprint_ready", "Blueprint ready.")

        if needs_db:
            log_fn("info", "This app needs a database — connect one to continue.")
            logger.info(
                "Prompt build withheld pending DB decision session=%s project=%s",
                session_id, project_id,
            )
            return {"withheld_for_db": True, "project_id": project_id}

        log_fn("info", "Starting the build…")
        from app.workers.build_worker import run_build
        run_build.apply_async(args=[session_id, project_id], queue="build")
        logger.info("Prompt build dispatched session=%s project=%s", session_id, project_id)
        return {"build_dispatched": True, "project_id": project_id}

    except Exception as exc:  # noqa: BLE001 — surface as a build error, don't crash silently
        logger.error("Prompt build failed session=%s: %s", session_id, exc, exc_info=True)
        with suppress(Exception):
            await db.collection("projects").document(project_id).update({
                "is_updating": False,
                "build_error": "We couldn't design your app from that prompt. Please try rephrasing.",
                "updated_at": datetime.now(UTC),
            })
            log_fn("error", "Could not design the app from that prompt.")
        raise


@celery_app.task(
    name="app.workers.prompt_build_worker.build_from_prompt", bind=True, max_retries=0
)
def build_from_prompt(self, session_id: str, project_id: str, description: str, user_id: str) -> dict:
    """Celery entry point — prompt → blueprint → dispatch build."""
    logger.info("Prompt build task started session=%s project=%s", session_id, project_id)
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            _build_from_prompt(session_id, project_id, description, user_id)
        )
    finally:
        loop.close()
