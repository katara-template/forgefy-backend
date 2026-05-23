"""Recall.ai webhook receiver.

Recall delivers two event types:
  • transcript.data   — real-time transcript segment for a bot
  • bot.status_change — bot lifecycle (joining → in_call_recording → call_ended …)

Security: every request carries the workspace verification secret in the
X-Recall-Workspace-Verification-Secret header.  If the secret is configured
we reject anything that doesn't match.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

import redis.asyncio as aioredis
from fastapi import APIRouter, Header, HTTPException, Request

from app.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()

_CHANNEL_PREFIX = "voxa:session:"

_STATUS_TO_SESSION = {
    "in_call_recording": "LISTENING",
    "joining_call": "JOINING",
    "in_call_not_recording": "JOINING",
}
_TERMINAL_STATUSES = {"call_ended", "done", "fatal"}


@router.post("/recall", status_code=204)
async def recall_webhook(
    request: Request,
    x_recall_workspace_verification_secret: str | None = Header(default=None),
) -> None:
    """Receive and process Recall.ai webhook events."""
    settings = get_settings()

    if settings.RECALL_WORKSPACE_VERIFICATION_SECRET:
        if x_recall_workspace_verification_secret != settings.RECALL_WORKSPACE_VERIFICATION_SECRET:
            raise HTTPException(status_code=401, detail="Invalid webhook secret")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    event_type: str = body.get("event", "")
    data: dict = body.get("data", {})

    logger.debug("Recall webhook event=%s", event_type)

    if event_type == "transcript.data":
        await _handle_transcript(data, settings)
    elif event_type == "bot.status_change":
        await _handle_status_change(data, settings)
    else:
        logger.debug("Unhandled Recall event type: %s", event_type)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

async def _handle_transcript(data: dict, settings) -> None:
    """Publish transcript segment to Redis and enqueue extraction."""
    bot_id: str = data.get("bot_id", "")
    transcript: dict = data.get("transcript", {})
    words: list[dict] = transcript.get("words", [])
    speaker: str = transcript.get("speaker", "")
    is_final: bool = bool(transcript.get("is_final", True))

    text = " ".join(w.get("text", "") for w in words).strip()
    if not text:
        return

    session_id = await _lookup_session(bot_id, settings.REDIS_URL)
    if not session_id:
        logger.warning("No session found for bot_id=%s", bot_id)
        return

    r = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    try:
        await r.publish(
            f"{_CHANNEL_PREFIX}{session_id}",
            json.dumps({
                "type": "transcript",
                "session_id": session_id,
                "text": text,
                "speaker": speaker,
                "is_final": is_final,
            }),
        )
    finally:
        await r.aclose()

    if is_final:
        from app.workers.extraction_worker import extract_requirements
        extract_requirements.apply_async(
            args=[session_id, text],
            queue="meeting.transcribe",
        )


async def _handle_status_change(data: dict, settings) -> None:
    """Transition session status and trigger blueprint generation when meeting ends."""
    bot: dict = data.get("bot", {})
    bot_id: str = bot.get("id", "")
    status_code: str = bot.get("status", {}).get("code", "")

    session_id = await _lookup_session(bot_id, settings.REDIS_URL)
    if not session_id:
        logger.warning("No session for bot_id=%s status=%s", bot_id, status_code)
        return

    logger.info("Recall bot_id=%s session=%s status=%s", bot_id, session_id, status_code)

    if status_code in _STATUS_TO_SESSION:
        target = _STATUS_TO_SESSION[status_code]
        await _try_transition_session(session_id, target, settings)

    elif status_code in _TERMINAL_STATUSES:
        await _end_session_from_bot(session_id, settings)
        await _clear_mapping(bot_id, session_id, settings.REDIS_URL)


async def _try_transition_session(
    session_id: str,
    target_status_value: str,
    settings,
) -> None:
    """Attempt a session status transition; ignore invalid-transition errors."""
    from app.db.firebase import get_firestore_client
    from app.db.models.enums import SessionStatus
    from app.modules.voxa.state_machine import MeetingStateMachine
    from app.core.exceptions import InvalidStateTransition

    db = get_firestore_client()
    target = SessionStatus(target_status_value)
    sm = MeetingStateMachine(db)
    try:
        await sm.transition(uuid.UUID(session_id), target)
    except InvalidStateTransition:
        pass  # already in target or ahead


async def _end_session_from_bot(session_id: str, settings) -> None:
    """Transition session to PROCESSING and dispatch blueprint generation."""
    from app.db.firebase import get_firestore_client
    from app.db.models.enums import SessionStatus
    from app.modules.voxa.state_machine import MeetingStateMachine
    from app.core.exceptions import InvalidStateTransition

    db = get_firestore_client()

    doc = await db.collection("sessions").document(session_id).get()
    if not doc.exists:
        return

    current_status = SessionStatus(doc.to_dict()["status"])
    endable = {SessionStatus.JOINING, SessionStatus.LISTENING}
    if current_status not in endable:
        return

    sm = MeetingStateMachine(db)
    try:
        await sm.transition(uuid.UUID(session_id), SessionStatus.PROCESSING)
        await db.collection("sessions").document(session_id).update(
            {"end_time": datetime.now(timezone.utc)}
        )
    except InvalidStateTransition:
        return

    from app.workers.blueprint_worker import generate_blueprint
    generate_blueprint.apply_async(args=[session_id], queue="meeting.extract")
    logger.info("Blueprint enqueued from bot end session=%s", session_id)


# ---------------------------------------------------------------------------
# Redis helpers
# ---------------------------------------------------------------------------

async def _lookup_session(bot_id: str, redis_url: str) -> str | None:
    r = aioredis.from_url(redis_url, decode_responses=True)
    try:
        return await r.get(f"recall:bot:{bot_id}")
    finally:
        await r.aclose()


async def _clear_mapping(bot_id: str, session_id: str, redis_url: str) -> None:
    r = aioredis.from_url(redis_url, decode_responses=True)
    try:
        await r.delete(f"recall:bot:{bot_id}", f"recall:session:{session_id}")
    finally:
        await r.aclose()
