"""Recall.ai webhook receiver.

Recall delivers two event types:
  • transcript.data   — real-time transcript segment for a bot
  • bot.status_change — bot lifecycle (joining → in_call_recording → call_ended …)

Security: every request carries the workspace verification secret in the
X-Recall-Workspace-Verification-Secret header.  If the secret is configured
we reject anything that doesn't match.
"""
from __future__ import annotations

import hmac
import json
import logging
import uuid
from datetime import UTC, datetime, timedelta

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

    if (
        settings.RECALL_WORKSPACE_VERIFICATION_SECRET
        and x_recall_workspace_verification_secret != settings.RECALL_WORKSPACE_VERIFICATION_SECRET
    ):
        raise HTTPException(status_code=401, detail="Invalid webhook secret")

    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from exc

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
    """Publish transcript segment to Redis and enqueue extraction.

    Recall only sends `transcript.data` (never `transcript.partial_data`, which
    we don't subscribe to) — every event here is already a finalized utterance.
    """
    bot_id: str = data.get("bot", {}).get("id", "")
    inner: dict = data.get("data", {})
    words: list[dict] = inner.get("words", [])
    participant: dict = inner.get("participant") or {}
    speaker: str = participant.get("name") or ""
    is_final = True

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
    from contextlib import suppress

    from app.core.exceptions import InvalidStateTransition
    from app.db.firebase import get_firestore_client
    from app.db.models.enums import SessionStatus
    from app.modules.voxa.state_machine import MeetingStateMachine

    db = get_firestore_client()
    target = SessionStatus(target_status_value)
    sm = MeetingStateMachine(db)
    with suppress(InvalidStateTransition):
        await sm.transition(uuid.UUID(session_id), target)


async def _end_session_from_bot(session_id: str, settings) -> None:
    """Transition session to PROCESSING and dispatch blueprint generation."""
    from app.core.exceptions import InvalidStateTransition
    from app.db.firebase import get_firestore_client
    from app.db.models.enums import SessionStatus
    from app.modules.voxa.state_machine import MeetingStateMachine

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
            {"end_time": datetime.now(UTC)}
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


# ---------------------------------------------------------------------------
# Notchpay webhook
# ---------------------------------------------------------------------------

# How subscriptions work with Notchpay (which has no native recurring billing):
#   - User pays once → tier set for 31 days (tier_expires_at)
#   - Frontend shows expiry date and a "Renew" button
#   - On renewal, user pays again → tier_expires_at extended by 31 days
#   - get_user_tier() auto-downgrades to free when tier_expires_at passes

_TIER_PRICES = {19: "starter", 49: "pro", 149: "team"}


@router.post("/notchpay", status_code=200)
async def notchpay_webhook(request: Request) -> dict:
    """Receive Notchpay payment events and update user tier."""
    settings = get_settings()
    raw_body = await request.body()

    # Verify webhook signature — Notchpay sends HMAC-SHA256 in x-notch-signature
    if settings.NOTCHPAY_SECRET_HASH:
        signature = request.headers.get("x-notch-signature", "")
        expected = hmac.digest(
            settings.NOTCHPAY_SECRET_HASH.encode(),
            raw_body,
            "sha256",
        ).hex()
        if not hmac.compare_digest(expected, signature):
            raise HTTPException(status_code=401, detail="Invalid webhook signature")

    try:
        body = json.loads(raw_body)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc

    event: str = body.get("event", "")
    data: dict = body.get("data", {})
    status: str = data.get("status", "")

    logger.info("Notchpay webhook event=%s status=%s", event, status)

    if event == "payment.complete" and status == "complete":
        await _handle_payment_complete(data, settings)

    return {"received": True}


async def _handle_payment_complete(data: dict, settings) -> None:
    """Activate or extend a user's subscription after a successful payment."""
    from app.db.firebase import get_firestore_client

    meta: dict = data.get("meta") or data.get("metadata") or {}
    user_id: str = meta.get("user_id", "")
    tier_key: str = meta.get("tier", "")

    # Fallback: infer tier from amount if meta wasn't stored
    if not tier_key:
        amount = int(data.get("amount", 0))
        tier_key = _TIER_PRICES.get(amount, "")

    if not user_id or not tier_key:
        logger.warning("Notchpay webhook: missing user_id or tier in meta — data=%s", data)
        return

    from app.core.tiers import TIERS
    if tier_key not in TIERS:
        logger.warning("Notchpay webhook: unknown tier %r for user %s", tier_key, user_id)
        return

    db = get_firestore_client()
    now = datetime.now(UTC)

    # Extend existing subscription if user already has one active; otherwise start fresh
    user_doc = await db.collection("users").document(user_id).get()
    current_expires = (user_doc.to_dict() or {}).get("tier_expires_at")

    base = current_expires if (current_expires and current_expires > now) else now
    new_expires = base + timedelta(days=31)

    await db.collection("users").document(user_id).update({
        "tier": tier_key,
        "tier_expires_at": new_expires,
        "updated_at": now,
    })

    logger.info(
        "Subscription activated user=%s tier=%s expires=%s",
        user_id, tier_key, new_expires.isoformat(),
    )
