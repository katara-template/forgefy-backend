"""Receiver for events posted by self-hosted Zoom bot containers.

Mounted at /api/v1/webhooks/zoom-bot. Kept separate from webhooks.py (which
serves Recall and Notchpay) so the two bot implementations stay independent,
but both converge on the same session handling — transcripts fan out on the
same Redis channel and drive the same state machine.

Only reachable when ZOOM_BOT_PROVIDER=self_hosted has actually launched a bot:
verification requires a per-session secret that only exists while a container
is running.

Security model: every bot is issued its own webhook secret at launch, stored
under zoombot:secret:{session_id}. A container can therefore only post events
for the one meeting it was spawned for, and the secret dies with the session.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
from datetime import UTC, datetime

import redis.asyncio as aioredis
from fastapi import APIRouter, Header, HTTPException, Request

logger = logging.getLogger(__name__)
router = APIRouter()

# Same channel the Recall path and the live WebSocket bridge already use.
_CHANNEL_PREFIX = "voxa:session:"

_STATUS_TO_SESSION = {
    "authenticating": "JOINING",
    "authenticated": "JOINING",
    "joining": "JOINING",
    "waiting_for_host": "JOINING",
    "in_waiting_room": "JOINING",
    "in_meeting": "JOINING",
    "awaiting_consent": "JOINING",
    "consent_granted": "JOINING",
    "reconnecting": "JOINING",
    # Only once raw audio is actually flowing is the session truly listening —
    # the bot can sit in a meeting for minutes awaiting consent before this.
    "recording": "LISTENING",
}

# Anything here means the bot will produce no further audio.
_TERMINAL_STATUSES = {
    "ended", "stopped", "error", "consent_denied", "consent_timeout", "suspended",
}

# Bounds how long a captured request stays replayable.
_MAX_SKEW_SECONDS = 300


@router.post("/zoom-bot", status_code=204)
async def zoom_bot_webhook(
    request: Request,
    timestamp: str | None = Header(default=None, alias="X-Forgefy-Timestamp"),
    signature: str | None = Header(default=None, alias="X-Forgefy-Signature"),
) -> None:
    """Receive transcript and lifecycle events from a self-hosted Zoom bot."""
    from app.config import get_settings

    settings = get_settings()
    raw_body = await request.body()

    try:
        body = json.loads(raw_body)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from exc

    session_id: str = body.get("session_id", "")
    if not session_id:
        raise HTTPException(status_code=400, detail="Missing session_id")

    _verify(session_id, raw_body, timestamp, signature, settings.REDIS_URL)

    event_type: str = body.get("type", "")
    if event_type == "transcript":
        await _handle_transcript(session_id, body, settings)
    elif event_type == "status":
        await _handle_status(session_id, body, settings)
    else:
        logger.debug("Unhandled zoom-bot event type: %s", event_type)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _verify(
    session_id: str,
    raw_body: bytes,
    timestamp: str | None,
    signature: str | None,
    redis_url: str,
) -> None:
    """Reject anything not signed by the secret issued to this session's bot."""
    from app.connectors.zoom_selfhosted import lookup_secret

    secret = lookup_secret(session_id, redis_url)
    if not secret:
        # No live bot for this session — it already finished, or this is forged.
        logger.warning("No zoom-bot secret for session=%s", session_id)
        raise HTTPException(status_code=401, detail="Unknown session")

    if not (timestamp and signature):
        raise HTTPException(status_code=401, detail="Missing signature headers")

    try:
        skew = abs(int(datetime.now(UTC).timestamp()) - int(timestamp))
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="Invalid timestamp") from exc

    if skew > _MAX_SKEW_SECONDS:
        raise HTTPException(status_code=401, detail="Stale request")

    expected = hmac.new(
        secret.encode(),
        f"{timestamp}.".encode() + raw_body,
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(expected, signature):
        raise HTTPException(status_code=401, detail="Invalid signature")


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

async def _handle_transcript(session_id: str, body: dict, settings) -> None:
    """Publish a transcript segment; enqueue extraction for final ones only."""
    text: str = (body.get("text") or "").strip()
    if not text:
        return

    is_final = bool(body.get("is_final"))
    speaker: str = body.get("speaker") or ""

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

    # Interim results exist to make the live UI feel responsive. Extracting
    # requirements from them would produce duplicates as the text firms up.
    if not is_final:
        return

    logger.info(
        "Zoom bot transcript session=%s chars=%d speaker=%s",
        session_id, len(text), speaker or "?",
    )

    from app.core.dispatch import dispatch
    from app.workers.extraction_worker import extract_requirements

    await dispatch(
        extract_requirements,
        args=[session_id, text],
        queue="meeting.transcribe",
    )


async def _handle_status(session_id: str, body: dict, settings) -> None:
    """Drive the session state machine from bot lifecycle events."""
    status: str = body.get("status", "")
    detail: str = body.get("detail", "")

    logger.info("Zoom bot session=%s status=%s %s", session_id, status, detail)

    if status in _STATUS_TO_SESSION:
        await _transition(session_id, _STATUS_TO_SESSION[status])
        return

    if status not in _TERMINAL_STATUSES:
        return

    await _end_session(session_id)

    # Reap the container. On the normal path the bot has already exited and
    # this is a no-op; it matters when the bot errored or was denied consent
    # and would otherwise sit idle in the meeting.
    from app.core.dispatch import dispatch
    from app.workers.zoom_bot_worker import zoom_bot_cleanup

    await dispatch(zoom_bot_cleanup, args=[session_id], queue="meeting.audio")


async def _transition(session_id: str, target_status: str) -> None:
    """Attempt a session status transition; ignore invalid-transition errors.

    The bot re-reports statuses on reconnect, so replaying a transition the
    session has already made is normal rather than exceptional.
    """
    import uuid
    from contextlib import suppress

    from app.core.exceptions import InvalidStateTransition
    from app.db.firebase import get_firestore_client
    from app.db.models.enums import SessionStatus
    from app.modules.voxa.state_machine import MeetingStateMachine

    sm = MeetingStateMachine(get_firestore_client())
    with suppress(InvalidStateTransition):
        await sm.transition(uuid.UUID(session_id), SessionStatus(target_status))


async def _end_session(session_id: str) -> None:
    """Run the same end-of-meeting pipeline as the frontend end button."""
    import uuid

    from app.db.firebase import get_firestore_client
    from app.modules.voxa.service import VoxaService

    service = VoxaService(get_firestore_client())
    session = await service.end_session_from_bot(uuid.UUID(session_id))
    if session:
        logger.info("Session auto-ended from zoom bot session=%s", session_id)
