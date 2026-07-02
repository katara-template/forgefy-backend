"""WebSocket gateway — /ws/voxa.

Auth: JWT passed as ?token= query param. The token is validated before
accepting the connection; invalid tokens are rejected with close code 4001.

Client→Server events: joinSession, streamAudio, endMeeting, ping
Server→Client events: transcript, featureDetected, blueprintReady, meetingStatus, pong, error

Audio path (physical live sessions):
  streamAudio chunks are decoded and forwarded directly to a per-session
  DeepgramLiveSession.  Transcripts are published to Redis, which the Redis
  subscriber task picks up and broadcasts to all connected browsers.

Redis pub/sub bridge: workers PUBLISH to voxa:session:{session_id};
this handler SUBSCRIBES and forwards each message to local WS connections.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import uuid
from typing import Any

import redis.asyncio as aioredis
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from pydantic import TypeAdapter, ValidationError

from app.api.ws.connection_manager import manager
from app.config import get_settings
from app.core.security import decode_access_token
from app.schemas.ws_events import (
    ClientEvent,
    EndMeetingEvent,
    JoinSessionEvent,
    PingEvent,
    PongEvent,
    StreamAudioEvent,
)
from app.transcription.deepgram_live import DeepgramLiveSession

logger = logging.getLogger(__name__)
router = APIRouter()

_client_event_adapter: TypeAdapter[ClientEvent] = TypeAdapter(ClientEvent)

_CHANNEL_PREFIX = "voxa:session:"

# Per-process registry of active Deepgram live sessions (physical mic path).
# Keyed by session_id UUID.
_live_sessions: dict[uuid.UUID, DeepgramLiveSession] = {}


def _channel(session_id: uuid.UUID) -> str:
    return f"{_CHANNEL_PREFIX}{session_id}"


# ── Redis subscriber task ─────────────────────────────────────────────────────


def _make_redis_client(redis_url: str) -> aioredis.Redis:
    """Create an aioredis client, stripping URL SSL params that redis-py ignores."""
    from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

    extra: dict = {}
    url = redis_url
    if redis_url.startswith("rediss://"):
        parsed = urlparse(redis_url)
        qs = parse_qs(parsed.query)
        qs.pop("ssl_cert_reqs", None)
        url = urlunparse(parsed._replace(query=urlencode(qs, doseq=True)))
        extra["ssl_cert_reqs"] = "none"
    return aioredis.from_url(url, decode_responses=True, **extra)


async def _redis_subscriber(session_id: uuid.UUID, settings_redis_url: str) -> None:
    """Subscribe to the Redis channel for session_id and forward messages.

    Uses get_message() polling instead of listen() so that idle-connection
    timeouts from managed Redis providers (Upstash etc.) cause a reconnect
    rather than a silent task crash.
    """
    channel = _channel(session_id)
    while True:
        redis: aioredis.Redis = _make_redis_client(settings_redis_url)
        try:
            async with redis.pubsub() as pubsub:
                await pubsub.subscribe(channel)
                while True:
                    raw = await pubsub.get_message(
                        ignore_subscribe_messages=True, timeout=0
                    )
                    if raw is not None:
                        try:
                            payload: dict[str, Any] = json.loads(raw["data"])
                            await manager.broadcast(session_id, payload)
                        except (json.JSONDecodeError, TypeError):
                            pass
                    await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.debug(
                "_redis_subscriber reconnecting session=%s: %s", session_id, exc
            )
            await asyncio.sleep(1.0)
        finally:
            await redis.aclose()


# ── WebSocket endpoint ────────────────────────────────────────────────────────


@router.websocket("/ws/voxa")
async def ws_voxa(
    ws: WebSocket,
    token: str = Query(..., description="Bearer access token for auth"),
) -> None:
    """Bidirectional WebSocket gateway for live meeting sessions."""
    settings = get_settings()

    try:
        user_id_str = decode_access_token(token, settings)
    except Exception:
        await ws.close(code=4001, reason="Unauthorized")
        return

    await ws.accept()
    logger.info("WS connected user=%s", user_id_str)

    session_id: uuid.UUID | None = None
    subscriber_task: asyncio.Task | None = None  # type: ignore[type-arg]

    try:
        async for raw_text in ws.iter_text():
            try:
                event = _client_event_adapter.validate_json(raw_text)
            except ValidationError as exc:
                await manager.send_to(ws, {"type": "error", "code": "invalid_event", "detail": str(exc)})
                continue

            if isinstance(event, JoinSessionEvent):
                if subscriber_task and not subscriber_task.done():
                    subscriber_task.cancel()
                if session_id:
                    manager.unregister(session_id, ws)

                session_id = event.session_id
                manager.register(session_id, ws)
                subscriber_task = asyncio.create_task(
                    _redis_subscriber(session_id, settings.REDIS_URL)
                )
                await manager.send_to(ws, {
                    "type": "meetingStatus",
                    "session_id": str(session_id),
                    "status": "joined",
                })
                logger.info("WS joined session=%s user=%s", session_id, user_id_str)

            elif isinstance(event, StreamAudioEvent):
                if session_id is None:
                    await manager.send_to(ws, {"type": "error", "code": "not_joined", "detail": "Send joinSession first"})
                    continue

                # Create Deepgram session on first chunk (lazy init)
                if session_id not in _live_sessions:
                    dg_session = DeepgramLiveSession(
                        session_id=str(session_id),
                        api_key=settings.DEEPGRAM_API_KEY,
                        model=settings.DEEPGRAM_MODEL,
                        redis_url=settings.REDIS_URL,
                    )
                    try:
                        await dg_session.start()
                        _live_sessions[session_id] = dg_session
                    except Exception as exc:
                        logger.error("Failed to start Deepgram session=%s: %s", session_id, exc)
                        await manager.send_to(ws, {"type": "error", "code": "deepgram_error", "detail": "Could not start transcription"})
                        continue

                try:
                    audio_bytes = base64.b64decode(event.chunk)
                    await _live_sessions[session_id].send(audio_bytes)
                except Exception as exc:
                    logger.warning("Audio send error session=%s: %s", session_id, exc)

            elif isinstance(event, EndMeetingEvent):
                if session_id is None:
                    await manager.send_to(ws, {"type": "error", "code": "not_joined", "detail": "Send joinSession first"})
                    continue

                dg_session = _live_sessions.pop(session_id, None)
                if dg_session:
                    await dg_session.finish()

                await manager.send_to(ws, {
                    "type": "meetingStatus",
                    "session_id": str(session_id),
                    "status": "ending",
                })

            elif isinstance(event, PingEvent):
                await manager.send_to(ws, PongEvent().model_dump())

    except WebSocketDisconnect:
        logger.info("WS disconnected user=%s session=%s", user_id_str, session_id)
    finally:
        # Clean up Deepgram session if still open
        if session_id:
            dg_session = _live_sessions.pop(session_id, None)
            if dg_session:
                await dg_session.finish()

        if subscriber_task and not subscriber_task.done():
            subscriber_task.cancel()
        if session_id:
            manager.unregister(session_id, ws)
