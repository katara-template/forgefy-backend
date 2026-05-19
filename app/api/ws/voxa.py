"""WebSocket gateway — /ws/voxa.

Auth: JWT passed as ?token= query param. The token is validated before
accepting the connection; invalid tokens are rejected with close code 4001.

Client→Server events: joinSession, streamAudio, endMeeting, ping
Server→Client events: transcript, featureDetected, blueprintReady, meetingStatus, pong, error

Redis pub/sub bridge: workers PUBLISH to voxa:session:{session_id};
this handler SUBSCRIBES and forwards each message to local WS connections.
"""
from __future__ import annotations

import asyncio
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
from app.workers.transcription_worker import (
    close_transcription_session,
    process_audio_chunk,
)
from app.schemas.ws_events import (
    ClientEvent,
    EndMeetingEvent,
    JoinSessionEvent,
    PingEvent,
    PongEvent,
    StreamAudioEvent,
)

logger = logging.getLogger(__name__)
router = APIRouter()

_client_event_adapter: TypeAdapter[ClientEvent] = TypeAdapter(ClientEvent)

# Redis channel prefix — workers publish here, WS handler subscribes.
_CHANNEL_PREFIX = "voxa:session:"


def _channel(session_id: uuid.UUID) -> str:
    return f"{_CHANNEL_PREFIX}{session_id}"


# ── Redis subscriber task ─────────────────────────────────────────────────────


async def _redis_subscriber(session_id: uuid.UUID, settings_redis_url: str) -> None:
    """Subscribe to the Redis channel for session_id and forward messages."""
    redis: aioredis.Redis = aioredis.from_url(settings_redis_url, decode_responses=True)
    pubsub = redis.pubsub()
    await pubsub.subscribe(_channel(session_id))
    try:
        async for raw in pubsub.listen():
            if raw["type"] != "message":
                continue
            try:
                payload: dict[str, Any] = json.loads(raw["data"])
            except (json.JSONDecodeError, TypeError):
                continue
            await manager.broadcast(session_id, payload)
    except asyncio.CancelledError:
        pass
    finally:
        await pubsub.unsubscribe(_channel(session_id))
        await pubsub.aclose()
        await redis.aclose()


# ── WebSocket endpoint ────────────────────────────────────────────────────────


@router.websocket("/ws/voxa")
async def ws_voxa(
    ws: WebSocket,
    token: str = Query(..., description="Bearer access token for auth"),
) -> None:
    """Bidirectional WebSocket gateway for live meeting sessions."""
    settings = get_settings()

    # Validate JWT before accepting — keeps unauthenticated sockets out.
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
                # Allow re-join (idempotent) — cancel previous subscriber first.
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
                process_audio_chunk.apply_async(
                    args=[str(session_id), event.chunk],
                    queue="meeting.audio",
                )

            elif isinstance(event, EndMeetingEvent):
                if session_id is None:
                    await manager.send_to(ws, {"type": "error", "code": "not_joined", "detail": "Send joinSession first"})
                    continue
                close_transcription_session.apply_async(
                    args=[str(session_id)],
                    queue="meeting.audio",
                )
                await manager.send_to(ws, {
                    "type": "meetingStatus",
                    "session_id": str(session_id),
                    "status": "ending",
                })
                # TODO Step 9 — call VoxaService.end_session and dispatch to Celery

            elif isinstance(event, PingEvent):
                await manager.send_to(ws, PongEvent().model_dump())

    except WebSocketDisconnect:
        logger.info("WS disconnected user=%s session=%s", user_id_str, session_id)
    finally:
        if subscriber_task and not subscriber_task.done():
            subscriber_task.cancel()
        if session_id:
            manager.unregister(session_id, ws)
