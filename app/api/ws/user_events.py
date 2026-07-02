"""WebSocket /ws/user/events — streams real-time events for the authenticated user.

Protocol (server → client):
  { "type": "quota_exceeded", "message": "...", "tier": "...", "limit": 500000 }
  { "type": "ping" }
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.config import get_settings
from app.core.security import decode_access_token

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/user/events")
async def ws_user_events(
    ws: WebSocket,
    token: str = Query(..., description="Bearer access token"),
) -> None:
    """Stream real-time user-scoped events (quota exceeded, etc.)."""
    settings = get_settings()
    try:
        user_id = decode_access_token(token, settings)
    except Exception:
        await ws.close(code=4001, reason="Unauthorized")
        return

    await ws.accept()
    logger.info("ws/user_events connected user=%s", user_id)

    channel = f"user:{user_id}:events"
    redis = ws.app.state.redis

    async def forward_events(pubsub) -> None:
        while True:
            try:
                msg = await pubsub.get_message(ignore_subscribe_messages=True)
                if msg and isinstance(msg.get("data"), str):
                    await ws.send_text(msg["data"])
            except Exception:
                break
            await asyncio.sleep(0.05)

    async with redis.pubsub() as pubsub:
        await pubsub.subscribe(channel)
        fwd_task = asyncio.create_task(forward_events(pubsub))

        try:
            while True:
                try:
                    await asyncio.wait_for(ws.receive_text(), timeout=25.0)
                except TimeoutError:
                    await ws.send_json({"type": "ping"})
                except WebSocketDisconnect:
                    break
        finally:
            fwd_task.cancel()
            await pubsub.unsubscribe(channel)

    logger.info("ws/user_events disconnected user=%s", user_id)
