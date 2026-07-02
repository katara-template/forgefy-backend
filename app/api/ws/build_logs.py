"""WebSocket /ws/projects/{project_id}/logs — streams real-time build activity.

Protocol (server → client):
  { "type": "thinking" | "tool" | "text" | "info" | "started" | "warning" | "error" | "done", "message": "..." }
  { "type": "ping" }   — sent every 25 s as a heartbeat
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.config import get_settings
from app.core.security import decode_access_token

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/projects/{project_id}/logs")
async def ws_build_logs(
    ws: WebSocket,
    project_id: str,
    token: str = Query(..., description="Bearer access token"),
) -> None:
    """Stream real-time build/update log events for a project."""
    settings = get_settings()
    try:
        decode_access_token(token, settings)
    except Exception:
        await ws.close(code=4001, reason="Unauthorized")
        return

    await ws.accept()
    logger.info("ws/build_logs connected project=%s", project_id)

    channel = f"build:{project_id}:logs"
    history_key = f"build:{project_id}:log_history"
    redis = ws.app.state.redis

    # Replay history before subscribing so the client never misses earlier entries.
    # Subscribe first, then replay — this ordering means any events emitted between
    # the two steps arrive via pub/sub and are de-duplicated on the client by ordering
    # (history entries all arrive before the first live message).
    async with redis.pubsub() as pubsub:
        await pubsub.subscribe(channel)

        history: list[str] = await redis.lrange(history_key, 0, -1)
        for entry in history:
            try:
                await ws.send_text(entry)
            except Exception:
                return  # client already gone

    async def forward_logs(pubsub) -> None:
        """Poll Redis pub/sub and forward events to the WebSocket."""
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
        fwd_task = asyncio.create_task(forward_logs(pubsub))

        try:
            while True:
                try:
                    await asyncio.wait_for(ws.receive_text(), timeout=25.0)
                except TimeoutError:
                    # Heartbeat so the connection stays alive
                    await ws.send_json({"type": "ping"})
                except WebSocketDisconnect:
                    break
        finally:
            fwd_task.cancel()
            await pubsub.unsubscribe(channel)

    logger.info("ws/build_logs disconnected project=%s", project_id)
