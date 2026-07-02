"""WebSocket endpoint — /ws/sessions

Pushes the authenticated user's session list immediately on connect, then
re-polls Firestore every 5 s and pushes again only when the list has changed.

Protocol (server → client):
  { "type": "sessions", "data": [ ...SessionOut... ] }
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.config import get_settings
from app.core.security import decode_access_token
from app.schemas.session import SessionOut

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/sessions")
async def ws_sessions(
    ws: WebSocket,
    token: str = Query(..., description="Bearer access token"),
) -> None:
    """Stream the caller's session list; polls every 5 s for changes."""
    settings = get_settings()
    try:
        user_id_str = decode_access_token(token, settings)
    except Exception:
        await ws.close(code=4001, reason="Unauthorized")
        return

    await ws.accept()
    logger.info("ws/sessions connected user=%s", user_id_str)
    db = ws.app.state.firestore
    last_data: list | None = None

    async def push() -> None:
        nonlocal last_data
        docs = await db.collection("sessions").where("user_id", "==", user_id_str).get()
        sessions = sorted(
            [d.to_dict() | {"id": d.id} for d in docs],
            key=lambda s: s.get("created_at", ""),
            reverse=True,
        )
        data = [SessionOut.model_validate(s).model_dump(mode="json") for s in sessions]
        if data != last_data:
            last_data = data
            await ws.send_json({"type": "sessions", "data": data})

    try:
        await push()
        while True:
            try:
                # Wait for any client message or the 5 s poll interval
                await asyncio.wait_for(ws.receive_text(), timeout=5.0)
            except TimeoutError:
                pass
            await push()
    except WebSocketDisconnect:
        logger.info("ws/sessions disconnected user=%s", user_id_str)
    except Exception:
        logger.exception("ws/sessions error user=%s", user_id_str)
