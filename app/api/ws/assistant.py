"""WebSocket /ws/assistant/chat — real-time assistant chat channel.

Protocol:
  client → server: { "message": "...", "page": "...", "mode": "chat"|"build",
                     "conversation_id": "...", "history": [...] }
  server → client: { "type": "thinking" }
                   { "type": "reply", "response", "links", "action",
                     "authenticated", "conversation_id" }
                   { "type": "error", "message": "..." }
                   { "type": "ping" }   — heartbeat every 25s

Auth is optional: a valid token identifies the user (enabling threads + memory);
without one the visitor is anonymous, exactly like the HTTP /chat endpoint. The
message logic itself is shared with that endpoint via process_chat().
"""
from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.api.v1.assistant import process_chat
from app.config import get_settings
from app.core.exceptions import ForgefyError
from app.deps import get_optional_user
from app.schemas.assistant import AssistantChatRequest

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/assistant/chat")
async def ws_assistant_chat(
    ws: WebSocket,
    token: str | None = Query(default=None),
) -> None:
    """Stream assistant chat turns. Token is optional (anonymous allowed)."""
    settings = get_settings()
    await ws.accept()

    db = ws.app.state.firestore
    try:
        user = await get_optional_user(token=token, db=db, settings=settings)
    except Exception:  # noqa: BLE001 — auth failure just means anonymous here
        user = None

    logger.info("ws/assistant_chat connected authed=%s", user is not None)

    try:
        while True:
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=25.0)
            except TimeoutError:
                await ws.send_json({"type": "ping"})
                continue
            except WebSocketDisconnect:
                break

            try:
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise ValueError("frame is not an object")
            except Exception:  # noqa: BLE001
                await ws.send_json({"type": "error", "message": "Malformed message."})
                continue

            if data.get("type") == "ping":  # client keepalive
                continue

            try:
                req = AssistantChatRequest(
                    message=str(data.get("message", "")),
                    page=data.get("page"),
                    mode=data.get("mode"),
                    conversation_id=data.get("conversation_id"),
                    history=data.get("history"),
                )
            except Exception:  # noqa: BLE001
                await ws.send_json({"type": "error", "message": "Invalid message payload."})
                continue

            await ws.send_json({"type": "thinking"})
            try:
                resp = await process_chat(db, user, req)
            except ForgefyError as exc:
                await ws.send_json({"type": "error", "message": exc.detail})
                continue
            except Exception:  # noqa: BLE001
                logger.exception("ws/assistant_chat process error")
                await ws.send_json(
                    {"type": "error", "message": "Something went wrong. Please try again."}
                )
                continue

            await ws.send_json({"type": "reply", **resp.model_dump()})
    finally:
        logger.info("ws/assistant_chat disconnected")
