"""WebSocket /ws/projects — streams the user's project list.

Protocol (server → client):
  { "type": "projects", "data": [ ...ProjectOut... ] }
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.config import get_settings
from app.core.security import decode_access_token
from app.schemas.project import ProjectOut

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/projects")
async def ws_projects(
    ws: WebSocket,
    token: str = Query(..., description="Bearer access token"),
) -> None:
    """Stream the caller's project list; polls every 5 s for changes."""
    settings = get_settings()
    try:
        user_id_str = decode_access_token(token, settings)
    except Exception:
        await ws.close(code=4001, reason="Unauthorized")
        return

    await ws.accept()
    logger.info("ws/projects connected user=%s", user_id_str)
    db = ws.app.state.firestore
    last_data: list | None = None

    async def push() -> None:
        nonlocal last_data
        docs = await db.collection("projects").where("owner_id", "==", user_id_str).get()
        projects = sorted(
            [d.to_dict() | {"id": d.id} for d in docs],
            key=lambda p: p.get("updated_at", ""),
            reverse=True,
        )
        data = [ProjectOut(**_coerce(p)).model_dump(mode="json") for p in projects]
        if data != last_data:
            last_data = data
            await ws.send_json({"type": "projects", "data": data})

    try:
        await push()
        while True:
            try:
                await asyncio.wait_for(ws.receive_text(), timeout=5.0)
            except asyncio.TimeoutError:
                pass
            await push()
    except WebSocketDisconnect:
        logger.info("ws/projects disconnected user=%s", user_id_str)
    except Exception:
        logger.exception("ws/projects error user=%s", user_id_str)


def _coerce(d: dict) -> dict:
    """Convert Firestore doc dict to ProjectOut-compatible dict."""
    import uuid
    return {
        "id": uuid.UUID(d["id"]),
        "owner_id": uuid.UUID(d["owner_id"]),
        "app_name": d["app_name"],
        "template_key": d["template_key"],
        "repo_full_name": d["repo_full_name"],
        "github_url": d["github_url"],
        "created_at": d["created_at"],
        "updated_at": d["updated_at"],
        "session_id": uuid.UUID(d["session_id"]) if d.get("session_id") else None,
        "blueprint_id": uuid.UUID(d["blueprint_id"]) if d.get("blueprint_id") else None,
        "preview_url": d.get("preview_url"),
        "artifact_url": d.get("artifact_url"),
        "is_updating": d.get("is_updating", False),
        "build_error": d.get("build_error"),
    }
