"""In-process WebSocket connection registry.

Maps session_id → set[WebSocket] so one server process can fan-out a single
Redis pub/sub message to every local subscriber of that session.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections import defaultdict

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Thread-safe (asyncio-safe) registry of active WebSocket connections."""

    def __init__(self) -> None:
        self._connections: dict[uuid.UUID, set[WebSocket]] = defaultdict(set)

    def register(self, session_id: uuid.UUID, ws: WebSocket) -> None:
        self._connections[session_id].add(ws)
        logger.debug("WS registered session=%s total=%d", session_id, len(self._connections[session_id]))

    def unregister(self, session_id: uuid.UUID, ws: WebSocket) -> None:
        bucket = self._connections.get(session_id)
        if bucket:
            bucket.discard(ws)
            if not bucket:
                del self._connections[session_id]
        logger.debug("WS unregistered session=%s", session_id)

    async def broadcast(self, session_id: uuid.UUID, payload: dict) -> None:
        """Send payload to every WebSocket subscribed to session_id."""
        bucket = self._connections.get(session_id)
        if not bucket:
            return
        text = json.dumps(payload)
        dead: list[WebSocket] = []
        for ws in list(bucket):
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.unregister(session_id, ws)

    async def send_to(self, ws: WebSocket, payload: dict) -> None:
        """Send payload to a single WebSocket connection."""
        await ws.send_text(json.dumps(payload))


# Module-level singleton shared across the process lifetime.
manager = ConnectionManager()
