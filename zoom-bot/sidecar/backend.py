"""Signed webhook client for reporting bot events back to the orchestrator.

The self-hosted bot reports the same way Recall.ai does — an HTTP POST per
event — so the orchestrator can drive session state, transcript fan-out and
requirement extraction through one code path regardless of which bot produced
the audio.

Requests are signed HMAC-SHA256 over "{timestamp}.{body}", matching the scheme
already used by the extract API's outbound webhooks.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time

import httpx

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 4
_TIMEOUT_SECONDS = 10


class BackendClient:
    """Posts transcript and lifecycle events to the orchestrator."""

    def __init__(self, url: str, secret: str, session_id: str) -> None:
        self._url = url
        self._secret = secret.encode()
        self._session_id = session_id
        self._client = httpx.AsyncClient(timeout=_TIMEOUT_SECONDS)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def send_status(self, status: str, detail: str = "") -> None:
        payload = {"type": "status", "status": status}
        if detail:
            payload["detail"] = detail
        await self._post(payload)

    async def send_transcript(self, text: str, *, is_final: bool, speaker: str = "") -> None:
        await self._post({
            "type": "transcript",
            "text": text,
            "is_final": is_final,
            "speaker": speaker,
        })

    # ── internals ────────────────────────────────────────────────────────────

    async def _post(self, payload: dict) -> None:
        payload["session_id"] = self._session_id
        body = json.dumps(payload, separators=(",", ":")).encode()
        timestamp = str(int(time.time()))

        signature = hmac.new(
            self._secret,
            f"{timestamp}.".encode() + body,
            hashlib.sha256,
        ).hexdigest()

        headers = {
            "Content-Type": "application/json",
            "X-Forgefy-Timestamp": timestamp,
            "X-Forgefy-Signature": signature,
        }

        for attempt in range(_MAX_ATTEMPTS):
            try:
                resp = await self._client.post(self._url, content=body, headers=headers)
                if resp.is_success:
                    return
                # A rejected signature or malformed body will fail identically
                # on every retry — only back off for transient server errors.
                if resp.status_code < 500:
                    logger.error(
                        "backend rejected %s event: %s %s",
                        payload["type"], resp.status_code, resp.text[:200],
                    )
                    return
                logger.warning("backend returned %s, retrying", resp.status_code)
            except Exception as exc:
                logger.warning("backend post failed (attempt %d): %s", attempt + 1, exc)

            if attempt < _MAX_ATTEMPTS - 1:
                await asyncio.sleep(2 ** attempt)

        # Losing an interim transcript is survivable; losing a terminal status
        # would strand the session, so make the failure loud.
        logger.error("giving up on %s event after %d attempts", payload["type"], _MAX_ATTEMPTS)
