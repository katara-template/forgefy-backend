"""Recall.ai connector — creates a meeting bot via REST API.

Non-blocking: join() makes one HTTP call to create the bot and stores
the bot_id → session_id mapping in Redis.  All transcription arrives
via the /api/v1/webhooks/recall webhook endpoint.
"""
from __future__ import annotations

import logging

import httpx
import redis as sync_redis

logger = logging.getLogger(__name__)

_BOT_NAME = "Forgefy Bot"


class RecallConnector:
    """Delegates meeting capture to Recall.ai cloud bots."""

    def __init__(
        self,
        api_key: str,
        region: str,
        redis_url: str,
        webhook_base_url: str,
    ) -> None:
        self._api_key = api_key
        self._base_url = f"https://{region}.recall.ai/api/v1"
        self._redis_url = redis_url
        self._webhook_url = webhook_base_url.rstrip("/") + "/api/v1/webhooks/recall"
        self._bot_id: str | None = None

    # ── Public interface ──────────────────────────────────────────────────────

    def join(self, meeting_url: str, session_id: str) -> None:
        """Create a Recall.ai bot for meeting_url; store bot↔session in Redis."""
        payload = {
            "meeting_url": meeting_url,
            "bot_name": _BOT_NAME,
            "real_time_transcription": {
                "destination_url": self._webhook_url,
                "partial_results": False,
            },
            "automatic_leave": {
                "everyone_left_timeout": 2,
            },
        }
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"{self._base_url}/bot/",
                json=payload,
                headers=self._auth_headers(),
            )
            resp.raise_for_status()
            bot_id = resp.json()["id"]

        self._bot_id = bot_id
        self._store_mapping(bot_id, session_id)
        logger.info("Recall bot created bot_id=%s session=%s", bot_id, session_id)

    def leave(self) -> None:
        """Remove the bot from the meeting."""
        if not self._bot_id:
            return
        remove_bot(self._bot_id, self._base_url, self._api_key)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Token {self._api_key}"}

    def _store_mapping(self, bot_id: str, session_id: str) -> None:
        r = sync_redis.from_url(self._redis_url)
        try:
            r.set(f"recall:bot:{bot_id}", session_id, ex=86_400)
            r.set(f"recall:session:{session_id}", bot_id, ex=86_400)
        finally:
            r.close()


def remove_bot(bot_id: str, base_url: str, api_key: str) -> None:
    """DELETE a Recall bot — safe to call even if already gone."""
    try:
        with httpx.Client(timeout=15) as client:
            client.delete(
                f"{base_url}/bot/{bot_id}/",
                headers={"Authorization": f"Token {api_key}"},
            )
        logger.info("Recall bot removed bot_id=%s", bot_id)
    except Exception as exc:
        logger.warning("Failed to remove Recall bot %s: %s", bot_id, exc)
