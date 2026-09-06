"""Recall.ai connector — creates a meeting bot via REST API.

Non-blocking: join() makes one HTTP call to create the bot and stores
the bot_id → session_id mapping in Redis.  All transcription arrives
via the /api/v1/webhooks/recall webhook endpoint.
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path

import httpx
import redis as sync_redis

logger = logging.getLogger(__name__)

_BOT_NAME = "Forgefy Bot"


class RecallBotCreationError(RuntimeError):
    """Recall refused to create the bot with a response that will not succeed on retry.

    Raised for 4xx other than 429 (bad meeting URL, wrong region, revoked key,
    rejected payload field) and for a 2xx body with no bot id. Carries Recall's
    own error text so the caller can surface it. Transient failures (429, 5xx,
    transport errors) are left as ``httpx`` exceptions so the task retries.
    """

# Conservative cap — Recall receives the image base64-inlined in the bot
# payload, so a large file slows every join and risks rejection.
_MAX_AVATAR_BYTES = 2 * 1024 * 1024

_BACKEND_ROOT = Path(__file__).resolve().parents[2]


def _load_avatar_b64(avatar_path: str | None) -> str | None:
    """Read the bot avatar JPEG and return it base64-encoded, or None.

    Missing file is not an error — the bot simply joins without an avatar.
    """
    if not avatar_path:
        return None

    path = Path(avatar_path)
    if not path.is_absolute():
        path = _BACKEND_ROOT / path

    if not path.is_file():
        logger.debug("Recall bot avatar not found at %s — joining without one", path)
        return None
    if path.suffix.lower() not in {".jpg", ".jpeg"}:
        logger.warning("Recall bot avatar must be a JPEG, got %s — skipping", path.name)
        return None

    data = path.read_bytes()
    if len(data) > _MAX_AVATAR_BYTES:
        logger.warning(
            "Recall bot avatar %s is %d bytes (max %d) — skipping",
            path.name, len(data), _MAX_AVATAR_BYTES,
        )
        return None

    return base64.b64encode(data).decode()


class RecallConnector:
    """Delegates meeting capture to Recall.ai cloud bots."""

    def __init__(
        self,
        api_key: str,
        region: str,
        redis_url: str,
        webhook_base_url: str,
        avatar_path: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = f"https://{region}.recall.ai/api/v1"
        self._redis_url = redis_url
        self._webhook_url = webhook_base_url.rstrip("/") + "/api/v1/webhooks/recall"
        self._avatar_path = avatar_path
        self._bot_id: str | None = None

    # ── Public interface ──────────────────────────────────────────────────────

    def join(self, meeting_url: str, session_id: str) -> None:
        """Create a Recall.ai bot for meeting_url; store bot↔session in Redis."""
        payload = {
            "meeting_url": meeting_url,
            "bot_name": _BOT_NAME,
            "recording_config": {
                "transcript": {
                    "provider": {
                        # prioritize_accuracy buffers a few seconds longer
                        # before finalizing, producing full sentences instead
                        # of word-fragments. Same engine and pricing as
                        # prioritize_low_latency — only the segmentation differs.
                        "recallai_streaming": {
                            "mode": "prioritize_accuracy",
                            "language_code": "en",
                        }
                    },
                },
                "realtime_endpoints": [
                    {
                        "type": "webhook",
                        "url": self._webhook_url,
                        "events": ["transcript.data"],
                    }
                ],
            },
            "automatic_leave": {
                "everyone_left_timeout": 2,
            },
        }

        # Show a branded image as the bot's camera feed instead of the
        # platform's default letter avatar.
        avatar_b64 = _load_avatar_b64(self._avatar_path)
        if avatar_b64:
            image = {"kind": "jpeg", "b64_data": avatar_b64}
            payload["automatic_video_output"] = {
                "in_call_recording": image,
                "in_call_not_recording": image,
            }

        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"{self._base_url}/bot/",
                json=payload,
                headers=self._auth_headers(),
            )

            # 429 / 5xx are worth another attempt — hand the caller an
            # httpx.HTTPStatusError so its retry policy fires.
            if resp.status_code == 429 or resp.status_code >= 500:
                logger.warning(
                    "Recall transient error session=%s status=%s body=%s",
                    session_id, resp.status_code, resp.text,
                )
                resp.raise_for_status()

            # Any other 4xx fails identically on retry — surface it as terminal
            # with Recall's own error body attached.
            if resp.is_error:
                logger.error(
                    "Recall bot creation failed session=%s status=%s body=%s",
                    session_id, resp.status_code, resp.text,
                )
                raise RecallBotCreationError(
                    f"Recall returned {resp.status_code}: {resp.text[:500]}"
                )

            try:
                bot_id = resp.json()["id"]
            except (ValueError, KeyError) as exc:
                raise RecallBotCreationError(
                    f"Recall response had no bot id: {resp.text[:300]}"
                ) from exc

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
    """Make a Recall bot leave the call — safe to call even if it's already gone."""
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.post(
                f"{base_url}/bot/{bot_id}/leave_call/",
                headers={"Authorization": f"Token {api_key}"},
            )
        if resp.is_error:
            logger.warning(
                "Recall bot removal failed bot_id=%s status=%s body=%s",
                bot_id, resp.status_code, resp.text,
            )
        else:
            logger.info("Recall bot removed bot_id=%s", bot_id)
    except Exception as exc:
        logger.warning("Failed to remove Recall bot %s: %s", bot_id, exc)


def delete_media(bot_id: str, base_url: str, api_key: str) -> None:
    """Purge all recording/transcript media Recall.ai stores for a bot.

    Transcripts stream to us in real time via webhook, so Recall's stored
    copy is never read — deleting it keeps user data off Recall's platform.

    Raises on failure so callers can retry: Recall rejects deletion while
    media is still processing shortly after the call ends. A 404 means the
    bot or its media is already gone, which is the desired end state.
    """
    with httpx.Client(timeout=15) as client:
        resp = client.post(
            f"{base_url}/bot/{bot_id}/delete_media/",
            headers={"Authorization": f"Token {api_key}"},
        )
    if resp.status_code == 404:
        logger.info("Recall media already gone bot_id=%s", bot_id)
        return
    if resp.is_error:
        logger.warning(
            "Recall media deletion failed bot_id=%s status=%s body=%s",
            bot_id, resp.status_code, resp.text,
        )
    resp.raise_for_status()
    logger.info("Recall media deleted bot_id=%s", bot_id)
