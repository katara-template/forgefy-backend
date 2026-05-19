"""Deepgram streaming transcription session.

Wraps a single Deepgram v1 WebSocket connection for one meeting session.
The listening loop runs in a daemon thread so the main thread (Celery worker)
can keep pushing audio chunks without blocking.

Transcripts are published to Redis channel ``voxa:session:{session_id}``
so the FastAPI WebSocket gateway can forward them to connected browsers.
"""
from __future__ import annotations

import json
import logging
import threading
from typing import TYPE_CHECKING

import redis as sync_redis
from deepgram import DeepgramClient
from deepgram.listen import ListenV1Results
from deepgram.listen.v1.socket_client import EventType

if TYPE_CHECKING:
    from deepgram.listen.v1.socket_client import V1SocketClient

logger = logging.getLogger(__name__)

_CHANNEL_PREFIX = "voxa:session:"


class TranscriptionSession:
    """Persistent Deepgram WS connection for a single meeting session.

    One instance lives per session in the Celery worker process.  Audio chunks
    are pushed via ``send_chunk``; the event loop thread forwards transcripts
    to Redis as they arrive.
    """

    def __init__(
        self,
        session_id: str,
        api_key: str,
        model: str,
        redis_url: str,
        *,
        sample_rate: int = 16_000,
        encoding: str = "linear16",
    ) -> None:
        self._session_id = session_id
        self._channel = f"{_CHANNEL_PREFIX}{session_id}"
        self._redis = sync_redis.from_url(redis_url, decode_responses=True)
        self._connection: V1SocketClient | None = None
        self._ctx = None
        self._closed = False

        dg = DeepgramClient(api_key=api_key)
        self._ctx = dg.listen.v1.connect(
            model=model,
            encoding=encoding,
            sample_rate=sample_rate,
            interim_results=True,
            punctuate=True,
            endpointing=200,
        )
        self._connection = self._ctx.__enter__()
        self._connection.on(EventType.MESSAGE, self._on_message)

        # start_listening() blocks; run it on a daemon thread.
        self._listener_thread = threading.Thread(
            target=self._connection.start_listening,
            name=f"dg-listener-{session_id}",
            daemon=True,
        )
        self._listener_thread.start()
        logger.info("TranscriptionSession opened session=%s", session_id)

    # ── Public API ────────────────────────────────────────────────────────────

    def send_chunk(self, chunk_bytes: bytes) -> None:
        if self._connection and not self._closed:
            self._connection.send_media(chunk_bytes)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._connection:
                self._connection.send_close_stream()
            if self._ctx:
                self._ctx.__exit__(None, None, None)
        except Exception as exc:
            logger.warning("Error closing Deepgram connection session=%s: %s", self._session_id, exc)
        finally:
            try:
                self._redis.close()
            except Exception:
                pass
            logger.info("TranscriptionSession closed session=%s", self._session_id)

    # ── Event callback ────────────────────────────────────────────────────────

    def _on_message(self, message: object) -> None:
        if not isinstance(message, ListenV1Results):
            return
        alternatives = message.channel.alternatives if message.channel else []
        if not alternatives:
            return
        text = alternatives[0].transcript
        if not text:
            return
        payload = json.dumps(
            {
                "type": "transcript",
                "session_id": self._session_id,
                "text": text,
                "is_final": bool(message.is_final),
            }
        )
        try:
            self._redis.publish(self._channel, payload)
        except Exception as exc:
            logger.warning("Redis publish failed session=%s: %s", self._session_id, exc)
