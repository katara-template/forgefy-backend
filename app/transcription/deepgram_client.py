"""Deepgram streaming transcription session.

Wraps a single Deepgram v1 WebSocket connection for one meeting session.
The listening loop runs in a daemon thread so the main thread (Celery worker)
can keep pushing audio chunks without blocking.

Transcripts are published to Redis channel ``voxa:session:{session_id}``
so the FastAPI WebSocket gateway can forward them to connected browsers.

Reconnection: if the connection drops, up to MAX_RECONNECT_ATTEMPTS are made
with exponential backoff (2 s, 5 s, 15 s) before giving up.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import TYPE_CHECKING

import redis as sync_redis
from deepgram import DeepgramClient
from deepgram.listen import ListenV1Results
from deepgram.listen.v1.socket_client import EventType

if TYPE_CHECKING:
    from deepgram.listen.v1.socket_client import V1SocketClient

logger = logging.getLogger(__name__)

_CHANNEL_PREFIX = "voxa:session:"
_RECONNECT_DELAYS = (2, 5, 15)  # seconds between attempts


class TranscriptionSession:
    """Persistent Deepgram WS connection for a single meeting session."""

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
        self._api_key = api_key
        self._model = model
        self._redis_url = redis_url
        self._sample_rate = sample_rate
        self._encoding = encoding
        self._redis = sync_redis.from_url(redis_url, decode_responses=True)
        self._connection: V1SocketClient | None = None
        self._ctx = None
        self._closed = False
        self._reconnect_lock = threading.Lock()

        self._open_connection()

    # ── Public API ────────────────────────────────────────────────────────────

    def send_chunk(self, chunk_bytes: bytes) -> None:
        if self._connection and not self._closed:
            try:
                self._connection.send_media(chunk_bytes)
            except Exception as exc:
                logger.warning("send_chunk error session=%s: %s — triggering reconnect", self._session_id, exc)
                self._schedule_reconnect()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_connection()
        try:
            self._redis.close()
        except Exception:
            pass
        logger.info("TranscriptionSession closed session=%s", self._session_id)

    # ── Connection management ─────────────────────────────────────────────────

    def _open_connection(self) -> None:
        dg = DeepgramClient(api_key=self._api_key)
        self._ctx = dg.listen.v1.connect(
            model=self._model,
            encoding=self._encoding,
            sample_rate=self._sample_rate,
            interim_results=True,
            punctuate=True,
            endpointing=200,
        )
        self._connection = self._ctx.__enter__()
        self._connection.on(EventType.MESSAGE, self._on_message)
        self._connection.on(EventType.ERROR, self._on_error)

        listener = threading.Thread(
            target=self._connection.start_listening,
            name=f"dg-listener-{self._session_id}",
            daemon=True,
        )
        listener.start()
        logger.info("TranscriptionSession opened session=%s", self._session_id)

    def _close_connection(self) -> None:
        try:
            if self._connection:
                self._connection.send_close_stream()
            if self._ctx:
                self._ctx.__exit__(None, None, None)
        except Exception as exc:
            logger.warning("Error closing Deepgram connection session=%s: %s", self._session_id, exc)

    def _schedule_reconnect(self) -> None:
        if self._closed:
            return
        t = threading.Thread(
            target=self._reconnect_loop,
            name=f"dg-reconnect-{self._session_id}",
            daemon=True,
        )
        t.start()

    def _reconnect_loop(self) -> None:
        with self._reconnect_lock:
            if self._closed:
                return
            self._close_connection()

            for delay in _RECONNECT_DELAYS:
                if self._closed:
                    return
                time.sleep(delay)
                try:
                    self._open_connection()
                    logger.info("Deepgram reconnected session=%s", self._session_id)
                    return
                except Exception as exc:
                    logger.warning("Deepgram reconnect attempt failed session=%s delay=%ds: %s", self._session_id, delay, exc)

            logger.error("Deepgram reconnect exhausted session=%s — transcription stopped", self._session_id)

    # ── Event callbacks ───────────────────────────────────────────────────────

    def _on_error(self, error: object) -> None:
        logger.warning("Deepgram error session=%s: %s", self._session_id, error)
        if not self._closed:
            self._schedule_reconnect()

    def _on_message(self, message: object) -> None:
        if not isinstance(message, ListenV1Results):
            return
        alternatives = message.channel.alternatives if message.channel else []
        if not alternatives:
            return
        text = alternatives[0].transcript
        if not text:
            return

        is_final = bool(message.is_final)
        payload = json.dumps({
            "type": "transcript",
            "session_id": self._session_id,
            "text": text,
            "is_final": is_final,
        })

        try:
            self._redis.publish(self._channel, payload)
        except Exception as exc:
            logger.warning("Redis publish failed session=%s: %s", self._session_id, exc)
            return

        if is_final:
            try:
                from app.workers.extraction_worker import extract_requirements
                extract_requirements.apply_async(
                    args=[self._session_id, text],
                    queue="meeting.transcribe",
                )
            except Exception as exc:
                logger.warning("Failed to enqueue extraction session=%s: %s", self._session_id, exc)
