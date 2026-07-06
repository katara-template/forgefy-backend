"""Deepgram transcription via chunked pre-recorded API.

Instead of a WebSocket (which the SDK 6.x live path rejects with 400), audio
is accumulated in a buffer and flushed to Deepgram's pre-recorded endpoint
every FLUSH_INTERVAL seconds.  Results are published to Redis so the existing
Redis→WebSocket bridge delivers them to the browser.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import uuid
import wave
from contextlib import suppress
from datetime import UTC, datetime

import redis.asyncio as aioredis
from deepgram import AsyncDeepgramClient

logger = logging.getLogger(__name__)

_CHANNEL_PREFIX = "voxa:session:"
_FLUSH_INTERVAL = 4.0          # seconds between automatic flushes
_MIN_CHUNK_BYTES = 16_000 * 2  # 1 second of 16 kHz mono PCM-16 (32 kB) — skip smaller


def _pcm16_to_wav(pcm_bytes: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Wrap raw PCM-16 bytes in a WAV container so Deepgram can parse them."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def _make_redis_client(redis_url: str) -> aioredis.Redis:
    """Create an aioredis client, stripping URL SSL params that redis-py ignores."""
    from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

    extra: dict = {}
    url = redis_url
    if redis_url.startswith("rediss://"):
        parsed = urlparse(redis_url)
        qs = parse_qs(parsed.query)
        qs.pop("ssl_cert_reqs", None)
        url = urlunparse(parsed._replace(query=urlencode(qs, doseq=True)))
        extra["ssl_cert_reqs"] = "none"
    return aioredis.from_url(url, decode_responses=True, **extra)


class DeepgramLiveSession:
    """Buffers PCM-16 audio and transcribes in periodic chunks via pre-recorded API."""

    def __init__(
        self,
        session_id: str,
        api_key: str,
        model: str,
        redis_url: str,
    ) -> None:
        self._session_id = session_id
        self._api_key = api_key
        self._model = model
        self._redis_url = redis_url
        self._buffer = bytearray()
        self._lock = asyncio.Lock()
        self._flush_task: asyncio.Task | None = None  # type: ignore[type-arg]
        self._running = False
        self._redis: aioredis.Redis | None = None

    async def start(self) -> None:
        self._running = True
        self._redis = _make_redis_client(self._redis_url)
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info("Deepgram chunked session started session=%s model=%s", self._session_id, self._model)

    async def send(self, audio_bytes: bytes) -> None:
        if not self._running:
            return
        async with self._lock:
            self._buffer.extend(audio_bytes)

    async def finish(self) -> None:
        self._running = False

        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._flush_task

        # Flush whatever remains in the buffer
        async with self._lock:
            remaining = bytes(self._buffer)
            self._buffer.clear()

        if len(remaining) >= _MIN_CHUNK_BYTES:
            await self._transcribe_chunk(remaining)

        if self._redis:
            await self._redis.aclose()
            self._redis = None

        logger.info("Deepgram chunked session finished session=%s", self._session_id)

    # ── internal ──────────────────────────────────────────────────────────────

    async def _persist_transcript(self, text: str) -> None:
        """Write transcript segment to Firestore immediately in the web-service process."""
        try:
            from app.db.firebase import get_firestore_client
            db = get_firestore_client()
            await (
                db.collection("sessions")
                .document(self._session_id)
                .collection("events")
                .document(str(uuid.uuid4()))
                .set({
                    "session_id": self._session_id,
                    "event_type": "transcript.segment",
                    "payload": {"text": text},
                    "timestamp": datetime.now(UTC),
                })
            )
        except Exception as exc:
            logger.warning("Failed to persist transcript session=%s: %s", self._session_id, exc)

    async def _flush_loop(self) -> None:
        """Periodic flush loop — keeps running even if individual flushes fail."""
        while self._running:
            try:
                await asyncio.sleep(_FLUSH_INTERVAL)
                await self._flush()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Flush loop error (continuing) session=%s: %s", self._session_id, exc)

    async def _flush(self) -> None:
        async with self._lock:
            if not self._buffer:
                return
            chunk = bytes(self._buffer)
            self._buffer.clear()

        if len(chunk) >= _MIN_CHUNK_BYTES:
            await self._transcribe_chunk(chunk)

    async def _transcribe_chunk(self, audio_bytes: bytes, *, _retries: int = 2) -> None:
        """Transcribe one audio chunk; retries up to _retries times on failure."""
        for attempt in range(_retries + 1):
            try:
                wav_bytes = _pcm16_to_wav(audio_bytes)
                dg = AsyncDeepgramClient(api_key=self._api_key)
                response = await dg.listen.v1.media.transcribe_file(
                    request=wav_bytes,
                    model=self._model,
                    punctuate=True,
                )

                if not (response.results and response.results.channels):
                    return
                alts = response.results.channels[0].alternatives
                if not alts:
                    return
                text = alts[0].transcript
                if not text:
                    return

                msg = {
                    "type": "transcript",
                    "session_id": self._session_id,
                    "text": text,
                    "is_final": True,
                }

                if self._redis:
                    await self._redis.publish(
                        f"{_CHANNEL_PREFIX}{self._session_id}",
                        json.dumps(msg),
                    )

                # Persist transcript segment immediately in the web-service process
                # so blueprint generation always has data regardless of how backed-up
                # the Celery extraction queue is.
                await self._persist_transcript(text)

                from app.workers.extraction_worker import extract_requirements
                extract_requirements.apply_async(
                    args=[self._session_id, text],
                    queue="meeting.transcribe",
                )

                logger.debug("Chunk transcribed session=%s chars=%d", self._session_id, len(text))
                return

            except Exception as exc:
                if attempt < _retries:
                    backoff = 2 ** attempt
                    logger.warning(
                        "Transcription attempt %d failed session=%s, retrying in %ds: %s",
                        attempt + 1, self._session_id, backoff, exc,
                    )
                    await asyncio.sleep(backoff)
                else:
                    logger.warning(
                        "Chunk transcription failed after %d attempts session=%s: %s",
                        _retries + 1, self._session_id, exc,
                    )
