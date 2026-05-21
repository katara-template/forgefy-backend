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
import wave

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

    async def start(self) -> None:
        self._running = True
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
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Flush whatever remains in the buffer
        async with self._lock:
            remaining = bytes(self._buffer)
            self._buffer.clear()

        if len(remaining) >= _MIN_CHUNK_BYTES:
            await self._transcribe_chunk(remaining)

        logger.info("Deepgram chunked session finished session=%s", self._session_id)

    # ── internal ──────────────────────────────────────────────────────────────

    async def _flush_loop(self) -> None:
        while self._running:
            await asyncio.sleep(_FLUSH_INTERVAL)
            await self._flush()

    async def _flush(self) -> None:
        async with self._lock:
            if not self._buffer:
                return
            chunk = bytes(self._buffer)
            self._buffer.clear()

        if len(chunk) >= _MIN_CHUNK_BYTES:
            await self._transcribe_chunk(chunk)

    async def _transcribe_chunk(self, audio_bytes: bytes) -> None:
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

            r = aioredis.from_url(self._redis_url, decode_responses=True)
            try:
                await r.publish(
                    f"{_CHANNEL_PREFIX}{self._session_id}",
                    json.dumps(msg),
                )
            finally:
                await r.aclose()

            # Enqueue LangGraph extraction
            from app.workers.extraction_worker import extract_requirements
            extract_requirements.apply_async(
                args=[self._session_id, text],
                queue="meeting.transcribe",
            )

            logger.debug("Chunk transcribed session=%s chars=%d", self._session_id, len(text))

        except Exception as exc:
            logger.warning(
                "Chunk transcription error session=%s: %s",
                self._session_id, exc, exc_info=True,
            )
