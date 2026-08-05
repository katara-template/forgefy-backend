"""Deepgram live streaming client.

Talks to wss://api.deepgram.com/v1/listen directly over a WebSocket rather than
through deepgram-sdk. The backend's existing transcription path fell back to
chunked pre-recorded calls because the SDK's live interface rejected the
handshake; going direct avoids that and restores true interim results.

The socket is treated as disposable: it can drop at any point in a long
meeting, so a supervisor loop reconnects with backoff while audio keeps
accumulating in a bounded queue.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from urllib.parse import urlencode

from websockets.asyncio.client import connect

logger = logging.getLogger(__name__)

_ENDPOINT = "wss://api.deepgram.com/v1/listen"

# Deepgram closes a connection that receives no audio for ~10s.
_KEEPALIVE_SECONDS = 5.0

# ~30s of 32 kHz mono PCM-16. Bounds memory if Deepgram is unreachable for a
# sustained period; oldest audio is dropped first.
_MAX_QUEUED_CHUNKS = 2000

_MAX_BACKOFF_SECONDS = 30.0

TranscriptHandler = Callable[[str, bool, str], Awaitable[None]]


class DeepgramStream:
    """Streams PCM to Deepgram and invokes a handler per transcript result."""

    def __init__(
        self,
        api_key: str,
        model: str,
        language: str,
        on_transcript: TranscriptHandler,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._language = language
        self._on_transcript = on_transcript

        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=_MAX_QUEUED_CHUNKS)
        self._task: asyncio.Task | None = None
        self._running = False
        self._url = ""
        self._dropped_chunks = 0

    def start(self, sample_rate: int, channels: int) -> None:
        """Begin streaming. Audio parameters come from the SDK, never hardcoded."""
        params = {
            "model": self._model,
            "language": self._language,
            "encoding": "linear16",     # Zoom's raw audio is PCM-16 little-endian
            "sample_rate": str(sample_rate),
            "channels": str(channels),
            "punctuate": "true",
            "interim_results": "true",
            "smart_format": "true",
            # Marks the end of an utterance so we can forward a final result
            # promptly instead of waiting for the next speaker.
            "endpointing": "300",
        }
        if channels > 1:
            # Keeps speakers separable when per-participant capture is enabled.
            params["multichannel"] = "true"

        self._url = f"{_ENDPOINT}?{urlencode(params)}"
        self._running = True
        self._task = asyncio.create_task(self._supervise())
        logger.info("deepgram stream starting at %dHz, %dch", sample_rate, channels)

    def send(self, pcm: bytes) -> None:
        """Enqueue audio. Never blocks — drops oldest if the queue is saturated."""
        if not self._running:
            return
        try:
            self._queue.put_nowait(pcm)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(pcm)
                self._dropped_chunks += 1
                if self._dropped_chunks % 100 == 1:
                    logger.warning(
                        "deepgram queue saturated, dropped %d chunks", self._dropped_chunks
                    )
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass

    async def finish(self) -> None:
        """Flush remaining audio, ask Deepgram to finalize, and stop."""
        if not self._running:
            return
        self._running = False
        await self._queue.put(None)  # sentinel unblocks the sender

        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=15)
            except TimeoutError:
                logger.warning("deepgram shutdown timed out; cancelling")
                self._task.cancel()
            self._task = None
        logger.info("deepgram stream finished")

    # ── internals ────────────────────────────────────────────────────────────

    async def _supervise(self) -> None:
        """Keep a connection alive for as long as the meeting is running."""
        backoff = 1.0
        while self._running:
            try:
                async with connect(
                    self._url,
                    additional_headers={"Authorization": f"Token {self._api_key}"},
                    max_size=None,
                ) as ws:
                    logger.info("deepgram connected")
                    backoff = 1.0
                    await self._pump(ws)

                    # _pump returning without an exception means we finished
                    # cleanly; nothing left to reconnect for.
                    if not self._running:
                        return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not self._running:
                    return
                logger.warning("deepgram connection lost (%s), reconnecting in %.0fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF_SECONDS)

    async def _pump(self, ws) -> None:
        """Run sender, receiver and keepalive until one of them stops."""
        sender = asyncio.create_task(self._send_loop(ws))
        receiver = asyncio.create_task(self._receive_loop(ws))
        keepalive = asyncio.create_task(self._keepalive_loop(ws))

        done, pending = await asyncio.wait(
            {sender, receiver, keepalive},
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

        # Surface a failure so the supervisor reconnects rather than exiting.
        for task in done:
            exc = task.exception()
            if exc is not None:
                raise exc

    async def _send_loop(self, ws) -> None:
        while True:
            chunk = await self._queue.get()

            if chunk is None:
                # End of meeting: CloseStream makes Deepgram flush and return
                # the final transcript instead of discarding buffered audio.
                await ws.send(json.dumps({"type": "CloseStream"}))
                # Give the receiver a moment to collect the trailing results.
                await asyncio.sleep(2)
                return

            await ws.send(chunk)

    async def _receive_loop(self, ws) -> None:
        async for message in ws:
            try:
                data = json.loads(message)
            except (TypeError, ValueError):
                continue

            if data.get("type") != "Results":
                continue

            alternatives = (data.get("channel") or {}).get("alternatives") or []
            if not alternatives:
                continue

            text = (alternatives[0].get("transcript") or "").strip()
            if not text:
                continue

            # is_final marks a stable segment; speech_final marks the end of an
            # utterance. Only speech_final results are worth extracting
            # requirements from — the rest are for live display.
            is_final = bool(data.get("is_final")) and bool(data.get("speech_final"))
            speaker = self._speaker_of(alternatives[0])

            await self._on_transcript(text, is_final, speaker)

    async def _keepalive_loop(self, ws) -> None:
        while True:
            await asyncio.sleep(_KEEPALIVE_SECONDS)
            if self._queue.empty():
                await ws.send(json.dumps({"type": "KeepAlive"}))

    @staticmethod
    def _speaker_of(alternative: dict) -> str:
        """Best-effort speaker label from Deepgram diarization, if present."""
        words = alternative.get("words") or []
        if words and "speaker" in words[0]:
            return f"Speaker {words[0]['speaker']}"
        return ""

    async def __aenter__(self) -> DeepgramStream:
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.finish()
