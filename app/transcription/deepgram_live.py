"""Async Deepgram live transcription — deepgram-sdk 6.x API.

Uses AsyncDeepgramClient.listen.v1.connect() (asynccontextmanager) to open
a streaming WebSocket, then forwards audio bytes and publishes transcripts
to Redis so the existing Redis→WebSocket bridge delivers them to the browser.
"""
from __future__ import annotations

import asyncio
import json
import logging

import redis.asyncio as aioredis
from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from deepgram.listen.v1.types.listen_v1results import ListenV1Results

logger = logging.getLogger(__name__)

_CHANNEL_PREFIX = "voxa:session:"


class DeepgramLiveSession:
    """One Deepgram async streaming connection per physical meeting session."""

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
        self._cm = None
        self._connection = None
        self._listen_task: asyncio.Task | None = None  # type: ignore[type-arg]

    async def start(self) -> None:
        dg = AsyncDeepgramClient(api_key=self._api_key)
        self._cm = dg.listen.v1.connect(
            model=self._model,
            encoding="linear16",
            sample_rate=16000,
            interim_results=True,
            punctuate=True,
            endpointing=300,
        )
        self._connection = await self._cm.__aenter__()
        self._connection.on(EventType.MESSAGE, self._on_transcript)
        # Run the listener loop as a background task
        self._listen_task = asyncio.create_task(self._connection.start_listening())
        logger.info("Deepgram live started session=%s model=%s", self._session_id, self._model)

    async def send(self, audio_bytes: bytes) -> None:
        if self._connection is not None:
            await self._connection.send_media(audio_bytes)

    async def finish(self) -> None:
        if self._connection is not None:
            try:
                await self._connection.send_close_stream()
            except Exception as exc:
                logger.warning("send_close_stream error session=%s: %s", self._session_id, exc)

        if self._listen_task and not self._listen_task.done():
            self._listen_task.cancel()
            try:
                await self._listen_task
            except (asyncio.CancelledError, Exception):
                pass

        if self._cm is not None:
            try:
                await self._cm.__aexit__(None, None, None)
            except Exception as exc:
                logger.warning("Deepgram context exit error session=%s: %s", self._session_id, exc)
            finally:
                self._cm = None
                self._connection = None

        logger.info("Deepgram live finished session=%s", self._session_id)

    async def _on_transcript(self, data) -> None:
        try:
            if not isinstance(data, ListenV1Results):
                return
            alternatives = data.channel.alternatives if data.channel else []
            if not alternatives:
                return
            text = alternatives[0].transcript
            if not text:
                return
            is_final = bool(data.is_final)

            msg = {
                "type": "transcript",
                "session_id": self._session_id,
                "text": text,
                "is_final": is_final,
            }

            # Publish to Redis — the WS gateway's Redis subscriber broadcasts to browsers
            r = aioredis.from_url(self._redis_url, decode_responses=True)
            try:
                await r.publish(
                    f"{_CHANNEL_PREFIX}{self._session_id}",
                    json.dumps(msg),
                )
            finally:
                await r.aclose()

            # Enqueue LangGraph extraction for every final segment
            if is_final:
                from app.workers.extraction_worker import extract_requirements
                extract_requirements.apply_async(
                    args=[self._session_id, text],
                    queue="meeting.transcribe",
                )
        except Exception as exc:
            logger.warning(
                "Deepgram transcript handler error session=%s: %s",
                self._session_id, exc, exc_info=True,
            )
