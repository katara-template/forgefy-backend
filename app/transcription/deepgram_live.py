"""Async Deepgram live transcription session — used directly by the WebSocket gateway.

Audio chunks arrive from the browser via WebSocket, are forwarded here without
any Celery intermediary, and transcripts are published to Redis so the existing
Redis→WebSocket bridge delivers them to the browser in real time.
"""
from __future__ import annotations

import json
import logging

import redis.asyncio as aioredis
from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents

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
        self._connection = None

    async def start(self) -> None:
        dg = DeepgramClient(self._api_key)
        self._connection = dg.listen.asynclive.v("1")
        self._connection.on(LiveTranscriptionEvents.Transcript, self._on_transcript)
        options = LiveOptions(
            model=self._model,
            encoding="linear16",
            sample_rate=16000,
            interim_results=True,
            punctuate=True,
            endpointing=300,
        )
        started = await self._connection.start(options)
        if not started:
            raise RuntimeError(f"Deepgram failed to start session={self._session_id}")
        logger.info("Deepgram live session started session=%s model=%s", self._session_id, self._model)

    async def send(self, audio_bytes: bytes) -> None:
        if self._connection is not None:
            await self._connection.send(audio_bytes)

    async def finish(self) -> None:
        if self._connection is not None:
            try:
                await self._connection.finish()
            except Exception as exc:
                logger.warning("Deepgram finish error session=%s: %s", self._session_id, exc)
            finally:
                self._connection = None
        logger.info("Deepgram live session finished session=%s", self._session_id)

    async def _on_transcript(self, result, **kwargs) -> None:
        try:
            alternatives = result.channel.alternatives if result.channel else []
            if not alternatives:
                return
            text = alternatives[0].transcript
            if not text:
                return
            is_final = bool(result.is_final)

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
