"""Extraction worker — runs the LangGraph agent pipeline on transcript segments.

Receives finalized transcript text on the ``meeting.transcribe`` queue,
runs the four-agent pipeline, publishes each extraction event to Redis
(for real-time WS delivery), and persists each event to meeting_events
(for blueprint aggregation in Step 8).
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid

import redis as sync_redis
from sqlalchemy.pool import NullPool

from app.config import get_settings
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

_CHANNEL_PREFIX = "voxa:session:"


async def _persist_events(
    session_id: str,
    events: list[dict],
    database_url: str,
    transcript_segment: str = "",
) -> None:
    """Write extraction events (and transcript segment) to meeting_events."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from app.db.models.meeting_event import MeetingEvent

    engine = create_async_engine(database_url, poolclass=NullPool)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with factory() as db:
            # Always store the raw transcript segment so the blueprint generator
            # can use it as a fallback if extraction yielded nothing.
            if transcript_segment:
                db.add(MeetingEvent(
                    session_id=uuid.UUID(session_id),
                    event_type="transcript.segment",
                    payload={"text": transcript_segment},
                ))
            for event in events:
                db.add(
                    MeetingEvent(
                        session_id=uuid.UUID(session_id),
                        event_type=event["sub_state"],
                        payload=event["payload"],
                    )
                )
            await db.commit()
    finally:
        await engine.dispose()


def _run_extraction(transcript: str, settings) -> list[dict]:
    """Dispatch to Ollama, Gemini, or Claude pipeline based on BP_MODEL setting."""
    if settings.BP_MODEL == "Qwen3":
        from app.ai.agents.ollama_synthesizer import run as ollama_run
        return ollama_run(transcript, settings.OLLAMA_URL, settings.OLLAMA_MODEL, timeout=settings.OLLAMA_TIMEOUT)

    if settings.BP_MODEL == "gemini":
        from app.ai.agents.gemini_synthesizer import run as gemini_run
        return gemini_run(transcript, settings.GEMINI_API_KEY, settings.GEMINI_MODEL)

    # Claude: LangGraph multi-agent pipeline with single-call fallback
    from app.ai.pipeline import run_pipeline
    events = run_pipeline(
        transcript=transcript,
        api_key=settings.ANTHROPIC_API_KEY,
        model=settings.ANTHROPIC_MODEL,
    )
    if not events:
        from app.ai.agents.synthesizer import run as claude_run
        events = claude_run(transcript, settings.ANTHROPIC_API_KEY, settings.ANTHROPIC_MODEL)
    return events


@celery_app.task(name="app.workers.extraction_worker.extract_requirements")
def extract_requirements(session_id: str, transcript_segment: str) -> None:
    """Run the LangGraph pipeline on a finalized transcript segment.

    Each extraction event (FEATURE_FOUND, QUESTION_FOUND, etc.) is:
    - Published to the session's Redis channel for real-time WS delivery
    - Persisted to meeting_events for blueprint aggregation
    """
    settings = get_settings()
    events = _run_extraction(transcript_segment, settings)

    if not events:
        logger.debug("No events extracted for session=%s", session_id)
        return

    # Publish to Redis for real-time delivery.
    r = sync_redis.from_url(settings.REDIS_URL, decode_responses=True)
    channel = f"{_CHANNEL_PREFIX}{session_id}"
    try:
        for event in events:
            payload = json.dumps(
                {
                    "type": "featureDetected",
                    "session_id": session_id,
                    "sub_state": event["sub_state"],
                    "payload": event["payload"],
                }
            )
            r.publish(channel, payload)
            logger.debug("Published %s for session=%s", event["sub_state"], session_id)
    finally:
        r.close()

    # Persist to DB for blueprint aggregation.
    try:
        asyncio.run(
            _persist_events(session_id, events, settings.DATABASE_URL, transcript_segment)
        )
    except Exception as exc:
        logger.warning("Failed to persist extraction events session=%s: %s", session_id, exc)
