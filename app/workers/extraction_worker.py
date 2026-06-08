"""Extraction worker — runs the LangGraph agent pipeline on transcript segments."""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone

import redis as sync_redis

from app.config import get_settings
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

_CHANNEL_PREFIX = "voxa:session:"


async def _persist_events(
    session_id: str,
    events: list[dict],
    transcript_segment: str = "",
) -> None:
    """Write extraction events (and transcript segment) to Firestore."""
    from app.db.firebase import refresh_async_firestore_client

    db = refresh_async_firestore_client()
    events_ref = db.collection("sessions").document(session_id).collection("events")
    now = datetime.now(timezone.utc)

    if transcript_segment:
        await events_ref.document(str(uuid.uuid4())).set({
            "session_id": session_id,
            "event_type": "transcript.segment",
            "payload": {"text": transcript_segment},
            "timestamp": now,
        })

    for event in events:
        await events_ref.document(str(uuid.uuid4())).set({
            "session_id": session_id,
            "event_type": event["sub_state"],
            "payload": event["payload"],
            "timestamp": now,
        })


def _run_extraction(transcript: str, settings) -> list[dict]:
    """Dispatch to Ollama, Gemini, or Claude pipeline based on BP_MODEL setting."""
    if settings.BP_MODEL == "Qwen3":
        from app.ai.agents.ollama_synthesizer import run as ollama_run
        return ollama_run(transcript, settings.OLLAMA_URL, settings.OLLAMA_MODEL, timeout=settings.OLLAMA_TIMEOUT)

    if settings.BP_MODEL == "gemini":
        from app.ai.agents.gemini_synthesizer import run as gemini_run
        return gemini_run(transcript, settings.GEMINI_API_KEY, settings.GEMINI_MODEL)

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
    """Run the LangGraph pipeline on a finalized transcript segment."""
    settings = get_settings()
    events = _run_extraction(transcript_segment, settings)

    if not events:
        logger.debug("No events extracted for session=%s", session_id)
        return

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

    try:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_persist_events(session_id, events, transcript_segment))
        finally:
            loop.close()
    except Exception as exc:
        logger.warning("Failed to persist extraction events session=%s: %s", session_id, exc)
