"""Extraction worker — runs the LangGraph agent pipeline on transcript segments.

Receives finalized transcript text on the ``meeting.transcribe`` queue,
runs the four-agent pipeline, and publishes each extraction event to
Redis so the WS gateway can push ``featureDetected`` events to clients.
"""
from __future__ import annotations

import json
import logging

import redis as sync_redis

from app.ai.pipeline import run_pipeline
from app.config import get_settings
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

_CHANNEL_PREFIX = "voxa:session:"


@celery_app.task(name="app.workers.extraction_worker.extract_requirements")
def extract_requirements(session_id: str, transcript_segment: str) -> None:
    """Run the LangGraph pipeline on a finalized transcript segment.

    Each extraction event (FEATURE_FOUND, QUESTION_FOUND, etc.) is published
    to the session's Redis channel so the WS gateway forwards it to browsers.
    """
    settings = get_settings()
    events = run_pipeline(
        transcript=transcript_segment,
        api_key=settings.ANTHROPIC_API_KEY,
        model=settings.ANTHROPIC_MODEL,
    )

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
            logger.debug(
                "Published %s for session=%s", event["sub_state"], session_id
            )
    finally:
        r.close()
