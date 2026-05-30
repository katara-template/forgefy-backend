"""Upload worker — transcribes a pre-recorded audio file and runs the full pipeline.

Physical-meeting flow:
  1. Deepgram pre-recorded API transcribes the uploaded file.
  2. LangGraph extraction pipeline runs on the full transcript.
  3. Extraction events are persisted and BlueprintAggregator generates the blueprint,
     transitioning the session to BLUEPRINT_READY.
  4. Redis publish events notify the WebSocket gateway for real-time UI updates.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import redis as sync_redis

from app.config import get_settings
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)
_CHANNEL_PREFIX = "voxa:session:"


@celery_app.task(name="app.workers.upload_worker.transcribe_upload")
def transcribe_upload(session_id: str, file_path: str) -> None:
    """Transcribe an uploaded recording and run extraction + blueprint pipeline."""
    settings = get_settings()
    path = Path(file_path)

    # 1. Transcribe
    try:
        transcript = _transcribe(file_path, settings)
    except Exception as exc:
        logger.error("Transcription failed session=%s: %s", session_id, exc, exc_info=True)
        return
    finally:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass

    if not transcript:
        logger.warning("Empty transcript session=%s — nothing to process", session_id)
        return

    logger.info("Transcript ready session=%s chars=%d", session_id, len(transcript))

    # 2. Publish full transcript to Redis (WebSocket picks it up)
    _publish_many(session_id, [{
        "type": "transcript",
        "session_id": session_id,
        "text": transcript,
        "is_final": True,
    }], settings)

    # 3. Run extraction pipeline
    from app.workers.extraction_worker import _run_extraction
    try:
        events = _run_extraction(transcript, settings)
    except Exception as exc:
        logger.error("Extraction failed session=%s: %s", session_id, exc, exc_info=True)
        events = []
    logger.info("Extraction done session=%s events=%d backend=%s", session_id, len(events), settings.BP_MODEL)

    # 4. Publish all feature events in one Redis round-trip
    feature_msgs = [
        {
            "type": "featureDetected",
            "session_id": session_id,
            "sub_state": ev["sub_state"],
            "payload": ev["payload"],
        }
        for ev in events
    ]
    if feature_msgs:
        _publish_many(session_id, feature_msgs, settings)

    blueprint_id = asyncio.run(
        _persist_and_generate(session_id, events, transcript, settings)
    )

    # 5. Notify frontend blueprint is ready
    _publish_many(session_id, [{
        "type": "blueprintReady",
        "session_id": session_id,
        "blueprint_id": blueprint_id,
    }], settings)

    logger.info("Physical pipeline complete session=%s blueprint=%s", session_id, blueprint_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _transcribe(file_path: str, settings) -> str:
    """Call Deepgram pre-recorded API and return the full transcript string."""
    async def _run() -> str:
        from deepgram import AsyncDeepgramClient
        dg = AsyncDeepgramClient(api_key=settings.DEEPGRAM_API_KEY)
        with open(file_path, "rb") as f:
            buffer = f.read()
        response = await dg.listen.v1.media.transcribe_file(
            request=buffer,
            model=settings.DEEPGRAM_MODEL,
            punctuate=True,
            smart_format=True,
        )
        if not (response.results and response.results.channels):
            return ""
        alts = response.results.channels[0].alternatives
        return alts[0].transcript if alts else ""

    return asyncio.run(_run())


def _publish_many(session_id: str, payloads: list[dict], settings) -> None:
    """Publish multiple events over a single Redis connection."""
    r = sync_redis.from_url(settings.REDIS_URL, decode_responses=True)
    channel = f"{_CHANNEL_PREFIX}{session_id}"
    try:
        pipe = r.pipeline(transaction=False)
        for payload in payloads:
            pipe.publish(channel, json.dumps(payload))
        pipe.execute()
    finally:
        r.close()


async def _persist_and_generate(
    session_id: str, events: list[dict], transcript: str, settings
) -> str:
    """Persist extraction events to Firestore (batched) and generate blueprint."""
    from app.db.firebase import get_firestore_client
    from app.build.blueprint_generator import BlueprintAggregator

    db = get_firestore_client()
    events_ref = db.collection("sessions").document(session_id).collection("events")
    now = datetime.now(timezone.utc)

    # Batch all Firestore writes into a single round-trip.
    batch = db.batch()

    if transcript:
        batch.set(events_ref.document(str(uuid.uuid4())), {
            "session_id": session_id,
            "event_type": "transcript.segment",
            "payload": {"text": transcript},
            "timestamp": now,
        })

    for ev in events:
        batch.set(events_ref.document(str(uuid.uuid4())), {
            "session_id": session_id,
            "event_type": ev["sub_state"],
            "payload": ev["payload"],
            "timestamp": now,
        })

    await batch.commit()

    aggregator = BlueprintAggregator(db)
    blueprint = await aggregator.generate(uuid.UUID(session_id))
    return str(blueprint.id)
