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
from pathlib import Path

import redis as sync_redis
from sqlalchemy.pool import NullPool

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
    _publish(session_id, {
        "type": "transcript",
        "session_id": session_id,
        "text": transcript,
        "is_final": True,
    }, settings)

    # 3. Run LangGraph extraction pipeline
    logger.info(
        "Transcript preview session=%s: %.200s…", session_id, transcript.replace("\n", " ")
    )
    from app.ai.pipeline import run_pipeline
    events = run_pipeline(
        transcript=transcript,
        api_key=settings.ANTHROPIC_API_KEY,
        model=settings.ANTHROPIC_MODEL,
    )
    logger.info("Extraction done session=%s events=%d", session_id, len(events))

    # Fallback: if per-segment pipeline extracted nothing, synthesise from the full transcript
    if not events:
        logger.warning(
            "Pipeline returned 0 events — running synthesis fallback session=%s", session_id
        )
        try:
            from app.ai.agents.synthesizer import run as synthesize
            events = synthesize(transcript, settings.ANTHROPIC_API_KEY, settings.ANTHROPIC_MODEL)
            logger.info(
                "Synthesis fallback done session=%s events=%d", session_id, len(events)
            )
        except Exception as exc:
            logger.error(
                "Synthesis fallback failed session=%s: %s", session_id, exc, exc_info=True
            )

    # 4. Publish feature events for real-time UI, persist to DB, generate blueprint
    for ev in events:
        _publish(session_id, {
            "type": "featureDetected",
            "session_id": session_id,
            "sub_state": ev["sub_state"],
            "payload": ev["payload"],
        }, settings)

    blueprint_id = asyncio.run(_persist_and_generate(session_id, events, transcript, settings))

    # 5. Notify frontend blueprint is ready
    _publish(session_id, {
        "type": "blueprintReady",
        "session_id": session_id,
        "blueprint_id": blueprint_id,
    }, settings)

    logger.info("Physical pipeline complete session=%s blueprint=%s", session_id, blueprint_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _transcribe(file_path: str, settings) -> str:
    """Call Deepgram pre-recorded API and return the full transcript string."""
    from deepgram import DeepgramClient, PrerecordedOptions

    dg = DeepgramClient(api_key=settings.DEEPGRAM_API_KEY)
    with open(file_path, "rb") as f:
        buffer = f.read()

    options = PrerecordedOptions(
        model=settings.DEEPGRAM_MODEL,
        punctuate=True,
        smart_format=True,
        paragraphs=True,
    )
    response = dg.listen.prerecorded.v("1").transcribe_file({"buffer": buffer}, options)
    return response.results.channels[0].alternatives[0].transcript


def _publish(session_id: str, payload: dict, settings) -> None:
    r = sync_redis.from_url(settings.REDIS_URL, decode_responses=True)
    try:
        r.publish(f"{_CHANNEL_PREFIX}{session_id}", json.dumps(payload))
    finally:
        r.close()


async def _persist_and_generate(session_id: str, events: list[dict], transcript: str, settings) -> str:
    """Persist extraction events and generate blueprint; return blueprint_id."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from app.build.blueprint_generator import BlueprintAggregator
    from app.db.models.meeting_event import MeetingEvent

    engine = create_async_engine(settings.DATABASE_URL, poolclass=NullPool)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with factory() as db:
            if transcript:
                db.add(MeetingEvent(
                    session_id=uuid.UUID(session_id),
                    event_type="transcript.segment",
                    payload={"text": transcript},
                ))
            for ev in events:
                db.add(MeetingEvent(
                    session_id=uuid.UUID(session_id),
                    event_type=ev["sub_state"],
                    payload=ev["payload"],
                ))
            await db.flush()

            aggregator = BlueprintAggregator(db)
            blueprint = await aggregator.generate(uuid.UUID(session_id))
            await db.commit()
            return str(blueprint.id)
    finally:
        await engine.dispose()
