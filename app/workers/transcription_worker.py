"""Transcription worker — streams audio chunks to Deepgram and emits transcript events.

Each Celery worker process holds a module-level registry of open
``TranscriptionSession`` objects, one per active meeting session.
Audio chunks arrive via the ``meeting.audio`` queue; the session's
background listener thread forwards Deepgram transcripts to Redis.
"""
from __future__ import annotations

import base64
import logging

from app.config import get_settings
from app.transcription.deepgram_client import TranscriptionSession
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

# Per-worker-process session registry.
_sessions: dict[str, TranscriptionSession] = {}


def _get_or_create_session(session_id: str) -> TranscriptionSession:
    if session_id not in _sessions:
        settings = get_settings()
        _sessions[session_id] = TranscriptionSession(
            session_id=session_id,
            api_key=settings.DEEPGRAM_API_KEY,
            model=settings.DEEPGRAM_MODEL,
            redis_url=settings.REDIS_URL,
        )
    return _sessions[session_id]


# Deliberately no acks_late/reject_on_worker_lost here: _sessions is an
# in-memory, per-worker-process registry tied to a live Deepgram WebSocket.
# Redelivering a chunk to a different worker after a crash wouldn't find
# that session — it would silently open a new Deepgram connection out of
# order, corrupting the transcript rather than safely retrying.
@celery_app.task(name="app.workers.transcription_worker.process_audio_chunk")
def process_audio_chunk(session_id: str, chunk_b64: str) -> None:
    """Decode a base64 PCM chunk and forward it to the Deepgram streaming session.

    The session is created on first chunk; subsequent chunks reuse the same
    Deepgram WebSocket connection for the lifetime of the Celery worker process.
    """
    chunk_bytes = base64.b64decode(chunk_b64)
    session = _get_or_create_session(session_id)
    session.send_chunk(chunk_bytes)
    logger.debug("Audio chunk forwarded session=%s bytes=%d", session_id, len(chunk_bytes))


@celery_app.task(name="app.workers.transcription_worker.close_transcription_session")
def close_transcription_session(session_id: str) -> None:
    """Close the Deepgram WebSocket for a session that has ended."""
    session = _sessions.pop(session_id, None)
    if session:
        session.close()
        logger.info("Transcription session closed session=%s", session_id)
    else:
        logger.warning(
            "close_transcription_session: session not found session=%s", session_id
        )
