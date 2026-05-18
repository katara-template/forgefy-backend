"""Transcription worker — pulls audio chunks and streams them to Deepgram.

Implemented in Step 6.
"""
from app.workers.celery_app import celery_app


@celery_app.task(name="app.workers.transcription_worker.process_audio_chunk")
def process_audio_chunk(session_id: str, chunk_b64: str) -> None:
    """Stream a base64-encoded audio chunk through Deepgram and emit transcript events."""
    raise NotImplementedError("Transcription worker — see Step 6")
