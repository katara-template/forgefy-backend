"""Extraction worker — runs the LangGraph agent pipeline on transcript segments.

Implemented in Step 7.
"""
from app.workers.celery_app import celery_app


@celery_app.task(name="app.workers.extraction_worker.extract_requirements")
def extract_requirements(session_id: str, transcript_segment: str) -> None:
    """Run the LangGraph agent pipeline on a finalized transcript segment."""
    raise NotImplementedError("Extraction worker — see Step 7")
