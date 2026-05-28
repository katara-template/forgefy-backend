"""Celery application instance and queue configuration."""
from celery import Celery

from app.config import get_settings

_s = get_settings()

celery_app = Celery(
    "forgefy",
    broker=_s.CELERY_BROKER_URL,
    backend=_s.CELERY_RESULT_BACKEND,
    include=[
        "app.workers.connector_worker",
        "app.workers.transcription_worker",
        "app.workers.extraction_worker",
        "app.workers.blueprint_worker",
        "app.workers.build_worker",
        "app.workers.update_worker",
        "app.workers.cleanup_worker",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_routes={
        "app.workers.connector_worker.*": {"queue": "meeting.audio"},
        "app.workers.transcription_worker.*": {"queue": "meeting.audio"},
        "app.workers.extraction_worker.*": {"queue": "meeting.transcribe"},
        "app.workers.blueprint_worker.*": {"queue": "meeting.extract"},
        "app.workers.build_worker.*": {"queue": "build"},
        "app.workers.update_worker.*": {"queue": "build"},
        "app.workers.cleanup_worker.*": {"queue": "build"},
    },
    beat_schedule={
        "cleanup-workspaces-daily": {
            "task": "app.workers.cleanup_worker.cleanup_workspaces",
            "schedule": 86400.0,  # every 24 hours
        },
    },
)
