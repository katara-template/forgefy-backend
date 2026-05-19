"""Connector worker — launches the platform bot for a meeting session.

Runs on the ``meeting.audio`` queue alongside the transcription worker so
the connector and transcription tasks share the same worker process pool.
"""
from __future__ import annotations

import logging

from app.connectors.factory import get_connector
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="app.workers.connector_worker.dispatch_connector")
def dispatch_connector(session_id: str, platform: str, meeting_url: str | None) -> None:
    """Instantiate the correct platform connector and join the meeting.

    Blocks until the meeting ends or an error occurs.  Designed to run as a
    long-running Celery task (acks_late recommended for production).
    """
    from app.db.models.enums import Platform

    try:
        plat = Platform(platform)
    except ValueError:
        logger.error("Unknown platform '%s' for session=%s", platform, session_id)
        return

    connector = get_connector(plat)
    try:
        connector.join(meeting_url or "", session_id)
        logger.info("Connector joined session=%s platform=%s", session_id, platform)
    except NotImplementedError as exc:
        logger.warning("Connector stub for session=%s: %s", session_id, exc)
    except Exception as exc:
        logger.error("Connector error session=%s: %s", session_id, exc, exc_info=True)
        try:
            connector.leave()
        except Exception:
            pass
