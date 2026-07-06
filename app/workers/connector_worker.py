"""Connector worker — creates the Recall.ai bot for a meeting session.

Runs on the ``meeting.audio`` queue.  dispatch_connector is now non-blocking
(just an HTTP call to Recall.ai); the bot lifecycle is managed via webhooks.
"""
from __future__ import annotations

import logging

import redis as sync_redis

from app.connectors.factory import get_connector
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(
    name="app.workers.connector_worker.dispatch_connector",
    acks_late=True,
    reject_on_worker_lost=True,
)
def dispatch_connector(session_id: str, platform: str, meeting_url: str | None) -> None:
    """Create a Recall.ai bot to join the meeting (non-blocking HTTP call)."""
    from app.db.models.enums import Platform

    try:
        plat = Platform(platform)
    except ValueError:
        logger.error("Unknown platform '%s' for session=%s", platform, session_id)
        return

    try:
        connector = get_connector(plat)
    except (NotImplementedError, RuntimeError) as exc:
        logger.warning("No connector for session=%s: %s", session_id, exc)
        return

    try:
        connector.join(meeting_url or "", session_id)
        logger.info("Recall bot dispatched session=%s platform=%s", session_id, platform)
    except Exception as exc:
        logger.error("Connector error session=%s: %s", session_id, exc, exc_info=True)


@celery_app.task(
    name="app.workers.connector_worker.recall_remove_bot",
    acks_late=True,
    reject_on_worker_lost=True,
)
def recall_remove_bot(session_id: str) -> None:
    """Look up the Recall bot for session_id and delete it."""
    from app.config import get_settings
    from app.connectors.recall import remove_bot

    settings = get_settings()
    if not settings.RECALL_API_KEY:
        return

    r = sync_redis.from_url(settings.REDIS_URL, decode_responses=True)
    try:
        bot_id = r.get(f"recall:session:{session_id}")
        if not bot_id:
            return
        remove_bot(
            bot_id=bot_id,
            base_url=f"https://{settings.RECALL_REGION}.recall.ai/api/v1",
            api_key=settings.RECALL_API_KEY,
        )
        r.delete(f"recall:session:{session_id}", f"recall:bot:{bot_id}")
    finally:
        r.close()
