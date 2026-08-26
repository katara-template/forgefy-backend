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

# Recall needs time to finalize media after a call ends before it can be
# deleted; the task's retry backoff covers bots that take longer.
DELETE_MEDIA_COUNTDOWN_SECONDS = 120


@celery_app.task(
    name="app.workers.connector_worker.dispatch_connector",
    acks_late=True,
    reject_on_worker_lost=True,
)
def dispatch_connector(
    session_id: str,
    platform: str,
    meeting_url: str | None,
    user_id: str | None = None,
) -> None:
    """Send a bot to join the meeting.

    Which bot depends on platform and ZOOM_BOT_PROVIDER — for Recall this is a
    non-blocking HTTP call, for a self-hosted Zoom bot it launches a container.
    user_id is optional so tasks queued before it was added still run.
    """
    from app.db.models.enums import Platform

    try:
        plat = Platform(platform)
    except ValueError:
        logger.error("Unknown platform '%s' for session=%s", platform, session_id)
        return

    try:
        connector = get_connector(plat, user_id=user_id)
    except (NotImplementedError, RuntimeError) as exc:
        logger.warning("No connector for session=%s: %s", session_id, exc)
        return

    try:
        connector.join(meeting_url or "", session_id)
        logger.info(
            "Bot dispatched session=%s platform=%s via=%s",
            session_id, platform, type(connector).__name__,
        )
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
        # Clearing the mapping stops the terminal-status webhook from acting
        # on this bot, so the media purge must be scheduled from here.
        recall_delete_media.apply_async(
            args=[bot_id],
            countdown=DELETE_MEDIA_COUNTDOWN_SECONDS,
            queue="meeting.audio",
        )
        r.delete(f"recall:session:{session_id}", f"recall:bot:{bot_id}")
    finally:
        r.close()


@celery_app.task(
    name="app.workers.connector_worker.recall_delete_media",
    acks_late=True,
    reject_on_worker_lost=True,
    autoretry_for=(Exception,),
    retry_backoff=60,
    retry_backoff_max=900,
    retry_jitter=True,
    max_retries=6,
)
def recall_delete_media(bot_id: str) -> None:
    """Purge the recording/transcript media Recall.ai stores for bot_id."""
    from app.config import get_settings
    from app.connectors.recall import delete_media

    settings = get_settings()
    if not settings.RECALL_API_KEY:
        return

    delete_media(
        bot_id=bot_id,
        base_url=f"https://{settings.RECALL_REGION}.recall.ai/api/v1",
        api_key=settings.RECALL_API_KEY,
    )
