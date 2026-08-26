"""Celery tasks for tearing down self-hosted Zoom bot containers.

Not yet registered. To link, either move this into app/workers/ or add its
module path to the Celery app's include list — the task names below are
already namespaced so they will not collide with the Recall tasks.

The existing teardown call site (VoxaService._end_session_common →
recall_remove_bot) assumes Recall. `remove_bot_for_session` is a drop-in
replacement for it that picks the right implementation by looking up which
mapping actually exists in Redis, so a single call site serves both bot types
and in-flight sessions keep working across a cutover.
"""
from __future__ import annotations

import logging

import redis as sync_redis

from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(
    name="app.workers.zoom_bot_worker.zoom_bot_cleanup",
    acks_late=True,
    reject_on_worker_lost=True,
)
def zoom_bot_cleanup(session_id: str) -> None:
    """Stop the bot container for session_id and clear its Redis mapping."""
    from app.config import get_settings
    from app.connectors.zoom_selfhosted import (
        clear_mapping,
        lookup_container,
        stop_container,
    )

    settings = get_settings()

    container_id = lookup_container(session_id, settings.REDIS_URL)
    if not container_id:
        logger.debug("No zoom bot container recorded for session=%s", session_id)
        return

    stop_container(container_id)

    # Clearing the secret is what actually revokes the container's ability to
    # post further events, so it must happen even if the stop above failed.
    clear_mapping(session_id, settings.REDIS_URL)
    logger.info("Zoom bot cleaned up session=%s", session_id)


@celery_app.task(
    name="app.workers.zoom_bot_worker.remove_bot_for_session",
    acks_late=True,
    reject_on_worker_lost=True,
)
def remove_bot_for_session(session_id: str) -> None:
    """Remove whichever bot is serving this session.

    Dispatches on the Redis mapping rather than on configuration: a session
    started before the self-hosted flag was flipped must still be torn down by
    the connector that created it.
    """
    from app.config import get_settings

    settings = get_settings()

    r = sync_redis.from_url(settings.REDIS_URL, decode_responses=True)
    try:
        is_self_hosted = bool(r.get(f"zoombot:session:{session_id}"))
    finally:
        r.close()

    if is_self_hosted:
        zoom_bot_cleanup.run(session_id)
        return

    from app.workers.connector_worker import recall_remove_bot
    recall_remove_bot.run(session_id)
