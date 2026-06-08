"""Blueprint worker — aggregates extraction events into a final blueprint."""
from __future__ import annotations

import asyncio
import json
import logging
import uuid

import redis as sync_redis

from app.config import get_settings
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

_CHANNEL_PREFIX = "voxa:session:"


async def _run_aggregation(session_id: str) -> str:
    """Run BlueprintAggregator in an async context; return blueprint_id string."""
    from app.db.firebase import refresh_async_firestore_client
    from app.build.blueprint_generator import BlueprintAggregator

    db = refresh_async_firestore_client()
    aggregator = BlueprintAggregator(db)
    blueprint = await aggregator.generate(uuid.UUID(session_id))
    return str(blueprint.id)


@celery_app.task(name="app.workers.blueprint_worker.generate_blueprint")
def generate_blueprint(session_id: str) -> None:
    """Aggregate requirements, create Blueprint document, notify via Redis."""
    settings = get_settings()

    loop = asyncio.new_event_loop()
    try:
        blueprint_id = loop.run_until_complete(_run_aggregation(session_id))
    except Exception as exc:
        from app.core.build_errors import sanitize_build_error
        build_err = sanitize_build_error(exc)
        logger.error("Blueprint generation FAILED session=%s: %s", session_id, exc, exc_info=True)
        r = sync_redis.from_url(settings.REDIS_URL, decode_responses=True)
        try:
            r.publish(
                f"{_CHANNEL_PREFIX}{session_id}",
                json.dumps({"type": "blueprintError", "session_id": session_id, "error": build_err.message}),
            )
        finally:
            r.close()
        return
    finally:
        loop.close()

    r = sync_redis.from_url(settings.REDIS_URL, decode_responses=True)
    try:
        r.publish(
            f"{_CHANNEL_PREFIX}{session_id}",
            json.dumps({"type": "blueprintReady", "session_id": session_id, "blueprint_id": blueprint_id}),
        )
        logger.info("blueprintReady published session=%s blueprint=%s", session_id, blueprint_id)
    finally:
        r.close()
