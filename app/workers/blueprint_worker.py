"""Blueprint worker — aggregates extraction events into a final blueprint.

Receives session_id on the ``meeting.extract`` queue after all transcript
segments have been processed.  Runs BlueprintAggregator to build the JSON
document, saves a Blueprint row, transitions the session to BLUEPRINT_READY,
and publishes a ``blueprintReady`` event to Redis.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid

import redis as sync_redis
from sqlalchemy.pool import NullPool

from app.config import get_settings
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

_CHANNEL_PREFIX = "voxa:session:"


async def _run_aggregation(session_id: str, database_url: str) -> str:
    """Run BlueprintAggregator in an async context; return blueprint_id string."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from app.build.blueprint_generator import BlueprintAggregator

    engine = create_async_engine(database_url, poolclass=NullPool)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with factory() as db:
            aggregator = BlueprintAggregator(db)
            blueprint = await aggregator.generate(uuid.UUID(session_id))
            await db.commit()
            return str(blueprint.id)
    finally:
        await engine.dispose()


@celery_app.task(name="app.workers.blueprint_worker.generate_blueprint")
def generate_blueprint(session_id: str) -> None:
    """Aggregate requirements, create Blueprint row, notify via Redis."""
    settings = get_settings()

    blueprint_id = asyncio.run(_run_aggregation(session_id, settings.DATABASE_URL))

    r = sync_redis.from_url(settings.REDIS_URL, decode_responses=True)
    try:
        r.publish(
            f"{_CHANNEL_PREFIX}{session_id}",
            json.dumps(
                {
                    "type": "blueprintReady",
                    "session_id": session_id,
                    "blueprint_id": blueprint_id,
                }
            ),
        )
        logger.info("blueprintReady published session=%s blueprint=%s", session_id, blueprint_id)
    finally:
        r.close()
