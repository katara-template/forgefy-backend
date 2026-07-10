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
    from app.build.blueprint_generator import BlueprintAggregator
    from app.db.firebase import refresh_async_firestore_client

    db = refresh_async_firestore_client()
    aggregator = BlueprintAggregator(db)
    blueprint = await aggregator.generate(uuid.UUID(session_id))
    return str(blueprint.id)


async def _mark_session_failed(session_id: str) -> None:
    from app.db.firebase import get_firestore_client
    from app.db.models.enums import SessionStatus

    db = get_firestore_client()
    await db.collection("sessions").document(session_id).update(
        {"status": SessionStatus.FAILED.value}
    )


async def _record_operator_alert(session_id: str, title: str, exc: Exception) -> None:
    """Surface a blueprint failure to the admin dashboard alerts feed."""
    from app.core.alerts import record_operator_alert
    from app.db.firebase import get_firestore_client

    db = get_firestore_client()
    await record_operator_alert(
        db,
        title=title,
        raw_detail=str(exc),
        source="blueprint",
        session_id=session_id,
    )


@celery_app.task(
    bind=True,
    name="app.workers.blueprint_worker.generate_blueprint",
    max_retries=2,
    time_limit=360,       # hard kill after 6 min
    soft_time_limit=300,  # SoftTimeLimitExceeded raised at 5 min
    acks_late=True,
    reject_on_worker_lost=True,
)
def generate_blueprint(self, session_id: str) -> None:
    """Aggregate requirements, create Blueprint document, notify via Redis."""
    settings = get_settings()

    from celery.exceptions import SoftTimeLimitExceeded  # type: ignore[import-untyped]

    loop = asyncio.new_event_loop()
    try:
        blueprint_id = loop.run_until_complete(_run_aggregation(session_id))
    except SoftTimeLimitExceeded as exc:
        loop.close()
        logger.warning("Blueprint task soft time limit hit session=%s, retrying", session_id)
        raise self.retry(exc=exc, countdown=15) from exc
    except Exception as exc:
        if self.request.retries < self.max_retries:
            # "No transcript data" means extraction tasks haven't landed yet — give
            # them time.  Other transient failures (API errors) get a shorter wait.
            countdown = 30 if "No transcript data" in str(exc) else 10
            raise self.retry(exc=exc, countdown=countdown) from exc

        # All retries exhausted — surface the failure.
        from app.core.build_errors import (
            GENERIC_OPERATOR_MESSAGE,
            is_ai_quota_error,
            sanitize_build_error,
        )
        build_err = sanitize_build_error(exc)
        logger.error("Blueprint generation FAILED session=%s: %s", session_id, exc, exc_info=True)

        # Alert the operator when the AI model ran out of tokens/credits/quota
        # (the user can't fix that) or on any other support-tier failure.
        user_message = build_err.message
        quota_hit = is_ai_quota_error(str(exc))
        if quota_hit or build_err.action == "support":
            title = (
                "AI model tokens/quota exhausted during blueprint generation"
                if quota_hit
                else build_err.message
            )
            try:
                loop.run_until_complete(_record_operator_alert(session_id, title, exc))
            except Exception as alert_exc:
                logger.warning(
                    "Could not record operator alert session=%s: %s", session_id, alert_exc
                )
            if build_err.action == "support":
                user_message = GENERIC_OPERATOR_MESSAGE

        try:
            loop.run_until_complete(_mark_session_failed(session_id))
        except Exception as mark_exc:
            logger.warning("Could not mark session FAILED session=%s: %s", session_id, mark_exc)
        r = sync_redis.from_url(settings.REDIS_URL, decode_responses=True)
        try:
            r.publish(
                f"{_CHANNEL_PREFIX}{session_id}",
                json.dumps({"type": "blueprintError", "session_id": session_id, "error": user_message}),
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
