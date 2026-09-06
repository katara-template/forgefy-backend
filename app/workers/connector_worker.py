"""Connector worker — creates the Recall.ai bot for a meeting session.

Runs on the ``meeting.audio`` queue.  dispatch_connector is now non-blocking
(just an HTTP call to Recall.ai); the bot lifecycle is managed via webhooks.
"""
from __future__ import annotations

import asyncio
import json
import logging

import redis as sync_redis

from app.connectors.factory import get_connector
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

# Session pub/sub channel the frontend's /ws/voxa listener (and its 5s poller)
# watch. Publishing a status nudge here makes the client react immediately.
_CHANNEL_PREFIX = "voxa:session:"

# Recall needs time to finalize media after a call ends before it can be
# deleted; the task's retry backoff covers bots that take longer.
DELETE_MEDIA_COUNTDOWN_SECONDS = 120


def _fail_session(
    session_id: str,
    user_message: str,
    alert: tuple[str, str] | None = None,
) -> None:
    """Mark a session FAILED because the bot never joined, and tell the client.

    The state machine has no JOINING->FAILED edge and the worker is sync, so
    this mirrors blueprint_worker's failure path: a direct status write on a
    fresh event loop plus a best-effort Redis publish. ``alert`` is an optional
    ``(title, detail)`` recorded on the operator dashboard for causes only the
    operator can fix (missing key, provider rejection). Every step is
    best-effort — a failure to report a failure must not crash the task.
    """
    from app.config import get_settings

    settings = get_settings()

    async def _write() -> None:
        from app.db.firebase import get_firestore_client, refresh_async_firestore_client
        from app.db.models.enums import SessionStatus

        refresh_async_firestore_client()
        db = get_firestore_client()
        await db.collection("sessions").document(session_id).update(
            {"status": SessionStatus.FAILED.value}
        )
        if alert is not None:
            from app.core.alerts import record_operator_alert

            title, detail = alert
            await record_operator_alert(
                db,
                title=title,
                raw_detail=detail,
                source="connector",
                session_id=session_id,
            )

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_write())
    except Exception as exc:
        logger.warning("Could not mark session FAILED session=%s: %s", session_id, exc)
    finally:
        loop.close()

    try:
        r = sync_redis.from_url(settings.REDIS_URL, decode_responses=True)
        try:
            r.publish(
                f"{_CHANNEL_PREFIX}{session_id}",
                json.dumps(
                    {
                        "type": "meetingStatus",
                        "session_id": session_id,
                        "status": "FAILED",
                        "error": user_message,
                    }
                ),
            )
        finally:
            r.close()
    except Exception as exc:
        logger.warning("Could not publish FAILED status session=%s: %s", session_id, exc)


@celery_app.task(
    bind=True,
    name="app.workers.connector_worker.dispatch_connector",
    acks_late=True,
    reject_on_worker_lost=True,
    max_retries=3,
)
def dispatch_connector(
    self,
    session_id: str,
    platform: str,
    meeting_url: str | None,
    user_id: str | None = None,
) -> None:
    """Send a bot to join the meeting.

    Which bot depends on platform and ZOOM_BOT_PROVIDER — for Recall this is a
    non-blocking HTTP call, for a self-hosted Zoom bot it launches a container.
    user_id is optional so tasks queued before it was added still run.

    Transient failures (transport errors, Recall 429/5xx) are retried with
    exponential backoff. A misconfigured server, a terminal provider rejection,
    or exhausted retries mark the session FAILED so the UI stops waiting on a
    bot that will never arrive.
    """
    from app.connectors.recall import RecallBotCreationError
    from app.db.models.enums import Platform

    try:
        plat = Platform(platform)
    except ValueError:
        logger.error("Unknown platform '%s' for session=%s", platform, session_id)
        _fail_session(session_id, "Unsupported meeting platform.")
        return

    if plat != Platform.PHYSICAL and not (meeting_url or "").strip():
        logger.error("No meeting_url for session=%s platform=%s", session_id, platform)
        _fail_session(session_id, "No meeting URL was provided for this session.")
        return

    try:
        connector = get_connector(plat, user_id=user_id)
    except NotImplementedError:
        return  # physical session: no bot, nothing to do
    except RuntimeError as exc:
        # RECALL_API_KEY / PUBLIC_API_BASE_URL missing from the worker's env —
        # the deploy is misconfigured, not something a retry will fix.
        logger.error("Meeting bot misconfigured session=%s: %s", session_id, exc)
        _fail_session(
            session_id,
            "The meeting bot is not configured on the server.",
            alert=("Meeting bot misconfigured", str(exc)),
        )
        return

    try:
        connector.join(meeting_url or "", session_id)
    except NotImplementedError as exc:
        # Stub connector (e.g. direct Zoom/Teams) — nothing to retry or fail.
        logger.warning("Connector has no join for session=%s: %s", session_id, exc)
        return
    except RecallBotCreationError as exc:
        logger.error("Recall rejected bot session=%s: %s", session_id, exc)
        _fail_session(
            session_id,
            "The meeting bot was rejected by the meeting provider.",
            alert=("Recall rejected meeting bot", str(exc)),
        )
        return
    except Exception as exc:
        logger.error(
            "Connector error session=%s (attempt %d/%d): %s",
            session_id, self.request.retries + 1, self.max_retries, exc, exc_info=True,
        )
        try:
            raise self.retry(exc=exc, countdown=min(60, 5 * 2 ** self.request.retries))
        except self.MaxRetriesExceededError:
            _fail_session(
                session_id, "The meeting bot could not join after several attempts."
            )
            return

    logger.info(
        "Bot dispatched session=%s platform=%s via=%s",
        session_id, platform, type(connector).__name__,
    )


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
