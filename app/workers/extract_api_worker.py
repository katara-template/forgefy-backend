"""Developer extract API worker — processes async extraction jobs.

Jobs are created by POST /api/v1/extract/jobs (app/api/v1/extract.py), which
writes an extract_jobs/{job_id} doc and dispatches process_extract_job. The
transcript travels in the Celery message, not Firestore — the job doc holds
metadata and, eventually, the result.

Webhook deliveries are HMAC-SHA256-signed with the per-job secret returned to
the caller at creation (X-Forgefy-Signature: sha256=<hex>), and retried
separately from extraction so a flaky receiver never re-runs the pipeline.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import json
import logging
import socket
from datetime import UTC, datetime, timedelta
from urllib.parse import urlparse

import httpx

from app.ai.pipeline import group_events, run_extraction
from app.config import get_settings
from app.core.usage import record_usage
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

_COLLECTION = "extract_jobs"
_JOB_TTL_DAYS = 30
_WEBHOOK_TIMEOUT = 10.0


def _run_coro(coro) -> None:
    """Run one async Firestore operation on a fresh loop (worker is sync)."""
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(coro)
    finally:
        loop.close()


async def _update_job_async(job_id: str, fields: dict) -> None:
    from app.db.firebase import refresh_async_firestore_client

    db = refresh_async_firestore_client()
    await db.collection(_COLLECTION).document(job_id).set(
        {**fields, "updated_at": datetime.now(UTC)}, merge=True
    )


def _update_job(job_id: str, fields: dict) -> None:
    _run_coro(_update_job_async(job_id, fields))


async def _record_usage_async(owner_user_id: str, tokens: int) -> None:
    from app.db.firebase import refresh_async_firestore_client

    db = refresh_async_firestore_client()
    await record_usage(db, owner_user_id, tokens)


def _extract(transcript: str, extractors: list[str], model_tier: str, settings) -> tuple[dict, dict]:
    """Run the requested tier; return (groups, usage). Raises on total failure."""
    if model_tier == "economy":
        from app.ai.qwen import run_qwen_synthesis

        events = run_qwen_synthesis(transcript)
        usage = {"input_tokens": 0, "output_tokens": 0}  # free model, unmetered
    else:
        result = run_extraction(
            transcript, settings.ANTHROPIC_API_KEY, settings.ANTHROPIC_MODEL,
            extractors=extractors,
        )
        events = result["events"]
        usage = result["usage"]
        if result["errors"] and not events:
            raise RuntimeError(f"all extractors errored: {result['errors']}")
    return group_events(events, extractors), usage


@celery_app.task(name="app.workers.extract_api_worker.process_extract_job")
def process_extract_job(
    job_id: str,
    transcript: str,
    extractors: list[str],
    model_tier: str,
    owner_user_id: str,
    webhook_url: str | None,
    webhook_secret: str | None,
) -> None:
    """Run one async extraction job end to end and persist the outcome.

    Deliberately not autoretried: a total pipeline failure here is a
    configuration or provider outage, not a transient blip — the job is
    marked failed and the caller decides whether to resubmit.
    """
    settings = get_settings()
    _update_job(job_id, {"status": "processing"})

    try:
        groups, usage = _extract(transcript, extractors, model_tier, settings)
    except Exception as exc:
        logger.error("Extract job failed job=%s: %s", job_id, exc)
        error = str(exc)[:500]
        _update_job(job_id, {"status": "failed", "error": error})
        if webhook_url and webhook_secret:
            deliver_extract_webhook.apply_async(
                args=[webhook_url, webhook_secret,
                      {"type": "extract.job.failed", "job_id": job_id,
                       "status": "failed", "error": error}]
            )
        return

    result = {**groups, "usage": usage}
    _update_job(job_id, {"status": "done", "result": result})

    tokens = usage["input_tokens"] + usage["output_tokens"]
    if tokens:
        try:  # metering must never fail a finished job
            _run_coro(_record_usage_async(owner_user_id, tokens))
        except Exception as exc:
            logger.warning("Usage recording failed job=%s: %s", job_id, exc)

    if webhook_url and webhook_secret:
        deliver_extract_webhook.apply_async(
            args=[webhook_url, webhook_secret,
                  {"type": "extract.job.completed", "job_id": job_id,
                   "status": "done", "result": result}]
        )

    logger.info("Extract job done job=%s tier=%s tokens=%d", job_id, model_tier, tokens)


class WebhookHostError(ValueError):
    """The webhook host resolves to a non-public address — never retried."""


def _assert_public_webhook_host(url: str) -> None:
    """SSRF guard: refuse deliveries to private/internal addresses.

    Checked at delivery time so a hostname that merely *looked* public when
    the job was created can't point us at the cloud metadata service or an
    internal host. Best-effort — the resolve here and httpx's resolve are
    separate lookups — but it shuts down the straightforward attacks. Skipped
    in development so localhost receivers keep working.
    """
    if get_settings().APP_ENV == "development":
        return
    host = urlparse(url).hostname or ""
    for info in socket.getaddrinfo(host, None):
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise WebhookHostError(f"webhook host {host!r} resolves to non-public {ip}")


@celery_app.task(
    name="app.workers.extract_api_worker.deliver_extract_webhook",
    autoretry_for=(Exception,),
    max_retries=3,
    retry_backoff=True,  # 1s, 2s, 4s
    retry_backoff_max=60,
    retry_jitter=True,
)
def deliver_extract_webhook(url: str, secret: str, payload: dict) -> None:
    """POST a signed job event to the caller's webhook endpoint."""
    try:
        _assert_public_webhook_host(url)
    except WebhookHostError as exc:
        # Permanent condition — drop the delivery instead of autoretrying it.
        logger.error("Webhook delivery refused job=%s: %s", payload.get("job_id"), exc)
        return

    body = json.dumps(payload, separators=(",", ":"))
    signature = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
    response = httpx.post(
        url,
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-Forgefy-Signature": f"sha256={signature}",
        },
        timeout=_WEBHOOK_TIMEOUT,
    )
    response.raise_for_status()
    logger.info("Webhook delivered job=%s status=%d", payload.get("job_id"), response.status_code)


# ── TTL cleanup (beat) ────────────────────────────────────────────────────────


async def _cleanup_async(cutoff: datetime) -> int:
    from app.db.firebase import refresh_async_firestore_client

    db = refresh_async_firestore_client()
    deleted = 0
    async for doc in db.collection(_COLLECTION).where("created_at", "<", cutoff).stream():
        await doc.reference.delete()
        deleted += 1
    return deleted


@celery_app.task(name="app.workers.extract_api_worker.cleanup_extract_jobs")
def cleanup_extract_jobs() -> dict:
    """Delete extract jobs older than the retention window."""
    cutoff = datetime.now(UTC) - timedelta(days=_JOB_TTL_DAYS)
    loop = asyncio.new_event_loop()
    try:
        deleted = loop.run_until_complete(_cleanup_async(cutoff))
    finally:
        loop.close()
    logger.info("Extract job cleanup: deleted=%d older_than=%s", deleted, cutoff.date())
    return {"deleted": deleted}
