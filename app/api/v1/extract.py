"""Developer extract API — transcript in, structured requirements out.

Machine-authed with an API key (see app.deps.get_api_key); dashboard users
mint keys at /api/v1/keys. Runs the same agent pipeline the meeting workers
use, synchronously:

  standard tier → Claude 4-agent LangGraph pipeline (metered per token)
  economy tier  → Qwen3 synthesis via OpenRouter/Ollama (free model, unmetered)

A key owner over their monthly token budget is handled with the shared quota
policy (app.core.usage): free tier is blocked with a 402, paid tiers are
silently served by the economy tier instead of failing.
"""
import logging
import secrets
import uuid
from contextlib import suppress
from datetime import UTC, datetime
from functools import partial
from typing import Annotated

import anyio
from fastapi import APIRouter, Header, Request
from google.api_core.exceptions import AlreadyExists

from app.ai.pipeline import EXTRACTORS, group_events, run_extraction
from app.core.dispatch import dispatch
from app.core.exceptions import (
    ExternalServiceError,
    NotFoundError,
    QuotaExceededError,
    RateLimitedError,
    ValidationError,
)
from app.core.rate_limit import api_key_ident, limiter
from app.core.usage import evaluate_quota, record_usage
from app.deps import ApiKeyDep, DBSession, SettingsDep
from app.schemas.extract import (
    ExtractJobCreatedResponse,
    ExtractJobRequest,
    ExtractJobStatusResponse,
    ExtractRequest,
    ExtractResponse,
    ExtractUsage,
)

logger = logging.getLogger(__name__)
router = APIRouter()

_ALL_GROUPS = tuple(EXTRACTORS)

# At most this many sync extractions in flight per process. Each standard-tier
# request fans out to one parallel model call per extractor and occupies a
# thread for its whole duration — beyond this, shed load with a 429 instead of queueing
# requests into timeout territory (batch work belongs on /extract/jobs).
# A plain counter, not an asyncio primitive: the event loop makes the
# check-then-increment atomic, and it survives the per-test loop churn that
# bites loop-bound Semaphores.
_SYNC_EXTRACT_CONCURRENCY = 8
_sync_in_flight = 0


def _run_economy(transcript: str) -> list[dict]:
    """Whole-transcript synthesis on the free Qwen3 backend (OpenRouter/Ollama)."""
    from app.ai.qwen import run_qwen_synthesis

    return run_qwen_synthesis(transcript)


@router.post("", response_model=ExtractResponse)
@limiter.limit("60/minute", key_func=api_key_ident)
async def extract(
    request: Request,
    body: ExtractRequest,
    db: DBSession,
    settings: SettingsDep,
    api_key: ApiKeyDep,
) -> ExtractResponse:
    """Extract features, questions, conflicts, and action items from a transcript."""
    owner_id = str(api_key.owner_user_id)
    requested = tuple(body.extractors) if body.extractors else _ALL_GROUPS

    tier = body.model_tier
    outcome = await evaluate_quota(db, settings, owner_id)
    if outcome.action == "block":
        raise QuotaExceededError(outcome.message)
    if outcome.action == "downgrade":
        # Same policy as builds: paid users over budget keep working on the
        # free model instead of hard-failing mid-month.
        tier = "economy"

    if tier == "standard" and not settings.ANTHROPIC_API_KEY:
        raise ValidationError("The standard extraction tier is not configured on this server")

    global _sync_in_flight
    if _sync_in_flight >= _SYNC_EXTRACT_CONCURRENCY:
        raise RateLimitedError(
            "Sync extraction capacity reached — retry shortly, or use POST /extract/jobs"
        )
    _sync_in_flight += 1
    try:
        if tier == "economy":
            try:
                events = await anyio.to_thread.run_sync(_run_economy, body.transcript)
            except Exception as exc:
                logger.error("Economy extraction failed key=%s: %s", api_key.id, exc)
                raise ExternalServiceError("Extraction failed on the economy backend") from exc
            # The synthesis path always extracts everything; free-model usage is
            # not metered (mirrors the downgrade policy in app.core.usage).
            usage = ExtractUsage()
        else:
            result = await anyio.to_thread.run_sync(
                partial(
                    run_extraction,
                    body.transcript,
                    settings.ANTHROPIC_API_KEY,
                    settings.ANTHROPIC_MODEL,
                    extractors=requested,
                )
            )
            events = result["events"]
            usage = ExtractUsage(**result["usage"])
            if result["errors"] and not events:
                # Partial agent failures still return what succeeded; only a
                # request where every agent failed is surfaced as an error.
                logger.error("All extractors failed key=%s: %s", api_key.id, result["errors"])
                raise ExternalServiceError("Extraction failed — all extractors errored")
    finally:
        _sync_in_flight -= 1

    groups = group_events(events, requested)

    total_tokens = usage.input_tokens + usage.output_tokens
    if total_tokens:
        with suppress(Exception):  # metering must never fail a served request
            await record_usage(db, owner_id, total_tokens)

    logger.info(
        "Extract served key=%s tier=%s extractors=%s tokens=%d",
        api_key.id, tier, ",".join(requested), total_tokens,
    )
    return ExtractResponse(id=uuid.uuid4(), model_tier=tier, usage=usage, **groups)


# ── Async jobs ────────────────────────────────────────────────────────────────

_JOBS_COLLECTION = "extract_jobs"


def _idempotent_job_id(owner_id: str, idempotency_key: str) -> str:
    """Deterministic job id per (owner, Idempotency-Key) — a replayed request
    maps to the same document, so dedupe is a plain get by id (no composite
    Firestore index)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"forgefy-extract-job:{owner_id}:{idempotency_key}"))


@router.post("/jobs", response_model=ExtractJobCreatedResponse, status_code=202)
@limiter.limit("60/minute", key_func=api_key_ident)
async def create_extract_job(
    request: Request,
    body: ExtractJobRequest,
    db: DBSession,
    settings: SettingsDep,
    api_key: ApiKeyDep,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> ExtractJobCreatedResponse:
    """Queue an async extraction job; poll GET /extract/jobs/{id} or receive a webhook."""
    owner_id = str(api_key.owner_user_id)
    requested = list(body.extractors) if body.extractors else list(EXTRACTORS)

    tier = body.model_tier
    outcome = await evaluate_quota(db, settings, owner_id)
    if outcome.action == "block":
        raise QuotaExceededError(outcome.message)
    if outcome.action == "downgrade":
        tier = "economy"
    if tier == "standard" and not settings.ANTHROPIC_API_KEY:
        raise ValidationError("The standard extraction tier is not configured on this server")

    job_id = (
        _idempotent_job_id(owner_id, idempotency_key)
        if idempotency_key
        else str(uuid.uuid4())
    )
    webhook_secret = secrets.token_urlsafe(32) if body.webhook_url else None
    now = datetime.now(UTC)

    try:
        await db.collection(_JOBS_COLLECTION).document(job_id).create({
            "owner_user_id": owner_id,
            "api_key_id": str(api_key.id),
            "status": "queued",
            "extractors": requested,
            "model_tier": tier,
            "webhook_url": body.webhook_url,
            "webhook_secret": webhook_secret,
            "idempotency_key": idempotency_key,
            "created_at": now,
            "updated_at": now,
            "result": None,
            "error": None,
        })
    except AlreadyExists:
        # Replay of the same Idempotency-Key — return the existing job without
        # dispatching a second run. The id is derived from (owner, key), so a
        # different owner can never collide into this document.
        existing = await db.collection(_JOBS_COLLECTION).document(job_id).get()
        data = existing.to_dict() or {}
        logger.info("Extract job replayed job=%s key=%s", job_id, api_key.id)
        return ExtractJobCreatedResponse(
            job_id=uuid.UUID(job_id),
            status=data.get("status", "queued"),
            webhook_secret=data.get("webhook_secret"),
        )

    from app.workers.extract_api_worker import process_extract_job

    try:
        await dispatch(
            process_extract_job,
            kwargs={
                "job_id": job_id,
                "transcript": body.transcript,
                "extractors": requested,
                "model_tier": tier,
                "owner_user_id": owner_id,
                "webhook_url": body.webhook_url,
                "webhook_secret": webhook_secret,
            },
        )
    except Exception as exc:
        # Broker unreachable — don't leave a job queued forever that no worker
        # will ever pick up.
        with suppress(Exception):
            await db.collection(_JOBS_COLLECTION).document(job_id).set(
                {"status": "failed", "error": "dispatch failed", "updated_at": datetime.now(UTC)},
                merge=True,
            )
        logger.error("Extract job dispatch failed job=%s: %s", job_id, exc)
        raise ExternalServiceError("Failed to queue the extraction job") from exc

    logger.info("Extract job queued job=%s key=%s tier=%s", job_id, api_key.id, tier)
    return ExtractJobCreatedResponse(
        job_id=uuid.UUID(job_id), status="queued", webhook_secret=webhook_secret
    )


@router.get("/jobs/{job_id}", response_model=ExtractJobStatusResponse)
@limiter.limit("60/minute", key_func=api_key_ident)
async def get_extract_job(
    request: Request,
    job_id: str,
    db: DBSession,
    api_key: ApiKeyDep,
) -> ExtractJobStatusResponse:
    """Return the status (and, once done, the result) of an extraction job."""
    doc = await db.collection(_JOBS_COLLECTION).document(job_id).get()
    data = doc.to_dict() if doc.exists else None
    # Same 404 for "doesn't exist" and "not yours" — don't leak job ids.
    if not data or data.get("owner_user_id") != str(api_key.owner_user_id):
        raise NotFoundError("Job not found")

    return ExtractJobStatusResponse(
        job_id=uuid.UUID(job_id),
        status=data.get("status", "queued"),
        model_tier=data.get("model_tier", "standard"),
        created_at=data["created_at"],
        result=data.get("result"),
        error=data.get("error"),
    )
