"""Pydantic schemas for the developer extract API."""
import uuid
from datetime import datetime
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, field_validator

Extractor = Literal["features", "questions", "conflicts", "action_items", "entities"]
JobStatus = Literal["queued", "processing", "done", "failed"]

# ~2-3 hours of speech; anything longer belongs on the async jobs endpoint,
# not a synchronous request.
_MAX_TRANSCRIPT_CHARS = 50_000

# Async job ceiling — a full workday of speech, still comfortably inside
# Firestore's 1 MB document limit for the stored result.
_MAX_JOB_TRANSCRIPT_CHARS = 200_000

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class ExtractRequest(BaseModel):
    transcript: str
    extractors: list[Extractor] | None = None  # None → all of them
    model_tier: Literal["standard", "economy"] = "standard"

    @field_validator("transcript")
    @classmethod
    def transcript_bounds(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Transcript must not be empty")
        if len(v) > _MAX_TRANSCRIPT_CHARS:
            raise ValueError(
                f"Transcript exceeds {_MAX_TRANSCRIPT_CHARS} characters — "
                "use the async jobs endpoint for long transcripts"
            )
        return v

    @field_validator("extractors")
    @classmethod
    def extractors_not_empty(cls, v: list[Extractor] | None) -> list[Extractor] | None:
        if v is not None and not v:
            raise ValueError("extractors must not be an empty list — omit it to run all")
        return v


class ExtractUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class ExtractResponse(BaseModel):
    id: uuid.UUID
    model_tier: str
    features: list[dict] = []
    questions: list[dict] = []
    conflicts: list[dict] = []
    action_items: list[dict] = []
    entities: list[dict] = []
    usage: ExtractUsage


# ── Async jobs ────────────────────────────────────────────────────────────────


class ExtractJobRequest(BaseModel):
    transcript: str
    extractors: list[Extractor] | None = None  # None → all of them
    model_tier: Literal["standard", "economy"] = "standard"
    webhook_url: str | None = None  # POSTed the result when the job finishes

    @field_validator("transcript")
    @classmethod
    def transcript_bounds(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Transcript must not be empty")
        if len(v) > _MAX_JOB_TRANSCRIPT_CHARS:
            raise ValueError(f"Transcript exceeds {_MAX_JOB_TRANSCRIPT_CHARS} characters")
        return v

    @field_validator("extractors")
    @classmethod
    def extractors_not_empty(cls, v: list[Extractor] | None) -> list[Extractor] | None:
        if v is not None and not v:
            raise ValueError("extractors must not be an empty list — omit it to run all")
        return v

    @field_validator("webhook_url")
    @classmethod
    def webhook_url_https(cls, v: str | None) -> str | None:
        if v is None:
            return None
        parsed = urlparse(v)
        if not parsed.netloc:
            raise ValueError("webhook_url must be an absolute URL")
        # https everywhere except local development targets.
        if parsed.scheme != "https" and parsed.hostname not in _LOCAL_HOSTS:
            raise ValueError("webhook_url must use https")
        return v


class ExtractJobCreatedResponse(BaseModel):
    job_id: uuid.UUID
    status: JobStatus
    # Returned when a webhook_url was given: HMAC-SHA256 key for verifying the
    # X-Forgefy-Signature header on the delivery. Returned only at creation
    # (and on idempotent replays of the same Idempotency-Key).
    webhook_secret: str | None = None


class ExtractJobStatusResponse(BaseModel):
    job_id: uuid.UUID
    status: JobStatus
    model_tier: str
    created_at: datetime
    result: dict | None = None  # groups + usage, set once status == "done"
    error: str | None = None  # set once status == "failed"
