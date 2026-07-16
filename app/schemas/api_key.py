"""Pydantic schemas for API key management endpoints."""
import uuid
from datetime import datetime

from pydantic import BaseModel, field_validator


class CreateApiKeyRequest(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def name_length(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Name must not be empty")
        if len(v) > 100:
            raise ValueError("Name must be at most 100 characters")
        return v


class ApiKeyOut(BaseModel):
    """A key as shown in lists — the full key is never recoverable."""

    id: uuid.UUID
    name: str
    prefix: str
    created_at: datetime
    last_used_at: datetime | None = None
    revoked_at: datetime | None = None


class ApiKeyCreatedResponse(ApiKeyOut):
    """Returned once, at creation — the only time `key` is ever visible."""

    key: str
