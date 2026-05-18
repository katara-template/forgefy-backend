"""Pydantic schemas for session endpoints."""
import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from app.db.models.enums import Platform, SessionStatus


class CreateSessionRequest(BaseModel):
    platform: Platform
    meeting_url: str | None = None


class JoinSessionRequest(BaseModel):
    session_id: uuid.UUID


class EndSessionRequest(BaseModel):
    session_id: uuid.UUID


class SessionEventOut(BaseModel):
    id: uuid.UUID
    event_type: str
    payload: dict[str, Any] | None
    timestamp: datetime

    model_config = {"from_attributes": True}


class SessionOut(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    status: SessionStatus
    platform: Platform
    meeting_url: str | None
    start_time: datetime | None
    end_time: datetime | None
    created_at: datetime

    model_config = {"from_attributes": True}


class SessionDetailOut(SessionOut):
    recent_events: list[SessionEventOut] = []
