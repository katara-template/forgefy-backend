"""MeetingSession model — Firestore document schema."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from app.db.models.enums import Platform, SessionStatus


@dataclass
class MeetingSession:
    id: uuid.UUID
    user_id: uuid.UUID
    status: SessionStatus
    platform: Platform
    meeting_url: str | None
    start_time: datetime | None
    end_time: datetime | None
    created_at: datetime
