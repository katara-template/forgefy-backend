"""MeetingEvent model — Firestore document schema."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class MeetingEvent:
    id: uuid.UUID
    session_id: uuid.UUID
    event_type: str
    payload: dict[str, Any] | None
    timestamp: datetime
