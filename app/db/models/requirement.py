"""Requirement model — Firestore document schema."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from app.db.models.enums import Priority


@dataclass
class Requirement:
    id: uuid.UUID
    session_id: uuid.UUID
    feature: str
    priority: Priority
    created_at: datetime
