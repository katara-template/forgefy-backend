"""User model — Firestore document schema."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime


@dataclass
class User:
    id: uuid.UUID
    email: str
    hashed_password: str
    created_at: datetime
    updated_at: datetime
