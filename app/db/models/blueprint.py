"""Blueprint model — Firestore document schema."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class Blueprint:
    id: uuid.UUID
    session_id: uuid.UUID
    json_output: dict[str, Any] | None
    approved: bool
    created_at: datetime
