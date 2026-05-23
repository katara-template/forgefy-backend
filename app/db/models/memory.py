"""Memory model — Firestore document schema for RAG transcript chunks."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Memory:
    id: uuid.UUID
    session_id: uuid.UUID
    content: str
    embedding: list[float] | None
    created_at: datetime
