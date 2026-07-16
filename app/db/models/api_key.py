"""ApiKey model — Firestore document schema.

Collection: api_keys/{key_id}. The raw key is never stored — key_hash is the
SHA-256 digest it is looked up by (see app/core/api_keys.py).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ApiKey:
    id: uuid.UUID
    owner_user_id: uuid.UUID
    name: str
    prefix: str  # plaintext display prefix, e.g. "fgy_live_k3J"
    key_hash: str  # hex SHA-256 of the full key
    created_at: datetime
    last_used_at: datetime | None = None
    revoked_at: datetime | None = None

    @property
    def revoked(self) -> bool:
        return self.revoked_at is not None
