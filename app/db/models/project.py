"""Project model — Firestore document schema."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Project:
    id: uuid.UUID
    owner_id: uuid.UUID
    app_name: str
    template_key: str
    repo_full_name: str
    github_url: str
    created_at: datetime
    updated_at: datetime
    session_id: uuid.UUID | None = None
    blueprint_id: uuid.UUID | None = None
    preview_url: str | None = None
    artifact_url: str | None = None
    is_updating: bool = False
    blueprint_context: dict[str, Any] | None = field(default=None, repr=False)
