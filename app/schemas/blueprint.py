"""Blueprint response schemas."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class BlueprintOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    session_id: uuid.UUID
    json_output: dict[str, Any] | None
    approved: bool
    created_at: datetime
    repo_url: str | None = None
    repo_name: str | None = None
    build_summary: str | None = None
    build_status: str | None = None
    artifact_url: str | None = None
    preview_url: str | None = None
