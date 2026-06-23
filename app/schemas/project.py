"""Project response schemas."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class ProjectOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    owner_id: uuid.UUID
    app_name: str
    template_key: str
    repo_full_name: str
    github_url: str
    repo_owner: str | None = None  # "platform" or "user"
    created_at: datetime
    updated_at: datetime
    session_id: uuid.UUID | None = None
    blueprint_id: uuid.UUID | None = None
    preview_url: str | None = None
    artifact_url: str | None = None
    is_updating: bool = False
    build_error: str | None = None
    build_error_action: str | None = None  # "retry" | "user_fix" | "support"


class UpdateProjectRequest(BaseModel):
    prompt: str


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    type: str          # "chat" | "update"
    response: str      # message to show the user
    update_queued: bool = False
