"""Project response schemas."""
from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


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
    supabase_project_ref: str | None = None
    supabase_url: str | None = None
    supabase_anon_key: str | None = None
    neon_project_id: str | None = None
    neon_data_api_url: str | None = None
    firebase_project_id: str | None = None
    firebase_api_key: str | None = None
    firebase_auth_domain: str | None = None
    firebase_storage_bucket: str | None = None
    firebase_messaging_sender_id: str | None = None
    firebase_app_id: str | None = None
    db_decision_pending: bool = False
    db_decision_reason: str | None = None
    # Schema-provisioning backstop state (app/integrations/db_migrations.py).
    # db_status: "ready" | "empty" | "error" | None. Exposed so the client can
    # show whether a connected database has real tables vs. code wired to an
    # empty one.
    db_schema_version: int | None = None
    db_schema_tables: list[str] | None = None
    db_status: str | None = None
    db_schema_error: str | None = None


class UpdateProjectRequest(BaseModel):
    prompt: str


class ConnectSupabaseRequest(BaseModel):
    organization_id: str


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    type: str          # "chat" | "update"
    response: str      # message to show the user
    update_queued: bool = False
    needs_database: bool = False
    # Present only when type is "clarify": exactly 2 items (a yes/no question) or
    # exactly 3 items (a multiple-choice question) — never free text the user has
    # to type a reply to. See chat_with_project's "ASKING CLARIFYING QUESTIONS" rules.
    clarify_options: list[str] | None = None


class ChatHistoryMessage(BaseModel):
    """One persisted transcript bubble.

    Field names are the client's camelCase on purpose — this document is written
    and read back by the dashboard only, and renaming it on the way through
    would just cost a mapping layer on both sides.

    ``needs_database`` and ``clarify_options`` carry a question the user has not
    answered yet ("Add a database / No thanks"). They are part of the stored
    shape rather than transient UI state: without them a reload restores the
    question but not its buttons, stranding the user on a prompt they cannot
    answer. ``answered_option`` records the choice so the row renders settled.
    """

    model_config = ConfigDict(populate_by_name=True)

    id: str
    role: str          # "user" | "assistant"
    text: str
    timestamp: str     # ISO 8601, as produced by Date.toISOString()
    needs_database: bool | None = Field(default=None, alias="needsDatabase")
    clarify_options: list[str] | None = Field(default=None, alias="clarifyOptions")
    answered_option: str | None = Field(default=None, alias="answeredOption")


class ChatHistoryRequest(BaseModel):
    messages: list[ChatHistoryMessage]
