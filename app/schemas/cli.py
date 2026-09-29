"""Pydantic schemas for the CLI-facing OpenAI-compatible chat endpoint.

Machine-authed with the same `fgy_live_…` keys as the developer extract API
(see app.deps.get_api_key) — the `forgefy` CLI provider (forgefy-cli's
config.py BUILTINS) points its base_url here. Requests are forwarded to
OpenRouter close to verbatim (app/api/v1/cli.py); these schemas only bound
what a caller may send.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator

Role = Literal["system", "user", "assistant", "tool"]

# Generous enough for a multi-turn coding/edit session with file contents
# attached, far below what would make a single request pathological upstream.
# Mirrors the CLI's own LIMIT (forgefy_cli/context.py) so a request that
# passed the client-side cap never gets rejected server-side.
_MAX_MESSAGES = 400
_MAX_TOTAL_CHARS = 200_000


class CliMessage(BaseModel):
    role: Role
    content: str | None = None
    # Present on an assistant turn that made tool calls, and echoed back by
    # the CLI's own history — forwarded to the model as-is, never inspected.
    tool_calls: list[dict] | None = None
    # Present on a "tool" role turn, pairing the result with its call.
    tool_call_id: str | None = None


class CliChatRequest(BaseModel):
    model: str
    messages: list[CliMessage]
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    stream: bool = False

    @field_validator("messages")
    @classmethod
    def messages_bounded(cls, v: list[CliMessage]) -> list[CliMessage]:
        if not v:
            raise ValueError("messages must not be empty")
        if len(v) > _MAX_MESSAGES:
            raise ValueError(f"messages exceeds {_MAX_MESSAGES} turns")
        total = sum(len(m.content or "") for m in v)
        if total > _MAX_TOTAL_CHARS:
            raise ValueError(f"messages exceed {_MAX_TOTAL_CHARS} characters combined")
        return v

    @field_validator("stream")
    @classmethod
    def stream_must_be_false(cls, v: bool) -> bool:
        if v:
            raise ValueError("Streaming is not supported on this endpoint")
        return v
