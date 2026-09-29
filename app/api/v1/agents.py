"""Coding-agent availability endpoints.

Lets the UI/backend query which coding agents are available ("Forgefy Agent" is
available when a provider backend is configured; "Claude Code" is available when
the Claude Code CLI is installed) without attempting a full build.
"""
from __future__ import annotations

from fastapi import APIRouter

from app.build.agents import all_agents_health

router = APIRouter()


@router.get("", response_model=list[dict])
async def list_agents() -> list[dict]:
    """Return availability for every supported coding agent (no secrets)."""
    return all_agents_health()