"""Conflict detector agent."""
from __future__ import annotations

from app.ai.agents.base import _load_prompt, call_claude

_SYSTEM = _load_prompt("conflict_detector")


def run(transcript: str, api_key: str, model: str, *, usage: list[dict] | None = None) -> dict:
    """Return conflicting requirements detected in a transcript segment."""
    return call_claude(_SYSTEM, transcript, api_key, model, usage=usage)
