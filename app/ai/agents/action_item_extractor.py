"""Action item extractor agent."""
from __future__ import annotations

from app.ai.agents.base import _load_prompt, call_claude

_SYSTEM = _load_prompt("action_item_extractor")


def run(transcript: str, api_key: str, model: str, *, usage: list[dict] | None = None) -> dict:
    """Return action items extracted from a transcript segment."""
    return call_claude(_SYSTEM, transcript, api_key, model, usage=usage)
