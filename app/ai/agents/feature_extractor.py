"""Feature extractor agent."""
from __future__ import annotations

from app.ai.agents.base import _load_prompt, call_claude

_SYSTEM = _load_prompt("feature_extractor")


def run(transcript: str, api_key: str, model: str) -> dict:
    """Return extracted features from a transcript segment."""
    return call_claude(_SYSTEM, transcript, api_key, model)
