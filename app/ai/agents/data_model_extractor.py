"""Data model extractor agent."""
from __future__ import annotations

from app.ai.agents.base import _load_prompt, call_claude

_SYSTEM = _load_prompt("data_model_extractor")


def run(transcript: str, api_key: str, model: str, *, usage: list[dict] | None = None) -> dict:
    """Return domain entities (fields + relationships) from a transcript segment."""
    # Entities carry nested fields and relationships, so a single segment can
    # produce several times the output of the flat extractors.
    return call_claude(_SYSTEM, transcript, api_key, model, max_tokens=2048, usage=usage)
