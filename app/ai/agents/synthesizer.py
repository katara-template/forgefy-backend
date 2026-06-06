"""Whole-transcript synthesizer — single Claude call that extracts all requirement types at once.

Used as a fallback when the per-segment pipeline returns no events (e.g., short segments,
API errors on individual agents, or transcripts that don't fit per-type prompts well).
"""
from __future__ import annotations

from app.ai.agents.base import _load_prompt, call_claude

_SYSTEM = _load_prompt("synthesizer")


def run(transcript: str, api_key: str, model: str) -> list[dict]:
    """Return a flat list of extraction events from a single Claude call.

    Each event has ``sub_state`` (FEATURE_FOUND etc.) and ``payload`` matching
    the same schema used by the per-segment pipeline.
    """
    result = call_claude(_SYSTEM, transcript, api_key, model, max_tokens=2048)

    events: list[dict] = []
    if app_name := result.get("app_name", ""):
        events.append({"sub_state": "APP_NAME", "payload": {"text": app_name}})
    if app_desc := result.get("app_description", ""):
        events.append({"sub_state": "APP_DESCRIPTION", "payload": {"text": app_desc}})
    for feature in result.get("features", []):
        events.append({"sub_state": "FEATURE_FOUND", "payload": feature})
    for question in result.get("questions", []):
        events.append({"sub_state": "QUESTION_FOUND", "payload": question})
    for conflict in result.get("conflicts", []):
        events.append({"sub_state": "CONFLICT_FOUND", "payload": conflict})
    for action_item in result.get("action_items", []):
        events.append({"sub_state": "ACTION_ITEM_FOUND", "payload": action_item})
    return events
