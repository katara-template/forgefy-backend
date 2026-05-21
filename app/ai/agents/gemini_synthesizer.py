"""Whole-transcript synthesizer using Gemini via REST API.

Single call that extracts all requirement types at once, identical output
schema to the Claude synthesizer so both can feed the same pipeline.
"""
from __future__ import annotations

import json
import logging

import requests

from app.ai.agents.base import _load_prompt

logger = logging.getLogger(__name__)

_SYSTEM = _load_prompt("synthesizer")
_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def call_gemini(
    system_prompt: str,
    user_content: str,
    api_key: str,
    model: str,
    *,
    max_tokens: int = 2048,
) -> dict:
    """Call Gemini and return the parsed JSON response."""
    url = _GEMINI_URL.format(model=model)
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": user_content}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
        },
    }
    resp = requests.post(url, params={"key": api_key}, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    raw = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Gemini returned non-JSON: %s", raw[:200])
        raise ValueError(f"Gemini response was not valid JSON: {exc}") from exc


def run(transcript: str, api_key: str, model: str) -> list[dict]:
    """Return a flat list of extraction events from a single Gemini call."""
    result = call_gemini(_SYSTEM, transcript, api_key, model, max_tokens=2048)

    events: list[dict] = []
    for feature in result.get("features", []):
        events.append({"sub_state": "FEATURE_FOUND", "payload": feature})
    for question in result.get("questions", []):
        events.append({"sub_state": "QUESTION_FOUND", "payload": question})
    for conflict in result.get("conflicts", []):
        events.append({"sub_state": "CONFLICT_FOUND", "payload": conflict})
    for action_item in result.get("action_items", []):
        events.append({"sub_state": "ACTION_ITEM_FOUND", "payload": action_item})
    return events
