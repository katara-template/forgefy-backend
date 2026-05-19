"""Shared base for all extraction agents."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import anthropic

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8").strip()


def call_claude(
    system_prompt: str,
    user_content: str,
    api_key: str,
    model: str,
    *,
    max_tokens: int = 1024,
) -> dict:
    """Call Claude and return the parsed JSON response.

    Raises ValueError if the response cannot be parsed as JSON.
    """
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
    )
    raw = message.content[0].text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Claude returned non-JSON: %s", raw[:200])
        raise ValueError(f"Claude response was not valid JSON: {exc}") from exc
