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
    timeout: float = 120.0,
    usage: list[dict] | None = None,
) -> dict:
    """Call Claude and return the parsed JSON response.

    When ``usage`` is given, a ``{"input_tokens", "output_tokens"}`` dict is
    appended to it — callers that meter consumption (the developer extract
    API) pass a list; everyone else ignores it.

    Raises ValueError if the response cannot be parsed as JSON.
    """
    client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
    )
    if usage is not None:
        usage.append({
            "input_tokens": int(getattr(message.usage, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(message.usage, "output_tokens", 0) or 0),
        })
    raw = message.content[0].text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Claude returned non-JSON: %s", raw[:200])
        raise ValueError(f"Claude response was not valid JSON: {exc}") from exc
