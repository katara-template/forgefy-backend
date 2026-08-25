"""Dispatcher for the "Qwen3" model setting.

BP_MODEL/BUILD_MODEL="Qwen3" has two possible backends:

  • OpenRouter (hosted) — used when OPENROUTER_API_KEY is set. Routes each
    action to the model best suited to it (see app/ai/openrouter.py), which may
    well be Nemotron, Gemma or DeepSeek rather than a literal Qwen model.
  • Ollama              — used when no OpenRouter key is set. Hits Ollama Cloud
    when OLLAMA_API_KEY is set, otherwise a local daemon (app/ai/ollama_http.py).

Call sites pass a `task` so the hosted backend can pick a fitting model; the
local backend ignores it and uses OLLAMA_MODEL for everything.
"""
from __future__ import annotations

import logging

from app.ai.openrouter import (  # noqa: F401  (re-exported for call sites)  # noqa: F401
    BLUEPRINT,
    CLASSIFY,
    CODE,
    DESIGN,
    FEATURES,
    NAMING,
    PLAN,
    SYNTHESIS,
)

logger = logging.getLogger(__name__)


def using_openrouter() -> bool:
    """True when the Qwen3 setting should resolve to the hosted OpenRouter backend.

    OLLAMA_API_KEY wins when both are present: setting it is an explicit request
    for Ollama Cloud, whereas an OpenRouter key is often left in .env as a
    leftover. Clear OLLAMA_API_KEY to fall back to OpenRouter.
    """
    from app.ai.ollama_http import using_cloud
    from app.config import get_settings

    if using_cloud():
        return False
    return bool((get_settings().OPENROUTER_API_KEY or "").strip())


def fallback_to_openrouter_enabled() -> bool:
    """True when a failing Ollama run may switch to OpenRouter mid-build.

    This is the automatic escape hatch for the opposite case of
    ``using_openrouter``: an OLLAMA_API_KEY that picks Ollama as primary, but
    whose endpoint rate-limits (Ollama Cloud 429s without warning) or is down.
    Requires OPENROUTER_API_KEY; without it there is nowhere to fall back to.
    """
    from app.config import get_settings

    settings = get_settings()
    if not settings.QWEN_FALLBACK_TO_OPENROUTER:
        return False
    return bool((settings.OPENROUTER_API_KEY or "").strip())


def call_qwen(
    system_prompt: str,
    user_content: str,
    *,
    task: str = BLUEPRINT,
    max_tokens: int = 2048,
) -> dict:
    """Single JSON-returning call on whichever Qwen3 backend is configured."""
    from app.config import get_settings

    settings = get_settings()

    if using_openrouter():
        from app.ai.openrouter import call_openrouter
        return call_openrouter(system_prompt, user_content, task=task, max_tokens=max_tokens)

    from app.ai.agents.ollama_synthesizer import call_ollama
    from app.ai.ollama_http import ollama_base_url
    return call_ollama(
        system_prompt=system_prompt,
        user_content=user_content,
        base_url=ollama_base_url(settings),
        model=settings.OLLAMA_MODEL,
        timeout=settings.OLLAMA_TIMEOUT,
    )


def run_qwen_synthesis(transcript: str) -> list[dict]:
    """Whole-transcript synthesis on whichever Qwen3 backend is configured."""
    from app.config import get_settings

    settings = get_settings()

    if using_openrouter():
        from app.ai.openrouter import run as openrouter_run
        return openrouter_run(transcript)

    from app.ai.agents.ollama_synthesizer import run as ollama_run
    from app.ai.ollama_http import ollama_base_url
    return ollama_run(
        transcript,
        ollama_base_url(settings),
        settings.OLLAMA_MODEL,
        timeout=settings.OLLAMA_TIMEOUT,
    )
