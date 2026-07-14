"""Dispatcher for the "Qwen3" model setting.

BP_MODEL/BUILD_MODEL="Qwen3" has two possible backends:

  • OpenRouter (hosted) — used when OPENROUTER_API_KEY is set. Routes each
    action to the model best suited to it (see app/ai/openrouter.py), which may
    well be Nemotron, Gemma or DeepSeek rather than a literal Qwen model.
  • Ollama (local)      — the original behaviour, used when no key is set.

Call sites pass a `task` so the hosted backend can pick a fitting model; the
local backend ignores it and uses OLLAMA_MODEL for everything.
"""
from __future__ import annotations

import logging

from app.ai.openrouter import BLUEPRINT, SYNTHESIS  # noqa: F401  (re-exported for call sites)
from app.ai.openrouter import CLASSIFY, CODE, DESIGN, FEATURES, NAMING, PLAN  # noqa: F401

logger = logging.getLogger(__name__)


def using_openrouter() -> bool:
    """True when the Qwen3 setting should resolve to the hosted OpenRouter backend."""
    from app.config import get_settings

    return bool((get_settings().OPENROUTER_API_KEY or "").strip())


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
    return call_ollama(
        system_prompt=system_prompt,
        user_content=user_content,
        base_url=settings.OLLAMA_URL,
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
    return ollama_run(
        transcript,
        settings.OLLAMA_URL,
        settings.OLLAMA_MODEL,
        timeout=settings.OLLAMA_TIMEOUT,
    )
