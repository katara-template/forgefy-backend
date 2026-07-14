"""OpenRouter backend for the "Qwen3" model setting.

Instead of pinning every request to one model, each *action* is routed to the
model best suited to it: transcript synthesis gets a long-context JSON model,
code generation gets a coding model, one-line classifications get a small fast
one. The chosen model is whatever ranks highest and actually answers — free
OpenRouter endpoints rate-limit upstream without warning, so every task defines
an ordered chain of candidates and we fall through it on failure.

Free models are tried first. Set OPENROUTER_ALLOW_PAID=true to append paid
fallbacks (DeepSeek/Gemini/Qwen-Coder, all cheap) so a build never dies just
because the free tier is saturated. OPENROUTER_MODEL pins a single model and
overrides all routing.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import requests

logger = logging.getLogger(__name__)

_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterError(RuntimeError):
    """Raised when no model in the chain could fulfil the request."""


# ── Tasks ────────────────────────────────────────────────────────────────────
# Each value is a routing key into _FREE_ROUTES / _PAID_ROUTES below.
SYNTHESIS = "synthesis"   # whole transcript → requirement events (long ctx, JSON)
BLUEPRINT = "blueprint"   # transcript/events → blueprint JSON
FEATURES = "features"     # description → inferred feature list
DESIGN = "design"         # app details → design-system JSON (creative)
NAMING = "naming"         # app details → 4-8 char brand name (tiny)
CLASSIFY = "classify"     # yes/no + reason (tiny)
PLAN = "plan"             # build execution plan (reasoning)
CODE = "code"             # code generation / fixes (coding + tool use)

# Free chains. Ordering is deliberate — first entry is the best fit for the
# task, the rest are progressively weaker but keep the pipeline alive.
# Capabilities verified against OpenRouter's /models catalog.
_FREE_ROUTES: dict[str, list[str]] = {
    # 1M ctx + native JSON + reasoning, and consistently the fastest responder.
    SYNTHESIS: [
        "nvidia/nemotron-3-super-120b-a12b:free",
        "qwen/qwen3-next-80b-a3b-instruct:free",
        "openai/gpt-oss-20b:free",
        "meta-llama/llama-3.3-70b-instruct:free",
    ],
    BLUEPRINT: [
        "nvidia/nemotron-3-super-120b-a12b:free",
        "qwen/qwen3-next-80b-a3b-instruct:free",
        "openai/gpt-oss-20b:free",
        "meta-llama/llama-3.3-70b-instruct:free",
    ],
    FEATURES: [
        "nvidia/nemotron-3-super-120b-a12b:free",
        "qwen/qwen3-next-80b-a3b-instruct:free",
        "google/gemma-4-31b-it:free",
    ],
    # Palette/typography work benefits from a model with stronger aesthetic
    # priors; Gemma is the best free instruct model for this.
    DESIGN: [
        "google/gemma-4-31b-it:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "qwen/qwen3-next-80b-a3b-instruct:free",
    ],
    # Micro-tasks: latency dominates. gpt-oss answers these directly (~5s) while
    # the Nemotrons spend 25-35s reasoning their way to the same one-liner.
    NAMING: [
        "openai/gpt-oss-20b:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "google/gemma-4-31b-it:free",
    ],
    CLASSIFY: [
        "openai/gpt-oss-20b:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "google/gemma-4-31b-it:free",
    ],
    # Planning rewards raw reasoning + a huge context window.
    PLAN: [
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "tencent/hy3:free",
    ],
    # Qwen3-Coder is the strongest free coding model: 1M ctx and tool calling.
    CODE: [
        "qwen/qwen3-coder:free",
        "cohere/north-mini-code:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
    ],
}

# Paid fallbacks, appended only when OPENROUTER_ALLOW_PAID is set. All support
# tools + structured outputs, and all are cheap (DeepSeek is ~$0.20/M input).
_PAID_ROUTES: dict[str, list[str]] = {
    SYNTHESIS: ["deepseek/deepseek-chat", "google/gemini-2.5-flash"],
    BLUEPRINT: ["deepseek/deepseek-chat", "google/gemini-2.5-flash"],
    FEATURES: ["deepseek/deepseek-chat", "google/gemini-2.5-flash"],
    DESIGN: ["google/gemini-2.5-flash", "deepseek/deepseek-chat"],
    NAMING: ["google/gemini-2.5-flash", "deepseek/deepseek-chat"],
    CLASSIFY: ["google/gemini-2.5-flash", "deepseek/deepseek-chat"],
    PLAN: ["deepseek/deepseek-r1", "google/gemini-2.5-flash"],
    CODE: ["qwen/qwen3-coder", "deepseek/deepseek-chat"],
}

# Models that accept response_format={"type":"json_object"}. For the rest we
# rely on the prompt plus _extract_json below.
_JSON_CAPABLE: frozenset[str] = frozenset({
    "nvidia/nemotron-3-super-120b-a12b:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "openai/gpt-oss-20b:free",
    "nvidia/nemotron-nano-9b-v2:free",
    "tencent/hy3:free",
    "deepseek/deepseek-chat",
    "deepseek/deepseek-r1",
    "google/gemini-2.5-flash",
    "qwen/qwen3-coder",
})

# Transient upstream conditions — try the next model in the chain.
_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}

# Almost every strong free model is a reasoning model, and reasoning tokens are
# billed against max_tokens *before* a single character of the answer is
# emitted. Asking for a 4-8 char app name with max_tokens=64 therefore returns
# content=None (the budget is gone by the time it starts answering), and asking
# with 256 returns truncated thinking-prose. Nemotron burned 395 reasoning
# tokens on a naming call even with reasoning explicitly disabled — the flag is
# advisory and these models ignore it. So we floor the budget instead. Output on
# the free tier costs nothing, so the only price of being generous is latency.
_MIN_MAX_TOKENS = 2048


def resolve_models(task: str) -> list[str]:
    """Return the ordered model chain for a task, honouring settings overrides."""
    from app.config import get_settings

    settings = get_settings()
    if pinned := (settings.OPENROUTER_MODEL or "").strip():
        return [pinned]

    chain = list(_FREE_ROUTES.get(task) or _FREE_ROUTES[BLUEPRINT])
    if settings.OPENROUTER_ALLOW_PAID:
        chain += [m for m in _PAID_ROUTES.get(task, []) if m not in chain]
    return chain


def _extract_json(raw: str) -> dict:
    """Parse a JSON object out of a model reply that may wrap it in noise.

    Handles what free models reliably do wrong: <think> preambles (Qwen/Nemotron
    leak reasoning traces straight into content), markdown fences, and prose on
    either side of the object.

    Scans for the first *balanced* object rather than regexing from the first
    "{" to the last "}" — a reply that quotes the schema before answering
    ('Reply ONLY with {"ok": true}. Here goes: {"ok": true}') makes the greedy
    span cover both objects and json.loads then dies on "Extra data".
    """
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) > 1:
            raw = parts[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and parsed:
            return parsed
    except json.JSONDecodeError:
        pass

    # Walk each "{" and keep the first object that both parses and carries data.
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(raw):
        if ch != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(raw[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and candidate:
            return candidate

    raise json.JSONDecodeError("no JSON object found in reply", raw[:200], 0)


def _post(
    model: str,
    system_prompt: str,
    user_content: str,
    *,
    api_key: str,
    max_tokens: int,
    timeout: int,
    json_mode: bool,
    referer: str,
    title: str,
) -> str:
    """One chat completion. Returns raw assistant text, or raises."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max(max_tokens, _MIN_MAX_TOKENS),
    }
    if json_mode and model in _JSON_CAPABLE:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # OpenRouter attributes usage to these; both are optional.
        "HTTP-Referer": referer,
        "X-Title": title,
    }

    resp = requests.post(_CHAT_URL, json=payload, headers=headers, timeout=timeout)

    if resp.status_code in _RETRYABLE_STATUS:
        raise OpenRouterError(f"{model} unavailable (HTTP {resp.status_code}): {resp.text[:200]}")
    if resp.status_code == 401:
        raise OpenRouterError("OpenRouter rejected the API key (HTTP 401). Check OPENROUTER_API_KEY.")
    if resp.status_code >= 400:
        raise OpenRouterError(f"{model} returned HTTP {resp.status_code}: {resp.text[:200]}")

    body = resp.json()
    # OpenRouter can return 200 with an error body when a provider fails mid-stream.
    if err := body.get("error"):
        raise OpenRouterError(f"{model} error: {str(err)[:200]}")

    choices = body.get("choices") or []
    if not choices:
        raise OpenRouterError(f"{model} returned no choices.")

    choice = choices[0]
    content = (choice.get("message") or {}).get("content") or ""

    # finish_reason="length" means the answer was cut mid-flight — on a reasoning
    # model that usually means reasoning ate the budget. Whatever came back is
    # truncated and unparseable, so move on rather than try to salvage it.
    if choice.get("finish_reason") == "length":
        raise OpenRouterError(
            f"{model} hit the token limit before finishing (reasoning likely consumed the budget)."
        )
    if not content.strip():
        raise OpenRouterError(f"{model} returned an empty message.")
    return content.strip()


def chat_openrouter(
    system_prompt: str,
    user_content: str,
    *,
    task: str = BLUEPRINT,
    max_tokens: int = 2048,
    json_mode: bool = True,
) -> str:
    """Run `task` against its model chain and return the raw assistant text.

    Walks the chain until a model answers; raises OpenRouterError only if every
    candidate fails.
    """
    from app.config import get_settings

    settings = get_settings()
    api_key = (settings.OPENROUTER_API_KEY or "").strip()
    if not api_key:
        raise OpenRouterError("OPENROUTER_API_KEY is not set.")

    chain = resolve_models(task)
    failures: list[str] = []

    for model in chain:
        try:
            text = _post(
                model,
                system_prompt,
                user_content,
                api_key=api_key,
                max_tokens=max_tokens,
                timeout=settings.OPENROUTER_TIMEOUT,
                json_mode=json_mode,
                referer=settings.OPENROUTER_SITE_URL,
                title=settings.OPENROUTER_APP_NAME,
            )
        except OpenRouterError as exc:
            failures.append(f"{model}: {exc}")
            logger.warning("OpenRouter task=%s model=%s failed, trying next: %s", task, model, exc)
            continue
        except requests.exceptions.RequestException as exc:
            failures.append(f"{model}: {exc}")
            logger.warning("OpenRouter task=%s model=%s network error: %s", task, model, exc)
            continue

        logger.info("OpenRouter task=%s served by model=%s", task, model)
        return text

    raise OpenRouterError(
        f"All {len(chain)} models failed for task '{task}'. Tried: " + " | ".join(failures)
    )


def call_openrouter(
    system_prompt: str,
    user_content: str,
    *,
    task: str = BLUEPRINT,
    max_tokens: int = 2048,
) -> dict:
    """Drop-in replacement for ollama_synthesizer.call_ollama: returns parsed JSON.

    A model that answers with unparseable JSON is treated as a failure and the
    chain moves on, so one chatty model can't sink the request.
    """
    from app.config import get_settings

    settings = get_settings()
    api_key = (settings.OPENROUTER_API_KEY or "").strip()
    if not api_key:
        raise OpenRouterError("OPENROUTER_API_KEY is not set.")

    chain = resolve_models(task)
    failures: list[str] = []

    for model in chain:
        try:
            text = _post(
                model,
                system_prompt,
                user_content,
                api_key=api_key,
                max_tokens=max_tokens,
                timeout=settings.OPENROUTER_TIMEOUT,
                json_mode=True,
                referer=settings.OPENROUTER_SITE_URL,
                title=settings.OPENROUTER_APP_NAME,
            )
            result = _extract_json(text)
        except (OpenRouterError, requests.exceptions.RequestException) as exc:
            failures.append(f"{model}: {exc}")
            logger.warning("OpenRouter task=%s model=%s failed, trying next: %s", task, model, exc)
            continue
        except json.JSONDecodeError as exc:
            failures.append(f"{model}: non-JSON reply ({exc})")
            logger.warning("OpenRouter task=%s model=%s returned non-JSON, trying next", task, model)
            continue

        if not isinstance(result, dict):
            failures.append(f"{model}: JSON was not an object")
            continue

        logger.info("OpenRouter task=%s served by model=%s", task, model)
        return result

    raise OpenRouterError(
        f"All {len(chain)} models failed for task '{task}'. Tried: " + " | ".join(failures)
    )


def run(transcript: str) -> list[dict]:
    """Whole-transcript synthesis → flat extraction events.

    Mirrors ollama_synthesizer.run so both Qwen3 backends feed the same pipeline.
    """
    from app.ai.agents.base import _load_prompt

    result = call_openrouter(
        _load_prompt("synthesizer"),
        transcript,
        task=SYNTHESIS,
        max_tokens=8192,  # a full transcript's requirements can be long
    )

    events: list[dict] = []
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
