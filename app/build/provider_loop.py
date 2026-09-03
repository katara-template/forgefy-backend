"""One agent loop behind a provider adapter (Part J), with three-zone prefix
stability (Part E1) and per-provider caching (Part E2).

Since Part L this is the ONLY agent loop: the five historical per-provider
loops (Anthropic ``_loop``, ``_ollama_agent_turns``, ``_gemini_loop``,
``_openai_loop``, ``_openrouter_loop``) were deleted after the unified path
soaked clean. They had drifted apart — round 1's guards were hand-copied into
``_loop`` and its caching landed only there, leaving the default provider
without it.

This module extracts the loop and parameterises it with a ``ProviderAdapter``.
The shared ``run_agent_loop`` owns every guard, the trimming policy (VOLATILE
only), cache placement and instrumentation exactly once. Adapters hold only
wire-format and cache-flag differences.

``cache_stats`` is part of the adapter contract: every adapter reports whatever
its provider exposes (Anthropic cache_read/creation, Gemini
cachedContentTokenCount, OpenAI/OpenRouter cached_tokens, Ollama prompt_eval)
normalised to ``{cached_tokens, uncached_tokens}``. A provider that reports
nothing returns zeros — the field is never omitted.
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.build.agent_tools import TOOLS, execute_tool
from app.build.build_agent import (
    _MAX_EXPLORE_STREAK,
    _MAX_ITERATIONS,
    _MAX_NUDGES,
    _MAX_OUTPUT_TOKENS_CHAT,
    _READ_ONLY_TOOLS,
    _TOOL_RESULT_LIMIT_LARGE_CTX,
    _TOOL_RESULT_LIMIT_SMALL_CTX,
    _WRITE_TOOLS,
    _truncate_tool_result,
)

logger = logging.getLogger(__name__)

# ── Normalised cache accounting ───────────────────────────────────────────────


@dataclass
class CacheStats:
    """Normalised ``{cached_tokens, uncached_tokens}`` for one turn.

    ``cached_tokens`` are input tokens served from a provider cache;
    ``uncached_tokens`` are everything billed at the non-cached rate (fresh
    context plus, for Anthropic, cache-creation tokens — writes still cost
    input, they are not reads). Every adapter fills both fields; a provider
    that exposes no cache numbers reports zeros.
    """

    cached_tokens: int = 0
    uncached_tokens: int = 0
    # Provider-native split, preserved for measurability (Part E3) without
    # pretending every provider reports the same shape.
    raw: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return self.cached_tokens + self.uncached_tokens

    def as_dict(self) -> dict[str, int]:
        return {"cached_tokens": self.cached_tokens, "uncached_tokens": self.uncached_tokens}


# Part E1: ANCHOR (system + tool schemas) and STABLE (blueprint / project map /
# git log / installed packages, all inside the first user message) are immutable
# for the whole phase; only VOLATILE (conversation history, tool results, the
# rolling window) may be trimmed. messages[:_ANCHOR_MSGS] is never dropped.
_ANCHOR_MSGS = 1


@dataclass
class TurnResult:
    """A provider call reduced to the loop's vocabulary."""

    text: str = ""
    # Each: {"id", "name", "arguments"} (arguments is a JSON-serialisable dict).
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    # Canonical: "end_turn" | "tool_use" | "max_tokens" | None
    stop_reason: str | None = None
    input_tokens: int = 0  # provider's total input tokens (cached + fresh + write)
    output_tokens: int = 0
    cache_stats: CacheStats = field(default_factory=CacheStats)
    # Model that actually served this turn (OpenRouter falls through a chain).
    model: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

# ── Adapter contract ──────────────────────────────────────────────────────────


class ProviderAdapter:
    """Wire-format packaging for one provider. No control logic lives here.

    The shared ``run_agent_loop`` calls ``send`` with a normalised message list
    and handles every guard; the adapter only (a) converts normalised messages
    and tools to the provider's payload, (b) streams / posts it, and (c) returns
    a normalised ``TurnResult`` carrying its cache split.
    """

    name: str = "provider"
    supports_images: bool = False  # Part G capability gate — never hardcode names.
    # Cost policy for image inputs, independent of supports_images: "standard"
    # unless the provider's image pricing is high enough that routing may want
    # to prefer a cheaper backend. Routing consults this; the capability flag
    # above never lies about what the model can do.
    image_cost_tier: str = "standard"
    supports_cache: bool = False   # Whether the provider exposes a cache to enable.
    # Rolling window, in assistant/tool-response pairs.
    history_pairs: int = 20

    def __init__(self, **cfg: Any) -> None:
        self.cfg = cfg

    def normalize_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        """Return the provider's wire tool definitions for `tools`."""
        return [dict(t) for t in (tools if tools is not None else TOOLS)]

    def trim_result_limit(self, tools: list[dict[str, Any]] | None = None) -> int:
        """How much tool-result text this provider's context can afford."""
        return _TOOL_RESULT_LIMIT_LARGE_CTX

    def format_tool_result(self, call_id: str, name: str, result: str) -> dict[str, Any]:
        """Provider-native shape of one tool result (a normalised tool turn)."""
        return {"role": "tool_turn", "tool_results": [{"call_id": call_id, "name": name, "content": result}]}

    def send(
        self,
        *,
        system: str,
        stable: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        anchor_end: int,
        will_trim: bool,
        log_fn: Callable[[str, str], None] | None = None,
        on_text: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
    ) -> TurnResult:
        """Post one request for `messages` and return a normalised TurnResult."""
        raise NotImplementedError


def _gemini_cache_ctx(usage: Any) -> CacheStats:
    """Normalise Gemini's cachedContentTokenCount into CacheStats."""
    cached = 0
    uncached = 0
    try:
        details = (usage or {}).get("promptTokensDetails") or []
        for d in details:
            t = d.get("tokenCount", 0) or 0
            if d.get("isCached"):
                cached += t
            else:
                uncached += t
    except Exception:
        return CacheStats()
    return CacheStats(cached_tokens=cached, uncached_tokens=uncached)


def _openai_cache_ctx(usage: Any) -> tuple[CacheStats, int, int]:
    """Normalise an OpenAI/OpenRouter usage object.

    Returns (cache_stats, input_tokens, output_tokens). Caching is automatic
    above ~1024 tokens for these providers; there is no opt-in flag, so the
    whole win is the shared ANCHOR/STABLE byte-stability (E1).
    """
    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    cached = 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached = int(getattr(details, "cached_tokens", 0) or 0)
    uncached = max(0, input_tokens - cached)
    return CacheStats(cached_tokens=cached, uncached_tokens=uncached), input_tokens, output_tokens


def _anthropic_cache_ctx(usage: Any) -> tuple[CacheStats, int, int]:
    """Normalise Anthropic's cache_read / cache_creation split."""
    fresh = int(getattr(usage, "input_tokens", 0) or 0)
    created = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    output = int(getattr(usage, "output_tokens", 0) or 0)
    # Cache-creation tokens are billed input, so they count as "uncached" here.
    stats = CacheStats(cached_tokens=read, uncached_tokens=fresh + created)
    return stats, fresh + created + read, output

# ── OpenAI / OpenRouter compatible wire format ────────────────────────────────


def _openai_wire_messages(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert the normalised list to OpenAI/OpenRouter chat-completions messages."""
    wire: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for m in messages:
        role = m.get("role")
        if role == "user":
            wire.append({"role": "user", "content": m.get("content", "")})
        elif role == "assistant":
            calls = m.get("tool_calls") or []
            entry: dict[str, Any] = {"role": "assistant", "content": m.get("content", "")}
            if calls:
                entry["tool_calls"] = [
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {"name": c["name"], "arguments": json.dumps(c.get("arguments") or {}, default=str)},
                    }
                    for c in calls
                ]
            wire.append(entry)
        elif role == "tool_turn":
            for tr in m.get("tool_results", []):
                wire.append({"role": "tool", "tool_call_id": tr["call_id"], "content": tr["content"]})
            if m.get("text"):
                wire.append({"role": "user", "content": m["text"]})
    return wire


def _safe_args(raw: Any) -> dict[str, Any]:
    """Parse tool-call arguments, repairing common LLM JSON errors.

    Handles:
      * Already-parsed dicts (passthrough).
      * Valid JSON strings.
      * Unescaped backslashes in string values (Windows paths, regex).
      * Unrepairable garbage → ``{}``.
    """
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        # Repair: unescaped backslashes inside string values are the most common
        # LLM emission error.  Walk the string and double any lone backslash.
        try:
            repaired = _repair_json_backslashes(raw)
            parsed = json.loads(repaired)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}


def _valid_args_json(raw: Any) -> str:
    """Return a JSON-serialisable string of ``raw``, always valid.

    Unlike ``_safe_args`` this returns a *string* (the wire format the loop
    echoes back into history) and guarantees a parseable result even for
    garbage input — the specific failure this guards against is a raw,
    provider-400-causing echo landing in ``messages[N]``.
    """
    if isinstance(raw, dict):
        return json.dumps(raw, ensure_ascii=False)
    if not isinstance(raw, str) or not raw.strip():
        return "{}"
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        try:
            repaired = _repair_json_backslashes(raw)
            json.loads(repaired)
            return repaired
        except (json.JSONDecodeError, TypeError):
            return "{}"


def _repair_json_backslashes(s: str) -> str:
    """Double lone backslashes inside JSON string values.

    The model emits ``C:\\Users\\app`` (two chars) where the wire wants
    ``C:\\\\Users\\\\app`` (four chars).  We can't just blanket-replace all
    backslashes because ``\\n`` and ``\\t`` are already valid escapes —
    so we walk the string, tracking whether we are inside a string value,
    and double any backslash that is not already part of a recognised
    escape sequence.
    """
    out: list[str] = []
    in_string = False
    valid_escapes = set('\"\\/bfnrtu')
    chars = list(s)
    i = 0
    while i < len(chars):
        ch = chars[i]
        if ch == '"' and (i == 0 or chars[i-1] != '\\'):
            in_string = not in_string
            out.append(ch)
            i += 1
            continue
        if ch == '\\' and in_string:
            # Check if this is already a valid JSON escape
            if i + 1 < len(chars) and chars[i+1] in valid_escapes:
                # Valid escape, keep as-is
                out.append(ch)
                out.append(chars[i+1])
                i += 2
                continue
            # Not a valid escape — double the backslash
            out.append('\\\\')
            i += 1
            continue
        out.append(ch)
        i += 1
    # If we ended mid-string, the model truncated its JSON; close it.
    result = ''.join(out)
    if in_string:
        result += '"'
    return result


class OpenAIAdapter(ProviderAdapter):
    """The ``gpt`` build backend. Caching is automatic above ~1024 tokens (no
    opt-in flag, Part E2) — the whole win is the byte-stable ANCHOR/STABLE
    prefix, verified through ``usage.prompt_tokens_details.cached_tokens``."""

    name = "openai"
    supports_images = True   # natively multimodal (Part G note).

    def __init__(self, *, api_key: str, model: str) -> None:
        super().__init__(api_key=api_key, model=model)

    def send(self, **kw: Any) -> TurnResult:
        from openai import OpenAI

        client = OpenAI(api_key=self.cfg["api_key"])
        messages = _openai_wire_messages(kw["system"], kw["messages"])
        tools = self.normalize_tools(kw["tools"])
        try:
            response = client.chat.completions.create(
                model=self.cfg["model"],
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=_MAX_OUTPUT_TOKENS_CHAT,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"OpenAI agent request failed: {exc}") from exc

        msg = response.choices[0].message
        tool_calls = [
            {"id": tc.id, "name": tc.function.name, "arguments": _safe_args(tc.function.arguments)}
            for tc in (msg.tool_calls or [])
        ]
        stats, in_tokens, out_tokens = _openai_cache_ctx(response.usage)
        if msg.content and kw.get("on_text"):
            kw["on_text"](msg.content)
        stop = "tool_use" if tool_calls else ("max_tokens" if response.choices[0].finish_reason == "length" else "end_turn")
        return TurnResult(
            text=msg.content or "",
            tool_calls=tool_calls,
            stop_reason=stop,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            cache_stats=stats,
            model=self.cfg["model"],
        )


class OpenRouterAdapter(OpenAIAdapter):
    """The hosted ``Qwen3`` build backend, driving the CODE chain with mid-build
    failover. Caching is per upstream model (Part E2), so every turn records who
    actually served it and the cache rate is reported per model, never per run."""

    name = "openrouter"

    def __init__(self, *, api_key: str, base_url: str = "", chain: list[str] | None = None,
                 model: str = "") -> None:
        super().__init__(api_key=api_key, model=model)
        self.cfg["base_url"] = base_url
        self.cfg["chain"] = list(chain or [])

    def send(self, **kw: Any) -> TurnResult:
        from openai import OpenAI

        from app.ai.openrouter import OPENROUTER_BASE_URL

        client = OpenAI(api_key=self.cfg["api_key"], base_url=self.cfg.get("base_url") or OPENROUTER_BASE_URL)
        chain = list(self.cfg["chain"])
        messages = _openai_wire_messages(kw["system"], kw["messages"])
        tools = self.normalize_tools(kw["tools"])
        log_fn = kw.get("log_fn")

        last_exc: Exception | None = None
        served: str | None = None
        resp_obj = None
        for idx, mdl in enumerate(chain):
            try:
                resp_obj = client.chat.completions.create(
                    model=mdl, messages=messages, tools=tools,
                    tool_choice="auto", max_tokens=_MAX_OUTPUT_TOKENS_CHAT,
                )
                served = mdl
                if idx > 0:
                    # Model names never reach the user feed; operators see them here.
                    logger.info("openrouter: serving with fallback model %s", mdl)
                    if log_fn:
                        log_fn("info", "Adjusting the AI route on our side…")
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                status = getattr(exc, "status_code", None)
                if status in {408, 409, 429, 500, 502, 503, 504}:
                    time.sleep(3)
                    continue
                break
        if resp_obj is None:
            raise RuntimeError(f"OpenRouter build agent request failed: {last_exc}")

        msg = resp_obj.choices[0].message
        tool_calls = [
            {"id": tc.id, "name": tc.function.name, "arguments": _safe_args(tc.function.arguments)}
            for tc in (msg.tool_calls or [])
        ]
        stats, in_tokens, out_tokens = _openai_cache_ctx(resp_obj.usage)
        if log_fn:
            try:
                reasoning = (getattr(msg, "reasoning", None) or "").strip()
                if reasoning:
                    log_fn("thinking", reasoning[:200])
            except Exception:
                pass
        if msg.content and kw.get("on_text"):
            kw["on_text"](msg.content)
        stop = "tool_use" if tool_calls else ("max_tokens" if resp_obj.choices[0].finish_reason == "length" else "end_turn")
        return TurnResult(
            text=msg.content or "",
            tool_calls=tool_calls,
            stop_reason=stop,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            cache_stats=stats,
            model=served,
        )


# ── Gemini wire format ────────────────────────────────────────────────────────


def _gemini_wire_contents(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert the normalised list to Gemini ``contents``."""
    contents: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role == "user":
            contents.append({"role": "user", "parts": [{"text": m.get("content", "")}]})
        elif role == "assistant":
            parts: list[dict[str, Any]] = []
            if m.get("content"):
                parts.append({"text": m["content"]})
            for c in m.get("tool_calls", []):
                parts.append({"functionCall": {"name": c["name"], "args": c.get("arguments") or {}}})
            contents.append({"role": "model", "parts": parts})
        elif role == "tool_turn":
            parts = []
            for tr in m.get("tool_results", []):
                parts.append({"functionResponse": {"name": tr["name"], "response": {"output": tr["content"]}}})
            if m.get("text"):
                parts.append({"text": m["text"]})
            contents.append({"role": "user", "parts": parts})
    return contents


def _gemini_wire_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Gemini's function-declaration tool shape."""
    return [{
        "functionDeclarations": [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {}),
            }
            for t in tools
        ]
    }]


class GeminiAdapter(ProviderAdapter):
    """The ``gemini`` build backend.

    Part E2: ANCHOR goes into ``systemInstruction``. Explicit ``cachedContent``
    (with a TTL covering the phase) is added when ``explicit_cache`` is enabled;
    if creation fails it falls back to implicit caching, and the handle never
    outlives the phase because the adapter instance is created per phase and
    ``close()`` deletes the handle. The cache split is read from
    ``usageMetadata.cachedContentTokenCount`` / ``promptTokensDetails``.
    """

    name = "gemini"
    supports_images = True   # natively multimodal (Part G).
    supports_cache = True

    def __init__(self, *, api_key: str, model: str, explicit_cache: bool = False,
                 cache_ttl: int = 3600) -> None:
        super().__init__(api_key=api_key, model=model,
                         explicit_cache=explicit_cache, cache_ttl=cache_ttl)
        self._cached_name: str | None = None
        self._cache_attempted = False

    def _maybe_create_cached_content(self, system: str, stable: str) -> None:
        """.cachedContents.create for the ANCHOR (system) + STABLE prefix."""
        if not self.cfg.get("explicit_cache") or self._cache_attempted:
            return
        self._cache_attempted = True
        import requests as _req

        url = "https://generativelanguage.googleapis.com/v1beta/cachedContents"
        try:
            resp = _req.post(
                url, params={"key": self.cfg["api_key"]},
                json={
                    "model": f"models/{self.cfg['model']}",
                    "systemInstruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "user", "parts": [{"text": stable}]}],
                    "ttl": f"{self.cfg.get('cache_ttl', 3600)}s",
                },
                timeout=30,
            )
            resp.raise_for_status()
            self._cached_name = (resp.json() or {}).get("name")
            logger.info("gemini: created %s for phase (ttl=%ss)", self._cached_name, self.cfg.get("cache_ttl", 3600))
        except Exception as exc:  # noqa: BLE001
            # Any failure — auth, quota, model support — falls back to implicit
            # caching; the request path below is unchanged.
            logger.warning("gemini: explicit cachedContent unavailable, using implicit: %s", exc)
            self._cached_name = None

    def close(self) -> None:
        """Delete a phase-scoped cachedContent handle so it cannot outlive the run."""
        name = self._cached_name
        self._cached_name = None
        if not name:
            return
        import requests as _req

        try:
            _req.delete(
                f"https://generativelanguage.googleapis.com/v1beta/{name}",
                params={"key": self.cfg["api_key"]}, timeout=15,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("gemini: could not delete %s (TTL will expire it): %s", name, exc)


    def send(self, **kw: Any) -> TurnResult:
        from app.build.build_agent import _GEMINI_URL, _gemini_post_with_retry

        model = self.cfg["model"]
        url = _GEMINI_URL.format(model=model)
        all_contents = _gemini_wire_contents(kw["messages"])
        tools = _gemini_wire_tools(self.normalize_tools(kw["tools"]))

        self._maybe_create_cached_content(kw["system"], kw["stable"])
        # With an active explicit cache, ANCHOR+STABLE live in the cachedContent,
        # so only VOLATILE history (messages after the STABLE seed) is inline.
        contents = all_contents[1:] if self._cached_name else all_contents

        payload: dict[str, Any] = {
            "system_instruction": {"parts": [{"text": kw["system"]}]},
            "contents": contents,
            "tools": tools,
            "generationConfig": {"maxOutputTokens": 8192},
        }
        if self._cached_name:
            payload["cachedContent"] = self._cached_name

        data = _gemini_post_with_retry(url, self.cfg["api_key"], payload, timeout=120, log_fn=kw.get("log_fn"))
        usage = data.get("usageMetadata") or {}
        total_tokens = usage.get("totalTokenCount", 0)
        stats = _gemini_cache_ctx(usage)

        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"Gemini returned no candidates: {data}")
        candidate = candidates[0]
        parts = candidate.get("content", {}).get("parts", [])
        finish = candidate.get("finishReason", "")

        text = ""
        tool_calls: list[dict[str, Any]] = []
        for part in parts:
            if "text" in part:
                text += part["text"]
            elif "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append({"id": fc.get("id", "") or fc.get("name", ""),
                                   "name": fc.get("name", ""), "arguments": fc.get("args") or {}})
        # Gemini functionCall parts carry no stable id; synthesise per-turn ids so
        # tool results can be correlated back to their calls.
        for idx, tc in enumerate(tool_calls):
            if not tc["id"]:
                tc["id"] = f"gemini-{idx}"

        if text and kw.get("on_text"):
            kw["on_text"](text[:160])
        stop = "tool_use" if tool_calls else ("max_tokens" if finish == "MAX_TOKENS" else "end_turn")
        return TurnResult(
            text=text,
            tool_calls=tool_calls,
            stop_reason=stop,
            input_tokens=total_tokens,
            output_tokens=0,
            cache_stats=stats,
            model=model,
        )


# ── Anthropic wire format ─────────────────────────────────────────────────────


def _anthropic_wire_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert the normalised list to Anthropic content-block messages."""
    wire: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role == "user":
            wire.append({"role": "user", "content": m.get("content", "")})
        elif role == "assistant":
            blocks: list[dict[str, Any]] = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for c in m.get("tool_calls", []):
                blocks.append({"type": "tool_use", "id": c["id"], "name": c["name"], "input": c.get("arguments") or {}})
            wire.append({"role": "assistant", "content": blocks})
        elif role == "tool_turn":
            blocks = []
            for tr in m.get("tool_results", []):
                blocks.append({"type": "tool_result", "tool_use_id": tr["call_id"], "content": tr["content"]})
            if m.get("text"):
                blocks.append({"type": "text", "text": m["text"]})
            wire.append({"role": "user", "content": blocks})
    return wire


class AnthropicAdapter(ProviderAdapter):
    """The ``claude`` build backend. Prompt caching is already on (Part E2); the
    breakpoint logic is shared with the old ``_loop`` via ``_mark_message_breakpoints``
    so it sits at the same ANCHOR/STABLE boundary rather than mid-history."""

    name = "anthropic"
    # Claude is fully multimodal — supports_images means CAN, not SHOULD.
    # Routing that wants to skip vision for cost reasons consults
    # image_cost_tier; silently hiding the capability would strand image
    # inputs on this backend (Part P).
    supports_images = True
    # Cost policy, separate from capability: providers whose per-image token
    # cost is high relative to text can be tagged so routing (Part P) can
    # prefer a cheaper backend when one is available. "standard" is the
    # default; "premium" marks this adapter.
    image_cost_tier = "premium"

    def __init__(self, *, api_key: str, model: str, client: Any = None) -> None:
        super().__init__(api_key=api_key, model=model)
        self.cfg["client"] = client

    def send(self, **kw: Any) -> TurnResult:
        import anthropic

        from app.build.build_agent import (
            _cached_system,
            _cached_tools_for,
            _mark_message_breakpoints,
            _stream_turn,
        )

        client = self.cfg.get("client")
        if client is None:
            client = anthropic.Anthropic(api_key=self.cfg["api_key"])
        model = self.cfg["model"]

        system_blocks = _cached_system(kw["system"])
        tool_blocks = _cached_tools_for(kw["tools"])
        wire = _anthropic_wire_messages(kw["messages"])
        _mark_message_breakpoints(wire, kw["anchor_end"], kw["will_trim"])

        response = _stream_turn(client, model, system_blocks, wire, kw.get("log_fn"), tool_blocks)
        stats, in_tokens, out_tokens = _anthropic_cache_ctx(response.usage)

        text = ""
        tool_calls: list[dict[str, Any]] = []
        for block in response.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                tool_calls.append({"id": block.id, "name": block.name, "arguments": block.input or {}})

        # NOTE: no on_text re-emit here — _stream_turn already flushes the feed
        # at sentence/newline boundaries; emitting the accumulated text again
        # would log every sentence twice.

        stop = ("tool_use" if tool_calls
                else ("max_tokens" if response.stop_reason == "max_tokens" else "end_turn"))
        return TurnResult(
            text=text,
            tool_calls=tool_calls,
            stop_reason=stop,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            cache_stats=stats,
            model=model,
        )


# ── Ollama wire format ────────────────────────────────────────────────────────


def _ollama_wire_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert the normalised list to Ollama /api/chat messages."""
    wire: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role == "user":
            wire.append({"role": "user", "content": m.get("content", "")})
        elif role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": m.get("content", "")}
            calls = m.get("tool_calls") or []
            if calls:
                entry["tool_calls"] = [
                    {"function": {"name": c["name"], "arguments": json.dumps(c.get("arguments") or {}, default=str)}}
                    for c in calls
                ]
            wire.append(entry)
        elif role == "tool_turn":
            for tr in m.get("tool_results", []):
                wire.append({"role": "tool", "tool_name": tr["name"], "content": tr["content"]})
            if m.get("text"):
                wire.append({"role": "user", "content": m["text"]})
    return wire


class OllamaAdapter(ProviderAdapter):
    """The local / Ollama-Cloud ``Qwen3`` backend (Part E2 note: local KV reuse
    depends on the byte-stable ANCHOR/STABLE prefix — E1 — rather than a flag).
    Cache accounting uses ``prompt_eval_count`` as the uncached input."""

    name = "ollama"
    supports_cache = False

    def __init__(self, *, base_url: str, model: str, timeout: int = 300) -> None:
        super().__init__(base_url=base_url, model=model, timeout=timeout)
        # Smaller local windows keep less history than cloud backends.
        self.history_pairs = 20

    def trim_result_limit(self, tools: list[dict[str, Any]] | None = None) -> int:
        # Ollama/cloud models get the large-context cap; small local daemons the
        # small one (mirrors the old loop's cloud-vs-local split).
        from app.ai.ollama_http import using_cloud

        return _TOOL_RESULT_LIMIT_LARGE_CTX if using_cloud() else _TOOL_RESULT_LIMIT_SMALL_CTX

    def send(self, **kw: Any) -> TurnResult:
        from app.ai.ollama_http import ollama_headers, ollama_options, open_chat_stream
        from app.build.build_agent import _tools_to_ollama_format

        base_url = self.cfg["base_url"]
        url = f"{base_url.rstrip('/')}/api/chat"
        headers = ollama_headers()
        options = ollama_options(num_ctx=8192, num_predict=4096)
        tools = _tools_to_ollama_format(self.normalize_tools(kw["tools"]))
        payload = {
            "model": self.cfg["model"],
            # The system prompt is byte-identical for the whole phase, so it
            # opens the wire and stays cacheable in Ollama's local KV (Part M).
            "messages": [{"role": "system", "content": kw["system"]}, *_ollama_wire_messages(kw["messages"])],
            "tools": tools,
            "stream": True,
            "options": options,
        }
        log_fn = kw.get("log_fn")
        content_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        prompt_eval = 0
        eval_count = 0

        try:
            # Shared retrying opener — one backoff policy for both loop paths.
            with open_chat_stream(
                url, payload=payload, headers=headers,
                timeout=self.cfg["timeout"], model=self.cfg["model"], log_fn=log_fn,
            ) as resp:
                stream_buf = ""
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    chunk = json.loads(raw_line)
                    msg = chunk.get("message", {})
                    if token := msg.get("content"):
                        content_parts.append(token)
                        stream_buf += token
                        if kw.get("on_text") and (
                            "\n" in stream_buf or stream_buf.endswith((".", "!", "?", "…")) or len(stream_buf) >= 120
                        ):
                            kw["on_text"](stream_buf.strip())
                            stream_buf = ""
                    if chunk_calls := msg.get("tool_calls"):
                        tool_calls.extend(chunk_calls)
                    if chunk.get("done"):
                        prompt_eval = chunk.get("prompt_eval_count", 0) or 0
                        eval_count = chunk.get("eval_count", 0) or 0
        except RuntimeError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Ollama build agent request failed: {exc}") from exc

        content_text = "".join(content_parts)
        if content_text and kw.get("on_text"):
            kw["on_text"](content_text.strip())

        normalised_calls = [
            {"id": f"o{i}", "name": fc.get("function", {}).get("name", ""), "arguments": _safe_args(fc.get("function", {}).get("arguments"))}
            for i, fc in enumerate(tool_calls)
        ]
        stats = CacheStats(cached_tokens=0, uncached_tokens=prompt_eval,
                           raw={"prompt_eval_count": prompt_eval})
        return TurnResult(
            text=content_text,
            tool_calls=normalised_calls,
            stop_reason="tool_use" if normalised_calls else "end_turn",
            input_tokens=prompt_eval,
            output_tokens=eval_count,
            cache_stats=stats,
            model=self.cfg["model"],
        )


class OllamaOpenRouterFallback(ProviderAdapter):
    """Qwen3 composite backend: Ollama (local daemon or cloud) first, OpenRouter
    as the automatic fallback.

    The first time Ollama fails on a turn — after its own retry budget is
    exhausted — the run switches to the OpenRouter CODE chain and sticks there
    for the rest of the phase rather than flip-flopping between backends. Work
    already done (files written, tool results in context) carries over because
    the shared loop owns the conversation; only the wire format changes.
    """

    name = "ollama+openrouter"

    def __init__(self, primary: ProviderAdapter, fallback: ProviderAdapter) -> None:
        self.primary = primary
        self.fallback = fallback
        self._active: ProviderAdapter | None = None  # None = still trying primary
        # Loop-level knobs come from the primary; both are chat-completions-ish
        # backends with comparable windows.
        self.history_pairs = primary.history_pairs
        self.cfg: dict[str, Any] = {}

    @property
    def supports_images(self) -> bool:
        active = self._active or self.primary
        return active.supports_images

    def trim_result_limit(self, tools: list[dict[str, Any]] | None = None) -> int:
        return (self._active or self.primary).trim_result_limit(tools)

    def close(self) -> None:
        for adapter in (self.primary, self.fallback):
            close = getattr(adapter, "close", None)
            if callable(close):
                close()

    def send(self, **kw: Any) -> TurnResult:
        if self._active is not None:
            return self._active.send(**kw)
        try:
            return self.primary.send(**kw)
        except Exception as exc:  # noqa: BLE001 — anything from the endpoint counts
            log_fn = kw.get("log_fn")
            logger.warning(
                "qwen: %s unavailable (%s) — falling back to %s for the rest of this run",
                self.primary.name, exc, self.fallback.name,
            )
            if log_fn:
                # Provider-agnostic for users; names stay in the operator log.
                log_fn("warning", "The AI service is switching to a backup automatically…")
            self._active = self.fallback
            return self.fallback.send(**kw)


def unified_loop_enabled() -> bool:
    """Part J flag: route the per-provider loops through the shared loop."""
    from app.config import get_settings

    return bool(get_settings().UNIFIED_AGENT_LOOP)


# ── The shared loop (Part J) ──────────────────────────────────────────────────


def run_agent_loop(
    adapter: ProviderAdapter,
    *,
    system: str,
    stable: str,
    workspace: Path,
    log_fn: Callable[[str, str], None] | None = None,
    cancel_fn: Callable[[], bool] | None = None,
    max_iterations: int = _MAX_ITERATIONS,
    tools: list[dict[str, Any]] | None = None,
    written: list[str] | None = None,
    cache_trace: list[dict[str, Any]] | None = None,
) -> tuple[str, int]:
    """One agent loop, shared by every provider. Returns (summary, total_tokens).

    Owns — exactly once, for all providers:
      * the ANCHOR/STABLE/VOLATILE split: ``messages[:_ANCHOR_MSGS]`` holds the
        byte-identical STABLE seed and is never trimmed; only the VOLATILE
        rolling window is (Part E1);
      * every guard — bounded DONE/no-write pushbacks, the exploration breaker,
        repeat-read suppression, tool-result truncation (Part J);
      * cache instrumentation, writing the normalised ``{cached, uncached}``
        split per iteration (Part E3);
      * phase teardown (``adapter.close()``) so no provider cache handle
        outlives the phase (Part E2 — Gemini).

    The adapter supplies only wire format and its cache split.
    """
    from app.build.build_logger import tool_message

    anchor_end = _ANCHOR_MSGS
    pairs = adapter.history_pairs
    result_limit = adapter.trim_result_limit(tools)

    # STABLE seed — the immutable head. Everything beyond is VOLATILE.
    messages: list[dict[str, Any]] = [{"role": "user", "content": stable}]
    total_tokens = 0
    write_calls = 0
    nudges = 0  # bounded pushbacks when the agent talks instead of writing
    # Read-only calls already answered since the last workspace mutation.
    seen_reads: set[str] = set()
    explore_streak = 0  # consecutive turns of tool use with no write_file
    last_text = ""
    trimming_started = False

    def _trimmed() -> list[dict[str, Any]]:
        """Immutable STABLE head + a rolling window that never orphans a result.

        Only VOLATILE may be trimmed (Part E1): the head (ANCHOR+STABLE) is
        returned untouched. The window starts on an assistant turn so a tool
        turn never loses the tool_use it answers.
        """
        rest = messages[anchor_end:][-(pairs * 2):]
        start = 0
        while start < len(rest) and rest[start].get("role") not in ("assistant", "user"):
            start += 1
        return messages[:anchor_end] + rest[start:]


    try:
        for iteration in range(max_iterations):
            if cancel_fn and cancel_fn():
                if log_fn:
                    log_fn("warning", "Agent stopped by user.")
                return "Stopped by user.", total_tokens
            warn_at = max(1, max_iterations - 10)
            if iteration == warn_at and log_fn:
                log_fn("warning", f"Build is complex ({iteration} steps so far) — finishing up…")

            will_trim = len(messages) > anchor_end + pairs * 2
            sent = _trimmed()
            dropped = len(messages) - len(sent)
            if dropped and not trimming_started:
                trimming_started = True
                logger.info("provider=%s first trim at iteration %d (%d messages dropped)",
                            adapter.name, iteration, dropped)

            def _on_text(t: str) -> None:
                if log_fn and t and t.strip():
                    log_fn("text", t.strip())

            def _on_thinking(t: str) -> None:
                if log_fn and t and t.strip():
                    log_fn("thinking", t.strip())

            result = adapter.send(
                system=system, stable=stable, messages=sent, tools=tools,
                anchor_end=anchor_end, will_trim=will_trim, log_fn=log_fn,
                on_text=_on_text, on_thinking=_on_thinking,
            )

            total_tokens += result.total_tokens
            cs = result.cache_stats
            logger.info(
                "provider=%s iter=%d trimmed=%d cached=%d uncached=%d model=%s",
                adapter.name, iteration, dropped, cs.cached_tokens, cs.uncached_tokens, result.model,
            )
            if cache_trace is not None:
                cache_trace.append({
                    "provider": adapter.name,
                    "iteration": iteration,
                    "trimmed": dropped,
                    "messages_sent": len(sent),
                    "input_tokens": result.input_tokens,
                    "cached_tokens": cs.cached_tokens,
                    "uncached_tokens": cs.uncached_tokens,
                    "model": result.model,
                })

            messages.append({"role": "assistant", "content": result.text, "tool_calls": result.tool_calls})
            if result.text:
                last_text = result.text

            if result.stop_reason == "max_tokens" and log_fn:
                log_fn("warning", "Response hit the output limit — the last file may be incomplete.")

            if not result.tool_calls:
                if result.stop_reason in (None, "end_turn") or result.text:
                    done = "DONE" in result.text.upper()
                    if done:
                        if write_calls == 0 and nudges < _MAX_NUDGES:
                            nudges += 1
                            if log_fn:
                                log_fn("info", "Agent said DONE without writing files — asking it to implement…")
                            messages.append({"role": "user", "content": (
                                "You said you were done but you haven't called write_file yet. "
                                "Please implement the changes now using write_file."
                            )})
                            continue
                        return result.text or last_text, total_tokens
                    if write_calls == 0 and nudges < _MAX_NUDGES:
                        nudges += 1
                        if log_fn:
                            log_fn("info", "No files written yet — asking agent to write the code…")
                        messages.append({"role": "user", "content": (
                            "You haven't written any files yet. "
                            "Use the write_file tool to implement the changes now."
                        )})
                        continue
                    return last_text or "Done.", total_tokens
                # A no-tool turn that didn't resolve is treated as end of work.
                continue

            # ── Process tool calls → one normalised tool turn. ──────────────
            results: list[dict[str, Any]] = []
            wrote_this_turn = False
            for call in result.tool_calls:
                tool_name = call.get("name", "")
                tool_input = call.get("arguments") or {}
                if tool_name in _WRITE_TOOLS:
                    write_calls += 1
                    wrote_this_turn = True
                    if log_fn:
                        log_fn("file_written", tool_input.get("path", ""))
                    if written is not None and (p := tool_input.get("path")):
                        written.append(str(p))

                call_key = f"{tool_name}:{json.dumps(tool_input, sort_keys=True, default=str)}"
                is_repeat = tool_name in _READ_ONLY_TOOLS and call_key in seen_reads
                if log_fn:
                    label = tool_message(tool_name, tool_input)
                    log_fn("tool", f"{label} — already read, skipping" if is_repeat else label)

                if is_repeat:
                    r = (
                        f"[You already ran {tool_name} with these exact arguments and "
                        "the workspace has not changed since. The result is the same. "
                        "Stop inspecting and make the change now.]"
                    )
                else:
                    r = execute_tool(tool_name, tool_input, workspace, log_fn)
                    if tool_name in _READ_ONLY_TOOLS:
                        seen_reads.add(call_key)
                    else:
                        # The workspace changed, so earlier reads may be stale.
                        seen_reads.clear()
                r = _truncate_tool_result(r, result_limit)
                results.append(adapter.format_tool_result(call.get("id", ""), tool_name, r))

            combined: dict[str, Any] = {"role": "tool_turn", "tool_results": [], "text": ""}
            for fmt in results:
                combined["tool_results"].extend(fmt.get("tool_results", []) or [])
            if not combined["tool_results"]:
                combined["tool_results"] = [
                    {"call_id": c.get("id", ""), "name": c.get("name", ""), "content": "(no result)"}
                    for c in result.tool_calls
                ]
            messages.append(combined)



            # Bounded exploration breaker — folded into the tool turn so Anthropic
            # never sees two adjacent user messages.
            explore_streak = 0 if wrote_this_turn else explore_streak + 1
            if explore_streak >= _MAX_EXPLORE_STREAK:
                if nudges < _MAX_NUDGES:
                    nudges += 1
                    explore_streak = 0
                    if log_fn:
                        log_fn("info", "Agent is still exploring — asking it to start writing…")
                    nudge = (
                        f"You have used {_MAX_EXPLORE_STREAK} turns inspecting the project "
                        "without writing anything. You have enough context. Implement the "
                        "change now with write_file, then reply DONE."
                    )
                    messages[-1]["text"] = (messages[-1].get("text", "") + "\n" + nudge).strip()
                else:
                    if log_fn:
                        log_fn("warning", "Agent kept exploring without making changes — stopping.")
                    return (
                        last_text or "Agent stopped: explored the project without making changes.",
                        total_tokens,
                    )
    finally:
        close = getattr(adapter, "close", None)
        if callable(close):
            close()

    if log_fn:
        log_fn("warning", "Agent reached iteration limit.")
    return "Agent reached iteration limit.", total_tokens

