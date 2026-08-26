"""Tests for the unified provider-adapter loop (Part J / Part E).

Run:
    venv/Scripts/python -m pytest tests/build/test_provider_loop.py --confcutdir=tests/build -v

Covers the parts that are testable offline:
  * all guards live in the shared loop (`run_agent_loop`) behind a scripted
    adapter — bounded pushbacks, the exploration breaker, repeat-read
    suppression, and tool-result truncation;
  * Part E1 — the STABLE head is byte-identical across iterations 1..N and
    trimming only touches VOLATILE;
  * Part J contract — every adapter normalises `cache_stats` to
    `{cached_tokens, uncached_tokens}` and declares its image capability;
  * Part E2 wiring — `_gemini_loop` routes through the shared loop when the flag
    is on.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.build import build_agent
from app.build import provider_loop as pl
from app.build.provider_loop import (
    CacheStats,
    OpenAIAdapter,
    OpenRouterAdapter,
    AnthropicAdapter,
    OllamaAdapter,
    TurnResult,
    _anthropic_cache_ctx,
    _gemini_cache_ctx,
    _openai_cache_ctx,
    run_agent_loop,
)


class _FakeAdapter(pl.ProviderAdapter):
    """A scripted adapter that records every request and never touches the wire."""

    def __init__(self, turns: list[TurnResult], history_pairs: int = 2) -> None:
        self.turns = turns
        self.history_pairs = history_pairs
        self.calls: list[list[dict[str, Any]]] = []  # messages sent each iteration
        self.systems: list[str] = []
        self.stables: list[str] = []

    def send(self, **kw: Any) -> TurnResult:
        self.calls.append(list(kw["messages"]))
        self.systems.append(kw["system"])
        self.stables.append(kw["stable"])
        idx = min(len(self.calls) - 1, len(self.turns) - 1)
        return self.turns[idx]


def _tool(name: str, *, id: str = "t1", **args: Any) -> dict[str, Any]:
    return {"id": id, "name": name, "arguments": args}


def _end(text: str = "DONE", stop: str = "end_turn") -> TurnResult:
    return TurnResult(text=text, stop_reason=stop, input_tokens=5, output_tokens=2,
                      cache_stats=CacheStats(cached_tokens=0, uncached_tokens=5))


def _tools(*call_dicts: dict[str, Any], text: str = "") -> TurnResult:
    return TurnResult(text=text, tool_calls=list(call_dicts), stop_reason="tool_use",
                      input_tokens=5, output_tokens=2,
                      cache_stats=CacheStats(cached_tokens=0, uncached_tokens=5))

# ── guards live in the shared loop ────────────────────────────────────────────


class TestNudgeBounding:
    def test_done_without_write_is_bounded(self, tmp_path):
        fake = _FakeAdapter([_end("DONE: nothing written")])
        summary, _ = run_agent_loop(
            fake, system="SYS", stable="seed", workspace=tmp_path, max_iterations=50,
        )
        assert summary == "DONE: nothing written"
        # initial turn + one pushback per nudge (bounded), then it gives up.
        assert len(fake.calls) == pl._MAX_NUDGES + 1
        assert fake.stables[0] == fake.stables[-1] == "seed"


class TestExplorationBreaker:
    def test_read_only_stall_is_stopped(self, tmp_path):
        turns = [_tools(_tool("read_file", path="a.txt"))]
        fake = _FakeAdapter(turns, history_pairs=2)
        summary, _ = run_agent_loop(
            fake, system="SYS", stable="seed", workspace=tmp_path, max_iterations=200,
        )
        assert "explored" in summary
        # The breaker fires after _MAX_EXPLORE_STREAK read-only turns.
        assert len(fake.calls) <= pl._MAX_EXPLORE_STREAK * (pl._MAX_NUDGES + 2)


class TestRepeatReadSuppression:
    def test_identical_read_is_served_from_a_notice(self, tmp_path):
        (tmp_path / "a.css").write_text("body { color: red; }", encoding="utf-8")
        fake = _FakeAdapter([
            _tools(_tool("read_file", path="a.css")),
            _tools(_tool("read_file", path="a.css")),
            _end("DONE"),
        ])
        run_agent_loop(fake, system="SYS", stable="seed", workspace=tmp_path, max_iterations=10)
        # The second tool turn is served from a notice rather than re-reading.
        tool_turn = fake.calls[2][-1]
        assert tool_turn["role"] == "tool_turn"
        assert any("already ran" in (r.get("content") or "") for r in tool_turn["tool_results"])


# ── Part E1: STABLE head is byte-identical; only VOLATILE is trimmed ─────────


class TestAnchorStability:
    def test_stable_head_is_identical_across_iterations(self, tmp_path):
        (tmp_path / "a.txt").write_text("x", encoding="utf-8")
        turns = [
            _tools(_tool("list_files", path=".")),
            _tools(_tool("list_files", path=".")),
            _tools(_tool("list_files", path=".")),
            _tools(_tool("write_file", path="out.txt", content="hi")),
            _end("DONE"),
        ]
        fake = _FakeAdapter(turns, history_pairs=2)
        run_agent_loop(fake, system="SYS", stable="STABLE_SEED", workspace=tmp_path, max_iterations=30)

        assert len(fake.calls) >= 2
        first_head = fake.calls[0][0]
        for sent in fake.calls[1:]:
            assert sent[0] == first_head, "the STABLE head must be byte-identical iteration 1..N"
            assert sent[0] == {"role": "user", "content": "STABLE_SEED"}


# ── Part J contract: cache_stats is always normalised, never omitted ─────────


class TestCacheStatsNormalisation:
    def test_anthropic(self):
        usage = SimpleNamespace(input_tokens=100, output_tokens=20,
                                cache_creation_input_tokens=50, cache_read_input_tokens=30)
        stats, inp, out = _anthropic_cache_ctx(usage)
        assert stats.as_dict() == {"cached_tokens": 30, "uncached_tokens": 150}
        assert (inp, out) == (180, 20)

    def test_openai(self):
        usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=200,
                                prompt_tokens_details=SimpleNamespace(cached_tokens=800))
        stats, inp, out = _openai_cache_ctx(usage)
        assert stats.as_dict() == {"cached_tokens": 800, "uncached_tokens": 200}
        assert (inp, out) == (1000, 200)

    def test_openai_reports_zeros_when_no_details(self):
        usage = SimpleNamespace(prompt_tokens=700, completion_tokens=50,
                                prompt_tokens_details=None)
        stats, inp, out = _openai_cache_ctx(usage)
        assert stats.as_dict() == {"cached_tokens": 0, "uncached_tokens": 700}

    def test_gemini_uses_cached_content_token_count(self):
        usage = {"promptTokensDetails": [
            {"tokenCount": 700, "isCached": True},
            {"tokenCount": 300, "isCached": False},
        ]}
        stats = _gemini_cache_ctx(usage)
        assert stats.as_dict() == {"cached_tokens": 700, "uncached_tokens": 300}

    def test_gemini_reports_zeros_when_nothing_exposed(self):
        stats = _gemini_cache_ctx({})
        assert stats.as_dict() == {"cached_tokens": 0, "uncached_tokens": 0}


class TestImageCapability:
    def test_native_multimodal_providers_declare_support(self):
        from app.build.provider_loop import GeminiAdapter
        assert GeminiAdapter(api_key="k", model="m").supports_images is True
        assert OpenAIAdapter(api_key="k", model="m").supports_images is True

    def test_anthropic_does_not_declare_image_support(self):
        assert AnthropicAdapter(api_key="k", model="m").supports_images is False

    def test_ollama_does_not_declare_image_support(self):
        assert OllamaAdapter(base_url="http://x", model="m").supports_images is False


# ── wire conversions ──────────────────────────────────────────────────────────


class TestWireConversions:
    def test_openai_wire_folds_tool_turns_into_tool_messages(self):
        messages = [
            {"role": "user", "content": "seed"},
            {"role": "assistant", "content": "think", "tool_calls": [_tool("read_file", id="c1", path="a.txt")]},
            {"role": "tool_turn", "tool_results": [{"call_id": "c1", "name": "read_file", "content": "ok"}], "text": ""},
        ]
        wire = pl._openai_wire_messages("SYS", messages)
        assert wire[0] == {"role": "system", "content": "SYS"}
        assert wire[2]["tool_calls"][0]["function"]["name"] == "read_file"
        assert wire[3] == {"role": "tool", "tool_call_id": "c1", "content": "ok"}

    def test_anthropic_wire_pairs_tool_results_with_tool_use(self):
        messages = [
            {"role": "user", "content": "seed"},
            {"role": "assistant", "content": "think", "tool_calls": [_tool("read_file", id="c1", path="a.txt")]},
            {"role": "tool_turn", "tool_results": [{"call_id": "c1", "name": "read_file", "content": "ok"}], "text": "act now"},
        ]
        wire = pl._anthropic_wire_messages(messages)
        assert wire[1]["content"][1]["type"] == "tool_use"
        assert wire[2]["content"][0]["type"] == "tool_result"
        assert wire[2]["content"][-1]["type"] == "text"


# ── Part E3: shared loop instruments the cache split ──────────────────────────


class TestInstrumentation:
    def test_cache_trace_records_every_iteration(self, tmp_path):
        fake = _FakeAdapter([_end("DONE")])
        trace: list[dict[str, Any]] = []
        run_agent_loop(fake, system="SYS", stable="seed", workspace=tmp_path,
                       max_iterations=5, cache_trace=trace)
        assert trace, "cache_trace must be populated"
        for entry in trace:
            assert "cached_tokens" in entry and "uncached_tokens" in entry


# ── Part J flag wiring: _gemini_loop routes through the shared loop ──────────


class TestFlagWiring:
    def test_gemini_loop_delegates_when_flag_on(self, tmp_path, monkeypatch):
        called: dict[str, Any] = {}

        def fake_unified(*args: Any, system, stable, workspace, **kw):
            called["system"] = system
            called["stable"] = stable
            called["workspace"] = workspace
            return "unified summary", 0

        # _gemini_loop imports these from provider_loop at call time.
        monkeypatch.setattr(pl, "unified_loop_enabled", lambda: True)
        monkeypatch.setattr(pl, "run_agent_loop", fake_unified)

        summary, _ = build_agent._gemini_loop("key", "model", "SYS", tmp_path, "seed")
        assert summary == "unified summary"
        assert called["system"] == "SYS"
        assert called["stable"] == "seed"


# ── shared Ollama transport retries rate limits ───────────────────────────────


class _FakeResp:
    def __init__(self, status: int, headers: dict | None = None) -> None:
        self.status_code = status
        self.reason = "Too Many Requests" if status == 429 else ""
        # Case-insensitive like requests' header dict.
        self._headers = {k.lower(): v for k, v in (headers or {}).items()}
        self.headers = self._headers

    def get(self, key: str, default=None):
        return self._headers.get(key.lower(), default)

    def raise_for_status(self) -> None:
        assert self.status_code < 400, self.status_code

    def close(self) -> None:
        pass


class TestOllamaChatRetry:
    def _run(self, monkeypatch, statuses, *, headers=None):
        """Drive open_chat_stream against a scripted status sequence."""
        import time as _time

        from app.ai.ollama_http import open_chat_stream

        seq = list(statuses)
        seen: list[int] = []
        sleeps: list[float] = []

        def fake_post(url, **kw):
            status = seq[min(len(seen), len(seq) - 1)]
            seen.append(status)
            return _FakeResp(status, headers)

        monkeypatch.setattr("requests.post", fake_post)
        monkeypatch.setattr(_time, "sleep", lambda s: sleeps.append(s))
        resp = open_chat_stream(
            "http://x/api/chat",
            payload={}, headers={}, timeout=30, model="m", log_fn=None,
        )
        return resp, seen, sleeps

    def test_429_is_retried_then_succeeds(self, monkeypatch):
        resp, seen, _ = self._run(monkeypatch, [429, 200])
        assert resp.status_code == 200
        assert seen == [429, 200]

    def test_backoff_doubles_between_attempts(self, monkeypatch):
        _, _, sleeps = self._run(monkeypatch, [429, 429, 200])
        assert sleeps[1] > sleeps[0] > 0

    def test_retry_after_header_wins_over_the_guess(self, monkeypatch):
        _, _, sleeps = self._run(monkeypatch, [429, 200], headers={"Retry-After": "12"})
        assert sleeps and sleeps[0] >= 12.0

    def test_gives_up_after_max_retries(self, monkeypatch):
        with pytest.raises(RuntimeError, match="attempts"):
            self._run(monkeypatch, [429])

    def test_missing_model_fails_fast_without_retrying(self, monkeypatch):
        # Pin the local-daemon hint so the assertion doesn't depend on whether
        # the ambient .env carries an OLLAMA_API_KEY.
        import app.ai.ollama_http as oh

        monkeypatch.setattr(oh, "using_cloud", lambda *a, **k: False)
        with pytest.raises(RuntimeError, match="ollama pull"):
            self._run(monkeypatch, [404, 200])

# ── malformed tool arguments must not poison history (OpenRouter/Nvidia 400) ──


class TestToolArgsSanitisation:
    def test_valid_json_round_trips(self):
        assert build_agent._safe_args('{"path": "src/a.ts"}') == {"path": "src/a.ts"}
        assert build_agent._valid_args_json({"path": "a"}) == '{"path": "a"}'

    def test_lone_backslash_is_repaired_not_dropped(self):
        # A Windows path or regex the model emitted without escaping.
        raw = '{"path": "C:\\Users\\app", "pattern": "\\d+"}'
        args = build_agent._safe_args(raw)
        assert args["path"].startswith("C:")
        assert build_agent._valid_args_json(raw).startswith("{")

    def test_unrepairable_garbage_becomes_empty_object(self):
        assert build_agent._safe_args("not json at all") == {}
        assert build_agent._valid_args_json("not json at all") == "{}"

    def test_echoed_history_never_contains_invalid_json(self):
        """The exact production failure: raw echo → provider 400 at messages[25]."""
        import json as _json

        raw = '{"path": "lib\\features", "old_string": "a\\b"}'
        echoed = _json.loads(build_agent._valid_args_json(raw))  # must not raise
        assert isinstance(echoed, dict)



# ── Qwen3 automatic failover: Ollama → OpenRouter ─────────────────────────────


class _RaisingAdapter(pl.ProviderAdapter):
    name = "failing"

    def send(self, **kw):  # noqa: ANN003
        raise RuntimeError("429 Too Many Requests for https://ollama.com/api/chat")


class _CountingAdapter(pl.ProviderAdapter):
    name = "counting"

    def __init__(self) -> None:
        self.calls = 0

    def send(self, **kw):  # noqa: ANN003
        self.calls += 1
        return TurnResult(text="DONE", stop_reason="end_turn",
                          input_tokens=1, output_tokens=1,
                          cache_stats=CacheStats())


class TestOllamaOpenRouterFallback:
    def test_falls_back_when_primary_fails(self, tmp_path):
        fallback = _CountingAdapter()
        composite = pl.OllamaOpenRouterFallback(_RaisingAdapter(), fallback)
        summary, _ = run_agent_loop(
            composite, system="SYS", stable="seed", workspace=tmp_path, max_iterations=5,
        )
        assert summary == "DONE"
        assert fallback.calls >= 1  # every turn after the switch goes to OpenRouter

    def test_switch_is_sticky(self, tmp_path):
        primary = _CountingAdapter()  # would succeed — proves stickiness instead
        fallback = _CountingAdapter()

        class _FailOnce(pl.ProviderAdapter):
            name = "fail-once"

            def __init__(self) -> None:
                self.n = 0
                self.history_pairs = 20

            def send(self, **kw):  # noqa: ANN003
                self.n += 1
                if self.n == 1:
                    raise RuntimeError("boom")
                return TurnResult(text="", stop_reason="tool_use", tool_calls=[],
                                  input_tokens=1, cache_stats=CacheStats())

        flaky = _FailOnce()
        composite = pl.OllamaOpenRouterFallback(flaky, fallback)
        run_agent_loop(composite, system="SYS", stable="seed", workspace=tmp_path, max_iterations=5)
        assert flaky.n == 1, "primary must not be retried once we have switched"
        assert fallback.calls >= 2

    def test_primary_is_used_while_healthy(self, tmp_path):
        primary = _CountingAdapter()
        fallback = _CountingAdapter()
        composite = pl.OllamaOpenRouterFallback(primary, fallback)
        run_agent_loop(composite, system="SYS", stable="seed", workspace=tmp_path, max_iterations=5)
        assert primary.calls >= 1 and fallback.calls == 0


class TestLegacyOllamaLoopFallback:
    def test_ollama_failure_switches_to_openrouter(self, tmp_path, monkeypatch):
        from app.build import build_agent as ba

        monkeypatch.setattr("app.ai.qwen.fallback_to_openrouter_enabled", lambda: True)
        monkeypatch.setattr(ba, "_ollama_agent_turns",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("429")))
        seen: dict[str, Any] = {}

        monkeypatch.setattr(
            ba, "_openrouter_loop",
            lambda system, workspace, msg, *a, **k: (seen.update(system=system), ("or summary", 0))[1],
        )

        summary, _ = ba._ollama_loop(
            base_url="http://x", model="m", system="SYS", workspace=tmp_path,
            initial_user_msg="seed",
        )
        assert summary == "or summary"
        assert seen["system"] == "SYS"

    def test_failure_is_fatal_when_fallback_disabled(self, tmp_path, monkeypatch):
        from app.build import build_agent as ba

        monkeypatch.setattr("app.ai.qwen.fallback_to_openrouter_enabled", lambda: False)
        monkeypatch.setattr(ba, "_ollama_agent_turns",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("429")))

        with pytest.raises(RuntimeError, match="429"):
            ba._ollama_loop(
                base_url="http://x", model="m", system="SYS", workspace=tmp_path,
                initial_user_msg="seed",
            )

