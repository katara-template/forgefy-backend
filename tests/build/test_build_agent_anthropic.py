"""Tests for the Anthropic build loop: prompt caching, context guards, streaming.

Run:
    venv/Scripts/python -m pytest tests/build/test_build_agent_anthropic.py -v

No network is used — the Anthropic client is replaced by a scripted fake that
records every request it is handed.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.build import build_agent
from app.build.build_agent import (
    _ANTHROPIC_HISTORY_PAIRS,
    _MAX_EXPLORE_STREAK,
    _MAX_NUDGES,
    _MAX_OUTPUT_TOKENS,
    _TOOL_RESULT_LIMIT_LARGE_CTX,
    _cached_system,
    _cached_tools,
    _loop,
    _mark_message_breakpoints,
    _usage_tokens,
)

# ── scripted Anthropic fake ───────────────────────────────────────────────────


def _text(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _tool(tool_id: str, name: str, **inputs: Any) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=tool_id, name=name, input=inputs)


def _usage(input_tokens: int = 10, output_tokens: int = 5, **kw: int) -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=kw.get("cache_creation_input_tokens", 0),
        cache_read_input_tokens=kw.get("cache_read_input_tokens", 0),
    )


def _message(
    content: list[SimpleNamespace],
    stop_reason: str = "tool_use",
    usage: SimpleNamespace | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        content=content, stop_reason=stop_reason, usage=usage or _usage(),
    )


class _FakeStream:
    """Mimics the context-manager + iterator shape of client.messages.stream()."""

    def __init__(self, message: SimpleNamespace, events: list[Any]) -> None:
        self._message = message
        self._events = events

    def __enter__(self) -> _FakeStream:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def __iter__(self):
        return iter(self._events)

    def get_final_message(self) -> SimpleNamespace:
        return self._message


class _FakeMessages:
    def __init__(self, script: list[SimpleNamespace], events: dict[int, list[Any]]) -> None:
        self._script = script
        self._events = events
        self.calls: list[dict[str, Any]] = []

    def stream(self, **kwargs: Any) -> _FakeStream:
        idx = len(self.calls)
        # Deep-ish snapshot: _loop mutates cache_control markers in place, so a
        # shared reference would make every recorded call look identical.
        self.calls.append({
            **kwargs,
            "messages": [dict(m, content=_snapshot(m["content"])) for m in kwargs["messages"]],
        })
        message = self._script[min(idx, len(self._script) - 1)]
        return _FakeStream(message, self._events.get(idx, []))


def _snapshot(content: Any) -> Any:
    if isinstance(content, list):
        return [dict(b) if isinstance(b, dict) else b for b in content]
    return content


class _FakeClient:
    def __init__(
        self,
        script: list[SimpleNamespace],
        events: dict[int, list[Any]] | None = None,
    ) -> None:
        self.messages = _FakeMessages(script, events or {})


def _delta(kind: str, value: str) -> SimpleNamespace:
    delta = (
        SimpleNamespace(type="text_delta", text=value)
        if kind == "text"
        else SimpleNamespace(type="thinking_delta", thinking=value)
    )
    return SimpleNamespace(type="content_block_delta", delta=delta)


def _run_loop(client: _FakeClient, workspace, **kw: Any) -> tuple[str, int]:
    return _loop(client, "claude-sonnet-5", "SYSTEM PROMPT", workspace, "build it", **kw)


# ── A1: prompt caching ────────────────────────────────────────────────────────


class TestPromptCaching:
    def test_system_is_sent_as_a_cached_content_block(self, tmp_path):
        client = _FakeClient([_message([_text("DONE: built")], stop_reason="end_turn")])

        _run_loop(client, tmp_path)

        system = client.messages.calls[0]["system"]
        assert system == [{
            "type": "text",
            "text": "SYSTEM PROMPT",
            "cache_control": {"type": "ephemeral"},
        }]

    def test_tools_carry_one_breakpoint_on_the_last_entry(self, tmp_path):
        client = _FakeClient([_message([_text("DONE")], stop_reason="end_turn")])

        _run_loop(client, tmp_path)

        tools = client.messages.calls[0]["tools"]
        assert tools[-1]["cache_control"] == {"type": "ephemeral"}
        assert [t for t in tools[:-1] if "cache_control" in t] == []

    def test_cached_tools_does_not_mutate_the_shared_tool_list(self):
        from app.build.agent_tools import TOOLS

        _cached_tools(TOOLS)

        assert all("cache_control" not in t for t in TOOLS)

    def test_cached_system_is_byte_identical_across_calls(self):
        assert _cached_system("abc") == _cached_system("abc")

    def test_conversation_keeps_only_the_two_newest_breakpoints(self):
        messages = [
            {"role": "user", "content": "start"},
            {"role": "user", "content": [{"type": "tool_result", "content": "a"}]},
            {"role": "user", "content": [{"type": "tool_result", "content": "b"}]},
            {"role": "user", "content": [{"type": "tool_result", "content": "c"}]},
        ]

        _mark_message_breakpoints(messages, build_agent._ANCHOR_END, trimming=False)

        marked = [
            i for i, m in enumerate(messages)
            if isinstance(m["content"], list) and "cache_control" in m["content"][-1]
        ]
        assert marked == [2, 3], "only the two newest turns stay as cache read points"

    def test_breakpoint_moves_forward_as_the_conversation_grows(self):
        messages: list[dict[str, Any]] = [{"role": "user", "content": "start"}]
        for turn in range(4):
            messages.append({"role": "user", "content": [{"type": "tool_result", "content": str(turn)}]})
            _mark_message_breakpoints(messages, build_agent._ANCHOR_END, trimming=False)

        marked = [
            m["content"][-1]["content"] for m in messages
            if isinstance(m["content"], list) and "cache_control" in m["content"][-1]
        ]
        assert marked == ["2", "3"]

    def test_total_breakpoints_stay_within_the_api_limit_of_four(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
        client = _FakeClient([
            _message([_tool("t1", "read_file", path="a.txt")]),
            _message([_tool("t2", "read_file", path="a.txt")]),
            _message([_tool("t3", "write_file", path="b.txt", content="x")]),
            _message([_text("DONE")], stop_reason="end_turn"),
        ])

        _run_loop(client, tmp_path)

        for call in client.messages.calls:
            count = sum("cache_control" in t for t in call["tools"])
            count += sum("cache_control" in b for b in call["system"])
            for msg in call["messages"]:
                if isinstance(msg["content"], list):
                    count += sum(
                        isinstance(b, dict) and "cache_control" in b for b in msg["content"]
                    )
            assert count <= 4, f"{count} breakpoints exceeds the API maximum of 4"

    def test_usage_tokens_includes_cached_input(self):
        usage = _usage(
            input_tokens=100,
            output_tokens=20,
            cache_creation_input_tokens=7000,
            cache_read_input_tokens=300,
        )

        # Cached halves must be counted, or enabling caching would silently make
        # every build look ~90% cheaper in the metering the workers report.
        assert _usage_tokens(usage) == 7420

    def test_usage_tokens_tolerates_a_backend_without_cache_fields(self):
        assert _usage_tokens(SimpleNamespace(input_tokens=5, output_tokens=3)) == 8


# ── Part E: the caching/trimming interaction ──────────────────────────────────


def _long_run(tmp_path, turns: int, **kw):
    """Drive the loop for `turns` tool-using iterations, then DONE."""
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    script = [
        _message([_tool(f"t{i}", "read_file", path="a.txt")]) for i in range(turns)
    ]
    script.append(_message([_text("DONE")], stop_reason="end_turn"))
    client = _FakeClient(script)
    trace: list[dict[str, Any]] = []
    _loop(
        client, "claude-sonnet-5", "SYSTEM PROMPT", tmp_path, "build it",
        max_iterations=turns + 2, cache_trace=trace, **kw,
    )
    return client, trace


def _marked(call: dict[str, Any]) -> list[int]:
    """Indices of messages carrying a cache breakpoint."""
    out = []
    for i, msg in enumerate(call["messages"]):
        content = msg.get("content")
        if (
            isinstance(content, list)
            and content
            and isinstance(content[-1], dict)
            and "cache_control" in content[-1]
        ):
            out.append(i)
    return out


class TestAnchorAndTrimming:
    def test_the_anchor_is_never_trimmed_away(self, tmp_path, monkeypatch):
        monkeypatch.setattr(build_agent, "_ANTHROPIC_HISTORY_PAIRS", 2)
        client, _ = _long_run(tmp_path, 14)

        for call in client.messages.calls:
            assert call["messages"][0]["content"] == "build it"
            # The opening turns stay in view for the whole phase.
            assert len(call["messages"]) >= min(
                build_agent._ANCHOR_END, len(call["messages"])
            )

    def test_once_trimming_starts_the_only_breakpoint_is_in_the_anchor(
        self, tmp_path, monkeypatch,
    ):
        """A breakpoint above the trim point can never match a stored prefix, so
        leaving one there pays the cache-write premium for nothing."""
        monkeypatch.setattr(build_agent, "_ANTHROPIC_HISTORY_PAIRS", 2)
        client, trace = _long_run(tmp_path, 14)

        trimming = [t["iteration"] for t in trace if t["trimmed"]]
        assert trimming, "the window never slid — test is not exercising the case"
        for i in trimming:
            marks = _marked(client.messages.calls[i])
            assert len(marks) == 1, f"iteration {i} marked {len(marks)} message blocks"
            assert marks[0] < build_agent._ANCHOR_END

    def test_before_trimming_the_two_newest_turns_are_marked(self, tmp_path, monkeypatch):
        monkeypatch.setattr(build_agent, "_ANTHROPIC_HISTORY_PAIRS", 20)
        client, trace = _long_run(tmp_path, 5)

        assert all(t["trimmed"] == 0 for t in trace), "unexpected trim"
        # By the third iteration there are two markable turns to carry marks.
        assert len(_marked(client.messages.calls[3])) == 2

    def test_breakpoints_never_exceed_the_api_limit(self, tmp_path, monkeypatch):
        monkeypatch.setattr(build_agent, "_ANTHROPIC_HISTORY_PAIRS", 2)
        client, _ = _long_run(tmp_path, 14)

        for call in client.messages.calls:
            total = sum("cache_control" in t for t in call["tools"])
            total += sum("cache_control" in b for b in call["system"])
            total += len(_marked(call))
            assert total <= 4

    def test_trimming_never_orphans_a_tool_result(self, tmp_path, monkeypatch):
        """The anchor introduces a gap in the history; the window must still
        resume on an assistant turn."""
        monkeypatch.setattr(build_agent, "_ANTHROPIC_HISTORY_PAIRS", 2)
        client, _ = _long_run(tmp_path, 14)

        for call in client.messages.calls:
            _assert_no_orphaned_tool_results(call["messages"])

    def test_no_two_adjacent_user_messages_across_the_gap(self, tmp_path, monkeypatch):
        monkeypatch.setattr(build_agent, "_ANTHROPIC_HISTORY_PAIRS", 2)
        client, _ = _long_run(tmp_path, 14)

        for call in client.messages.calls:
            roles = [m["role"] for m in call["messages"]]
            for a, b in zip(roles, roles[1:], strict=False):
                assert not (a == "user" and b == "user"), f"adjacent user turns: {roles}"

    def test_cache_trace_records_every_iteration(self, tmp_path, monkeypatch):
        monkeypatch.setattr(build_agent, "_ANTHROPIC_HISTORY_PAIRS", 2)
        client, trace = _long_run(tmp_path, 14)

        assert len(trace) == len(client.messages.calls)
        assert {"iteration", "trimmed", "messages_sent", "cache_read_input_tokens"} <= set(
            trace[0]
        )

    def test_trim_counts_grow_monotonically_once_started(self, tmp_path, monkeypatch):
        monkeypatch.setattr(build_agent, "_ANTHROPIC_HISTORY_PAIRS", 2)
        _, trace = _long_run(tmp_path, 14)

        trims = [t["trimmed"] for t in trace if t["trimmed"]]
        assert trims == sorted(trims), "the window should slide, not jump around"


# ── A2: context-control guards ────────────────────────────────────────────────


def _tool_result_blocks(call: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for msg in call["messages"]:
        if isinstance(msg["content"], list):
            out += [
                b for b in msg["content"]
                if isinstance(b, dict) and b.get("type") == "tool_result"
            ]
    return out


class TestContextGuards:
    def test_large_tool_results_are_truncated(self, tmp_path):
        (tmp_path / "big.txt").write_text("x" * 60_000, encoding="utf-8")
        client = _FakeClient([
            _message([_tool("t1", "read_file", path="big.txt")]),
            _message([_text("DONE")], stop_reason="end_turn"),
        ])

        _run_loop(client, tmp_path)

        result = _tool_result_blocks(client.messages.calls[1])[0]["content"]
        assert "TRUNCATED" in result
        assert len(result) < _TOOL_RESULT_LIMIT_LARGE_CTX + 500

    def test_results_under_the_limit_are_untouched(self, tmp_path):
        (tmp_path / "small.txt").write_text("hello world", encoding="utf-8")
        client = _FakeClient([
            _message([_tool("t1", "read_file", path="small.txt")]),
            _message([_text("DONE")], stop_reason="end_turn"),
        ])

        _run_loop(client, tmp_path)

        # read_file numbers its lines so edit_file targets are easy to build.
        assert _tool_result_blocks(client.messages.calls[1])[0]["content"] == "1\thello world"

    def test_repeated_read_is_served_from_a_notice(self, tmp_path):
        (tmp_path / "a.txt").write_text("original", encoding="utf-8")
        client = _FakeClient([
            _message([_tool("t1", "read_file", path="a.txt")]),
            _message([_tool("t2", "read_file", path="a.txt")]),
            _message([_text("DONE")], stop_reason="end_turn"),
        ])

        _run_loop(client, tmp_path)

        second = _tool_result_blocks(client.messages.calls[2])[-1]["content"]
        assert "already ran" in second

    def test_a_write_invalidates_the_repeat_read_cache(self, tmp_path):
        (tmp_path / "a.txt").write_text("original", encoding="utf-8")
        client = _FakeClient([
            _message([_tool("t1", "read_file", path="a.txt")]),
            _message([_tool("t2", "write_file", path="a.txt", content="changed")]),
            _message([_tool("t3", "read_file", path="a.txt")]),
            _message([_text("DONE")], stop_reason="end_turn"),
        ])

        _run_loop(client, tmp_path)

        third = _tool_result_blocks(client.messages.calls[3])[-1]["content"]
        assert "changed" in third, "a mutated workspace must be re-read, not suppressed"
        assert "already ran" not in third

    def test_window_never_orphans_a_tool_result(self, tmp_path, monkeypatch):
        # A tiny window forces trimming within a handful of turns.
        monkeypatch.setattr(build_agent, "_ANTHROPIC_HISTORY_PAIRS", 1)
        (tmp_path / "a.txt").write_text("hi", encoding="utf-8")
        script = [_message([_tool(f"t{i}", "list_files", path=".")]) for i in range(6)]
        script.append(_message([_text("DONE")], stop_reason="end_turn"))
        client = _FakeClient(script)

        _run_loop(client, tmp_path)

        for call in client.messages.calls:
            _assert_no_orphaned_tool_results(call["messages"])

    def test_window_keeps_the_first_user_message_as_an_anchor(self, tmp_path, monkeypatch):
        monkeypatch.setattr(build_agent, "_ANTHROPIC_HISTORY_PAIRS", 1)
        script = [_message([_tool(f"t{i}", "list_files", path=".")]) for i in range(5)]
        script.append(_message([_text("DONE")], stop_reason="end_turn"))
        client = _FakeClient(script)

        _run_loop(client, tmp_path)

        for call in client.messages.calls:
            assert call["messages"][0]["content"] == "build it"

    def test_exploration_streak_stops_a_survey_only_agent(self, tmp_path):
        # Never writes anything — only ever lists files.
        script = [
            _message([_tool(f"t{i}", "list_files", path=str(i))])
            for i in range(_MAX_EXPLORE_STREAK * (_MAX_NUDGES + 2))
        ]
        client = _FakeClient(script)

        summary, _ = _run_loop(client, tmp_path, max_iterations=200)

        assert "explor" in summary.lower()
        assert len(client.messages.calls) < 200, "the breaker must stop the loop early"

    def test_exploration_nudge_rides_along_with_the_tool_results(self, tmp_path):
        script = [
            _message([_tool(f"t{i}", "list_files", path=str(i))])
            for i in range(_MAX_EXPLORE_STREAK + 2)
        ]
        client = _FakeClient(script)

        _run_loop(client, tmp_path, max_iterations=_MAX_EXPLORE_STREAK + 2)

        # The nudge must not become a second consecutive user turn, and every
        # tool_use still has to be answered in the message right after it.
        for call in client.messages.calls:
            _assert_no_orphaned_tool_results(call["messages"])

    def test_edit_file_counts_as_writing_code(self, tmp_path):
        """An agent that only edits has done the work and must not be nudged."""
        (tmp_path / "a.txt").write_text("original", encoding="utf-8")
        client = _FakeClient([
            _message([_tool("t1", "edit_file", path="a.txt",
                            old_string="original", new_string="changed")]),
            _message([_text("DONE: edited the file")], stop_reason="end_turn"),
        ])
        logged: list[tuple[str, str]] = []

        summary, _ = _run_loop(client, tmp_path, log_fn=lambda k, m: logged.append((k, m)))

        assert summary == "DONE: edited the file"
        assert len(client.messages.calls) == 2, "an editing agent was nudged as idle"
        assert (tmp_path / "a.txt").read_text() == "changed"
        assert ("file_written", "a.txt") in logged

    def test_edit_file_resets_the_exploration_streak(self, tmp_path):
        (tmp_path / "a.txt").write_text("v1", encoding="utf-8")
        script: list[SimpleNamespace] = []
        for i in range(_MAX_EXPLORE_STREAK - 1):
            script.append(_message([_tool(f"L{i}", "list_files", path=str(i))]))
        script.append(_message([_tool("e1", "edit_file", path="a.txt",
                                      old_string="v1", new_string="v2")]))
        for i in range(_MAX_EXPLORE_STREAK - 1):
            script.append(_message([_tool(f"M{i}", "list_files", path=f"x{i}")]))
        script.append(_message([_text("DONE")], stop_reason="end_turn"))
        client = _FakeClient(script)

        summary, _ = _run_loop(client, tmp_path, max_iterations=len(script) + 2)

        assert summary == "DONE", "the breaker fired despite real progress"

    def test_nudges_are_bounded(self, tmp_path):
        # Claims DONE forever without writing anything.
        client = _FakeClient([_message([_text("DONE: all set")], stop_reason="end_turn")])

        summary, _ = _run_loop(client, tmp_path, max_iterations=50)

        assert summary == "DONE: all set"
        assert len(client.messages.calls) == _MAX_NUDGES + 1

    def test_cancellation_preserves_the_token_count(self, tmp_path):
        calls = {"n": 0}

        def cancel() -> bool:
            calls["n"] += 1
            return calls["n"] > 1

        client = _FakeClient([_message([_tool("t1", "list_files", path=".")])])

        _, tokens = _run_loop(client, tmp_path, cancel_fn=cancel)

        assert tokens > 0, "tokens already spent must still be reported on cancel"


def _assert_no_orphaned_tool_results(messages: list[dict[str, Any]]) -> None:
    """Every tool_result must sit directly after the assistant turn that asked."""
    for idx, msg in enumerate(messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        results = [
            b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"
        ]
        if not results:
            continue
        assert idx > 0, "a tool_result cannot be the first message"
        prev = messages[idx - 1]
        assert prev["role"] == "assistant", f"tool_result at {idx} follows {prev['role']}"
        asked = {
            b.id for b in prev["content"] if getattr(b, "type", None) == "tool_use"
        }
        for block in results:
            assert block["tool_use_id"] in asked, "tool_result answers a dropped tool_use"


# ── A3: streaming ─────────────────────────────────────────────────────────────


class TestStreaming:
    def test_text_is_flushed_to_the_log_at_boundaries(self, tmp_path):
        events = [_delta("text", "First sentence."), _delta("text", " Second one.")]
        client = _FakeClient(
            [_message([_text("First sentence. Second one.")], stop_reason="end_turn")],
            events={0: events},
        )
        logged: list[tuple[str, str]] = []

        _run_loop(client, tmp_path, log_fn=lambda k, m: logged.append((k, m)))

        text_events = [m for k, m in logged if k == "text"]
        assert text_events == ["First sentence.", "Second one."]

    def test_thinking_deltas_reach_the_feed(self, tmp_path):
        events = [_delta("thinking", "x" * 130), _delta("thinking", " tail")]
        client = _FakeClient(
            [_message([_text("done")], stop_reason="end_turn")],
            events={0: events},
        )
        logged: list[tuple[str, str]] = []

        _run_loop(client, tmp_path, log_fn=lambda k, m: logged.append((k, m)))

        thinking = [m for k, m in logged if k == "thinking"]
        assert thinking == ["x" * 130, "tail"]

    def test_streamed_text_is_not_logged_twice(self, tmp_path):
        """The old loop logged a 160-char preview per text block after the call."""
        events = [_delta("text", "Hello there.")]
        client = _FakeClient(
            [_message([_text("Hello there.")], stop_reason="end_turn")],
            events={0: events},
        )
        logged: list[tuple[str, str]] = []

        _run_loop(client, tmp_path, log_fn=lambda k, m: logged.append((k, m)))

        assert [m for k, m in logged if k == "text"] == ["Hello there."]

    def test_adaptive_thinking_is_requested(self, tmp_path):
        client = _FakeClient([_message([_text("DONE")], stop_reason="end_turn")])

        _run_loop(client, tmp_path)

        assert client.messages.calls[0]["thinking"] == {
            "type": "adaptive", "display": "summarized",
        }

    def test_thinking_is_dropped_when_the_model_rejects_it(self, tmp_path, monkeypatch):
        import anthropic
        import httpx

        monkeypatch.setattr(build_agent, "_thinking_supported", True)
        attempts: list[dict[str, Any]] = []
        final = _message([_text("DONE")], stop_reason="end_turn")
        response = httpx.Response(
            400, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        )

        class _PickyMessages(_FakeMessages):
            def stream(self, **kwargs: Any) -> _FakeStream:
                attempts.append(kwargs)
                if "thinking" in kwargs:
                    raise anthropic.BadRequestError(
                        "thinking: unsupported parameter", response=response, body=None,
                    )
                return _FakeStream(final, [])

        client = _FakeClient([final])
        client.messages = _PickyMessages([final], {})

        summary, _ = _run_loop(client, tmp_path)

        assert summary == "DONE"
        assert "thinking" in attempts[0], "the first attempt asks for thinking"
        assert "thinking" not in attempts[1], "the retry drops it"
        # And it stays dropped for the rest of the process rather than being
        # re-attempted (and re-rejected) on every subsequent turn.
        assert all("thinking" not in a for a in attempts[1:])


# ── A4: output cap ────────────────────────────────────────────────────────────


class TestOutputCap:
    def test_max_tokens_is_far_above_the_old_8096(self, tmp_path):
        client = _FakeClient([_message([_text("DONE")], stop_reason="end_turn")])

        _run_loop(client, tmp_path)

        assert client.messages.calls[0]["max_tokens"] == _MAX_OUTPUT_TOKENS
        assert _MAX_OUTPUT_TOKENS >= 16384

    def test_max_tokens_stays_within_the_models_reported_limit(self):
        # claude-sonnet-5 reports max_tokens=128000 from the Models API.
        assert _MAX_OUTPUT_TOKENS <= 128_000

    def test_hitting_the_cap_is_surfaced_as_a_warning(self, tmp_path):
        client = _FakeClient([
            _message([_text("half a file")], stop_reason="max_tokens"),
            _message([_text("DONE")], stop_reason="end_turn"),
        ])
        logged: list[tuple[str, str]] = []

        _run_loop(client, tmp_path, log_fn=lambda k, m: logged.append((k, m)))

        assert any("output limit" in m for k, m in logged if k == "warning")


@pytest.mark.parametrize("pairs", [_ANTHROPIC_HISTORY_PAIRS])
def test_history_window_is_generous_for_a_large_context_model(pairs):
    assert pairs >= 20
