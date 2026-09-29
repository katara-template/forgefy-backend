"""Tests for the Claude Code agent, with the SDK's query() mocked.

Everything here runs offline: the official claude-agent-sdk is monkeypatched at
``claude_agent_sdk.query`` so the tests exercise the real driver, event mapper,
outcome taxonomy, timeouts, cancellation and environment scrubbing without a
provider or CLI.

Run:
    venv/Scripts/python -m pytest tests/build/test_claude_code_agent.py -v
"""
from __future__ import annotations

import asyncio
from dataclasses import field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import claude_agent_sdk
from app.build.agents import ClaudeCodeAgent
from app.build.agents.claude_code import (
    ClaudeCodeCancelled,
    ClaudeCodeError,
    ClaudeCodeTimeout,
    _build_options,
    _StreamParser,
    run_claude,
)


# ── SDK message builders (real claude_agent_sdk types) ────────────────────────

from claude_agent_sdk import ResultMessage as _SDKResult  # noqa: E402
from claude_agent_sdk import StreamEvent as _SDKStream  # noqa: E402


def _se(event: dict[str, Any]) -> _SDKStream:
    return _SDKStream(uuid="u", session_id="s", event=event)


def _rm(**overrides: Any) -> _SDKResult:
    base: dict[str, Any] = dict(
        subtype="success",
        duration_ms=10,
        duration_api_ms=5,
        is_error=False,
        num_turns=1,
        session_id="s",
        result="DONE: built the app",
        usage={"input_tokens": 10, "output_tokens": 5},
        errors=None,
    )
    base.update(overrides)
    return _SDKResult(**base)


def _stream_events() -> list[_SDKStream]:
    """A realistic slice of stream-json wire events (SDK-parsed form)."""
    return [
        _se({"type": "message_start"}),
        _se({"type": "content_block_start", "index": 0,
             "content_block": {"type": "text", "text": ""}}),
        _se({"type": "content_block_delta", "index": 0,
             "delta": {"type": "text_delta", "text": "Building the app. "}}),
        _se({"type": "content_block_stop", "index": 0}),
        _se({"type": "content_block_start", "index": 1,
             "content_block": {"type": "thinking", "thinking": ""}}),
        _se({"type": "content_block_delta", "index": 1,
             "delta": {"type": "thinking_delta", "thinking": "Planning files"}}),
        _se({"type": "content_block_stop", "index": 1}),
        _se({"type": "content_block_start", "index": 2,
             "content_block": {"type": "tool_use", "id": "t1", "name": "Write", "input": {}}}),
        _se({"type": "content_block_delta", "index": 2,
             "delta": {"type": "input_json_delta", "partial_json": '{"file_path": "app/page.tsx"}'}}),
        _se({"type": "content_block_stop", "index": 2}),
    ]


def _fake_query_factory(messages: list, sink: dict | None = None, delay: float = 0.0):
    """Build a stand-in for claude_agent_sdk.query capturing its options."""

    def fake_query(*, prompt: str, options=None, transport=None):
        if sink is not None:
            sink["prompt"] = prompt
            sink["options"] = options

        async def _gen():
            if delay:
                await asyncio.sleep(delay)
            for m in messages:
                yield m

        return _gen()

    return fake_query


@pytest.fixture
def cc_settings(monkeypatch):
    s = SimpleNamespace(
        CLAUDE_CODE_ENABLED=True,
        CLAUDE_CODE_COMMAND="claude",
        CLAUDE_CODE_CLI_PATH="",
        CLAUDE_CODE_MODEL="",
        CLAUDE_CODE_MAX_TURNS=0,
        CLAUDE_CODE_PERMISSION_MODE="bypassPermissions",
        CLAUDE_CODE_ALLOWED_TOOLS="",
        CLAUDE_CODE_ADD_DIR=True,
        CLAUDE_CODE_TIMEOUT=60,
        CLAUDE_CODE_IDLE_TIMEOUT=30,
        CLAUDE_CODE_API_KEY="",
        CLAUDE_CODE_BASE_URL="",
        CLAUDE_CODE_AUTH_TOKEN="",
        ANTHROPIC_API_KEY="test-key",
    )
    monkeypatch.setattr("app.config.get_settings", lambda: s)
    yield s


# ── options / env construction ────────────────────────────────────────────────


class TestOptions:
    def test_workspace_pins_cwd_and_add_dirs(self, cc_settings, tmp_path: Path) -> None:
        opts = _build_options(settings=cc_settings, workspace=tmp_path, prompt="p")
        assert opts.cwd == str(tmp_path.resolve())
        assert opts.add_dirs == [str(tmp_path.resolve())]

    def test_headless_flags(self, cc_settings, tmp_path: Path) -> None:
        opts = _build_options(settings=cc_settings, workspace=tmp_path, prompt="p")
        assert opts.permission_mode == "bypassPermissions"
        assert opts.include_partial_messages is True

    def test_model_and_turn_caps_applied(self, cc_settings, tmp_path: Path) -> None:
        cc_settings.CLAUDE_CODE_MODEL = "qwen3-coder"
        cc_settings.CLAUDE_CODE_MAX_TURNS = 40
        opts = _build_options(settings=cc_settings, workspace=tmp_path, prompt="p")
        assert opts.model == "qwen3-coder"
        assert opts.max_turns == 40

    def test_system_prompt_is_preset_with_forgefy_append(self, cc_settings, tmp_path: Path) -> None:
        opts = _build_options(settings=cc_settings, workspace=tmp_path, prompt="p", system_extra="CTX")
        assert opts.system_prompt == {"type": "preset", "preset": "claude_code", "append": "CTX"}

    def test_env_is_whitelisted_never_the_whole_process_env(
        self, cc_settings, tmp_path: Path, monkeypatch
    ) -> None:
        """Forgefy secrets must not ride along into the agent's environment."""
        monkeypatch.setenv("GITHUB_TOKEN", "gh-secret")
        monkeypatch.setenv("FIREBASE_CREDENTIALS", "secret")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        opts = _build_options(settings=cc_settings, workspace=tmp_path, prompt="p")
        env = opts.env
        assert env["ANTHROPIC_API_KEY"] == "test-key"
        assert "GITHUB_TOKEN" not in env
        assert "FIREBASE_CREDENTIALS" not in env
        assert "PATH" in env  # toolchain vars survive so builds still work

    def test_ollama_provider_env_is_passed_through(self, cc_settings, tmp_path: Path) -> None:
        cc_settings.CLAUDE_CODE_AUTH_TOKEN = "ollama"
        cc_settings.CLAUDE_CODE_BASE_URL = "http://localhost:11434"
        cc_settings.CLAUDE_CODE_MODEL = "qwen3-coder"
        opts = _build_options(settings=cc_settings, workspace=tmp_path, prompt="p")
        assert opts.env["ANTHROPIC_AUTH_TOKEN"] == "ollama"
        assert opts.env["ANTHROPIC_BASE_URL"] == "http://localhost:11434"
        assert opts.env["ANTHROPIC_MODEL"] == "qwen3-coder"


# ── execution (mocked SDK) ────────────────────────────────────────────────────


class TestExecution:
    def test_success_streams_events_and_returns_summary(
        self, cc_settings, tmp_path: Path, monkeypatch
    ) -> None:
        sink: dict = {}
        monkeypatch.setattr(
            claude_agent_sdk, "query",
            _fake_query_factory(_stream_events() + [_rm()], sink=sink),
        )
        events: list[tuple[str, str]] = []
        outcome = run_claude(workspace=tmp_path, prompt="build it", log_fn=lambda k, m: events.append((k, m)))
        assert outcome.state == "success"
        assert outcome.summary == "DONE: built the app"
        assert outcome.tokens == 15
        kinds = [k for k, _ in events]
        assert "text" in kinds
        assert "thinking" in kinds
        assert "tool" in kinds
        assert "file_written" in kinds
        # Streamed incrementally — not one lump at the end.
        assert len(events) >= 4

    def test_prompt_and_options_reach_the_sdk(self, cc_settings, tmp_path: Path, monkeypatch) -> None:
        sink: dict = {}
        monkeypatch.setattr(
            claude_agent_sdk, "query",
            _fake_query_factory([_rm()], sink=sink),
        )
        run_claude(workspace=tmp_path, prompt="the task", system_extra="CTX")
        assert sink["prompt"] == "the task"
        assert sink["options"].cwd == str(tmp_path.resolve())
        assert sink["options"].system_prompt["append"] == "CTX"

    def test_error_result_raises_claude_code_error(self, cc_settings, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(
            claude_agent_sdk, "query",
            _fake_query_factory([_rm(is_error=True, errors=["auth failed"])]),
        )
        with pytest.raises(ClaudeCodeError, match="auth failed"):
            run_claude(workspace=tmp_path, prompt="x")

    def test_no_result_message_is_a_failure(self, cc_settings, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(claude_agent_sdk, "query", _fake_query_factory(_stream_events()))
        with pytest.raises(ClaudeCodeError, match="without a final result"):
            run_claude(workspace=tmp_path, prompt="x")

    def test_sdk_provider_error_is_mapped_and_secret_free(
        self, cc_settings, tmp_path: Path, monkeypatch
    ) -> None:
        from claude_agent_sdk import ResultError

        def failing_query(*, prompt: str, options=None, transport=None):
            async def _gen():
                raise ResultError("Credit balance is too low", exit_code=1)
                yield  # pragma: no cover

            return _gen()

        monkeypatch.setattr(claude_agent_sdk, "query", failing_query)
        with pytest.raises(ClaudeCodeError, match="Credit balance"):
            run_claude(workspace=tmp_path, prompt="x")

    def test_cli_not_found_maps_to_a_clear_error(self, cc_settings, tmp_path: Path, monkeypatch) -> None:
        from claude_agent_sdk import CLINotFoundError

        def failing_query(*, prompt: str, options=None, transport=None):
            async def _gen():
                raise CLINotFoundError()
                yield  # pragma: no cover

            return _gen()

        monkeypatch.setattr(claude_agent_sdk, "query", failing_query)
        with pytest.raises(ClaudeCodeError, match="CLI could not be found"):
            run_claude(workspace=tmp_path, prompt="x")

    def test_malformed_or_unknown_events_do_not_crash(
        self, cc_settings, tmp_path: Path, monkeypatch
    ) -> None:
        junk = [
            _se({"type": "ping"}),
            _se({"type": "content_block_delta", "index": 99,
                             "delta": {"type": "input_json_delta", "partial_json": "{broken"}}),
            _se({"unexpected": "shape"}),
        ]
        monkeypatch.setattr(
            claude_agent_sdk, "query",
            _fake_query_factory(junk + [_rm()]),
        )
        outcome = run_claude(workspace=tmp_path, prompt="x")
        assert outcome.state == "success"


# ── process management: timeout / cancellation / teardown ─────────────────────


class TestProcessManagement:
    def test_overall_timeout_terminates_the_run(self, cc_settings, tmp_path: Path, monkeypatch) -> None:
        """A hanging agent is stopped and reported as a timeout, not a crash."""
        cc_settings.CLAUDE_CODE_TIMEOUT = 1
        cc_settings.CLAUDE_CODE_IDLE_TIMEOUT = 60
        monkeypatch.setattr(claude_agent_sdk, "query", _fake_query_factory([_rm()], delay=30))
        with pytest.raises(ClaudeCodeTimeout):
            run_claude(workspace=tmp_path, prompt="x")

    def test_idle_timeout_fires_when_output_stops(
        self, cc_settings, tmp_path: Path, monkeypatch
    ) -> None:
        cc_settings.CLAUDE_CODE_TIMEOUT = 60
        cc_settings.CLAUDE_CODE_IDLE_TIMEOUT = 1
        monkeypatch.setattr(claude_agent_sdk, "query", _fake_query_factory([_rm()], delay=30))
        with pytest.raises(ClaudeCodeTimeout):
            run_claude(workspace=tmp_path, prompt="x")

    def test_user_cancellation_raises_cancelled(self, cc_settings, tmp_path: Path, monkeypatch) -> None:
        cc_settings.CLAUDE_CODE_TIMEOUT = 60
        monkeypatch.setattr(claude_agent_sdk, "query", _fake_query_factory([_rm()], delay=30))
        with pytest.raises(ClaudeCodeCancelled):
            run_claude(workspace=tmp_path, prompt="x", cancel_fn=lambda: True)

    def test_force_stop_teardown_safety_net(self, cc_settings, tmp_path: Path, monkeypatch) -> None:
        """kill_workspace_claude stops a run even with no cancel_fn wired."""
        import threading

        from app.build.agents.claude_code import _FORCE_STOP, kill_workspace_claude

        cc_settings.CLAUDE_CODE_TIMEOUT = 60
        monkeypatch.setattr(claude_agent_sdk, "query", _fake_query_factory([_rm()], delay=30))

        kill_flag = threading.Event()
        timer = threading.Timer(0.2, kill_workspace_claude, args=(tmp_path,))
        timer.start()
        try:
            with pytest.raises(ClaudeCodeCancelled):
                run_claude(workspace=tmp_path, prompt="x", cancel_fn=kill_flag.is_set)
        finally:
            timer.join()
        # The registry is cleaned up after the run ends.
        assert str(tmp_path.resolve()) not in _FORCE_STOP


# ── the CodingAgent facade ────────────────────────────────────────────────────


class TestClaudeCodeAgentFacade:
    def test_run_build_builds_a_blueprint_prompt(
        self, cc_settings, tmp_path: Path, monkeypatch
    ) -> None:
        sink: dict = {}
        monkeypatch.setattr(
            claude_agent_sdk, "query",
            _fake_query_factory([_rm(result="DONE: x")], sink=sink),
        )
        summary, tokens = ClaudeCodeAgent().run_build(
            workspace=tmp_path, blueprint={"app_name": "demo"}, app_name="demo",
            template_key="next", log_fn=None, cancel_fn=None,
        )
        assert summary == "DONE: x"
        assert tokens == 15
        assert "Blueprint:" in sink["prompt"]
        assert "demo" in sink["prompt"]
        assert sink["options"].system_prompt["append"].startswith("You are implementing a next")

    def test_run_update_passes_the_prompt_through(
        self, cc_settings, tmp_path: Path, monkeypatch
    ) -> None:
        sink: dict = {}
        monkeypatch.setattr(
            claude_agent_sdk, "query",
            _fake_query_factory([_rm(result="DONE: dark mode added")], sink=sink),
        )
        summary, _ = ClaudeCodeAgent().run_update(
            workspace=tmp_path, prompt="add dark mode", blueprint={},
            app_name="demo", template_key="next",
        )
        assert summary == "DONE: dark mode added"
        assert sink["prompt"] == "add dark mode"

    def test_run_fix_passes_the_compile_error_prompt(
        self, cc_settings, tmp_path: Path, monkeypatch
    ) -> None:
        sink: dict = {}
        monkeypatch.setattr(
            claude_agent_sdk, "query",
            _fake_query_factory([_rm(result="DONE: fixed")], sink=sink),
        )
        summary, _ = ClaudeCodeAgent().run_fix(
            workspace=tmp_path, prompt="lib/x.dart:12: Error: boom",
            app_name="demo", template_key="flutter",
        )
        assert summary == "DONE: fixed"
        assert "lib/x.dart:12" in sink["prompt"]


# ── event mapping unit tests ──────────────────────────────────────────────────


class TestStreamParser:
    def test_write_tool_emits_tool_and_file_written(self) -> None:
        events: list[tuple[str, str]] = []
        parser = _StreamParser(lambda k, m: events.append((k, m)))
        parser.handle_stream_event({"type": "content_block_start", "index": 0,
                                    "content_block": {"type": "tool_use", "name": "Write"}})
        parser.handle_stream_event({"type": "content_block_delta", "index": 0,
                                    "delta": {"type": "input_json_delta",
                                              "partial_json": '{"file_path": "src/app.tsx"}'}})
        parser.handle_stream_event({"type": "content_block_stop", "index": 0})
        assert ("tool", "Writing file · src/app.tsx") in events
        assert ("file_written", "Wrote src/app.tsx") in events

    def test_bash_tool_shows_the_command(self) -> None:
        events: list[tuple[str, str]] = []
        parser = _StreamParser(lambda k, m: events.append((k, m)))
        parser.handle_stream_event({"type": "content_block_start", "index": 0,
                                    "content_block": {"type": "tool_use", "name": "Bash"}})
        parser.handle_stream_event({"type": "content_block_delta", "index": 0,
                                    "delta": {"type": "input_json_delta",
                                              "partial_json": '{"command": "npm test"}'}})
        parser.handle_stream_event({"type": "content_block_stop", "index": 0})
        assert ("tool", "Running `npm test`") in events

    def test_broken_tool_input_json_does_not_crash(self) -> None:
        events: list[tuple[str, str]] = []
        parser = _StreamParser(lambda k, m: events.append((k, m)))
        parser.handle_stream_event({"type": "content_block_start", "index": 0,
                                    "content_block": {"type": "tool_use", "name": "Write"}})
        parser.handle_stream_event({"type": "content_block_delta", "index": 0,
                                    "delta": {"type": "input_json_delta", "partial_json": "{oops"}})
        parser.handle_stream_event({"type": "content_block_stop", "index": 0})
        assert ("tool", "Writing file …") in events

    def test_non_dict_events_are_ignored(self) -> None:
        parser = _StreamParser(None)
        parser.handle_stream_event("not a dict")  # type: ignore[arg-type]
        assert parser.text_buf == []