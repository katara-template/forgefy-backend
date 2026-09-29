"""Tests for the coding-agent registry and the shared CodingAgent contract.

Covers agent selection (forgefy / claude_code), backwards compatibility
(no agent specified → Forgefy Agent), delegation of the Forgefy agent to the
existing unchanged internal loops, and health checks.

Run:
    venv/Scripts/python -m pytest tests/build/test_agents_registry.py -v
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.build.agents import (
    DEFAULT_AGENT,
    SUPPORTED_AGENTS,
    ClaudeCodeAgent,
    ForgefyAgent,
    agent_health,
    get_coding_agent,
    list_agents,
    resolve_agent,
)


# ── registry / selection ──────────────────────────────────────────────────────


class TestAgentSelection:
    def test_forgefy_resolves_to_forgefy_agent(self) -> None:
        assert isinstance(get_coding_agent("forgefy"), ForgefyAgent)

    def test_claude_code_resolves_to_claude_code_agent(self) -> None:
        assert isinstance(get_coding_agent("claude_code"), ClaudeCodeAgent)

    def test_none_defaults_to_forgefy(self) -> None:
        """Existing projects without an agent field keep using the Forgefy agent."""
        assert isinstance(get_coding_agent(None), ForgefyAgent)

    def test_unknown_agent_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown-agent"):
            get_coding_agent("unknown-agent")

    def test_case_and_whitespace_are_normalised(self) -> None:
        assert isinstance(get_coding_agent("  CLAUDE_CODE "), ClaudeCodeAgent)

    def test_resolve_agent_falls_back_rather_than_raising(self) -> None:
        assert resolve_agent(None) == "forgefy"
        assert resolve_agent("claude_code") == "claude_code"
        assert resolve_agent("garbage") == DEFAULT_AGENT == "forgefy"

    def test_supported_agents_are_exactly_the_two_shipped(self) -> None:
        assert SUPPORTED_AGENTS == ("forgefy", "claude_code")

    def test_list_agents_returns_both(self) -> None:
        assert {a.key for a in list_agents()} == {"forgefy", "claude_code"}

    def test_agents_expose_display_names(self) -> None:
        assert ForgefyAgent().display_name == "Forgefy Agent"
        assert ClaudeCodeAgent().display_name == "Claude Code"


# ── the Forgefy agent delegates to the existing, unchanged loops ─────────────


class TestForgefyAgentDelegation:
    def test_run_build_delegates_to_run_build_agent(self, monkeypatch, tmp_path: Path) -> None:
        calls: dict = {}

        def fake_run_build_agent(**kwargs):
            calls.update(kwargs)
            return "DONE: built", 123

        monkeypatch.setattr("app.build.build_agent.run_build_agent", fake_run_build_agent)

        summary, tokens = ForgefyAgent().run_build(
            workspace=tmp_path, blueprint={"a": 1}, app_name="demo",
            template_key="next", log_fn=None, cancel_fn=None,
        )
        assert (summary, tokens) == ("DONE: built", 123)
        assert calls["blueprint"] == {"a": 1}
        assert calls["workspace"] == tmp_path

    def test_run_update_delegates_to_run_update_agent(self, monkeypatch, tmp_path: Path) -> None:
        calls: dict = {}

        def fake_run_update_agent(**kwargs):
            calls.update(kwargs)
            return "VALIDATED: ok", 5

        monkeypatch.setattr("app.build.build_agent.run_update_agent", fake_run_update_agent)

        summary, tokens = ForgefyAgent().run_update(
            workspace=tmp_path, prompt="add dark mode", blueprint={},
            app_name="demo", template_key="next", build_model="claude",
        )
        assert (summary, tokens) == ("VALIDATED: ok", 5)
        assert calls["build_model"] == "claude"

    def test_run_fix_delegates_to_run_fix_agent(self, monkeypatch, tmp_path: Path) -> None:
        calls: dict = {}

        def fake_run_fix_agent(**kwargs):
            calls.update(kwargs)
            return "DONE", 7

        monkeypatch.setattr("app.build.build_agent.run_fix_agent", fake_run_fix_agent)

        summary, tokens = ForgefyAgent().run_fix(
            workspace=tmp_path, prompt="fix it", app_name="demo", template_key="flutter",
        )
        assert (summary, tokens) == ("DONE", 7)
        assert calls["template_key"] == "flutter"


# ── health checks ─────────────────────────────────────────────────────────────


class TestHealth:
    def _settings(self, **overrides):
        base = dict(
            ANTHROPIC_API_KEY="key",
            GEMINI_API_KEY="",
            OPENAI_API_KEY="",
            OLLAMA_URL="http://x",
            OPENROUTER_API_KEY="",
            BUILD_MODEL="claude",
            CLAUDE_CODE_ENABLED=True,
            CLAUDE_CODE_CLI_PATH="",
            CLAUDE_CODE_COMMAND="claude",
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_forgefy_available_when_provider_configured(self, monkeypatch) -> None:
        monkeypatch.setattr("app.config.get_settings", lambda: self._settings())
        health = ForgefyAgent().health_check()
        assert health.available is True
        assert health.status == "available"

    def test_forgefy_not_configured_without_any_provider(self, monkeypatch) -> None:
        s = self._settings(ANTHROPIC_API_KEY="", OLLAMA_URL="")
        monkeypatch.setattr("app.config.get_settings", lambda: s)
        health = agent_health("forgefy")
        assert health.available is False
        assert health.status == "not_configured"

    def test_claude_code_disabled_reports_not_configured(self, monkeypatch) -> None:
        s = self._settings(CLAUDE_CODE_ENABLED=False)
        monkeypatch.setattr("app.config.get_settings", lambda: s)
        health = ClaudeCodeAgent().health_check()
        assert health.available is False
        assert "disabled" in health.detail.lower()

    def test_claude_code_not_configured_without_sdk(self, monkeypatch) -> None:
        s = self._settings()
        monkeypatch.setattr("app.config.get_settings", lambda: s)
        import builtins

        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "claude_agent_sdk":
                raise ImportError("no sdk")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        health = ClaudeCodeAgent().health_check()
        assert health.available is False
        assert health.status == "not_configured"

    def test_claude_code_available_with_resolved_cli(self, monkeypatch) -> None:
        s = self._settings()
        monkeypatch.setattr("app.config.get_settings", lambda: s)
        monkeypatch.setattr("app.build.agents.claude_code.resolve_cli", lambda _s: "C:/x/claude.exe")
        health = ClaudeCodeAgent().health_check()
        assert health.available is True
        assert health.label == "Available"

    def test_agent_health_never_raises(self, monkeypatch) -> None:
        def boom(_key):
            raise RuntimeError("probe exploded")

        monkeypatch.setattr("app.build.agents.get_coding_agent", boom)
        health = agent_health("forgefy")
        assert health.available is False
        assert health.status == "error"