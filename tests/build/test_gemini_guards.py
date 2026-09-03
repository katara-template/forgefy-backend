"""Context-control guards on the Gemini backend.

Run:
    venv/Scripts/python -m pytest tests/build/test_gemini_guards.py -v

Regression cover for the read/edit treadmill seen in production: a 1,500-char
tool-result cap meant a stylesheet came back as a fragment, so the agent paged
through the same file with read_file offsets for a whole phase. Since Part L
the loop itself is shared (run_agent_loop); these tests drive the GeminiAdapter
wire through it.
"""
from __future__ import annotations

from typing import Any

import pytest

from app.build import build_agent
from app.build.build_agent import (
    _MAX_EXPLORE_STREAK,
    _MAX_NUDGES,
    _TOOL_RESULT_LIMIT_LARGE_CTX,
    _TOOL_RULES,
    _with_tool_rules,
)
from app.build.provider_loop import GeminiAdapter, run_agent_loop


def _call(name: str, **args: Any) -> dict[str, Any]:
    return {"functionCall": {"name": name, "args": args}}


def _turn(*parts: dict[str, Any], finish: str = "STOP") -> dict[str, Any]:
    return {
        "candidates": [{"content": {"parts": list(parts)}, "finishReason": finish}],
        "usageMetadata": {"totalTokenCount": 10},
    }


def _text(t: str) -> dict[str, Any]:
    return {"text": t}


@pytest.fixture
def gemini(monkeypatch):
    """Scripted Gemini transport; records every request body."""
    sent: list[dict[str, Any]] = []

    def make(script: list[dict[str, Any]]):
        def fake_post(url, api_key, body, timeout=120, log_fn=None):
            sent.append(body)
            return script[min(len(sent) - 1, len(script) - 1)]

        monkeypatch.setattr(build_agent, "_gemini_post_with_retry", fake_post)
        return sent

    return make


def _tool_outputs(body: dict[str, Any]) -> list[str]:
    out = []
    for turn in body["contents"]:
        for part in turn.get("parts", []):
            if "functionResponse" in part:
                out.append(part["functionResponse"]["response"]["output"])
    return out


def _run(workspace, script, gemini, *, pairs: int | None = None, **kw):
    sent = gemini(script)
    adapter = GeminiAdapter(api_key="fake-key", model="gemini-2.5-flash")
    if pairs is not None:
        adapter.history_pairs = pairs
    summary, tokens = run_agent_loop(
        adapter, system="SYSTEM", stable="do the thing", workspace=workspace, **kw
    )
    return summary, sent


# ── the actual production bug ─────────────────────────────────────────────────


class TestToolResultLimit:
    def test_gemini_uses_the_large_context_limit(self):
        """1,500 chars starved a 1M-token model and caused the paging loop."""
        assert GeminiAdapter(api_key="k", model="m").trim_result_limit() == _TOOL_RESULT_LIMIT_LARGE_CTX
        assert _TOOL_RESULT_LIMIT_LARGE_CTX >= 24000

    def test_a_realistic_stylesheet_arrives_whole(self, tmp_path, gemini):
        # ~8 KB, the size that was being fragmented in production.
        css = "\n".join(f".cls-{i} {{ color: #00{i:04d}; }}" for i in range(300))
        (tmp_path / "globals.css").write_text(css, encoding="utf-8")
        script = [
            _turn(_call("read_file", path="globals.css")),
            _turn(_text("DONE"), finish="STOP"),
        ]

        _, sent = _run(tmp_path, script, gemini)

        output = _tool_outputs(sent[1])[0]
        assert "TRUNCATED" not in output, "the file was fragmented again"
        assert ".cls-299" in output, "the tail never reached the model"


class TestRepeatReadSuppression:
    def test_identical_read_is_served_from_a_notice(self, tmp_path, gemini):
        (tmp_path / "a.css").write_text("body { color: red; }", encoding="utf-8")
        script = [
            _turn(_call("read_file", path="a.css")),
            _turn(_call("read_file", path="a.css")),
            _turn(_text("DONE"), finish="STOP"),
        ]

        _, sent = _run(tmp_path, script, gemini)

        assert "already ran" in _tool_outputs(sent[2])[-1]

    def test_an_edit_makes_a_re_read_legitimate(self, tmp_path, gemini):
        (tmp_path / "a.css").write_text("body { color: red; }", encoding="utf-8")
        script = [
            _turn(_call("read_file", path="a.css")),
            _turn(_call("edit_file", path="a.css", old_string="red", new_string="blue")),
            _turn(_call("read_file", path="a.css")),
            _turn(_text("DONE"), finish="STOP"),
        ]

        _, sent = _run(tmp_path, script, gemini)

        last = _tool_outputs(sent[3])[-1]
        assert "already ran" not in last
        assert "blue" in last


class TestExplorationBreaker:
    def test_survey_only_agent_is_stopped(self, tmp_path, gemini):
        script = [
            _turn(_call("list_files", path="."))
            for _ in range(_MAX_EXPLORE_STREAK * (_MAX_NUDGES + 2))
        ]

        summary, sent = _run(tmp_path, script, gemini, max_iterations=200)

        assert "explor" in summary.lower()
        assert len(sent) < 200, "the breaker never fired"

    def test_editing_resets_the_streak(self, tmp_path, gemini):
        (tmp_path / "a.css").write_text("body { color: red; }", encoding="utf-8")
        script = [_turn(_call("list_files", path=str(i))) for i in range(_MAX_EXPLORE_STREAK - 1)]
        script.append(
            _turn(_call("edit_file", path="a.css", old_string="red", new_string="blue"))
        )
        script.append(_turn(_text("DONE"), finish="STOP"))

        summary, _ = _run(tmp_path, script, gemini, max_iterations=len(script) + 2)

        assert summary == "DONE"


class TestRollingWindow:
    def test_window_never_orphans_a_function_response(self, tmp_path, gemini):
        script = [_turn(_call("list_files", path=str(i))) for i in range(6)]
        script.append(_turn(_text("DONE"), finish="STOP"))

        _, sent = _run(tmp_path, script, gemini, pairs=1)

        for body in sent:
            turns = body["contents"]
            for idx, turn in enumerate(turns):
                has_response = any("functionResponse" in p for p in turn.get("parts", []))
                if not has_response:
                    continue
                assert idx > 0, "a functionResponse cannot open the history"
                prev = turns[idx - 1]
                assert prev["role"] == "model"
                assert any("functionCall" in p for p in prev["parts"]), (
                    "functionResponse answers a dropped functionCall"
                )

    def test_first_user_turn_is_kept_as_an_anchor(self, tmp_path, gemini):
        script = [_turn(_call("list_files", path=str(i))) for i in range(5)]
        script.append(_turn(_text("DONE"), finish="STOP"))

        _, sent = _run(tmp_path, script, gemini, pairs=1)

        for body in sent:
            assert body["contents"][0]["parts"][0]["text"] == "do the thing"


class TestNudgeBounding:
    def test_nudges_are_bounded(self, tmp_path, gemini):
        script = [_turn(_text("DONE: all set"), finish="STOP")]

        summary, sent = _run(tmp_path, script, gemini, max_iterations=50)

        assert summary == "DONE: all set"
        assert len(sent) == _MAX_NUDGES + 1


# ── shared tool rules reach every phase ───────────────────────────────────────


class TestToolRules:
    def test_rules_tell_the_agent_not_to_re_read_after_editing(self):
        assert "DO NOT RE-READ A FILE YOU JUST EDITED" in _TOOL_RULES
        assert "edit_file" in _TOOL_RULES
        assert "grep" in _TOOL_RULES

    def test_build_system_includes_them(self):
        assert _TOOL_RULES in build_agent._build_system("next")

    @pytest.mark.parametrize(
        "prompt_name",
        [
            "_DESIGN_AGENT_SYSTEM",
            "_SCHEMA_SYSTEM",
            "_UPDATE_SYSTEM",
            "_VALIDATOR_SYSTEM",
            "_TEST_SYSTEM",
            "_SECURITY_SYSTEM",
        ],
    )
    def test_every_phase_prompt_gets_them(self, prompt_name):
        """The design agent has its own prompt — guidance in only one is guidance
        the other five agents never see."""
        base = getattr(build_agent, prompt_name)

        assert _TOOL_RULES in _with_tool_rules(base)

    def test_wrapping_is_byte_stable_so_caching_still_works(self):
        assert _with_tool_rules("abc") == _with_tool_rules("abc")
