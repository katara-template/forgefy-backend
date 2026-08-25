"""Tests for _run_update_sequence — the shared update pipeline.

Run:
    venv/Scripts/python -m pytest tests/build/test_update_sequence.py -v

Every backend (Claude, Ollama, Gemini, OpenAI, OpenRouter) drives this one
sequence through a `run_phase` closure, so these cover all five. The fake
run_phase records the system prompt of each phase, which is what identifies it.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from app.build.agent_tools import TOOLS, execute_tool
from app.build.build_agent import (
    _DESIGN_AGENT_SYSTEM,
    _REPORT_INSTRUCTION,
    _SCHEMA_SYSTEM,
    _SECURITY_SYSTEM,
    _TEST_SYSTEM,
    _TOOL_RULES,
    _UPDATE_SYSTEM,
    _VALIDATOR_SYSTEM,
    _run_update_sequence,
)

_ENTITY = {"name": "Invoice", "description": "A bill", "fields": [], "relationships": []}

# Phase labels keyed by system prompt, so assertions read as a phase sequence.
_LABELS = {
    _SCHEMA_SYSTEM: "schema",
    _DESIGN_AGENT_SYSTEM: "design",
    _UPDATE_SYSTEM: "exec",  # also the fix pass — disambiguated by position
    _VALIDATOR_SYSTEM: "validate",
    _TEST_SYSTEM: "test",
    _SECURITY_SYSTEM: "security",
}

# A validator summary that signals a clean run (no fix pass).
_CLEAN = "VALIDATED: all good"


# Reviewing phases file their result through a per-workspace store, so the fake
# needs a real directory to key it by. One shared dir is safe: each reviewing
# phase resets the store before it runs.
_WS = Path(tempfile.mkdtemp(prefix="forgefy_seq_"))

_REVIEWING = ("validate", "test", "security")


class _Recorder:
    """Fake run_phase that records calls and returns scripted summaries.

    `reports` maps a reviewing phase's label to what it should file:
      omitted / "clean" -> a clean report (the common case)
      list[dict]        -> those findings, i.e. issues_found
      None              -> file nothing, simulating a phase that forgot
    """

    def __init__(
        self,
        summaries: dict[str, str] | None = None,
        tokens: int = 10,
        reports: dict[str, Any] | None = None,
    ):
        self.calls: list[tuple[str, str, int]] = []  # (label, user_msg, iters)
        self.tools_seen: dict[str, list[str]] = {}
        self._summaries = summaries or {}
        self._tokens = tokens
        self._reports = reports or {}

    def __call__(
        self, system: str, user_msg: str, iters: int, tools: list[dict] | None = None,
    ) -> tuple[str, int]:
        # Every phase is dispatched with the shared tool-usage rules appended, so
        # the design/schema/test/security agents all learn to prefer edit_file
        # and not to re-read a file they just edited.
        base = system
        if base.endswith(_REPORT_INSTRUCTION):
            base = base[: -len(_REPORT_INSTRUCTION)]
        assert base.endswith(_TOOL_RULES), "phase dispatched without the tool rules"
        label = _LABELS.get(base[: -len(_TOOL_RULES)], "?")
        # The fix pass reuses _UPDATE_SYSTEM; it is the second "exec" seen.
        if label == "exec" and any(c[0] in ("exec", "fix") for c in self.calls):
            label = "fix"
        self.calls.append((label, user_msg, iters))
        self.tools_seen[label] = [t["name"] for t in (tools or TOOLS)]

        if label in _REVIEWING:
            self._file_report(label)
        return self._summaries.get(label, f"{label} done"), self._tokens

    def _file_report(self, label: str) -> None:
        spec = self._reports.get(label, "clean")
        if spec is None:
            return  # phase forgot to report
        findings = [] if spec == "clean" else spec
        execute_tool(
            "report_findings",
            {
                "status": "clean" if not findings else "issues_found",
                "findings": findings,
                "summary": f"{label} report",
            },
            _WS,
        )

    @property
    def phases(self) -> list[str]:
        return [c[0] for c in self.calls]

    def msg(self, label: str) -> str:
        return next(c[1] for c in self.calls if c[0] == label)


_A_FINDING = [{"severity": "high", "file": "app/page.tsx", "line": 12,
               "summary": "Missing import for Card"}]


def _run(recorder: _Recorder, plan: dict | None, blueprint: dict | None = None, **kw):
    return _run_update_sequence(
        recorder,
        workspace=_WS,
        prompt="add invoices",
        blueprint=blueprint if blueprint is not None else {},
        app_name="Acme",
        template_key="next",
        plan=plan,
        **kw,
    )


class TestPhaseOrder:
    def test_full_sequence_when_nothing_is_skipped(self):
        rec = _Recorder({"validate": _CLEAN})
        plan = {"design_impact": {"new_components_needed": ["Card"]}}

        _run(rec, plan, {"entities": [_ENTITY]})

        assert rec.phases == ["schema", "design", "exec", "validate", "test", "security"]

    def test_planner_skips_are_honoured(self):
        rec = _Recorder({"validate": _CLEAN})
        plan = {
            "skip_schema_agent": True,
            "skip_design_agent": True,
            "skip_security_agent": True,
        }

        _run(rec, plan)

        assert rec.phases == ["exec", "validate", "test"]

    def test_missing_plan_runs_every_phase(self):
        """Both skip gates return False without a plan — no planner signal means
        run everything rather than silently drop a phase."""
        rec = _Recorder({"validate": _CLEAN})

        _run(rec, None)

        assert rec.phases == ["schema", "design", "exec", "validate", "test", "security"]

    def test_fix_pass_runs_when_validator_is_not_clean(self):
        rec = _Recorder(reports={"validate": _A_FINDING})

        _run(rec, {"skip_schema_agent": True, "skip_security_agent": True})

        # Tests run after the fix pass, against code that now compiles.
        assert rec.phases == ["exec", "validate", "fix", "test"]

    def test_iteration_budgets_are_per_phase(self):
        from app.build.build_agent import _ITERS_EXEC, _ITERS_SCHEMA

        rec = _Recorder({"validate": _CLEAN})
        plan = {"skip_design_agent": True, "skip_security_agent": True, "skip_test_agent": True}
        _run(rec, plan, {"entities": [_ENTITY]})

        by_label = {c[0]: c[2] for c in rec.calls}
        assert by_label["schema"] == _ITERS_SCHEMA
        assert by_label["exec"] == _ITERS_EXEC


class TestTestPhase:
    def test_skipped_only_when_planner_says_nothing_testable_changed(self):
        rec = _Recorder({"validate": _CLEAN})

        _run(rec, {"skip_test_agent": True, "skip_schema_agent": True, "skip_security_agent": True})

        assert "test" not in rec.phases

    def test_runs_after_the_fix_pass_not_before(self):
        """Tests must assert against code that already compiles."""
        rec = _Recorder(reports={"validate": _A_FINDING})

        _run(rec, {"skip_schema_agent": True, "skip_security_agent": True})

        assert rec.phases.index("test") > rec.phases.index("fix")

    def test_executor_summary_reaches_the_test_agent(self):
        rec = _Recorder({"exec": "Added invoice total calculation", "validate": _CLEAN})

        _run(rec, {"skip_schema_agent": True, "skip_security_agent": True})

        assert "Added invoice total calculation" in rec.msg("test")

    def test_brief_is_not_fed_into_later_phases(self):
        """The test brief is evidence for the user, not input to security."""
        rec = _Recorder({"test": "TESTS: 4 written, all passing", "validate": _CLEAN})

        _run(rec, {"skip_schema_agent": True})

        assert "TESTS: 4 written" not in rec.msg("security")

    def test_summary_is_logged_for_the_user(self):
        logged: list[tuple[str, str]] = []
        rec = _Recorder({"test": "TESTS: 3 written, all passing", "validate": _CLEAN})

        _run(
            rec,
            {"skip_schema_agent": True, "skip_security_agent": True},
            log_fn=lambda level, msg: logged.append((level, msg)),
        )

        assert any("TESTS: 3 written" in msg for _, msg in logged)


class TestBriefPlumbing:
    def test_schema_and_design_briefs_reach_the_executor(self):
        rec = _Recorder({
            "schema": "SCHEMA READY: invoices(id, amount)",
            "design": "DESIGN READY: use AppCard",
            "validate": _CLEAN,
        })
        plan = {"design_impact": {"new_components_needed": ["Card"]}}

        _run(rec, plan, {"entities": [_ENTITY]})

        exec_msg = rec.msg("exec")
        assert "invoices(id, amount)" in exec_msg
        assert "use AppCard" in exec_msg
        # Schema first — the executor needs tables before components.
        assert exec_msg.index("DATABASE SCHEMA BRIEF") < exec_msg.index("DESIGN SYSTEM BRIEF")

    def test_skipped_phase_contributes_no_brief(self):
        rec = _Recorder({"validate": _CLEAN})

        _run(rec, {"skip_schema_agent": True, "skip_design_agent": True})

        assert "DATABASE SCHEMA BRIEF" not in rec.msg("exec")
        assert "DESIGN SYSTEM BRIEF" not in rec.msg("exec")


class TestTokensAndReturn:
    def test_tokens_sum_across_every_phase(self):
        rec = _Recorder({"validate": _CLEAN}, tokens=7)
        plan = {"design_impact": {"new_components_needed": ["Card"]}}

        _, tokens = _run(rec, plan, {"entities": [_ENTITY]})

        assert tokens == 7 * 6  # schema, design, exec, validate, test, security

    def test_returns_validator_summary(self):
        rec = _Recorder({"validate": _CLEAN})
        summary, _ = _run(rec, {"skip_schema_agent": True, "skip_security_agent": True, "skip_test_agent": True})
        assert summary == _CLEAN

    def test_fix_summary_supersedes_validator_summary(self):
        rec = _Recorder({"fix": "FIXED: done"}, reports={"validate": _A_FINDING})
        summary, _ = _run(rec, {"skip_schema_agent": True, "skip_security_agent": True, "skip_test_agent": True})
        assert summary == "FIXED: done"

    def test_falls_back_to_exec_summary_when_validator_is_empty(self):
        rec = _Recorder({"validate": "", "fix": ""})
        summary, _ = _run(rec, {"skip_schema_agent": True, "skip_security_agent": True, "skip_test_agent": True})
        assert summary == "exec done"


class TestCancellation:
    def test_cancel_before_any_phase_runs_nothing(self):
        rec = _Recorder()
        summary, tokens = _run(rec, None, cancel_fn=lambda: True)

        assert rec.phases == []
        assert summary == "Stopped by user."
        assert tokens == 0

    def test_cancel_after_first_phase_stops_and_keeps_tokens(self):
        rec = _Recorder(tokens=5)
        calls = {"n": 0}

        def cancel() -> bool:
            calls["n"] += 1
            return calls["n"] > 1  # allow the initial check, cancel after phase 1

        summary, tokens = _run(rec, None, {"entities": [_ENTITY]}, cancel_fn=cancel)

        assert rec.phases == ["schema"]
        assert tokens == 5
        assert summary == "schema done"
