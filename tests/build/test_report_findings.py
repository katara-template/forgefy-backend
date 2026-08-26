"""Structured phase reports: the report_findings tool and the fix-pass gate.

Run:
    venv/Scripts/python -m pytest tests/build/test_report_findings.py -v

Replaces a gate that sniffed the validator's prose for a "VALIDATED:" prefix and
spent a 100-iteration fix pass whenever the model phrased its sign-off naturally.
"""
from __future__ import annotations

from typing import Any

import pytest

from app.build.agent_tools import (
    REPORT_FINDINGS_TOOL,
    TOOLS,
    execute_tool,
    missing_report,
    reset_report,
    take_report,
)
from app.build.build_agent import (
    _REPORT_INSTRUCTION,
    _REVIEW_TOOLS,
    _format_findings,
    _make_fix_prompt,
    _needs_fix_pass,
)


def _report(workspace, **inputs: Any) -> str:
    return execute_tool("report_findings", inputs, workspace)


_FINDING = {"severity": "high", "file": "app/page.tsx", "line": 12,
            "summary": "Card is used but never imported"}


# ── the tool ──────────────────────────────────────────────────────────────────


class TestReportFindingsTool:
    def test_it_is_not_in_the_default_tool_set(self):
        """Only reviewing phases may report; the executor must not self-certify."""
        assert "report_findings" not in [t["name"] for t in TOOLS]

    def test_review_tools_add_it_to_everything_else(self):
        names = [t["name"] for t in _REVIEW_TOOLS]

        assert "report_findings" in names
        assert len(names) == len(TOOLS) + 1
        assert names[: len(TOOLS)] == [t["name"] for t in TOOLS]

    def test_schema_requires_status_and_summary(self):
        required = REPORT_FINDINGS_TOOL["input_schema"]["required"]

        assert set(required) == {"status", "summary"}

    def test_clean_report_is_recorded(self, tmp_path):
        result = _report(tmp_path, status="clean", findings=[], summary="All good")

        assert "clean" in result
        report = take_report(tmp_path)
        assert report["status"] == "clean"
        assert report["findings"] == []
        assert report["reported"] is True

    def test_issues_report_keeps_every_field(self, tmp_path):
        _report(tmp_path, status="issues_found", findings=[_FINDING], summary="1 issue")

        report = take_report(tmp_path)
        assert report["status"] == "issues_found"
        assert report["findings"] == [{
            "severity": "high", "file": "app/page.tsx", "line": 12,
            "summary": "Card is used but never imported",
        }]

    def test_taking_a_report_consumes_it(self, tmp_path):
        _report(tmp_path, status="clean", summary="ok")

        assert take_report(tmp_path) is not None
        assert take_report(tmp_path) is None, "a stale report must not leak into the next phase"

    def test_reset_clears_a_previous_report(self, tmp_path):
        _report(tmp_path, status="clean", summary="ok")
        reset_report(tmp_path)

        assert take_report(tmp_path) is None

    def test_reports_are_scoped_per_workspace(self, tmp_path):
        one, two = tmp_path / "one", tmp_path / "two"
        one.mkdir()
        two.mkdir()
        _report(one, status="clean", summary="ok")

        assert take_report(two) is None
        assert take_report(one) is not None

    def test_clean_status_with_findings_is_treated_as_dirty(self, tmp_path):
        """A contradictory report must not drop a real problem."""
        _report(tmp_path, status="clean", findings=[_FINDING], summary="mislabelled")

        assert take_report(tmp_path)["status"] == "issues_found"

    def test_missing_line_defaults_to_zero(self, tmp_path):
        _report(tmp_path, status="issues_found", summary="x", findings=[
            {"severity": "low", "file": "a.ts", "summary": "nit"},
        ])

        assert take_report(tmp_path)["findings"][0]["line"] == 0

    @pytest.mark.parametrize("status", ["ok", "CLEAN", "", "passed", "issues"])
    def test_invalid_status_is_rejected(self, tmp_path, status):
        result = _report(tmp_path, status=status, summary="x")

        assert "ERROR" in result
        assert take_report(tmp_path) is None

    def test_invalid_severity_is_rejected(self, tmp_path):
        result = _report(tmp_path, status="issues_found", summary="x", findings=[
            {"severity": "blocker", "file": "a.ts", "summary": "boom"},
        ])

        assert "ERROR" in result and "severity" in result

    def test_finding_without_a_summary_is_rejected(self, tmp_path):
        result = _report(tmp_path, status="issues_found", summary="x", findings=[
            {"severity": "high", "file": "a.ts", "summary": "   "},
        ])

        assert "ERROR" in result

    def test_findings_must_be_a_list(self, tmp_path):
        result = _report(tmp_path, status="issues_found", summary="x", findings="nope")

        assert "ERROR" in result

    def test_the_log_reports_the_count(self, tmp_path):
        events: list[tuple[str, str]] = []
        execute_tool(
            "report_findings",
            {"status": "issues_found", "findings": [_FINDING], "summary": "x"},
            tmp_path,
            lambda k, m: events.append((k, m)),
        )

        assert any("1 issue(s) found" in m for _, m in events)


# ── the gate ──────────────────────────────────────────────────────────────────


class TestFixPassGate:
    def test_clean_report_skips_the_fix_pass(self):
        assert _needs_fix_pass({"status": "clean", "findings": []}) is False

    def test_issues_found_triggers_it(self):
        assert _needs_fix_pass({"status": "issues_found", "findings": [_FINDING]}) is True

    def test_a_phase_that_never_reported_is_treated_as_unresolved(self):
        report = missing_report("validate")

        assert _needs_fix_pass(report) is True
        assert report["reported"] is False
        assert "did not" in report["summary"] or "without calling" in report["findings"][0]["summary"]

    @pytest.mark.parametrize("summary", [
        "Validation complete — no issues found.",
        "The implementation is correct and compiles cleanly.",
        "No problems were found in the changes.",
        "✅ VALIDATED: clean",
        "Everything looks good — DONE.",
        "",
        "Done.",
    ])
    def test_prose_that_used_to_trigger_a_false_fix_pass_no_longer_does(self, summary):
        """Each of these fired the 100-iteration pass under the string-prefix gate.

        The status field is now what decides, so how the model words its sign-off
        is irrelevant.
        """
        assert _needs_fix_pass({"status": "clean", "findings": [], "summary": summary}) is False


# ── the fix prompt ────────────────────────────────────────────────────────────


class TestFixPrompt:
    def test_findings_are_numbered_with_file_and_line(self):
        prompt = _make_fix_prompt({
            "status": "issues_found", "summary": "2 issues",
            "findings": [
                _FINDING,
                {"severity": "low", "file": "lib/api.ts", "line": 3, "summary": "unused var"},
            ],
        })

        assert "1. [high] app/page.tsx:12 — Card is used but never imported" in prompt
        assert "2. [low] lib/api.ts:3 — unused var" in prompt

    def test_severity_orders_the_list(self):
        rendered = _format_findings([
            {"severity": "low", "file": "a", "line": 0, "summary": "nit"},
            {"severity": "critical", "file": "b", "line": 0, "summary": "crash"},
            {"severity": "medium", "file": "c", "line": 0, "summary": "meh"},
        ])

        assert rendered.index("crash") < rendered.index("meh") < rendered.index("nit")

    def test_a_line_of_zero_is_omitted(self):
        rendered = _format_findings([
            {"severity": "high", "file": "a.ts", "line": 0, "summary": "boom"},
        ])

        assert "a.ts —" in rendered
        assert "a.ts:0" not in rendered

    def test_missing_file_is_labelled_not_blank(self):
        rendered = _format_findings([
            {"severity": "high", "file": "", "line": 0, "summary": "boom"},
        ])

        assert "file not specified" in rendered

    def test_empty_findings_still_produce_a_usable_prompt(self):
        prompt = _make_fix_prompt({"status": "issues_found", "findings": [], "summary": "unclear"})

        assert "no individual findings" in prompt
        assert "unclear" in prompt


class TestReportInstruction:
    def test_it_demands_exactly_one_call(self):
        assert "report_findings" in _REPORT_INSTRUCTION
        assert "exactly once" in _REPORT_INSTRUCTION

    def test_it_says_fixed_issues_are_not_findings(self):
        """Otherwise the validator reports what it repaired and triggers a
        pointless fix pass over already-correct code."""
        assert "NOT a finding" in _REPORT_INSTRUCTION
