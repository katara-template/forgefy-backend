"""Tests for resuming an interrupted run.

Run:
    venv/Scripts/python -m pytest tests/build/test_resume_state.py -v

The risky behaviour here is substitution: a bare "continue" is replaced by the
original request. Getting that boundary wrong silently runs the wrong task, so
most of these tests are about what must NOT be treated as a continuation.
"""
from __future__ import annotations

import pytest

from app.build import resume_state


def _unfinished(request="Add a checkout flow", reason="step_limit", progress="Built the cart page."):
    return resume_state.record(request, reason, progress)[resume_state.FIELD]


class TestContinueDetection:
    @pytest.mark.parametrize("prompt", [
        "continue", "Continue", "  continue  ", "continue.", "resume",
        "keep going", "carry on", "go on", "finish it", "finish",
        "pick up where you left off",
    ])
    def test_bare_continuations_are_recognised(self, prompt):
        assert resume_state.is_continue_command(prompt)

    @pytest.mark.parametrize("prompt", [
        "continue the checkout flow",
        "resume the payment integration and add Stripe",
        "keep going with the dashboard redesign",
        "finish the login page then add signup",
        "add a settings page",
        "",
    ])
    def test_real_instructions_are_not_continuations(self, prompt):
        """A prompt carrying its own task must never be swallowed."""
        assert not resume_state.is_continue_command(prompt)


class TestRequestResolution:
    def test_bare_continue_resolves_to_the_original_request(self):
        assert resume_state.resolve_request("continue", _unfinished()) == "Add a checkout flow"

    def test_a_new_instruction_is_taken_at_face_value(self):
        """The user may change direction; that must win over the old request."""
        assert resume_state.resolve_request("add dark mode", _unfinished()) == "add dark mode"

    def test_continue_without_pending_work_is_left_alone(self):
        assert resume_state.resolve_request("continue", None) == "continue"


class TestRecordAndClear:
    def test_record_captures_what_is_needed_to_resume(self):
        rec = resume_state.record("Add checkout", "step_limit", "Did the cart")[resume_state.FIELD]
        assert rec["request"] == "Add checkout"
        assert rec["reason"] == "step_limit"
        assert rec["progress"] == "Did the cart"
        assert rec["recorded_at"]

    def test_unknown_reason_falls_back_rather_than_storing_junk(self):
        rec = resume_state.record("x", "banana")[resume_state.FIELD]
        assert rec["reason"] == "error"

    def test_long_values_are_truncated(self):
        """This record is injected into every later prompt — it must stay small."""
        rec = resume_state.record("r" * 5000, "error", "p" * 9000)[resume_state.FIELD]
        assert len(rec["request"]) <= 500
        assert len(rec["progress"]) <= 1200

    def test_clear_nulls_the_field(self):
        assert resume_state.clear() == {resume_state.FIELD: None}

    def test_pending_ignores_empty_or_malformed_records(self):
        assert resume_state.pending({}) is None
        assert resume_state.pending({resume_state.FIELD: None}) is None
        assert resume_state.pending({resume_state.FIELD: "nonsense"}) is None
        assert resume_state.pending({resume_state.FIELD: {"request": ""}}) is None

    def test_pending_returns_a_usable_record(self):
        rec = _unfinished()
        assert resume_state.pending({resume_state.FIELD: rec}) == rec


class TestResumeContext:
    def test_no_pending_work_adds_nothing_to_the_prompt(self):
        assert resume_state.resume_context(None) == ""

    def test_context_states_the_request_the_reason_and_the_progress(self):
        block = resume_state.resume_context(_unfinished())
        assert "Add a checkout flow" in block
        assert "maximum number of steps" in block
        assert "Built the cart page." in block

    def test_context_tells_the_agent_not_to_start_over(self):
        """Without this the agent redoes finished work and burns the budget again."""
        block = resume_state.resume_context(_unfinished()).lower()
        assert "do not start over" in block
        assert "already in this workspace" in block

    def test_handles_a_run_that_reported_no_progress(self):
        block = resume_state.resume_context(_unfinished(progress=""))
        assert "before reporting any progress" in block

    @pytest.mark.parametrize("reason,expected", [
        ("quota", "token budget"),
        ("stopped", "user stopped"),
        ("error", "unexpected error"),
        ("step_limit", "maximum number of steps"),
    ])
    def test_each_reason_reads_naturally(self, reason, expected):
        assert expected in resume_state.resume_context(_unfinished(reason=reason))
