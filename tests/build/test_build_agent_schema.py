"""Tests for the Schema Agent phase: gating, user message, and executor brief.

Run:
    venv/Scripts/python -m pytest tests/build/test_build_agent_schema.py -v

These cover the pure functions only — the phase's tool loop is exercised by the
live build pipeline, not here.
"""
from __future__ import annotations

from app.build.build_agent import (
    _schema_agent_user_msg,
    _should_skip_schema,
    _update_user_msg_with_brief,
)

_ENTITY = {
    "name": "Invoice",
    "description": "A bill sent to a customer",
    "fields": [{"name": "amount", "type": "decimal", "required": True, "notes": ""}],
    "relationships": [{"kind": "belongs_to", "target": "Customer", "notes": ""}],
}


# ── _should_skip_schema ───────────────────────────────────────────────────────


class TestShouldSkipSchema:
    def test_no_plan_runs_the_phase(self):
        """Missing plan must not skip — a missed schema fails at runtime, not build time."""
        assert _should_skip_schema(None, {}) is False

    def test_planner_skip_without_entities_skips(self):
        assert _should_skip_schema({"skip_schema_agent": True}, {}) is True

    def test_planner_skip_is_overridden_by_blueprint_entities(self):
        """Entities in the blueprint are direct evidence something is stored."""
        plan = {"skip_schema_agent": True}
        assert _should_skip_schema(plan, {"entities": [_ENTITY]}) is False

    def test_planner_not_skipping_runs_the_phase(self):
        assert _should_skip_schema({"skip_schema_agent": False}, {}) is False

    def test_empty_entities_list_does_not_override(self):
        assert _should_skip_schema({"skip_schema_agent": True}, {"entities": []}) is True


# ── _schema_agent_user_msg ────────────────────────────────────────────────────


class TestSchemaAgentUserMsg:
    def test_includes_entities_and_count(self):
        msg = _schema_agent_user_msg(
            "Acme", "next", {"entities": [_ENTITY]}, {"summary": "add billing"}, "add invoices"
        )

        assert "1 entities" in msg
        assert "Invoice" in msg
        assert "belongs_to" in msg
        assert "Next.js" in msg
        assert msg.rstrip().endswith("SCHEMA READY:")

    def test_no_entities_tells_agent_not_to_invent(self):
        msg = _schema_agent_user_msg("Acme", "flutter", {}, None, "make the button blue")

        assert "carries no extracted entities" in msg
        assert "write no migration if nothing needs persisting" in msg

    def test_uses_current_request_section_when_present(self):
        prompt = "OLD CONTEXT\nstale stuff\nCURRENT REQUEST\nadd an invoices table"
        msg = _schema_agent_user_msg("Acme", "next", {"entities": [_ENTITY]}, None, prompt)

        assert "add an invoices table" in msg
        assert "stale stuff" not in msg


# ── executor brief plumbing ───────────────────────────────────────────────────


class TestUpdateUserMsgWithBrief:
    def _base_args(self):
        return ("Acme", "next", {"features": []}, "do the thing", None, None)

    def test_schema_brief_is_prepended(self):
        msg = _update_user_msg_with_brief(*self._base_args(), "", "SCHEMA READY: invoices(id, amount)")

        assert "DATABASE SCHEMA BRIEF" in msg
        assert "invoices(id, amount)" in msg

    def test_schema_brief_precedes_design_brief(self):
        msg = _update_user_msg_with_brief(
            *self._base_args(), "DESIGN READY: use AppCard", "SCHEMA READY: invoices"
        )

        assert msg.index("DATABASE SCHEMA BRIEF") < msg.index("DESIGN SYSTEM BRIEF")

    def test_omitted_schema_brief_matches_previous_behaviour(self):
        """The arg defaults to empty so the four non-schema call sites are unchanged."""
        with_default = _update_user_msg_with_brief(*self._base_args(), "DESIGN READY: x")
        explicit_empty = _update_user_msg_with_brief(*self._base_args(), "DESIGN READY: x", "")

        assert with_default == explicit_empty
        assert "DATABASE SCHEMA BRIEF" not in with_default

    def test_both_briefs_empty_returns_bare_base(self):
        msg = _update_user_msg_with_brief(*self._base_args(), "", "")

        assert "DATABASE SCHEMA BRIEF" not in msg
        assert "DESIGN SYSTEM BRIEF" not in msg
