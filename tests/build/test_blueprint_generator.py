"""Tests for BlueprintAggregator.generate() and its helper methods.

Run:
    # this file only
    venv/Scripts/python -m pytest tests/build/test_blueprint_generator.py -v

    # both blueprint test files together
    venv/Scripts/python -m pytest tests/workers/test_blueprint_worker.py tests/build/test_blueprint_generator.py -v

    # all project tests
    venv/Scripts/python -m pytest -v

All Firestore I/O, state machine transitions, and AI API calls are mocked —
no live Firebase or Anthropic credentials are needed.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.build.blueprint_generator import BlueprintAggregator
from app.db.models.enums import Platform, SessionStatus
from app.db.models.meeting_session import MeetingSession

SESSION_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
USER_ID    = uuid.UUID("22222222-2222-2222-2222-222222222222")

_FAKE_SETTINGS = MagicMock(
    ANTHROPIC_API_KEY="fake-api-key",
    ANTHROPIC_MODEL="claude-test",
    BP_MODEL="claude",
    GEMINI_API_KEY="",
    GEMINI_MODEL="",
    OLLAMA_URL="",
    OLLAMA_MODEL="",
    OLLAMA_TIMEOUT=60,
)


# ── Firestore mock helpers ────────────────────────────────────────────────────


def _session_doc(status: str = "PROCESSING", platform: str = "physical") -> MagicMock:
    doc = MagicMock()
    doc.exists = True
    doc.to_dict.return_value = {
        "user_id": str(USER_ID),
        "status": status,
        "platform": platform,
        "meeting_url": None,
        "start_time": None,
        "end_time": None,
        "created_at": datetime.now(UTC),
    }
    return doc


def _event_doc(event_type: str, payload: dict) -> MagicMock:
    doc = MagicMock()
    doc.to_dict.return_value = {
        "event_type": event_type,
        "payload": payload,
        "timestamp": datetime.now(UTC),
    }
    return doc


def _make_db(session_doc: MagicMock, events: list[MagicMock]) -> MagicMock:
    """Build a Firestore mock that returns the given session doc and events list."""
    db = MagicMock()

    # sessions/SESSION_ID  →  session_doc  (fetched in generate())
    sess_doc_ref = MagicMock()
    sess_doc_ref.get = AsyncMock(return_value=session_doc)
    sess_doc_ref.update = AsyncMock()

    # sessions/SESSION_ID/events (order_by → get)
    events_col = MagicMock()
    events_col.order_by.return_value.get = AsyncMock(return_value=events)
    events_col.document.return_value.set = AsyncMock()
    sess_doc_ref.collection.return_value = events_col

    sessions_col = MagicMock()
    sessions_col.document.return_value = sess_doc_ref

    # blueprints/BLUEPRINT_ID
    bp_doc_ref = MagicMock()
    bp_doc_ref.set = AsyncMock()
    blueprints_col = MagicMock()
    blueprints_col.document.return_value = bp_doc_ref

    def _col_router(name: str):
        return sessions_col if name == "sessions" else blueprints_col

    db.collection.side_effect = _col_router
    return db


def _session_obj(status: SessionStatus = SessionStatus.BLUEPRINT_READY) -> MeetingSession:
    return MeetingSession(
        id=SESSION_ID,
        user_id=USER_ID,
        status=status,
        platform=Platform.PHYSICAL,
        meeting_url=None,
        start_time=None,
        end_time=None,
        created_at=datetime.now(UTC),
    )


# ── aggregator generate() tests ───────────────────────────────────────────────


class TestBlueprintGeneratorGenerate:
    async def test_builds_blueprint_from_extraction_events(self):
        """When FEATURE_FOUND events exist, blueprint JSON must contain those features."""
        feat = _event_doc(
            "FEATURE_FOUND",
            {"title": "User Authentication", "description": "Login system", "priority": "high"},
        )
        db = _make_db(_session_doc(), [feat])

        with (
            patch("app.build.blueprint_generator.MeetingStateMachine") as mock_sm_cls,
            patch.object(BlueprintAggregator, "_infer_app_name", new=AsyncMock(return_value="AuthApp")),
        ):
            mock_sm_cls.return_value.transition = AsyncMock(return_value=_session_obj())
            blueprint = await BlueprintAggregator(db).generate(SESSION_ID)

        assert blueprint.json_output["features"][0]["title"] == "User Authentication"

    async def test_transitions_session_to_blueprint_ready(self):
        """generate() must end with a BLUEPRINT_READY state transition."""
        feat = _event_doc("FEATURE_FOUND", {"title": "F", "description": "d", "priority": "low"})
        db = _make_db(_session_doc(), [feat])

        with (
            patch("app.build.blueprint_generator.MeetingStateMachine") as mock_sm_cls,
            patch.object(BlueprintAggregator, "_infer_app_name", new=AsyncMock(return_value="App")),
        ):
            mock_sm = MagicMock()
            mock_sm.transition = AsyncMock(return_value=_session_obj())
            mock_sm_cls.return_value = mock_sm
            await BlueprintAggregator(db).generate(SESSION_ID)

        mock_sm.transition.assert_called_once()
        _, kwargs = mock_sm.transition.call_args
        # Second positional arg (or new_status kwarg) is BLUEPRINT_READY
        args = mock_sm.transition.call_args[0]
        assert args[1] == SessionStatus.BLUEPRINT_READY

    async def test_falls_back_to_synthesizer_when_no_extraction_events(self):
        """Without FEATURE_FOUND/etc. events, the synthesizer must be called."""
        transcript = _event_doc("transcript.segment", {"text": "We need a login page"})
        db = _make_db(_session_doc(), [transcript])

        synth_events = [
            {"sub_state": "APP_DESCRIPTION", "payload": {"text": "A team collaboration tool"}},
            {"sub_state": "FEATURE_FOUND", "payload": {"title": "Login", "description": "Auth", "priority": "high"}},
        ]

        with (
            patch("app.build.blueprint_generator.MeetingStateMachine") as mock_sm_cls,
            patch("app.config.get_settings", return_value=_FAKE_SETTINGS),
            patch("app.ai.agents.synthesizer.run", return_value=synth_events) as mock_synth,
        ):
            mock_sm_cls.return_value.transition = AsyncMock(return_value=_session_obj())
            blueprint = await BlueprintAggregator(db).generate(SESSION_ID)

        mock_synth.assert_called_once()
        assert blueprint.json_output["app_description"] == "A team collaboration tool"

    async def test_prefers_full_transcript_synthesis_over_noisy_fragment_events(self):
        """Regression test: when BOTH raw transcript segments AND incremental
        per-fragment extraction events exist (the normal case for a real
        Recall/physical meeting), the final blueprint must come from one
        synthesis pass over the full transcript — not from concatenating the
        per-fragment events, which are each extracted from an isolated,
        often too-short chunk and can be noisy or hallucinated junk.
        """
        transcript = _event_doc("transcript.segment", {"text": "We need a login page"})
        junk_feature = _event_doc(
            "FEATURE_FOUND",
            {"title": "Junk", "description": "hallucinated from a 2-word fragment", "priority": "low"},
        )
        db = _make_db(_session_doc(), [transcript, junk_feature])

        synth_events = [
            {"sub_state": "APP_DESCRIPTION", "payload": {"text": "A real product description"}},
            {"sub_state": "FEATURE_FOUND", "payload": {"title": "Login", "description": "Auth", "priority": "high"}},
        ]

        with (
            patch("app.build.blueprint_generator.MeetingStateMachine") as mock_sm_cls,
            patch("app.config.get_settings", return_value=_FAKE_SETTINGS),
            patch("app.ai.agents.synthesizer.run", return_value=synth_events) as mock_synth,
        ):
            mock_sm_cls.return_value.transition = AsyncMock(return_value=_session_obj())
            blueprint = await BlueprintAggregator(db).generate(SESSION_ID)

        mock_synth.assert_called_once()
        assert blueprint.json_output["app_description"] == "A real product description"
        assert blueprint.json_output["features"][0]["title"] == "Login"

    async def test_raises_when_no_transcript_data_at_all(self):
        """Empty events collection → ValueError mentioning 'No transcript data'."""
        db = _make_db(_session_doc(), [])  # no events whatsoever

        with (
            patch("app.build.blueprint_generator.MeetingStateMachine"),
            pytest.raises(ValueError, match="No transcript data"),
        ):
            await BlueprintAggregator(db).generate(SESSION_ID)

    async def test_raises_when_synthesizer_returns_nothing_useful(self):
        """Synthesizer producing no description AND no features → ValueError."""
        transcript = _event_doc("transcript.segment", {"text": "um yeah"})
        db = _make_db(_session_doc(), [transcript])

        with (
            patch("app.build.blueprint_generator.MeetingStateMachine"),
            patch("app.config.get_settings", return_value=_FAKE_SETTINGS),
            patch("app.ai.agents.synthesizer.run", return_value=[]),
            pytest.raises(ValueError, match="No requirements could be extracted"),
        ):
            await BlueprintAggregator(db).generate(SESSION_ID)

    async def test_platform_both_saves_two_blueprints(self):
        """Content with both mobile and web keywords → two blueprint docs written."""
        events = [
            _event_doc("FEATURE_FOUND", {
                "title": "Mobile Feed",
                "description": "A native mobile app for real-time updates",
                "priority": "high",
            }),
            _event_doc("FEATURE_FOUND", {
                "title": "Admin Dashboard",
                "description": "A web dashboard for administrators",
                "priority": "med",
            }),
        ]
        db = _make_db(_session_doc(), events)
        # grab the blueprints doc ref so we can count set() calls
        blueprints_doc_ref = db.collection("blueprints").document.return_value

        with (
            patch("app.build.blueprint_generator.MeetingStateMachine") as mock_sm_cls,
            patch.object(BlueprintAggregator, "_infer_app_name", new=AsyncMock(return_value="DualApp")),
        ):
            mock_sm_cls.return_value.transition = AsyncMock(return_value=_session_obj())
            await BlueprintAggregator(db).generate(SESSION_ID)

        assert blueprints_doc_ref.set.call_count == 2

    async def test_single_web_blueprint_uses_next_template(self):
        """Pure web content → single blueprint with template='next'."""
        events = [
            _event_doc("FEATURE_FOUND", {
                "title": "Dashboard",
                "description": "A web dashboard with analytics",
                "priority": "high",
            }),
        ]
        db = _make_db(_session_doc(), events)

        with (
            patch("app.build.blueprint_generator.MeetingStateMachine") as mock_sm_cls,
            patch.object(BlueprintAggregator, "_infer_app_name", new=AsyncMock(return_value="WebApp")),
        ):
            mock_sm_cls.return_value.transition = AsyncMock(return_value=_session_obj())
            blueprint = await BlueprintAggregator(db).generate(SESSION_ID)

        assert blueprint.json_output["template"] == "next"

    async def test_single_mobile_blueprint_uses_flutter_template(self):
        """Ambiguous / mobile-only content → single blueprint with template='flutter'."""
        events = [
            _event_doc("FEATURE_FOUND", {
                "title": "Push Notifications",
                "description": "Send push notifications to mobile users",
                "priority": "high",
            }),
        ]
        db = _make_db(_session_doc(), events)

        with (
            patch("app.build.blueprint_generator.MeetingStateMachine") as mock_sm_cls,
            patch.object(BlueprintAggregator, "_infer_app_name", new=AsyncMock(return_value="MobileApp")),
        ):
            mock_sm_cls.return_value.transition = AsyncMock(return_value=_session_obj())
            blueprint = await BlueprintAggregator(db).generate(SESSION_ID)

        assert blueprint.json_output["template"] == "flutter"


# ── timeout enforcement tests ─────────────────────────────────────────────────


class TestAnthropicTimeoutEnforcement:
    """Verify the timeout=120.0 fix on methods that were previously hanging."""

    async def test_derive_features_creates_anthropic_client_with_timeout_120(self):
        """_derive_features must pass timeout=120.0 to the Anthropic constructor."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(text='[{"title":"Login","description":"Auth","priority":"high"}]')
        ]

        with (
            patch("app.config.get_settings", return_value=_FAKE_SETTINGS),
            patch("anthropic.Anthropic") as mock_cls,
        ):
            mock_cls.return_value.messages.create.return_value = mock_response
            # Instantiate without __init__ to test only the helper method
            agg = object.__new__(BlueprintAggregator)
            result = await agg._derive_features("A task management app for remote teams")

        mock_cls.assert_called_once_with(api_key="fake-api-key", timeout=120.0)
        assert len(result) == 1
        assert result[0]["title"] == "Login"

    async def test_infer_app_name_creates_anthropic_client_with_timeout_120(self):
        """_infer_app_name must pass timeout=120.0 to the Anthropic constructor."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="TaskFlow")]

        with (
            patch("app.config.get_settings", return_value=_FAKE_SETTINGS),
            patch("anthropic.Anthropic") as mock_cls,
        ):
            mock_cls.return_value.messages.create.return_value = mock_response
            agg = object.__new__(BlueprintAggregator)
            result = await agg._infer_app_name({
                "app_description": "A project management tool",
                "features": [{"title": "Kanban Board"}],
            })

        mock_cls.assert_called_once_with(api_key="fake-api-key", timeout=120.0)
        assert result == "TaskFlow"

    async def test_derive_features_returns_empty_list_on_api_error(self):
        """A failed AI call must return [] gracefully, not propagate the exception."""
        with (
            patch("app.config.get_settings", return_value=_FAKE_SETTINGS),
            patch("anthropic.Anthropic") as mock_cls,
        ):
            mock_cls.return_value.messages.create.side_effect = RuntimeError("API down")
            agg = object.__new__(BlueprintAggregator)
            result = await agg._derive_features("Some description")

        assert result == []

    async def test_infer_app_name_returns_empty_string_on_api_error(self):
        """A failed AI call must return '' gracefully, not propagate the exception."""
        with (
            patch("app.config.get_settings", return_value=_FAKE_SETTINGS),
            patch("anthropic.Anthropic") as mock_cls,
        ):
            mock_cls.return_value.messages.create.side_effect = RuntimeError("Timeout")
            agg = object.__new__(BlueprintAggregator)
            result = await agg._infer_app_name({"app_description": "An app", "features": []})

        assert result == ""


class TestClassifyNeedsDatabase:
    """The gate that decides whether the pipeline should ask for database consent
    (see app/api/v1/blueprints.py's approve_blueprint and app/api/v1/projects.py's
    chat_with_project)."""

    async def test_returns_true_with_reason(self):
        from app.build.blueprint_generator import classify_needs_database

        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(text='{"needs_database": true, "reason": "Tracks user orders."}')
        ]

        with (
            patch("app.config.get_settings", return_value=_FAKE_SETTINGS),
            patch("anthropic.Anthropic") as mock_cls,
        ):
            mock_cls.return_value.messages.create.return_value = mock_response
            needs_db, reason = await classify_needs_database(
                "An e-commerce app", [{"title": "Orders", "description": "Track orders"}]
            )

        assert needs_db is True
        assert reason == "Tracks user orders."

    async def test_returns_false_for_ephemeral_app(self):
        from app.build.blueprint_generator import classify_needs_database

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"needs_database": false, "reason": ""}')]

        with (
            patch("app.config.get_settings", return_value=_FAKE_SETTINGS),
            patch("anthropic.Anthropic") as mock_cls,
        ):
            mock_cls.return_value.messages.create.return_value = mock_response
            needs_db, reason = await classify_needs_database("A simple stopwatch", [])

        assert needs_db is False
        assert reason == ""

    async def test_defaults_to_false_on_api_error(self):
        """A failed classifier call must never block a build — default to no."""
        from app.build.blueprint_generator import classify_needs_database

        with (
            patch("app.config.get_settings", return_value=_FAKE_SETTINGS),
            patch("anthropic.Anthropic") as mock_cls,
        ):
            mock_cls.return_value.messages.create.side_effect = RuntimeError("API down")
            needs_db, reason = await classify_needs_database("An app", [])

        assert needs_db is False
        assert reason == ""

    async def test_defaults_to_false_on_malformed_json(self):
        from app.build.blueprint_generator import classify_needs_database

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="not json at all")]

        with (
            patch("app.config.get_settings", return_value=_FAKE_SETTINGS),
            patch("anthropic.Anthropic") as mock_cls,
        ):
            mock_cls.return_value.messages.create.return_value = mock_response
            needs_db, reason = await classify_needs_database("An app", [])

        assert needs_db is False
        assert reason == ""
