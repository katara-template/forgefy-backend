"""Agent pipeline and extraction worker tests.

All Anthropic and Redis I/O is mocked — no live API calls.
"""
from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock, patch

import pytest

# ── Agent base tests ──────────────────────────────────────────────────────────


class TestCallClaude:
    def test_returns_parsed_json(self):
        from app.ai.agents.base import call_claude

        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text='{"features": []}')]

        with patch("app.ai.agents.base.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.create.return_value = mock_msg
            result = call_claude("system", "user text", api_key="fake", model="claude-sonnet-4-5")

        assert result == {"features": []}

    def test_raises_on_non_json_response(self):
        from app.ai.agents.base import call_claude

        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="This is not JSON")]

        with (
            patch("app.ai.agents.base.anthropic.Anthropic") as mock_cls,
            pytest.raises(ValueError),
        ):
            mock_cls.return_value.messages.create.return_value = mock_msg
            call_claude("system", "user text", api_key="fake", model="claude-sonnet-4-5")


# ── Individual agent tests ────────────────────────────────────────────────────


class TestAgents:
    def test_feature_extractor_returns_features(self):
        from app.ai.agents import feature_extractor

        expected = {"features": [{"title": "Login", "description": "User login", "priority": "high"}]}
        with patch("app.ai.agents.feature_extractor.call_claude", return_value=expected):
            result = feature_extractor.run("some transcript", "fake", "model")
        assert result == expected

    def test_question_detector_returns_questions(self):
        from app.ai.agents import question_detector

        expected = {"questions": [{"text": "Who owns this?", "context": "auth"}]}
        with patch("app.ai.agents.question_detector.call_claude", return_value=expected):
            result = question_detector.run("transcript", "fake", "model")
        assert result == expected

    def test_conflict_detector_returns_conflicts(self):
        from app.ai.agents import conflict_detector

        expected = {"conflicts": [{"description": "conflict", "side_a": "A", "side_b": "B"}]}
        with patch("app.ai.agents.conflict_detector.call_claude", return_value=expected):
            result = conflict_detector.run("transcript", "fake", "model")
        assert result == expected

    def test_action_item_extractor_returns_items(self):
        from app.ai.agents import action_item_extractor

        expected = {"action_items": [{"task": "Deploy", "owner": "Alice", "due": None}]}
        with patch("app.ai.agents.action_item_extractor.call_claude", return_value=expected):
            result = action_item_extractor.run("transcript", "fake", "model")
        assert result == expected


# ── Pipeline integration tests ────────────────────────────────────────────────


class TestPipeline:
    def _patch_agents(self, features=None, questions=None, conflicts=None, actions=None):
        return [
            patch("app.ai.pipeline.feature_extractor.run", return_value={"features": features or []}),
            patch("app.ai.pipeline.question_detector.run", return_value={"questions": questions or []}),
            patch("app.ai.pipeline.conflict_detector.run", return_value={"conflicts": conflicts or []}),
            patch("app.ai.pipeline.action_item_extractor.run", return_value={"action_items": actions or []}),
        ]

    def test_run_pipeline_returns_events(self):
        from app.ai.pipeline import run_pipeline

        feature = {"title": "OAuth login", "description": "SSO support", "priority": "high"}
        question = {"text": "What about mobile?", "context": "platform"}

        patches = self._patch_agents(features=[feature], questions=[question])
        for p in patches:
            p.start()

        try:
            events = run_pipeline("some transcript", "fake", "model")
        finally:
            for p in patches:
                p.stop()

        sub_states = [e["sub_state"] for e in events]
        assert "FEATURE_FOUND" in sub_states
        assert "QUESTION_FOUND" in sub_states

        feature_events = [e for e in events if e["sub_state"] == "FEATURE_FOUND"]
        assert feature_events[0]["payload"]["title"] == "OAuth login"

    def test_run_pipeline_empty_returns_empty_list(self):
        from app.ai.pipeline import run_pipeline

        patches = self._patch_agents()
        for p in patches:
            p.start()
        try:
            events = run_pipeline("quiet transcript", "fake", "model")
        finally:
            for p in patches:
                p.stop()

        assert events == []

    def test_node_error_does_not_abort_pipeline(self):
        """A failing node should log and continue; other nodes still run."""
        from app.ai.pipeline import run_pipeline

        action = {"task": "Fix it", "owner": "Bob", "due": None}
        patches = [
            patch("app.ai.pipeline.feature_extractor.run", side_effect=RuntimeError("API down")),
            patch("app.ai.pipeline.question_detector.run", return_value={"questions": []}),
            patch("app.ai.pipeline.conflict_detector.run", return_value={"conflicts": []}),
            patch("app.ai.pipeline.action_item_extractor.run", return_value={"action_items": [action]}),
        ]
        for p in patches:
            p.start()
        try:
            events = run_pipeline("transcript", "fake", "model")
        finally:
            for p in patches:
                p.stop()

        sub_states = [e["sub_state"] for e in events]
        assert "ACTION_ITEM_FOUND" in sub_states
        assert "FEATURE_FOUND" not in sub_states

    def test_all_four_event_types_present(self):
        from app.ai.pipeline import run_pipeline

        patches = self._patch_agents(
            features=[{"title": "F", "description": "d", "priority": "low"}],
            questions=[{"text": "Q?", "context": "c"}],
            conflicts=[{"description": "C", "side_a": "A", "side_b": "B"}],
            actions=[{"task": "T", "owner": "O", "due": None}],
        )
        for p in patches:
            p.start()
        try:
            events = run_pipeline("full transcript", "fake", "model")
        finally:
            for p in patches:
                p.stop()

        sub_states = {e["sub_state"] for e in events}
        assert sub_states == {"FEATURE_FOUND", "QUESTION_FOUND", "CONFLICT_FOUND", "ACTION_ITEM_FOUND"}


# ── Extraction worker tests ───────────────────────────────────────────────────


class TestExtractionWorker:
    def test_publishes_events_to_redis(self):
        from app.workers.extraction_worker import extract_requirements

        session_id = str(uuid.uuid4())
        events = [
            {"sub_state": "FEATURE_FOUND", "payload": {"title": "X", "description": "Y", "priority": "high"}},
        ]
        mock_redis = MagicMock()

        with (
            patch("app.workers.extraction_worker._run_extraction", return_value=events),
            patch("app.workers.extraction_worker.sync_redis.from_url", return_value=mock_redis),
        ):
            extract_requirements(session_id, "some transcript")

        mock_redis.publish.assert_called_once()
        channel, payload_str = mock_redis.publish.call_args[0]
        assert channel == f"voxa:session:{session_id}"
        data = json.loads(payload_str)
        assert data["type"] == "featureDetected"
        assert data["sub_state"] == "FEATURE_FOUND"
        mock_redis.close.assert_called_once()

    def test_no_events_skips_redis(self):
        from app.workers.extraction_worker import extract_requirements

        session_id = str(uuid.uuid4())
        mock_redis = MagicMock()

        with (
            patch("app.workers.extraction_worker._run_extraction", return_value=[]),
            patch("app.workers.extraction_worker.sync_redis.from_url", return_value=mock_redis),
        ):
            extract_requirements(session_id, "silent meeting")

        mock_redis.publish.assert_not_called()
