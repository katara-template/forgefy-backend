"""Tests for the developer extract API: usage capture, pipeline subsetting, endpoint."""
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import AsyncClient

from app.config import get_settings
from app.core.usage import QuotaOutcome
from app.db.models.api_key import ApiKey
from app.deps import get_api_key
from app.main import app

_FEATURE = {"title": "OAuth login", "description": "SSO support", "priority": "high"}
_QUESTION = {"text": "What about mobile?", "context": "platform"}
_ACTION = {"task": "Deploy", "owner": "Alice", "due": None}

# ── call_claude usage capture ─────────────────────────────────────────────────


class TestCallClaudeUsage:
    def _mock_message(self) -> MagicMock:
        msg = MagicMock()
        msg.content = [MagicMock(text='{"features": []}')]
        msg.usage.input_tokens = 120
        msg.usage.output_tokens = 30
        return msg

    def test_appends_usage_when_list_given(self):
        from app.ai.agents.base import call_claude

        usage: list[dict] = []
        with patch("app.ai.agents.base.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.create.return_value = self._mock_message()
            call_claude("system", "text", api_key="fake", model="m", usage=usage)

        assert usage == [{"input_tokens": 120, "output_tokens": 30}]

    def test_no_usage_list_is_backwards_compatible(self):
        from app.ai.agents.base import call_claude

        with patch("app.ai.agents.base.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.create.return_value = self._mock_message()
            result = call_claude("system", "text", api_key="fake", model="m")

        assert result == {"features": []}


# ── run_extraction: subsetting + usage aggregation ────────────────────────────


def _agent_stub(result: dict, tokens: tuple[int, int] = (100, 25)):
    """A fake agent.run that reports token usage like the real one."""

    def run(transcript, api_key, model, *, usage=None):
        if usage is not None:
            usage.append({"input_tokens": tokens[0], "output_tokens": tokens[1]})
        return result

    return run


class TestRunExtraction:
    def test_runs_only_requested_extractors(self):
        from app.ai.pipeline import run_extraction

        feature_run = MagicMock(side_effect=_agent_stub({"features": [_FEATURE]}))
        question_run = MagicMock(side_effect=_agent_stub({"questions": [_QUESTION]}))

        with (
            patch("app.ai.pipeline.feature_extractor.run", feature_run),
            patch("app.ai.pipeline.question_detector.run", question_run),
        ):
            result = run_extraction("transcript", "fake", "m", extractors=["features"])

        feature_run.assert_called_once()
        question_run.assert_not_called()
        assert [e["sub_state"] for e in result["events"]] == ["FEATURE_FOUND"]

    def test_aggregates_usage_across_agents(self):
        from app.ai.pipeline import run_extraction

        with (
            patch("app.ai.pipeline.feature_extractor.run", _agent_stub({"features": []}, (100, 25))),
            patch("app.ai.pipeline.question_detector.run", _agent_stub({"questions": []}, (80, 15))),
        ):
            result = run_extraction("t", "fake", "m", extractors=["features", "questions"])

        assert result["usage"] == {"input_tokens": 180, "output_tokens": 40}

    def test_failed_agent_reports_error_but_keeps_others(self):
        from app.ai.pipeline import run_extraction

        with (
            patch("app.ai.pipeline.feature_extractor.run", side_effect=RuntimeError("down")),
            patch("app.ai.pipeline.action_item_extractor.run", _agent_stub({"action_items": [_ACTION]})),
        ):
            result = run_extraction("t", "fake", "m", extractors=["features", "action_items"])

        assert [e["sub_state"] for e in result["events"]] == ["ACTION_ITEM_FOUND"]
        assert len(result["errors"]) == 1

    def test_run_pipeline_unchanged_for_worker(self):
        from app.ai.pipeline import run_pipeline

        with (
            patch("app.ai.pipeline.feature_extractor.run", _agent_stub({"features": [_FEATURE]})),
            patch("app.ai.pipeline.question_detector.run", _agent_stub({"questions": []})),
            patch("app.ai.pipeline.conflict_detector.run", _agent_stub({"conflicts": []})),
            patch("app.ai.pipeline.action_item_extractor.run", _agent_stub({"action_items": []})),
        ):
            events = run_pipeline("transcript", "fake", "m")

        assert [e["sub_state"] for e in events] == ["FEATURE_FOUND"]


# ── /api/v1/extract endpoint ──────────────────────────────────────────────────


def _pipeline_result(events: list[dict], usage=(100, 40), errors=None) -> dict:
    return {
        "events": events,
        "usage": {"input_tokens": usage[0], "output_tokens": usage[1]},
        "errors": errors or [],
    }


class TestExtractEndpoint:
    async def test_requires_api_key(self, client: AsyncClient) -> None:
        resp = await client.post("/api/v1/extract", json={"transcript": "hello world"})
        assert resp.status_code == 401

    async def test_standard_extraction_grouped_and_metered(
        self, api_client: AsyncClient, test_user
    ) -> None:
        events = [
            {"sub_state": "FEATURE_FOUND", "payload": _FEATURE},
            {"sub_state": "QUESTION_FOUND", "payload": _QUESTION},
            {"sub_state": "ACTION_ITEM_FOUND", "payload": _ACTION},
        ]
        record = AsyncMock()
        with (
            patch("app.api.v1.extract.run_extraction", return_value=_pipeline_result(events)),
            patch("app.api.v1.extract.record_usage", record),
        ):
            resp = await api_client.post(
                "/api/v1/extract", json={"transcript": "we need oauth login"}
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["model_tier"] == "standard"
        assert body["features"] == [_FEATURE]
        assert body["questions"] == [_QUESTION]
        assert body["action_items"] == [_ACTION]
        assert body["conflicts"] == []
        assert body["usage"] == {"input_tokens": 100, "output_tokens": 40}
        record.assert_awaited_once()
        assert record.await_args[0][1:] == (str(test_user.id), 140)

    async def test_extractor_subset_passed_to_pipeline(
        self, api_client: AsyncClient
    ) -> None:
        events = [
            {"sub_state": "FEATURE_FOUND", "payload": _FEATURE},
            {"sub_state": "QUESTION_FOUND", "payload": _QUESTION},
        ]
        with patch(
            "app.api.v1.extract.run_extraction", return_value=_pipeline_result(events)
        ) as run:
            resp = await api_client.post(
                "/api/v1/extract",
                json={"transcript": "we need oauth login", "extractors": ["features"]},
            )

        assert resp.status_code == 200
        assert run.call_args.kwargs["extractors"] == ("features",)
        body = resp.json()
        assert body["features"] == [_FEATURE]
        assert body["questions"] == []  # not requested → filtered even if present

    async def test_quota_block_returns_402(self, api_client: AsyncClient) -> None:
        blocked = QuotaOutcome("block", "You've used all your tokens", None, "free")
        with patch("app.api.v1.extract.evaluate_quota", AsyncMock(return_value=blocked)):
            resp = await api_client.post(
                "/api/v1/extract", json={"transcript": "hello world again"}
            )
        assert resp.status_code == 402

    async def test_quota_downgrade_serves_economy(self, api_client: AsyncClient) -> None:
        downgraded = QuotaOutcome("downgrade", "switched you to the free model", "Qwen3", "pro")
        events = [{"sub_state": "FEATURE_FOUND", "payload": _FEATURE}]
        record = AsyncMock()
        with (
            patch("app.api.v1.extract.evaluate_quota", AsyncMock(return_value=downgraded)),
            patch("app.api.v1.extract._run_economy", return_value=events),
            patch("app.api.v1.extract.record_usage", record),
        ):
            resp = await api_client.post(
                "/api/v1/extract", json={"transcript": "we need oauth login"}
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["model_tier"] == "economy"
        assert body["features"] == [_FEATURE]
        assert body["usage"] == {"input_tokens": 0, "output_tokens": 0}
        record.assert_not_awaited()  # free model is unmetered

    async def test_explicit_economy_tier_filters_requested_extractors(
        self, api_client: AsyncClient
    ) -> None:
        events = [
            {"sub_state": "FEATURE_FOUND", "payload": _FEATURE},
            {"sub_state": "QUESTION_FOUND", "payload": _QUESTION},
        ]
        with patch("app.api.v1.extract._run_economy", return_value=events):
            resp = await api_client.post(
                "/api/v1/extract",
                json={
                    "transcript": "we need oauth login",
                    "model_tier": "economy",
                    "extractors": ["questions"],
                },
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["model_tier"] == "economy"
        assert body["questions"] == [_QUESTION]
        assert body["features"] == []

    async def test_all_agents_failed_returns_502(self, api_client: AsyncClient) -> None:
        result = _pipeline_result([], usage=(0, 0), errors=["feature_extractor: boom"])
        with patch("app.api.v1.extract.run_extraction", return_value=result):
            resp = await api_client.post(
                "/api/v1/extract", json={"transcript": "hello world again"}
            )
        assert resp.status_code == 502

    async def test_economy_backend_failure_returns_502(self, api_client: AsyncClient) -> None:
        with patch("app.api.v1.extract._run_economy", side_effect=RuntimeError("ollama down")):
            resp = await api_client.post(
                "/api/v1/extract",
                json={"transcript": "hello world again", "model_tier": "economy"},
            )
        assert resp.status_code == 502

    async def test_standard_tier_unconfigured_rejected(
        self, client: AsyncClient, test_api_key: ApiKey
    ) -> None:
        settings = MagicMock(ANTHROPIC_API_KEY="", ANTHROPIC_MODEL="claude-test")
        app.dependency_overrides[get_api_key] = lambda: test_api_key
        app.dependency_overrides[get_settings] = lambda: settings
        resp = await client.post("/api/v1/extract", json={"transcript": "hello world"})
        assert resp.status_code == 422

    async def test_empty_transcript_rejected(self, api_client: AsyncClient) -> None:
        resp = await api_client.post("/api/v1/extract", json={"transcript": "   "})
        assert resp.status_code == 422

    async def test_oversized_transcript_rejected(self, api_client: AsyncClient) -> None:
        resp = await api_client.post("/api/v1/extract", json={"transcript": "x" * 50_001})
        assert resp.status_code == 422

    async def test_empty_extractor_list_rejected(self, api_client: AsyncClient) -> None:
        resp = await api_client.post(
            "/api/v1/extract", json={"transcript": "hello world", "extractors": []}
        )
        assert resp.status_code == 422

    async def test_unknown_extractor_rejected(self, api_client: AsyncClient) -> None:
        resp = await api_client.post(
            "/api/v1/extract", json={"transcript": "hello world", "extractors": ["bugs"]}
        )
        assert resp.status_code == 422
