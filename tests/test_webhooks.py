"""Tests for the Recall.ai webhook handler — payload shape is load-bearing here:
Recall's `transcript.data` event nests bot id under `bot.id` and the words/speaker
under a *second* `data` level (`data.data.words`, `data.data.participant.name`),
not the flatter shape the handler used to assume.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.v1.webhooks import _handle_transcript


def _recall_transcript_payload(bot_id: str, words: list[str], speaker: str | None) -> dict:
    """Build a payload matching Recall's current transcript.data event shape."""
    return {
        "bot": {"id": bot_id},
        "data": {
            "words": [{"text": w} for w in words],
            "language_code": "en",
            "participant": {"id": 1, "name": speaker},
        },
        "realtime_endpoint": {"id": "endpoint-1"},
        "transcript": {"id": "transcript-1"},
        "recording": {"id": "recording-1"},
    }


def _mock_redis(get_return: str | None) -> AsyncMock:
    r = AsyncMock()
    r.get.return_value = get_return
    return r


class TestHandleTranscript:
    async def test_extracts_words_and_speaker_from_current_payload_shape(self) -> None:
        settings = MagicMock(REDIS_URL="redis://test")
        redis_mock = _mock_redis(get_return="session-1")

        with (
            patch("app.api.v1.webhooks.aioredis.from_url", return_value=redis_mock),
            patch("app.workers.extraction_worker.extract_requirements") as mock_extract,
        ):
            data = _recall_transcript_payload("bot-1", ["Hello", "world"], "Alice")
            await _handle_transcript(data, settings)

        redis_mock.get.assert_awaited_once_with("recall:bot:bot-1")
        publish_args = redis_mock.publish.call_args
        assert publish_args[0][0] == "voxa:session:session-1"
        import json

        payload = json.loads(publish_args[0][1])
        assert payload["text"] == "Hello world"
        assert payload["speaker"] == "Alice"
        assert payload["is_final"] is True
        mock_extract.apply_async.assert_called_once_with(
            args=["session-1", "Hello world"], queue="meeting.transcribe"
        )

    async def test_empty_words_does_not_publish(self) -> None:
        settings = MagicMock(REDIS_URL="redis://test")
        redis_mock = _mock_redis(get_return="session-1")

        with patch("app.api.v1.webhooks.aioredis.from_url", return_value=redis_mock):
            data = _recall_transcript_payload("bot-1", [], None)
            await _handle_transcript(data, settings)

        redis_mock.get.assert_not_awaited()
        redis_mock.publish.assert_not_awaited()

    async def test_unknown_bot_id_does_not_publish(self) -> None:
        settings = MagicMock(REDIS_URL="redis://test")
        redis_mock = _mock_redis(get_return=None)

        with patch("app.api.v1.webhooks.aioredis.from_url", return_value=redis_mock):
            data = _recall_transcript_payload("bot-unknown", ["Hi"], "Bob")
            await _handle_transcript(data, settings)

        redis_mock.publish.assert_not_awaited()

    async def test_missing_speaker_name_defaults_to_empty_string(self) -> None:
        settings = MagicMock(REDIS_URL="redis://test")
        redis_mock = _mock_redis(get_return="session-1")

        with (
            patch("app.api.v1.webhooks.aioredis.from_url", return_value=redis_mock),
            patch("app.workers.extraction_worker.extract_requirements"),
        ):
            data = _recall_transcript_payload("bot-1", ["Hi"], None)
            await _handle_transcript(data, settings)

        import json

        payload = json.loads(redis_mock.publish.call_args[0][1])
        assert payload["speaker"] == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
