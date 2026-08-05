from __future__ import annotations

from unittest.mock import patch

from app.ai.agents import gemini_synthesizer


def test_run_emits_app_name_event_when_gemini_returns_it() -> None:
    with patch.object(
        gemini_synthesizer,
        "call_gemini",
        return_value={
            "app_name": "TaskFlow",
            "app_description": "A task management app",
            "features": [],
            "questions": [],
            "conflicts": [],
            "action_items": [],
        },
    ):
        events = gemini_synthesizer.run("transcript", "api-key", "gemini-model")

    assert {event["sub_state"] for event in events} == {"APP_NAME", "APP_DESCRIPTION"}
    assert any(event["sub_state"] == "APP_NAME" and event["payload"] == {"text": "TaskFlow"} for event in events)


# ── Output-cap behaviour ──────────────────────────────────────────────────────
# A long meeting produces a large JSON document. A cap the response outgrows
# truncates it mid-object, which used to surface as "not valid JSON" and fail
# the whole blueprint.


def _gemini_response(text: str, finish_reason: str = "STOP") -> dict:
    return {
        "candidates": [
            {
                "content": {"parts": [{"text": text}]},
                "finishReason": finish_reason,
            }
        ]
    }


def test_synthesis_sends_no_output_cap() -> None:
    """Whole-transcript synthesis must be uncapped, or long meetings fail."""
    with patch.object(gemini_synthesizer, "call_gemini", return_value={}) as call:
        gemini_synthesizer.run("transcript", "api-key", "gemini-model")

    assert call.call_args.kwargs["max_tokens"] is None


def test_none_max_tokens_omits_the_field_entirely() -> None:
    """Sending maxOutputTokens: null would be rejected — it must be absent."""
    with patch.object(gemini_synthesizer, "requests") as requests_mock:
        requests_mock.post.return_value.json.return_value = _gemini_response("{}")
        requests_mock.exceptions = __import__("requests").exceptions

        gemini_synthesizer.call_gemini(
            "system", "content", "key", "model", max_tokens=None
        )

    config = requests_mock.post.call_args.kwargs["json"]["generationConfig"]
    assert "maxOutputTokens" not in config
    assert config["responseMimeType"] == "application/json"


def test_explicit_max_tokens_is_still_sent() -> None:
    """Sub-steps like app-name generation rely on keeping a tight cap."""
    with patch.object(gemini_synthesizer, "requests") as requests_mock:
        requests_mock.post.return_value.json.return_value = _gemini_response("{}")
        requests_mock.exceptions = __import__("requests").exceptions

        gemini_synthesizer.call_gemini(
            "system", "content", "key", "model", max_tokens=32
        )

    config = requests_mock.post.call_args.kwargs["json"]["generationConfig"]
    assert config["maxOutputTokens"] == 32


def test_truncated_response_reports_the_token_limit() -> None:
    """MAX_TOKENS arrives as a 200 with plausible-looking text, so without an
    explicit check it masquerades as malformed JSON."""
    import pytest

    with patch.object(gemini_synthesizer, "requests") as requests_mock:
        # Truncated mid-object, exactly as a cut-off response arrives.
        requests_mock.post.return_value.json.return_value = _gemini_response(
            '{"features": [{"title": "Refer patient to phar',
            finish_reason="MAX_TOKENS",
        )
        requests_mock.exceptions = __import__("requests").exceptions

        with pytest.raises(ValueError, match="output token limit"):
            gemini_synthesizer.call_gemini(
                "system", "content", "key", "model", max_tokens=2048
            )
