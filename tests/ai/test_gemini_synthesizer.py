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
