"""Unit tests for prompt-build helpers (app/workers/prompt_build_worker.py)."""
import pytest

from app.workers.prompt_build_worker import _explicit_name


@pytest.mark.parametrize(
    "prompt, expected",
    [
        ("build an app called FitTrack for logging workouts", "FitTrack"),
        ("make an app named RecipeBox to save family recipes", "RecipeBox"),
        ("a meditation app, call it Zenly, for daily calm", "Zenly"),
        ('an app called "Bloom" that reminds you to water plants', "Bloom"),
        ("name it Pulse — a heart-rate tracker", "Pulse"),
    ],
)
def test_extracts_user_named_app(prompt: str, expected: str) -> None:
    assert _explicit_name(prompt) == expected


@pytest.mark.parametrize(
    "prompt",
    [
        "a fitness app to track my daily workouts and progress",
        "an app for organizing recipes and planning weekly meals",
        "something to help me budget my monthly spending",
    ],
)
def test_no_name_returns_empty(prompt: str) -> None:
    # No explicit name → the caller falls back to the AI name generator.
    assert _explicit_name(prompt) == ""
