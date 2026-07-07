"""Tests for the extract_requirements Celery task.

Covers the short-fragment guard added after a production bug: Recall's
realtime transcript provider can finalize utterances as short as 2-3 words,
and running full AI extraction on an isolated fragment that small produced
hallucinated junk (the model, given nothing real to describe, drifted into
describing its own extraction-task instructions instead of a product).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.workers.extraction_worker import extract_requirements

SESSION_ID = "11111111-1111-1111-1111-111111111111"


@pytest.fixture(autouse=True)
def _patch_persist():
    with patch(
        "app.workers.extraction_worker._persist_transcript", new=AsyncMock(return_value=None)
    ) as mock_persist:
        yield mock_persist


class TestShortFragmentGuard:
    def test_skips_extraction_for_a_two_word_fragment(self, _patch_persist) -> None:
        with patch("app.workers.extraction_worker._run_extraction") as mock_run:
            extract_requirements(SESSION_ID, "the field")

        mock_run.assert_not_called()
        _patch_persist.assert_awaited_once_with(SESSION_ID, "the field")

    def test_skips_extraction_right_at_the_boundary(self, _patch_persist) -> None:
        with patch("app.workers.extraction_worker._run_extraction") as mock_run:
            extract_requirements(SESSION_ID, "one two three four five")  # 5 words

        mock_run.assert_not_called()

    def test_runs_extraction_once_enough_words_are_present(self, _patch_persist) -> None:
        with (
            patch("app.workers.extraction_worker._run_extraction", return_value=[]) as mock_run,
            patch("app.workers.extraction_worker.get_settings"),
        ):
            extract_requirements(SESSION_ID, "we need a login page for admins")  # 7 words

        mock_run.assert_called_once()

    def test_transcript_is_always_persisted_even_when_extraction_is_skipped(
        self, _patch_persist
    ) -> None:
        with patch("app.workers.extraction_worker._run_extraction") as mock_run:
            extract_requirements(SESSION_ID, "hi")

        _patch_persist.assert_awaited_once()
        mock_run.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
