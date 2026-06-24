"""Tests for the generate_blueprint Celery task.

Run:
    # this file only
    venv/Scripts/python -m pytest tests/workers/test_blueprint_worker.py -v

    # both blueprint test files together
    venv/Scripts/python -m pytest tests/workers/test_blueprint_worker.py tests/build/test_blueprint_generator.py -v

    # all project tests
    venv/Scripts/python -m pytest -v

Covers: success path, retry logic (countdown values per error type),
soft-time-limit handling, and exhausted-retries → FAILED + blueprintError.

The task is called directly (bypassing the broker) with a mock self so that
retry / failure assertions don't require a running Celery worker or Redis.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from celery.exceptions import Retry, SoftTimeLimitExceeded

SESSION_ID = "11111111-1111-1111-1111-111111111111"
BLUEPRINT_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"

_FAKE_SETTINGS = MagicMock(REDIS_URL="redis://fake:6379/0")


# ── helpers ───────────────────────────────────────────────────────────────────


def _task_self(retries: int = 0, max_retries: int = 2) -> MagicMock:
    """Minimal Celery task-self mock.

    self.retry() raises Retry (as the real Celery does) so pytest.raises(Retry)
    works without a broker.  Call args are still recorded for assertions.
    """
    m = MagicMock()
    m.request.retries = retries
    m.max_retries = max_retries
    m.retry.side_effect = Retry()
    return m


def _run(mock_self: MagicMock, sid: str = SESSION_ID) -> None:
    # generate_blueprint is a Celery Task instance; .run is a bound method that
    # would prepend the real task instance as self.  .__func__ gives us the raw
    # unbound function so we can inject our mock_self without the extra arg.
    from app.workers.blueprint_worker import generate_blueprint
    generate_blueprint.run.__func__(mock_self, sid)  # type: ignore[attr-defined]


# ── success ───────────────────────────────────────────────────────────────────


class TestGenerateBlueprintSuccess:
    def test_publishes_blueprint_ready_to_redis(self):
        mock_conn = MagicMock()

        with (
            patch("app.workers.blueprint_worker._run_aggregation", new=AsyncMock(return_value=BLUEPRINT_ID)),
            patch("app.workers.blueprint_worker.get_settings", return_value=_FAKE_SETTINGS),
            patch("app.workers.blueprint_worker.sync_redis") as mock_redis,
        ):
            mock_redis.from_url.return_value = mock_conn
            _run(_task_self())

        assert mock_conn.publish.called
        channel, payload_str = mock_conn.publish.call_args[0]
        msg = json.loads(payload_str)
        assert msg["type"] == "blueprintReady"
        assert msg["blueprint_id"] == BLUEPRINT_ID
        assert msg["session_id"] == SESSION_ID

    def test_publishes_to_correct_redis_channel(self):
        mock_conn = MagicMock()

        with (
            patch("app.workers.blueprint_worker._run_aggregation", new=AsyncMock(return_value=BLUEPRINT_ID)),
            patch("app.workers.blueprint_worker.get_settings", return_value=_FAKE_SETTINGS),
            patch("app.workers.blueprint_worker.sync_redis") as mock_redis,
        ):
            mock_redis.from_url.return_value = mock_conn
            _run(_task_self())

        channel = mock_conn.publish.call_args[0][0]
        assert channel == f"voxa:session:{SESSION_ID}"

    def test_closes_redis_connection_after_success(self):
        mock_conn = MagicMock()

        with (
            patch("app.workers.blueprint_worker._run_aggregation", new=AsyncMock(return_value=BLUEPRINT_ID)),
            patch("app.workers.blueprint_worker.get_settings", return_value=_FAKE_SETTINGS),
            patch("app.workers.blueprint_worker.sync_redis") as mock_redis,
        ):
            mock_redis.from_url.return_value = mock_conn
            _run(_task_self())

        mock_conn.close.assert_called_once()


# ── retry logic ───────────────────────────────────────────────────────────────


class TestGenerateBlueprintRetry:
    def test_no_transcript_error_uses_countdown_30(self):
        """'No transcript data' errors get a longer retry gap to wait for Celery queue drain."""
        err = ValueError("No transcript data found for session xyz. Nothing was recorded.")

        with (
            patch("app.workers.blueprint_worker._run_aggregation", new=AsyncMock(side_effect=err)),
            patch("app.workers.blueprint_worker.get_settings", return_value=_FAKE_SETTINGS),
        ):
            mock_self = _task_self(retries=0)
            with pytest.raises(Retry):
                _run(mock_self)

        assert mock_self.retry.call_args[1]["countdown"] == 30

    def test_generic_error_uses_countdown_10(self):
        err = RuntimeError("Anthropic API 500")

        with (
            patch("app.workers.blueprint_worker._run_aggregation", new=AsyncMock(side_effect=err)),
            patch("app.workers.blueprint_worker.get_settings", return_value=_FAKE_SETTINGS),
        ):
            mock_self = _task_self(retries=0)
            with pytest.raises(Retry):
                _run(mock_self)

        assert mock_self.retry.call_args[1]["countdown"] == 10

    def test_soft_time_limit_uses_countdown_15(self):
        """SoftTimeLimitExceeded is caught separately and retried with countdown=15."""
        with (
            patch("app.workers.blueprint_worker._run_aggregation", new=AsyncMock(side_effect=SoftTimeLimitExceeded())),
            patch("app.workers.blueprint_worker.get_settings", return_value=_FAKE_SETTINGS),
        ):
            mock_self = _task_self(retries=0)
            with pytest.raises(Retry):
                _run(mock_self)

        assert mock_self.retry.call_args[1]["countdown"] == 15

    def test_retries_still_happen_on_second_attempt(self):
        """First retry (retries=1) must also call self.retry(), not take the failure path."""
        err = ValueError("AI synthesis failed")

        with (
            patch("app.workers.blueprint_worker._run_aggregation", new=AsyncMock(side_effect=err)),
            patch("app.workers.blueprint_worker.get_settings", return_value=_FAKE_SETTINGS),
        ):
            mock_self = _task_self(retries=1)
            with pytest.raises(Retry):
                _run(mock_self)

        mock_self.retry.assert_called_once()

    def test_no_retry_when_max_retries_exhausted(self):
        """After max_retries the task must NOT call self.retry() — it goes to the failure path."""
        mock_conn = MagicMock()
        mock_self = _task_self(retries=2, max_retries=2)

        with (
            patch("app.workers.blueprint_worker._run_aggregation", new=AsyncMock(side_effect=ValueError("err"))),
            patch("app.workers.blueprint_worker._mark_session_failed", new=AsyncMock()),
            patch("app.workers.blueprint_worker.get_settings", return_value=_FAKE_SETTINGS),
            patch("app.workers.blueprint_worker.sync_redis") as mock_redis,
        ):
            mock_redis.from_url.return_value = mock_conn
            _run(mock_self)

        mock_self.retry.assert_not_called()


# ── failure (exhausted retries) ───────────────────────────────────────────────


class TestGenerateBlueprintFailed:
    """Tests for the exhausted-retries path: FAILED state + blueprintError event."""

    def _run_exhausted(self, exc: Exception, mark_side_effect=None):
        """Run the task at max retries; return (mock_mark, mock_redis_conn)."""
        mock_conn = MagicMock()
        mock_mark = AsyncMock(side_effect=mark_side_effect)

        with (
            patch("app.workers.blueprint_worker._run_aggregation", new=AsyncMock(side_effect=exc)),
            patch("app.workers.blueprint_worker._mark_session_failed", new=mock_mark),
            patch("app.workers.blueprint_worker.get_settings", return_value=_FAKE_SETTINGS),
            patch("app.workers.blueprint_worker.sync_redis") as mock_redis,
        ):
            mock_redis.from_url.return_value = mock_conn
            _run(_task_self(retries=2, max_retries=2))

        return mock_mark, mock_conn

    def test_marks_session_failed_in_firestore(self):
        mock_mark, _ = self._run_exhausted(ValueError("blueprint is empty"))
        mock_mark.assert_called_once_with(SESSION_ID)

    def test_publishes_blueprint_error_to_redis(self):
        _, mock_conn = self._run_exhausted(ValueError("blueprint is empty"))

        assert mock_conn.publish.called
        _, payload_str = mock_conn.publish.call_args[0]
        msg = json.loads(payload_str)
        assert msg["type"] == "blueprintError"
        assert msg["session_id"] == SESSION_ID

    def test_blueprint_error_includes_message(self):
        _, mock_conn = self._run_exhausted(ValueError("blueprint is empty"))

        _, payload_str = mock_conn.publish.call_args[0]
        msg = json.loads(payload_str)
        assert "error" in msg
        assert msg["error"]  # non-empty

    def test_closes_redis_after_failure(self):
        _, mock_conn = self._run_exhausted(ValueError("blueprint is empty"))
        mock_conn.close.assert_called_once()

    def test_still_publishes_error_when_mark_failed_raises(self):
        """Even if the Firestore FAILED update itself errors, blueprintError must be published."""
        _, mock_conn = self._run_exhausted(
            exc=ValueError("blueprint empty"),
            mark_side_effect=RuntimeError("Firestore unavailable"),
        )

        assert mock_conn.publish.called
        msg = json.loads(mock_conn.publish.call_args[0][1])
        assert msg["type"] == "blueprintError"

    def test_error_message_sanitised_for_rate_limit(self):
        """A rate-limit error must become a user-friendly message (via sanitize_build_error)."""
        _, mock_conn = self._run_exhausted(RuntimeError("rate limit exceeded"))

        msg = json.loads(mock_conn.publish.call_args[0][1])
        assert "rate" not in msg["error"].lower() or "try again" in msg["error"].lower()
