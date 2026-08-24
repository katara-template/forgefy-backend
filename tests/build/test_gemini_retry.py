"""Gemini retry classification, pacing, and error messaging.

Run:
    venv/Scripts/python -m pytest tests/build/test_gemini_retry.py -v

Regression cover for a solo local user seeing "High demand right now" followed by
a generic failure: the cause was our own unpaced request rate against a free-tier
per-minute limit, not anyone else's load.
"""
from __future__ import annotations

from typing import Any

import pytest
import requests

from app.build import build_agent
from app.build.build_agent import _GEMINI_MAX_RETRIES, _gemini_post_with_retry


@pytest.fixture(autouse=True)
def _reset_pacing(monkeypatch):
    """Pacing is module state; never leak it between tests, and never sleep."""
    monkeypatch.setattr(build_agent, "_gemini_min_interval_s", 0.0)
    monkeypatch.setattr(build_agent, "_gemini_last_request_at", 0.0)
    monkeypatch.setattr(build_agent.time, "sleep", lambda _s: None)


class _Resp:
    def __init__(self, status: int, body: dict[str, Any] | None = None, text: str = ""):
        self.status_code = status
        self.reason = "Too Many Requests" if status == 429 else "Error"
        self._body = body if body is not None else {}
        self.text = text or str(self._body)

    def json(self) -> dict[str, Any]:
        return self._body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code}")


def _rate_limit_body(quota_id: str, retry_delay: str | None = "37s") -> dict[str, Any]:
    details: list[dict[str, Any]] = [{
        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
        "violations": [{"quotaId": quota_id, "quotaMetric": "generate_requests"}],
    }]
    if retry_delay:
        details.append({
            "@type": "type.googleapis.com/google.rpc.RetryInfo",
            "retryDelay": retry_delay,
        })
    return {"error": {
        "code": 429, "status": "RESOURCE_EXHAUSTED",
        "message": "Quota exceeded for quota metric.", "details": details,
    }}


def _post(monkeypatch, responses: list[Any]):
    """Script the transport; returns the log events the call produced."""
    calls = {"n": 0}

    def fake_post(url, params=None, json=None, timeout=None):
        idx = min(calls["n"], len(responses) - 1)
        calls["n"] += 1
        item = responses[idx]
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(requests, "post", fake_post)
    events: list[tuple[str, str]] = []
    return events, calls


class TestRateLimitMessaging:
    def test_per_minute_limit_does_not_blame_demand(self, monkeypatch):
        ok = _Resp(200, {"candidates": []})
        events, _ = _post(monkeypatch, [
            _Resp(429, _rate_limit_body("GenerateRequestsPerMinutePerProjectPerModel-FreeTier")),
            ok,
        ])

        _gemini_post_with_retry(
            "http://x", "k", {}, 60, lambda kind, msg: events.append((kind, msg))
        )

        text = " ".join(m for _, m in events)
        assert "High demand" not in text, "a solo user's own rate limit is not demand"
        assert "rate limit" in text.lower()
        assert "billing" in text.lower(), "the message must say how to fix it"

    def test_server_overload_still_reports_demand(self, monkeypatch):
        events, _ = _post(monkeypatch, [_Resp(503, {}), _Resp(200, {"candidates": []})])

        _gemini_post_with_retry(
            "http://x", "k", {}, 60, lambda kind, msg: events.append((kind, msg))
        )

        assert any("High demand" in m for _, m in events), "5xx really is Google's load"

    def test_connection_error_reports_a_connection_problem(self, monkeypatch):
        events, _ = _post(monkeypatch, [
            requests.exceptions.ConnectionError("no route"),
            _Resp(200, {"candidates": []}),
        ])

        _gemini_post_with_retry(
            "http://x", "k", {}, 60, lambda kind, msg: events.append((kind, msg))
        )

        text = " ".join(m for _, m in events).lower()
        assert "reach" in text or "connection" in text
        assert "High demand" not in " ".join(m for _, m in events)

    def test_the_user_is_told_only_once(self, monkeypatch):
        events, _ = _post(monkeypatch, [
            _Resp(429, _rate_limit_body("PerMinute-FreeTier")),
            _Resp(429, _rate_limit_body("PerMinute-FreeTier")),
            _Resp(200, {"candidates": []}),
        ])

        _gemini_post_with_retry(
            "http://x", "k", {}, 60, lambda kind, msg: events.append((kind, msg))
        )

        assert len(events) == 1, "one notice per call, not one per retry"


class TestPermanentFailures:
    def test_daily_quota_fails_fast_instead_of_retrying(self, monkeypatch):
        _, calls = _post(monkeypatch, [
            _Resp(429, _rate_limit_body("GenerateRequestsPerDayPerProjectPerModel-FreeTier")),
        ])

        with pytest.raises(RuntimeError, match="daily request quota"):
            _gemini_post_with_retry("http://x", "k", {}, 60)

        assert calls["n"] == 1, "a per-day quota will not clear — retrying is waste"

    def test_daily_quota_message_names_the_alternatives(self, monkeypatch):
        _post(monkeypatch, [
            _Resp(429, _rate_limit_body("GenerateRequestsPerDayPerProjectPerModel-FreeTier")),
        ])

        with pytest.raises(RuntimeError) as exc:
            _gemini_post_with_retry("http://x", "k", {}, 60)

        assert "BUILD_MODEL" in str(exc.value)

    def test_depleted_credits_still_fails_fast(self, monkeypatch):
        body = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                          "message": "You exceeded your current credit balance."}}
        _, calls = _post(monkeypatch, [_Resp(429, body, text="RESOURCE_EXHAUSTED credit")])

        with pytest.raises(RuntimeError, match="credits are depleted"):
            _gemini_post_with_retry("http://x", "k", {}, 60)

        assert calls["n"] == 1

    def test_a_per_minute_limit_is_not_mistaken_for_a_daily_one(self, monkeypatch):
        _, calls = _post(monkeypatch, [
            _Resp(429, _rate_limit_body("GenerateRequestsPerMinutePerProjectPerModel-FreeTier")),
            _Resp(200, {"candidates": []}),
        ])

        _gemini_post_with_retry("http://x", "k", {}, 60)

        assert calls["n"] == 2, "a per-minute limit must be retried, not fatal"


class TestPacing:
    def test_a_rate_limit_starts_client_side_pacing(self, monkeypatch):
        _post(monkeypatch, [
            _Resp(429, _rate_limit_body("PerMinute-FreeTier", retry_delay="12s")),
            _Resp(200, {"candidates": []}),
        ])

        _gemini_post_with_retry("http://x", "k", {}, 60)

        assert build_agent._gemini_min_interval_s == pytest.approx(12.0)

    def test_pacing_falls_back_to_the_free_tier_rate_without_retry_info(self, monkeypatch):
        _post(monkeypatch, [
            _Resp(429, _rate_limit_body("PerMinute-FreeTier", retry_delay=None)),
            _Resp(200, {"candidates": []}),
        ])

        _gemini_post_with_retry("http://x", "k", {}, 60)

        # ~10 requests/minute.
        assert build_agent._gemini_min_interval_s == pytest.approx(6.0)

    def test_pacing_is_capped(self, monkeypatch):
        _post(monkeypatch, [
            _Resp(429, _rate_limit_body("PerMinute-FreeTier", retry_delay="9999s")),
            _Resp(200, {"candidates": []}),
        ])

        _gemini_post_with_retry("http://x", "k", {}, 60)

        assert build_agent._gemini_min_interval_s <= build_agent._GEMINI_MAX_INTERVAL_S

    def test_a_healthy_key_never_pays_for_pacing(self, monkeypatch):
        _post(monkeypatch, [_Resp(200, {"candidates": []})])

        _gemini_post_with_retry("http://x", "k", {}, 60)

        assert build_agent._gemini_min_interval_s == 0.0

    def test_pacing_decays_as_requests_succeed(self, monkeypatch):
        monkeypatch.setattr(build_agent, "_gemini_min_interval_s", 10.0)
        _post(monkeypatch, [_Resp(200, {"candidates": []})])

        _gemini_post_with_retry("http://x", "k", {}, 60)

        assert build_agent._gemini_min_interval_s < 10.0


class TestRetryDelayIsHonoured:
    def test_google_supplied_delay_overrides_the_guess(self, monkeypatch):
        slept: list[float] = []
        monkeypatch.setattr(build_agent.time, "sleep", lambda s: slept.append(s))
        _post(monkeypatch, [
            _Resp(429, _rate_limit_body("PerMinute-FreeTier", retry_delay="45s")),
            _Resp(200, {"candidates": []}),
        ])

        _gemini_post_with_retry("http://x", "k", {}, 60)

        assert any(s >= 45.0 for s in slept), "ignored Google's own retryDelay"

    def test_exhausting_retries_surfaces_googles_message(self, monkeypatch):
        _post(monkeypatch, [
            _Resp(429, _rate_limit_body("PerMinute-FreeTier")),
        ] * (_GEMINI_MAX_RETRIES + 1))

        with pytest.raises(RuntimeError) as exc:
            _gemini_post_with_retry("http://x", "k", {}, 60)

        assert "Quota exceeded" in str(exc.value), "the real cause must reach the log"
