"""Tests for the Forgefy Python SDK — all HTTP mocked via httpx.MockTransport."""
import hashlib
import hmac
import json

import httpx
import pytest
from forgefy import (
    AuthenticationError,
    Forgefy,
    JobFailedError,
    JobTimeoutError,
    QuotaExceededError,
    RateLimitError,
    ServerError,
    ValidationError,
    verify_signature,
)

BASE = "https://api.example.com"

EXTRACT_OK = {
    "id": "e1",
    "model_tier": "standard",
    "features": [{"title": "OAuth login", "description": "SSO", "priority": "high"}],
    "questions": [],
    "conflicts": [],
    "action_items": [],
    "usage": {"input_tokens": 100, "output_tokens": 40},
}


def problem(status: int, detail: str) -> httpx.Response:
    return httpx.Response(
        status, json={"type": "about:blank", "title": "err", "status": status, "detail": detail}
    )


def make_client(handler, **kwargs) -> Forgefy:
    transport = httpx.MockTransport(handler)
    return Forgefy(
        "fgy_live_test",
        base_url=BASE,
        retry_delay=0,  # keep retry tests instant
        http_client=httpx.Client(transport=transport),
        **kwargs,
    )


class TestExtract:
    def test_sends_key_and_body_and_parses(self):
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=EXTRACT_OK)

        client = make_client(handler)
        result = client.extract("we need oauth", extractors=["features"], model_tier="standard")

        assert result["features"][0]["title"] == "OAuth login"
        request = seen[0]
        assert str(request.url) == f"{BASE}/api/v1/extract"
        assert request.headers["authorization"] == "Bearer fgy_live_test"
        assert json.loads(request.content) == {
            "transcript": "we need oauth",
            "extractors": ["features"],
            "model_tier": "standard",
        }

    def test_omits_optional_fields(self):
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=EXTRACT_OK)

        make_client(handler).extract("hello world")
        assert json.loads(seen[0].content) == {"transcript": "hello world"}

    @pytest.mark.parametrize(
        ("status", "error_cls"),
        [(401, AuthenticationError), (402, QuotaExceededError), (422, ValidationError)],
    )
    def test_maps_problem_to_typed_errors(self, status, error_cls):
        client = make_client(lambda request: problem(status, "nope"))
        with pytest.raises(error_cls) as exc_info:
            client.extract("t")
        assert exc_info.value.status == status
        assert exc_info.value.detail == "nope"

    def test_retries_429_then_succeeds(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return problem(429, "slow down")
            return httpx.Response(200, json=EXTRACT_OK)

        result = make_client(handler).extract("t")
        assert result["id"] == "e1"
        assert calls["n"] == 2

    def test_gives_up_after_max_retries(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return problem(429, "slow down")

        client = make_client(handler, max_retries=1)
        with pytest.raises(RateLimitError):
            client.extract("t")
        assert calls["n"] == 2  # initial + 1 retry

    def test_does_not_retry_sync_extract_on_5xx(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return problem(502, "all extractors errored")

        with pytest.raises(ServerError):
            make_client(handler).extract("t")
        assert calls["n"] == 1  # a retry would double-bill tokens


class TestJobs:
    def test_create_autogenerates_idempotency_key_and_retries_5xx(self):
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            if len(seen) == 1:
                return problem(502, "queue hiccup")
            return httpx.Response(
                202, json={"job_id": "j1", "status": "queued", "webhook_secret": None}
            )

        job = make_client(handler).jobs.create("long one")

        assert job["job_id"] == "j1"
        keys = [r.headers["idempotency-key"] for r in seen]
        assert keys[0]  # auto-generated
        assert keys[0] == keys[1]  # the retry reuses the same key — no double job

    def test_create_passes_explicit_key_and_webhook(self):
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(
                202, json={"job_id": "j1", "status": "queued", "webhook_secret": "s3cret"}
            )

        job = make_client(handler).jobs.create(
            "t", webhook_url="https://example.com/hook", idempotency_key="import-42"
        )

        assert job["webhook_secret"] == "s3cret"
        assert seen[0].headers["idempotency-key"] == "import-42"
        assert json.loads(seen[0].content)["webhook_url"] == "https://example.com/hook"

    def test_wait_for_polls_until_done(self):
        statuses = iter(["queued", "processing", "done"])

        def handler(request: httpx.Request) -> httpx.Response:
            status = next(statuses)
            return httpx.Response(200, json={
                "job_id": "j1", "status": status, "model_tier": "standard",
                "created_at": "2026-07-16T00:00:00Z",
                "result": {"features": []} if status == "done" else None,
                "error": None,
            })

        job = make_client(handler).jobs.wait_for("j1", poll_interval=0)
        assert job["status"] == "done"

    def test_wait_for_raises_on_failed(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "job_id": "j1", "status": "failed", "model_tier": "standard",
                "created_at": "2026-07-16T00:00:00Z", "result": None,
                "error": "provider down",
            })

        with pytest.raises(JobFailedError) as exc_info:
            make_client(handler).jobs.wait_for("j1", poll_interval=0)
        assert "provider down" in str(exc_info.value)

    def test_wait_for_times_out(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "job_id": "j1", "status": "processing", "model_tier": "standard",
                "created_at": "2026-07-16T00:00:00Z", "result": None, "error": None,
            })

        with pytest.raises(JobTimeoutError):
            make_client(handler).jobs.wait_for("j1", poll_interval=0.001, timeout=0.005)


class TestUsage:
    def test_fetches_quota_snapshot(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert str(request.url) == f"{BASE}/api/v1/usage"
            return httpx.Response(200, json={
                "tier": "starter", "tier_name": "Starter",
                "monthly_tokens": 5_000_000, "tokens_used": 1_200_000,
                "tokens_remaining": 3_800_000, "resets_at": "2026-08-01T00:00:00+00:00",
            })

        usage = make_client(handler).usage()
        assert usage["tokens_remaining"] == 3_800_000


class TestVerifySignature:
    secret = "s3cret"
    body = json.dumps({"type": "extract.job.completed", "job_id": "j1"})
    good = "sha256=" + hmac.new(b"s3cret", body.encode(), hashlib.sha256).hexdigest()

    def test_accepts_valid_signature(self):
        assert verify_signature(self.body, self.good, self.secret) is True

    def test_accepts_bytes_payload_and_bare_hex(self):
        assert verify_signature(self.body.encode(), self.good.removeprefix("sha256="), self.secret) is True

    def test_rejects_tampered_body(self):
        assert verify_signature(self.body + "x", self.good, self.secret) is False

    def test_rejects_wrong_secret(self):
        assert verify_signature(self.body, self.good, "other") is False

    def test_rejects_garbage_headers(self):
        assert verify_signature(self.body, None, self.secret) is False
        assert verify_signature(self.body, "", self.secret) is False
        assert verify_signature(self.body, "sha256=nothex", self.secret) is False
