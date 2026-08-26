"""Tests for async extract jobs: endpoints, worker task, webhook delivery, cleanup."""
import hashlib
import hmac
import json
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from google.api_core.exceptions import AlreadyExists
from httpx import AsyncClient

from app.core.usage import QuotaOutcome
from tests.conftest import make_doc_snapshot

_FEATURE = {"title": "OAuth login", "description": "SSO support", "priority": "high"}
_TRANSCRIPT = "we need oauth login and a billing page"


def _job_doc_data(owner: str, **overrides) -> dict:
    data = {
        "owner_user_id": owner,
        "api_key_id": str(uuid.uuid4()),
        "status": "queued",
        "extractors": ["features"],
        "model_tier": "standard",
        "webhook_url": None,
        "webhook_secret": None,
        "idempotency_key": None,
        "created_at": datetime.now(UTC),
        "updated_at": datetime.now(UTC),
        "result": None,
        "error": None,
    }
    data.update(overrides)
    return data


# ── POST /api/v1/extract/jobs ─────────────────────────────────────────────────


class TestCreateExtractJob:
    async def test_creates_job_and_dispatches(
        self, api_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        doc_ref = mock_db.collection.return_value.document.return_value
        dispatch = AsyncMock()
        with patch("app.api.v1.extract.dispatch", dispatch):
            resp = await api_client.post(
                "/api/v1/extract/jobs", json={"transcript": _TRANSCRIPT}
            )

        assert resp.status_code == 202
        body = resp.json()
        assert body["status"] == "queued"
        assert body["webhook_secret"] is None

        stored = doc_ref.create.await_args[0][0]
        assert stored["status"] == "queued"
        assert stored["model_tier"] == "standard"
        assert stored["owner_user_id"] == str(test_user.id)
        # transcript travels in the Celery message, never stored in Firestore
        assert _TRANSCRIPT not in str(stored)

        task_kwargs = dispatch.await_args.kwargs["kwargs"]
        assert task_kwargs["job_id"] == body["job_id"]
        assert task_kwargs["transcript"] == _TRANSCRIPT

    async def test_webhook_secret_returned_and_stored(
        self, api_client: AsyncClient, mock_db: MagicMock
    ) -> None:
        doc_ref = mock_db.collection.return_value.document.return_value
        with patch("app.api.v1.extract.dispatch", AsyncMock()):
            resp = await api_client.post(
                "/api/v1/extract/jobs",
                json={"transcript": _TRANSCRIPT, "webhook_url": "https://example.com/hook"},
            )

        assert resp.status_code == 202
        secret = resp.json()["webhook_secret"]
        assert secret
        assert doc_ref.create.await_args[0][0]["webhook_secret"] == secret

    async def test_idempotent_replay_returns_existing_without_dispatch(
        self, api_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        doc_ref = mock_db.collection.return_value.document.return_value
        doc_ref.create.side_effect = AlreadyExists("exists")
        existing = _job_doc_data(str(test_user.id), status="processing", webhook_secret="s3cret")
        doc_ref.get.return_value = make_doc_snapshot(existing)

        dispatch = AsyncMock()
        with patch("app.api.v1.extract.dispatch", dispatch):
            resp = await api_client.post(
                "/api/v1/extract/jobs",
                json={"transcript": _TRANSCRIPT},
                headers={"Idempotency-Key": "retry-abc"},
            )

        assert resp.status_code == 202
        body = resp.json()
        assert body["status"] == "processing"
        assert body["webhook_secret"] == "s3cret"
        dispatch.assert_not_awaited()

    async def test_same_idempotency_key_derives_same_job_id(self) -> None:
        from app.api.v1.extract import _idempotent_job_id

        owner = str(uuid.uuid4())
        assert _idempotent_job_id(owner, "k1") == _idempotent_job_id(owner, "k1")
        assert _idempotent_job_id(owner, "k1") != _idempotent_job_id(owner, "k2")
        assert _idempotent_job_id(owner, "k1") != _idempotent_job_id(str(uuid.uuid4()), "k1")

    async def test_quota_block_returns_402(self, api_client: AsyncClient) -> None:
        blocked = QuotaOutcome("block", "over budget", None, "free")
        with patch("app.api.v1.extract.evaluate_quota", AsyncMock(return_value=blocked)):
            resp = await api_client.post(
                "/api/v1/extract/jobs", json={"transcript": _TRANSCRIPT}
            )
        assert resp.status_code == 402

    async def test_quota_downgrade_queues_economy(
        self, api_client: AsyncClient, mock_db: MagicMock
    ) -> None:
        doc_ref = mock_db.collection.return_value.document.return_value
        downgraded = QuotaOutcome("downgrade", "free model", "Qwen3", "pro")
        with (
            patch("app.api.v1.extract.evaluate_quota", AsyncMock(return_value=downgraded)),
            patch("app.api.v1.extract.dispatch", AsyncMock()),
        ):
            resp = await api_client.post(
                "/api/v1/extract/jobs", json={"transcript": _TRANSCRIPT}
            )

        assert resp.status_code == 202
        assert doc_ref.create.await_args[0][0]["model_tier"] == "economy"

    async def test_dispatch_failure_marks_job_failed(
        self, api_client: AsyncClient, mock_db: MagicMock
    ) -> None:
        doc_ref = mock_db.collection.return_value.document.return_value
        with patch(
            "app.api.v1.extract.dispatch", AsyncMock(side_effect=RuntimeError("broker down"))
        ):
            resp = await api_client.post(
                "/api/v1/extract/jobs", json={"transcript": _TRANSCRIPT}
            )

        assert resp.status_code == 502
        marked = doc_ref.set.await_args[0][0]
        assert marked["status"] == "failed"

    async def test_http_webhook_rejected(self, api_client: AsyncClient) -> None:
        resp = await api_client.post(
            "/api/v1/extract/jobs",
            json={"transcript": _TRANSCRIPT, "webhook_url": "http://example.com/hook"},
        )
        assert resp.status_code == 422

    async def test_localhost_http_webhook_allowed_for_dev(
        self, api_client: AsyncClient
    ) -> None:
        with patch("app.api.v1.extract.dispatch", AsyncMock()):
            resp = await api_client.post(
                "/api/v1/extract/jobs",
                json={"transcript": _TRANSCRIPT, "webhook_url": "http://localhost:9000/hook"},
            )
        assert resp.status_code == 202

    async def test_oversized_transcript_rejected(self, api_client: AsyncClient) -> None:
        resp = await api_client.post(
            "/api/v1/extract/jobs", json={"transcript": "x" * 200_001}
        )
        assert resp.status_code == 422

    async def test_requires_api_key(self, client: AsyncClient) -> None:
        resp = await client.post("/api/v1/extract/jobs", json={"transcript": _TRANSCRIPT})
        assert resp.status_code == 401


# ── GET /api/v1/extract/jobs/{id} ─────────────────────────────────────────────


class TestGetExtractJob:
    async def test_done_job_returns_result(
        self, api_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        job_id = str(uuid.uuid4())
        result = {"features": [_FEATURE], "usage": {"input_tokens": 10, "output_tokens": 5}}
        data = _job_doc_data(str(test_user.id), status="done", result=result)
        mock_db.collection.return_value.document.return_value.get.return_value = (
            make_doc_snapshot(data, doc_id=job_id)
        )

        resp = await api_client.get(f"/api/v1/extract/jobs/{job_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "done"
        assert body["result"] == result
        assert body["error"] is None

    async def test_missing_job_404(self, api_client: AsyncClient, mock_db: MagicMock) -> None:
        mock_db.collection.return_value.document.return_value.get.return_value = (
            make_doc_snapshot(None)
        )
        resp = await api_client.get(f"/api/v1/extract/jobs/{uuid.uuid4()}")
        assert resp.status_code == 404

    async def test_other_owners_job_404(
        self, api_client: AsyncClient, mock_db: MagicMock
    ) -> None:
        data = _job_doc_data(str(uuid.uuid4()))  # different owner
        mock_db.collection.return_value.document.return_value.get.return_value = (
            make_doc_snapshot(data)
        )
        resp = await api_client.get(f"/api/v1/extract/jobs/{uuid.uuid4()}")
        assert resp.status_code == 404


# ── Worker: process_extract_job ───────────────────────────────────────────────


def _pipeline_result(events, usage=(100, 40), errors=None) -> dict:
    return {
        "events": events,
        "usage": {"input_tokens": usage[0], "output_tokens": usage[1]},
        "errors": errors or [],
    }


class TestProcessExtractJob:
    def test_success_persists_result_and_delivers_webhook(self):
        from app.workers import extract_api_worker as mod

        events = [{"sub_state": "FEATURE_FOUND", "payload": _FEATURE}]
        job_id = str(uuid.uuid4())

        with (
            patch.object(mod, "_update_job") as update,
            patch.object(mod, "run_extraction", return_value=_pipeline_result(events)),
            patch.object(mod, "_run_coro") as run_coro,
            patch.object(mod, "_record_usage_async", MagicMock(return_value=None)),
            patch.object(mod.deliver_extract_webhook, "apply_async") as deliver,
        ):
            mod.process_extract_job(
                job_id, _TRANSCRIPT, ["features"], "standard",
                "owner-1", "https://example.com/hook", "sec",
            )

        statuses = [call.args[1].get("status") for call in update.call_args_list]
        assert statuses == ["processing", "done"]
        result = update.call_args_list[1].args[1]["result"]
        assert result["features"] == [_FEATURE]
        assert result["usage"] == {"input_tokens": 100, "output_tokens": 40}
        run_coro.assert_called_once()  # usage recording
        payload = deliver.call_args.kwargs["args"][2]
        assert payload["type"] == "extract.job.completed"
        assert payload["result"]["features"] == [_FEATURE]

    def test_failure_marks_failed_and_delivers_failure_webhook(self):
        from app.workers import extract_api_worker as mod

        job_id = str(uuid.uuid4())
        with (
            patch.object(mod, "_update_job") as update,
            patch.object(mod, "run_extraction", side_effect=RuntimeError("provider down")),
            patch.object(mod.deliver_extract_webhook, "apply_async") as deliver,
        ):
            mod.process_extract_job(
                job_id, _TRANSCRIPT, ["features"], "standard",
                "owner-1", "https://example.com/hook", "sec",
            )

        final = update.call_args_list[-1].args[1]
        assert final["status"] == "failed"
        assert "provider down" in final["error"]
        payload = deliver.call_args.kwargs["args"][2]
        assert payload["type"] == "extract.job.failed"

    def test_no_webhook_skips_delivery(self):
        from app.workers import extract_api_worker as mod

        with (
            patch.object(mod, "_update_job"),
            patch.object(mod, "run_extraction", return_value=_pipeline_result([])),
            patch.object(mod, "_run_coro"),
            patch.object(mod, "_record_usage_async", MagicMock(return_value=None)),
            patch.object(mod.deliver_extract_webhook, "apply_async") as deliver,
        ):
            mod.process_extract_job(
                str(uuid.uuid4()), _TRANSCRIPT, ["features"], "standard",
                "owner-1", None, None,
            )

        deliver.assert_not_called()

    def test_economy_tier_uses_qwen_and_is_unmetered(self):
        from app.workers import extract_api_worker as mod

        events = [{"sub_state": "FEATURE_FOUND", "payload": _FEATURE}]
        with (
            patch.object(mod, "_update_job") as update,
            patch("app.ai.qwen.run_qwen_synthesis", return_value=events),
            patch.object(mod, "_run_coro") as run_coro,
        ):
            mod.process_extract_job(
                str(uuid.uuid4()), _TRANSCRIPT, ["features"], "economy",
                "owner-1", None, None,
            )

        result = update.call_args_list[-1].args[1]["result"]
        assert result["features"] == [_FEATURE]
        assert result["usage"] == {"input_tokens": 0, "output_tokens": 0}
        run_coro.assert_not_called()  # zero tokens → nothing to meter


# ── Worker: deliver_extract_webhook ───────────────────────────────────────────


class TestDeliverWebhook:
    def test_posts_signed_payload(self):
        from app.workers import extract_api_worker as mod

        payload = {"type": "extract.job.completed", "job_id": "j1", "status": "done"}
        response = MagicMock(status_code=200)
        with patch.object(mod.httpx, "post", return_value=response) as post:
            mod.deliver_extract_webhook("https://example.com/hook", "sec", payload)

        response.raise_for_status.assert_called_once()
        kwargs = post.call_args.kwargs
        body = kwargs["content"]
        expected_sig = hmac.new(b"sec", body.encode(), hashlib.sha256).hexdigest()
        assert kwargs["headers"]["X-Forgefy-Signature"] == f"sha256={expected_sig}"
        assert json.loads(body) == payload


# ── Worker: cleanup_extract_jobs ──────────────────────────────────────────────


class TestCleanupExtractJobs:
    def test_deletes_expired_jobs(self):
        from app.workers import extract_api_worker as mod

        docs = []
        for _ in range(2):
            doc = MagicMock()
            doc.reference.delete = AsyncMock()
            docs.append(doc)

        async def _agen():
            for d in docs:
                yield d

        db = MagicMock()
        db.collection.return_value.where.return_value.stream = MagicMock(
            return_value=_agen()
        )
        with patch("app.db.firebase.refresh_async_firestore_client", return_value=db):
            result = mod.cleanup_extract_jobs()

        assert result == {"deleted": 2}
        for doc in docs:
            doc.reference.delete.assert_awaited_once()
