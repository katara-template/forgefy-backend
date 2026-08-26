"""Tests for build-queue admission control.

Run:
    venv/Scripts/python -m pytest tests/test_backpressure.py -v

Builds cannot be queued without bound: each is minutes of LLM work, so an
unbounded backlog grows Redis until it OOMs and takes the broker down. Shedding
with 429 keeps the failure inside one endpoint.
"""
from __future__ import annotations

import pytest

from app.core import backpressure
from app.core import dispatch as dispatch_mod
from app.core.exceptions import RateLimitedError


class FakeRedis:
    def __init__(self, depth: int = 0, fail: bool = False):
        self._depth = depth
        self._fail = fail
        self.queried: list[str] = []

    async def llen(self, key: str) -> int:
        self.queried.append(key)
        if self._fail:
            raise ConnectionError("redis down")
        return self._depth


@pytest.fixture
def limit(monkeypatch):
    """Set BUILD_QUEUE_MAX_DEPTH without touching the real settings cache."""
    def _set(value: int):
        from types import SimpleNamespace
        monkeypatch.setattr(
            backpressure, "get_settings",
            lambda: SimpleNamespace(BUILD_QUEUE_MAX_DEPTH=value),
        )
    return _set


class TestCapacityCheck:
    async def test_allows_when_queue_is_shallow(self, limit):
        limit(500)
        await backpressure.ensure_build_capacity(FakeRedis(depth=10))  # no raise

    async def test_sheds_when_queue_is_full(self, limit):
        limit(100)
        with pytest.raises(RateLimitedError) as exc:
            await backpressure.ensure_build_capacity(FakeRedis(depth=100))
        assert exc.value.status_code == 429
        assert "not lost" in exc.value.detail, "message should tell users to retry"

    async def test_sheds_well_past_the_limit(self, limit):
        limit(500)
        with pytest.raises(RateLimitedError):
            await backpressure.ensure_build_capacity(FakeRedis(depth=50_000))

    async def test_zero_disables_the_check(self, limit):
        limit(0)
        redis = FakeRedis(depth=999_999)
        await backpressure.ensure_build_capacity(redis)
        assert not redis.queried, "disabled check should not even probe"

    async def test_unreachable_redis_does_not_block_builds(self, limit):
        """A broken probe must not become an outage of its own."""
        limit(100)
        await backpressure.ensure_build_capacity(FakeRedis(fail=True))  # no raise

    async def test_depth_probe_reads_the_build_queue(self, limit):
        limit(500)
        redis = FakeRedis(depth=1)
        await backpressure.ensure_build_capacity(redis)
        assert redis.queried == [backpressure.BUILD_QUEUE]


class TestDispatchGate:
    async def test_build_queue_is_gated(self, monkeypatch):
        called: list = []
        monkeypatch.setattr(dispatch_mod, "_redis", lambda: FakeRedis(depth=0))
        monkeypatch.setattr(
            "app.core.backpressure.ensure_build_capacity",
            lambda r: called.append(r) or _noop(),
        )

        class Task:
            def apply_async(self, *a, **k):
                return "queued"

        assert await dispatch_mod.dispatch(Task(), queue="build") == "queued"
        assert called, "build dispatch bypassed admission control"

    async def test_other_queues_are_not_gated(self, monkeypatch):
        """Meeting tasks are short and drop live audio if shed — never gate them."""
        called: list = []
        monkeypatch.setattr(
            "app.core.backpressure.ensure_build_capacity",
            lambda r: called.append(r) or _noop(),
        )

        class Task:
            def apply_async(self, *a, **k):
                return "queued"

        await dispatch_mod.dispatch(Task(), queue="meeting.audio")
        assert not called, "meeting work must not be admission-controlled"

    async def test_a_full_queue_stops_the_enqueue(self, monkeypatch):
        enqueued: list = []
        monkeypatch.setattr(dispatch_mod, "_redis", lambda: FakeRedis(depth=10_000))
        from types import SimpleNamespace
        monkeypatch.setattr(
            backpressure, "get_settings",
            lambda: SimpleNamespace(BUILD_QUEUE_MAX_DEPTH=10),
        )

        class Task:
            def apply_async(self, *a, **k):
                enqueued.append(k)

        with pytest.raises(RateLimitedError):
            await dispatch_mod.dispatch(Task(), queue="build")
        assert not enqueued, "task was queued despite being over capacity"


async def _noop():
    return None


class TestProbeTarget:
    def test_probe_connects_to_the_broker_not_the_app_redis(self, monkeypatch):
        """The build queue lives in the broker database, not REDIS_URL.

        With the documented defaults these are different databases (/0 vs /1),
        so probing REDIS_URL reads a key that does not exist and reports depth 0
        forever — the 429 would never fire and the bound would be decorative.
        """
        from types import SimpleNamespace

        captured: dict = {}
        monkeypatch.setattr(dispatch_mod, "_redis_client", None, raising=False)
        monkeypatch.setattr(
            "app.config.get_settings",
            lambda: SimpleNamespace(
                REDIS_URL="redis://localhost:6379/0",
                CELERY_BROKER_URL="redis://localhost:6379/1",
            ),
        )

        import redis.asyncio as aioredis
        monkeypatch.setattr(
            aioredis, "from_url",
            lambda url, **kw: captured.setdefault("url", url) or object(),
        )

        dispatch_mod._redis()
        assert captured["url"].endswith("/1"), (
            f"probe targets {captured['url']!r}; the build queue is in the broker db"
        )
        monkeypatch.setattr(dispatch_mod, "_redis_client", None, raising=False)


class TestCeleryTuning:
    def test_prefetch_is_one_for_long_tasks(self):
        from app.workers.celery_app import celery_app
        assert celery_app.conf.worker_prefetch_multiplier == 1

    def test_visibility_timeout_exceeds_the_longest_build(self):
        """Below the build duration, Redis redelivers and two agents share a workspace."""
        from app.workers.celery_app import celery_app
        vt = celery_app.conf.broker_transport_options["visibility_timeout"]
        assert vt > 3600, "must exceed Celery's 1h Redis default"

    def test_workers_are_recycled(self):
        from app.workers.celery_app import celery_app
        assert celery_app.conf.worker_max_tasks_per_child > 0

    def test_concurrency_comes_from_config(self):
        from app.workers.celery_app import celery_app
        assert celery_app.conf.worker_concurrency > 0

    def test_no_entrypoint_passes_concurrency(self):
        """--concurrency overrides celery_app.conf, so neither may pass it.

        docker-compose.yml and Dockerfile.worker are separate images with
        separate launch commands; a flag in either silently wins over config
        and lets the two drift apart, which is exactly what happened before.
        """
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        for name in ("docker-compose.yml", "Dockerfile.worker", "Dockerfile"):
            path = root / name
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for line in text.splitlines():
                if line.lstrip().startswith("#"):
                    continue  # explanatory comments may name the flag
                assert "--concurrency" not in line, (
                    f"{name} passes --concurrency, which overrides "
                    f"CELERY_CONCURRENCY: {line.strip()}"
                )
