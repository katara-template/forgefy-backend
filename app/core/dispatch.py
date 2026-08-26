"""Dispatch Celery tasks from async endpoints without blocking the event loop.

task.apply_async() talks to the Redis broker over a synchronous socket — called
directly inside an async handler it freezes every other request on that worker
for a full broker round trip. Request-handling code must dispatch through
here; workers and other sync contexts can keep calling apply_async directly.

This is also where build work is admission-controlled. Gating here rather than
at each call site means a new build endpoint cannot forget the check.
"""
from functools import partial

import anyio

_redis_client = None


def _redis():
    """Lazily built async client for the BROKER, pooled internally by redis-py.

    Must be the broker, not REDIS_URL: the build queue is a Celery structure and
    lives in whatever database CELERY_BROKER_URL points at. The defaults put the
    app on /0 and the broker on /1, so probing REDIS_URL would read a key that
    does not exist and report depth 0 forever — admission control that silently
    never fires.
    """
    global _redis_client
    if _redis_client is None:
        import redis.asyncio as aioredis

        from app.config import get_settings

        _redis_client = aioredis.from_url(
            get_settings().CELERY_BROKER_URL, decode_responses=True
        )
    return _redis_client


async def dispatch(task, *args, **kwargs):
    """Awaitable task.apply_async(*args, **kwargs), run in a worker thread.

    Build work is rejected with 429 when the backlog is already too deep — see
    app/core/backpressure.py for why builds specifically cannot be queued
    without bound.
    """
    if kwargs.get("queue") == "build":
        from app.core.backpressure import ensure_build_capacity
        await ensure_build_capacity(_redis())

    return await anyio.to_thread.run_sync(partial(task.apply_async, *args, **kwargs))
