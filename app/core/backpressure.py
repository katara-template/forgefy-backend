"""Queue backpressure for build work.

A build is minutes of LLM inference plus a workspace, so the build queue can
never absorb a traffic spike the way a read endpoint can. Without a bound, a
burst is accepted in full: Redis grows until it OOMs and takes the broker — and
therefore every queue — down with it. The tasks that survive that are ones users
gave up waiting for anyway.

So we bound the queue and reject past it. A 429 with Retry-After is a far better
outcome than a queue position no one will ever reach, and it keeps the failure
in one endpoint instead of spreading to the whole broker.

This is admission control, not rate limiting: slowapi throttles how fast one
caller may ask, this caps how much unstarted work the system holds in total.
"""
from __future__ import annotations

import logging

from app.config import get_settings
from app.core.exceptions import RateLimitedError

logger = logging.getLogger(__name__)

BUILD_QUEUE = "build"


async def build_queue_depth(redis) -> int | None:
    """Number of tasks waiting in the build queue, or None if unknown.

    Celery's Redis transport stores a queue as a list keyed by the queue name,
    so LLEN is the pending count. Running tasks are not in the list — this is
    unstarted backlog, which is exactly what we want to bound.
    """
    try:
        return int(await redis.llen(BUILD_QUEUE))
    except Exception:
        # Never fail a request because the depth probe failed; an unreachable
        # Redis will surface at enqueue time with a clearer error.
        logger.warning("build queue depth probe failed", exc_info=True)
        return None


async def ensure_build_capacity(redis) -> None:
    """Raise RateLimitedError when the build backlog is already too deep.

    Called before enqueuing build work. Deliberately not applied to the meeting
    queues: those tasks are short and bursty, and shedding them loses live audio.
    """
    max_depth = get_settings().BUILD_QUEUE_MAX_DEPTH
    if max_depth <= 0:  # 0 disables admission control
        return

    depth = await build_queue_depth(redis)
    if depth is None or depth < max_depth:
        return

    logger.warning("build queue full: depth=%d max=%d — shedding", depth, max_depth)
    raise RateLimitedError(
        f"The build queue is at capacity ({depth} jobs waiting). "
        "Your request was not lost — please retry shortly."
    )
