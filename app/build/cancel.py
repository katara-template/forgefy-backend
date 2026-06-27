"""Cancellation signal for long-running agent and build tasks.

A lightweight Redis flag: set it to cancel any running task for a project.
The agent loops and build loops check this flag on every iteration and stop
gracefully when they see it.
"""
from __future__ import annotations

import redis as _redis

_TTL = 600  # seconds — auto-expires so stale flags don't block future runs


def _r(redis_url: str) -> _redis.Redis:
    return _redis.Redis.from_url(redis_url, decode_responses=True)


def _key(project_id: str) -> str:
    return f"cancel:{project_id}"


def request_cancel(redis_url: str, project_id: str) -> None:
    """Set the cancellation flag for a project."""
    _r(redis_url).set(_key(project_id), "1", ex=_TTL)


def is_cancelled(redis_url: str, project_id: str) -> bool:
    """Return True if a cancellation has been requested."""
    return bool(_r(redis_url).exists(_key(project_id)))


def clear_cancel(redis_url: str, project_id: str) -> None:
    """Remove the cancellation flag (call after honouring it)."""
    _r(redis_url).delete(_key(project_id))
