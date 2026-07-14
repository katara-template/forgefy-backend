"""Dispatch Celery tasks from async endpoints without blocking the event loop.

task.apply_async() talks to the Redis broker over a synchronous socket — called
directly inside an async handler it freezes every other request on that worker
for a full broker round trip. Request-handling code must dispatch through
here; workers and other sync contexts can keep calling apply_async directly.
"""
from functools import partial

import anyio


async def dispatch(task, *args, **kwargs):
    """Awaitable task.apply_async(*args, **kwargs), run in a worker thread."""
    return await anyio.to_thread.run_sync(partial(task.apply_async, *args, **kwargs))
