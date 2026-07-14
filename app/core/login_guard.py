"""Per-account login lockout, backed by Redis.

Rate limiting in app/core/rate_limit.py is IP-based and trivially bypassed
with rotating IPs/proxies. This tracks failed attempts per email so a
credential-stuffing run against one account gets locked out regardless of
how many source IPs it comes from.
"""
import redis.asyncio as aioredis

from app.core.exceptions import RateLimitedError

_MAX_ATTEMPTS = 10
_LOCKOUT_SECONDS = 30 * 60


def _key(email: str) -> str:
    return f"login_fail:{email.lower()}"


async def check_not_locked_out(redis: aioredis.Redis, email: str) -> None:
    """Raise RateLimitedError if this email has too many recent failed attempts."""
    attempts = await redis.get(_key(email))
    if attempts is not None and int(attempts) >= _MAX_ATTEMPTS:
        raise RateLimitedError(
            f"Too many failed login attempts. Try again in {_LOCKOUT_SECONDS // 60} minutes."
        )


async def record_failed_attempt(redis: aioredis.Redis, email: str) -> None:
    """Increment the failed-attempt counter, starting the lockout window on the first failure."""
    key = _key(email)
    count = await redis.incr(key)
    if count == 1:
        await redis.expire(key, _LOCKOUT_SECONDS)


async def clear_failed_attempts(redis: aioredis.Redis, email: str) -> None:
    """Reset the counter on a successful login."""
    await redis.delete(_key(email))
