"""Per-account login lockout, backed by Redis.

Rate limiting in app/core/rate_limit.py is IP-based and trivially bypassed
with rotating IPs/proxies. This tracks failed attempts per email so a
credential-stuffing run against one account gets locked out regardless of
how many source IPs it comes from.

Every Redis call here fails open. The lockout is defense-in-depth layered on
top of the IP rate limiter and the password check itself — neither of which
depends on Redis — so an unreachable cache should cost us that extra layer,
not authentication as a whole. Letting redis-py's ConnectionError escape
instead turned a deleted Redis instance into a 500 on every single login.
"""
import logging

import redis.asyncio as aioredis
from redis.exceptions import RedisError

from app.core.exceptions import RateLimitedError

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 10
_LOCKOUT_SECONDS = 30 * 60


def _key(email: str) -> str:
    return f"login_fail:{email.lower()}"


async def check_not_locked_out(redis: aioredis.Redis, email: str) -> None:
    """Raise RateLimitedError if this email has too many recent failed attempts.

    Skips the check (rather than blocking the login) when Redis is unreachable.
    """
    try:
        attempts = await redis.get(_key(email))
    except RedisError as exc:
        logger.warning("Lockout check skipped — Redis unavailable: %s", exc)
        return

    # Outside the try: a genuine lockout must not be swallowed as a Redis fault.
    if attempts is not None and int(attempts) >= _MAX_ATTEMPTS:
        raise RateLimitedError(
            f"Too many failed login attempts. Try again in {_LOCKOUT_SECONDS // 60} minutes."
        )


async def record_failed_attempt(redis: aioredis.Redis, email: str) -> None:
    """Increment the failed-attempt counter, starting the lockout window on the first failure.

    A dropped increment only costs this attempt's contribution to the lockout;
    the caller still rejects the bad credentials.
    """
    key = _key(email)
    try:
        count = await redis.incr(key)
        if count == 1:
            await redis.expire(key, _LOCKOUT_SECONDS)
    except RedisError as exc:
        logger.warning("Failed-attempt counter not recorded — Redis unavailable: %s", exc)


async def clear_failed_attempts(redis: aioredis.Redis, email: str) -> None:
    """Reset the counter on a successful login.

    A dropped reset leaves a stale counter that expires on its own within
    _LOCKOUT_SECONDS — worth far less than failing a valid login.
    """
    try:
        await redis.delete(_key(email))
    except RedisError as exc:
        logger.warning("Failed-attempt counter not cleared — Redis unavailable: %s", exc)
