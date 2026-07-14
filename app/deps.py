"""Shared FastAPI dependencies."""
import time
import uuid
import weakref
from typing import Annotated

import redis.asyncio as aioredis
from fastapi import Depends, Request
from fastapi.security import OAuth2PasswordBearer
from google.cloud.firestore import AsyncClient

from app.config import Settings, get_settings
from app.core.exceptions import ForbiddenError, UnauthorizedError
from app.core.security import decode_access_token
from app.db.models.user import User

# ── Settings ──────────────────────────────────────────────────────────────────

SettingsDep = Annotated[Settings, Depends(get_settings)]


# ── Firestore client ──────────────────────────────────────────────────────────


def get_db(request: Request) -> AsyncClient:
    """Return the app-level Firestore async client."""
    return request.app.state.firestore


DBSession = Annotated[AsyncClient, Depends(get_db)]


# ── Redis client ─────────────────────────────────────────────────────────────


def get_redis(request: Request) -> aioredis.Redis:
    """Return the app-level async Redis client."""
    return request.app.state.redis


RedisDep = Annotated[aioredis.Redis, Depends(get_redis)]


# ── Current user ──────────────────────────────────────────────────────────────

_oauth2 = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


# How long a resolved User may be served from cache before re-reading
# Firestore. Keeps the per-request user-doc read off the hot path; the
# trade-off is that tier/is_admin changes take up to this long to be seen.
_USER_CACHE_TTL_SECONDS = 30.0
# Hard cap so a scan of random-but-valid tokens can't grow the dict unbounded.
_USER_CACHE_MAX_ENTRIES = 10_000

# One cache per Firestore client instance — each worker process (and each
# test's mock client) gets its own, and a cache dies with its client.
_user_caches: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


async def get_current_user(
    token: Annotated[str, Depends(_oauth2)],
    db: DBSession,
    settings: SettingsDep,
) -> User:
    """Validate the Bearer JWT and return the corresponding User document.

    The Firestore user doc is cached briefly per Firestore client (attached to
    the client instance, so each worker process — and each test's mock client —
    gets its own isolated cache).
    """
    user_id_str = decode_access_token(token, settings)
    try:
        uid = uuid.UUID(user_id_str)
    except ValueError as exc:
        raise UnauthorizedError("Token subject is not a valid UUID") from exc

    cache: dict[str, tuple[User, float]] | None = _user_caches.get(db)
    if cache is None:
        cache = {}
        _user_caches[db] = cache
    now = time.monotonic()
    cached = cache.get(str(uid))
    if cached and now - cached[1] < _USER_CACHE_TTL_SECONDS:
        return cached[0]

    doc = await db.collection("users").document(str(uid)).get()
    if not doc.exists:
        cache.pop(str(uid), None)
        raise UnauthorizedError("User not found")

    data = doc.to_dict()
    user = User(
        id=uid,
        email=data["email"],
        hashed_password=data.get("hashed_password", ""),
        created_at=data["created_at"],
        updated_at=data["updated_at"],
        tier=data.get("tier", "free"),
        is_admin=data.get("is_admin", False),
    )
    if len(cache) >= _USER_CACHE_MAX_ENTRIES:
        cache.clear()
    cache[str(uid)] = (user, now)
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


async def get_admin_user(user: CurrentUser) -> User:
    """Require the current user to have is_admin set."""
    if not user.is_admin:
        raise ForbiddenError("Admin privileges required")
    return user


AdminUser = Annotated[User, Depends(get_admin_user)]
