"""Shared FastAPI dependencies."""
import logging
import time
import uuid
import weakref
from contextlib import suppress
from datetime import UTC, datetime
from typing import Annotated

import redis.asyncio as aioredis
from fastapi import Depends, Header, Request
from fastapi.security import OAuth2PasswordBearer
from google.cloud.firestore import AsyncClient

from app.config import Settings, get_settings
from app.core.api_keys import hash_api_key, looks_like_api_key
from app.core.exceptions import ForbiddenError, UnauthorizedError
from app.core.security import decode_access_token
from app.db.models.api_key import ApiKey
from app.db.models.user import User

try:  # optional — set_tag on an uninitialized SDK is a no-op
    import sentry_sdk
except ImportError:  # pragma: no cover
    sentry_sdk = None

logger = logging.getLogger(__name__)

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


# Same tokenUrl as _oauth2, but auto_error=False so a missing/invalid token
# yields None instead of a 401 — for endpoints (like the help assistant) that
# serve both signed-in and anonymous visitors.
_oauth2_optional = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)


async def get_optional_user(
    token: Annotated[str | None, Depends(_oauth2_optional)],
    db: DBSession,
    settings: SettingsDep,
) -> User | None:
    """Resolve the current user if a valid token is present, else None.

    Never raises on auth failure — an anonymous visitor is a valid caller here.
    """
    if not token:
        return None
    try:
        return await get_current_user(token=token, db=db, settings=settings)
    except UnauthorizedError:
        return None


OptionalUser = Annotated[User | None, Depends(get_optional_user)]


async def get_admin_user(user: CurrentUser) -> User:
    """Require the current user to have is_admin set."""
    if not user.is_admin:
        raise ForbiddenError("Admin privileges required")
    return user


AdminUser = Annotated[User, Depends(get_admin_user)]


# ── API key auth (developer API) ──────────────────────────────────────────────

# Same shape and rationale as the user cache above: keep the per-request
# api_keys read off the hot path; revocation takes up to the TTL to be seen.
_API_KEY_CACHE_TTL_SECONDS = 30.0
_API_KEY_CACHE_MAX_ENTRIES = 10_000

_api_key_caches: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _tag_sentry(api_key: ApiKey) -> None:
    """Tag the current Sentry scope so errors are debuggable per customer.

    The prefix is the displayable, non-secret part of the key — never the
    hash, never the key itself.
    """
    if sentry_sdk is not None:
        with suppress(Exception):
            sentry_sdk.set_tag("api_key_prefix", api_key.prefix)


def _api_key_from_doc(doc) -> ApiKey:
    data = doc.to_dict()
    return ApiKey(
        id=uuid.UUID(doc.id),
        owner_user_id=uuid.UUID(data["owner_user_id"]),
        name=data.get("name", ""),
        prefix=data.get("prefix", ""),
        key_hash=data["key_hash"],
        created_at=data["created_at"],
        last_used_at=data.get("last_used_at"),
        revoked_at=data.get("revoked_at"),
    )


async def get_api_key(
    db: DBSession,
    authorization: Annotated[str | None, Header()] = None,
) -> ApiKey:
    """Validate an `Authorization: Bearer fgy_live_…` header and return the ApiKey.

    Machine auth for the developer API — parallel to get_current_user's JWT
    auth for dashboard users. Resolved keys are cached briefly per Firestore
    client; last_used_at is stamped on cache misses only, which throttles the
    extra write to at most once per TTL per process.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise UnauthorizedError("Missing API key")
    token = authorization.removeprefix("Bearer ").strip()
    if not looks_like_api_key(token):
        raise UnauthorizedError("Invalid API key")

    digest = hash_api_key(token)

    cache: dict[str, tuple[ApiKey, float]] | None = _api_key_caches.get(db)
    if cache is None:
        cache = {}
        _api_key_caches[db] = cache
    now = time.monotonic()
    cached = cache.get(digest)
    if cached and now - cached[1] < _API_KEY_CACHE_TTL_SECONDS:
        _tag_sentry(cached[0])
        return cached[0]

    docs = await db.collection("api_keys").where("key_hash", "==", digest).limit(1).get()
    if not docs:
        cache.pop(digest, None)
        raise UnauthorizedError("Invalid API key")

    api_key = _api_key_from_doc(docs[0])
    if api_key.revoked:
        cache.pop(digest, None)
        raise UnauthorizedError("API key has been revoked")

    with suppress(Exception):  # telemetry only — never fail the request over it
        await docs[0].reference.set({"last_used_at": datetime.now(UTC)}, merge=True)

    if len(cache) >= _API_KEY_CACHE_MAX_ENTRIES:
        cache.clear()
    cache[digest] = (api_key, now)
    _tag_sentry(api_key)
    return api_key


ApiKeyDep = Annotated[ApiKey, Depends(get_api_key)]
