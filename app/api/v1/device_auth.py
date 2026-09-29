"""Browser device-code login for the `forgefy` CLI (`forgefy login`).

Same shape as GitHub/Google CLI device flows:

  1. POST /auth/device/start (no auth) — CLI gets a `device_code` (its own
     secret, long and opaque — never shown to the person) and a short
     `user_code` to display.
  2. The CLI opens `verification_uri_complete` in a browser. The person is
     already logged into the web app there; they approve, which calls
     POST /auth/device/approve (JWT-authed) with just the `user_code`.
  3. Approving mints a normal dashboard API key (same api_keys/{id} shape
     and MAX_ACTIVE_KEYS_PER_USER cap as app/api/v1/keys.py) and stashes the
     raw key on the pending device_logins doc.
  4. The CLI has been polling POST /auth/device/poll (no auth — `device_code`
     itself is the bearer secret) every `interval` seconds. The first poll to
     see "approved" receives the raw key and the doc is deleted immediately
     — the key is delivered exactly once, never persisted here longer than
     necessary (it already lives on, and is retrievable from, the api_keys
     collection like any other key).

Firestore collections used:
  device_logins/{device_code} — user_code, status, owner_user_id, api_key
                                 (raw, cleared on delivery), key_id,
                                 created_at, expires_at
"""
import logging
import secrets
import uuid
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Request

from app.core.api_keys import (
    MAX_ACTIVE_KEYS_PER_USER,
    count_active_api_keys,
    display_prefix,
    generate_api_key,
    hash_api_key,
)
from app.core.exceptions import NotFoundError, ValidationError
from app.core.rate_limit import limiter
from app.deps import CurrentUser, DBSession, SettingsDep
from app.schemas.device_auth import (
    DeviceApproveRequest,
    DevicePollRequest,
    DevicePollResponse,
    DeviceStartResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()

_EXPIRES_IN_SECONDS = 600  # 10 minutes, same order as GitHub's device flow
_POLL_INTERVAL_SECONDS = 5
# Excludes 0/O/1/I so a person reading the code aloud or typing it can't
# confuse characters.
_USER_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def _generate_user_code() -> str:
    chars = [secrets.choice(_USER_CODE_ALPHABET) for _ in range(8)]
    return "".join(chars[:4]) + "-" + "".join(chars[4:])


@router.post("/start", response_model=DeviceStartResponse)
@limiter.limit("20/minute")
async def start_device_login(
    request: Request, db: DBSession, settings: SettingsDep
) -> DeviceStartResponse:
    """Begin a device login; the CLI polls /poll with the returned device_code."""
    device_code = secrets.token_urlsafe(32)
    user_code = _generate_user_code()
    now = datetime.now(UTC)

    await db.collection("device_logins").document(device_code).set({
        "user_code": user_code,
        "status": "pending",
        "owner_user_id": None,
        "api_key": None,
        "key_id": None,
        "created_at": now,
        "expires_at": now + timedelta(seconds=_EXPIRES_IN_SECONDS),
    })

    base = settings.FRONTEND_URL.rstrip("/")
    return DeviceStartResponse(
        device_code=device_code,
        user_code=user_code,
        verification_uri=f"{base}/cli-auth",
        verification_uri_complete=f"{base}/cli-auth?user_code={user_code}",
        expires_in=_EXPIRES_IN_SECONDS,
        interval=_POLL_INTERVAL_SECONDS,
    )


@router.post("/poll", response_model=DevicePollResponse)
@limiter.limit("30/minute")
async def poll_device_login(
    request: Request, body: DevicePollRequest, db: DBSession
) -> DevicePollResponse:
    """Return the current status; delivers the minted API key exactly once."""
    doc_ref = db.collection("device_logins").document(body.device_code)
    doc = await doc_ref.get()
    if not doc.exists:
        return DevicePollResponse(status="expired")

    data = doc.to_dict() or {}
    if data.get("expires_at", datetime.min.replace(tzinfo=UTC)) < datetime.now(UTC):
        await doc_ref.delete()
        return DevicePollResponse(status="expired")

    status = data.get("status", "pending")
    if status == "approved":
        api_key = data.get("api_key")
        # Delivered exactly once: the doc is gone immediately after, so a
        # repeated poll (e.g. a race between two CLI processes) sees
        # "expired" rather than handing out the key twice.
        await doc_ref.delete()
        if not api_key:
            return DevicePollResponse(status="expired")
        return DevicePollResponse(status="approved", api_key=api_key)
    if status == "denied":
        await doc_ref.delete()
        return DevicePollResponse(status="denied")
    return DevicePollResponse(status="pending")


@router.post("/approve", status_code=204)
@limiter.limit("30/minute")
async def approve_device_login(
    request: Request, body: DeviceApproveRequest, db: DBSession, user: CurrentUser
) -> None:
    """Approve a pending device login as the currently signed-in user.

    Called by the /cli-auth confirmation page, never by the CLI itself.
    """
    user_code = body.user_code.strip().upper()
    candidates = await db.collection("device_logins").where("user_code", "==", user_code).limit(5).get()

    now = datetime.now(UTC)
    match = None
    for candidate in candidates:
        data = candidate.to_dict() or {}
        if data.get("status") == "pending" and data.get("expires_at", now) >= now:
            match = candidate
            break

    if match is None:
        raise NotFoundError("This code has expired or was already used. Run 'forgefy login' again.")

    active = await count_active_api_keys(db, str(user.id))
    if active >= MAX_ACTIVE_KEYS_PER_USER:
        raise ValidationError(
            f"Active API key limit reached ({MAX_ACTIVE_KEYS_PER_USER}). "
            "Revoke an unused key in Developers → API keys first, then try again."
        )

    key = generate_api_key()
    key_id = str(uuid.uuid4())
    await db.collection("api_keys").document(key_id).set({
        "owner_user_id": str(user.id),
        "name": "CLI login",
        "prefix": display_prefix(key),
        "key_hash": hash_api_key(key),
        "created_at": now,
        "last_used_at": None,
        "revoked_at": None,
    })

    await match.reference.set(
        {"status": "approved", "owner_user_id": str(user.id), "api_key": key, "key_id": key_id},
        merge=True,
    )
    logger.info("CLI device login approved: key=%s user=%s", key_id, user.id)
