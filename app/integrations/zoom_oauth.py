"""Zoom OAuth — link a host's Zoom account so our bot may join their meetings.

Since 2026-03-02 a Meeting SDK app can no longer join meetings hosted on other
Zoom accounts using only its SDK JWT. It must additionally present one of:

  • an OBF ("on behalf of") token — the app joins as itself, as an automated
    participant. This is what a notetaker bot wants.
  • a ZAK token — the app joins *as the user*, carrying their identity.
  • RTMS, a separate Zoom product we do not use.

Both tokens are minted from the host's OAuth access token, so the host must
have installed and authorized the Forgefy app once. Meetings hosted on our own
Zoom account still work without any of this.

Token lifetimes worth knowing:
  access token   1 hour       refreshed here, transparently
  refresh token  long-lived   BUT rotates on every use — see _refresh()
  OBF / ZAK      ~2 hours     minted per meeting, never stored
  local recording join token   120 seconds — minted immediately before the
                               bot launches, never earlier
"""
from __future__ import annotations

import base64
import logging
from datetime import UTC, datetime, timedelta
from urllib.parse import urlencode

import httpx

logger = logging.getLogger(__name__)

_AUTHORIZE_URL = "https://zoom.us/oauth/authorize"
_TOKEN_URL = "https://zoom.us/oauth/token"
_API_BASE = "https://api.zoom.us/v2"

# user:read:token   → OBF tokens (the requirement that started all this)
# user:read         → ZAK tokens and the account identity we display
# meeting:read:local_recording_token → pre-authorize recording, skipping the
#                                      in-meeting consent prompt
_SCOPES = (
    "user:read:token "
    "user:read "
    "meeting:read:local_recording_token"
)

# Refresh a little early: a token that expires mid-request is indistinguishable
# from a revoked one, and the retry would be wasted.
_EXPIRY_SKEW = timedelta(minutes=5)

_TIMEOUT = 30

# Firestore fields on the user document.
FIELD_ACCESS = "zoom_access_token"
FIELD_REFRESH = "zoom_refresh_token"
FIELD_EXPIRES = "zoom_token_expires_at"
FIELD_ACCOUNT = "zoom_account_email"
FIELD_USER_ID = "zoom_user_id"


class ZoomNotLinked(Exception):
    """The user has never authorized Forgefy against their Zoom account."""


class ZoomAuthError(Exception):
    """Zoom rejected a token exchange or refresh."""


# ---------------------------------------------------------------------------
# Install flow
# ---------------------------------------------------------------------------

def redirect_uri(settings) -> str:
    """The callback Zoom sends the authorization code to.

    Must match a redirect URL registered on the Marketplace app exactly,
    including scheme and trailing path.
    """
    return f"{settings.PUBLIC_API_BASE_URL}/api/v1/zoom/callback"


def authorize_url(state: str, settings) -> str:
    """Where to send the host so they can grant Forgefy access."""
    params = urlencode({
        "response_type": "code",
        "client_id": _oauth_client_id(settings),
        "redirect_uri": redirect_uri(settings),
        "scope": _SCOPES,
        "state": state,
    })
    return f"{_AUTHORIZE_URL}?{params}"


async def exchange_code(code: str, settings) -> dict:
    """Trade the authorization code for the initial token pair."""
    return await _token_request(
        {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri(settings),
        },
        settings,
    )


# ---------------------------------------------------------------------------
# Token storage and refresh
# ---------------------------------------------------------------------------

async def store_tokens(db, user_id: str, tokens: dict, settings) -> None:
    """Persist a token pair against the user, encrypted at rest."""
    from app.core.crypto import encrypt

    expires_at = datetime.now(UTC) + timedelta(seconds=int(tokens.get("expires_in", 3600)))

    update = {
        FIELD_ACCESS: encrypt(tokens["access_token"]),
        FIELD_REFRESH: encrypt(tokens["refresh_token"]),
        FIELD_EXPIRES: expires_at,
    }

    # Best effort — a failure to read the profile must not fail the install.
    try:
        profile = await _get_json(
            f"{_API_BASE}/users/me", tokens["access_token"]
        )
        update[FIELD_ACCOUNT] = profile.get("email", "")
        update[FIELD_USER_ID] = profile.get("id", "")
    except Exception as exc:
        logger.warning("Could not read Zoom profile after install: %s", exc)

    await db.collection("users").document(user_id).update(update)


async def unlink(db, user_id: str) -> None:
    """Forget the user's Zoom tokens. Their bots stop joining external meetings."""
    await db.collection("users").document(user_id).update({
        FIELD_ACCESS: "",
        FIELD_REFRESH: "",
        FIELD_EXPIRES: None,
        FIELD_ACCOUNT: "",
        FIELD_USER_ID: "",
    })


async def is_linked(db, user_id: str) -> dict:
    """Return link status for the UI."""
    doc = await db.collection("users").document(user_id).get()
    data = doc.to_dict() if doc.exists else {}
    return {
        "linked": bool(data.get(FIELD_REFRESH)),
        "email": data.get(FIELD_ACCOUNT) or None,
    }


async def get_access_token(db, user_id: str, settings) -> str:
    """Return a currently-valid access token, refreshing if necessary.

    Raises ZoomNotLinked if the user never authorized us, and ZoomAuthError if
    the stored grant is no longer accepted (revoked, or a rotated refresh token
    was lost).
    """
    from app.core.crypto import decrypt

    doc = await db.collection("users").document(user_id).get()
    data = doc.to_dict() if doc.exists else {}

    refresh_token = data.get(FIELD_REFRESH)
    if not refresh_token:
        raise ZoomNotLinked(f"User {user_id} has not linked a Zoom account")

    expires_at = data.get(FIELD_EXPIRES)
    access_token = data.get(FIELD_ACCESS)

    if access_token and expires_at and _still_valid(expires_at):
        return decrypt(access_token)

    return await _refresh(db, user_id, decrypt(refresh_token), settings)


async def _refresh(db, user_id: str, refresh_token: str, settings) -> str:
    """Exchange a refresh token for a new pair and persist it.

    Zoom ROTATES refresh tokens: each refresh invalidates the one used and
    returns a replacement. Losing the replacement means the user must re-install
    the app, so persistence happens before the token is handed back — and a
    Redis lock keeps two concurrent meetings from racing, where the slower
    refresh would present an already-spent token and unlink a working account.
    """
    from app.core.crypto import decrypt, encrypt

    lock = _refresh_lock(user_id, settings)
    with lock:
        # Another worker may have refreshed while we waited for the lock.
        doc = await db.collection("users").document(user_id).get()
        data = doc.to_dict() if doc.exists else {}
        if data.get(FIELD_ACCESS) and _still_valid(data.get(FIELD_EXPIRES)):
            return decrypt(data[FIELD_ACCESS])

        current = data.get(FIELD_REFRESH)
        if current:
            refresh_token = decrypt(current)

        try:
            tokens = await _token_request(
                {"grant_type": "refresh_token", "refresh_token": refresh_token},
                settings,
            )
        except ZoomAuthError:
            logger.warning("Zoom refresh rejected for user=%s — grant revoked", user_id)
            raise

        expires_at = datetime.now(UTC) + timedelta(
            seconds=int(tokens.get("expires_in", 3600))
        )
        await db.collection("users").document(user_id).update({
            FIELD_ACCESS: encrypt(tokens["access_token"]),
            FIELD_REFRESH: encrypt(tokens["refresh_token"]),
            FIELD_EXPIRES: expires_at,
        })
        logger.info("Refreshed Zoom tokens user=%s", user_id)
        return tokens["access_token"]


# ---------------------------------------------------------------------------
# Per-meeting tokens
# ---------------------------------------------------------------------------

async def get_obf_token(access_token: str) -> str:
    """Mint an On-Behalf-Of token so the bot may join an external meeting."""
    data = await _get_json(f"{_API_BASE}/users/me/token?type=onbehalf", access_token)
    return data.get("token", "")


async def get_zak_token(access_token: str) -> str:
    """Mint a ZAK token — the bot would join carrying the host's identity."""
    data = await _get_json(f"{_API_BASE}/users/me/token?type=zak", access_token)
    return data.get("token", "")


async def get_local_recording_token(
    access_token: str,
    meeting_number: str,
    *,
    bypass_waiting_room: bool = False,
) -> str:
    """Pre-authorize recording so the bot need not prompt the host mid-meeting.

    Only valid for ~120 seconds and for one meeting occurrence, so this must be
    called immediately before the bot launches — never cached.
    """
    url = (
        f"{_API_BASE}/meetings/{meeting_number}/jointoken/local_recording"
        f"?bypass_waiting_room={'true' if bypass_waiting_room else 'false'}"
    )
    data = await _get_json(url, access_token)
    return data.get("token", "")


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _oauth_client_id(settings) -> str:
    """OAuth credentials, falling back to the Meeting SDK app's.

    A Marketplace "General App" with the Meeting SDK feature enabled uses one
    Client ID for both, so most setups leave the OAuth-specific settings blank.
    """
    return settings.ZOOM_OAUTH_CLIENT_ID or settings.ZOOM_SDK_CLIENT_ID


def _oauth_client_secret(settings) -> str:
    return settings.ZOOM_OAUTH_CLIENT_SECRET or settings.ZOOM_SDK_CLIENT_SECRET


def _still_valid(expires_at) -> bool:
    if not expires_at:
        return False
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    return datetime.now(UTC) + _EXPIRY_SKEW < expires_at


async def _token_request(form: dict, settings) -> dict:
    """POST to Zoom's token endpoint with HTTP Basic client authentication."""
    client_id = _oauth_client_id(settings)
    client_secret = _oauth_client_secret(settings)
    if not (client_id and client_secret):
        raise ZoomAuthError(
            "Zoom OAuth is not configured — set ZOOM_SDK_CLIENT_ID/SECRET "
            "(or ZOOM_OAUTH_CLIENT_ID/SECRET) in the backend .env"
        )

    basic = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(
            _TOKEN_URL,
            data=form,
            headers={
                "Authorization": f"Basic {basic}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )

    if resp.is_error:
        # Zoom puts the useful part in the body; the status alone is rarely
        # enough to tell a bad code from a revoked grant.
        raise ZoomAuthError(f"Zoom token request failed {resp.status_code}: {resp.text[:300]}")

    tokens = resp.json()
    if "access_token" not in tokens or "refresh_token" not in tokens:
        raise ZoomAuthError(f"Zoom token response missing fields: {sorted(tokens)}")
    return tokens


async def _get_json(url: str, access_token: str) -> dict:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(url, headers={"Authorization": f"Bearer {access_token}"})

    if resp.is_error:
        raise ZoomAuthError(f"Zoom API {url} failed {resp.status_code}: {resp.text[:300]}")
    return resp.json()


def _refresh_lock(user_id: str, settings):
    """Cross-worker lock guarding refresh-token rotation."""
    import redis as sync_redis

    client = sync_redis.from_url(settings.REDIS_URL, decode_responses=True)
    # blocking_timeout bounds the wait so a stuck holder cannot hang a join;
    # timeout expires the lock itself if a worker dies mid-refresh.
    return client.lock(
        f"zoom:oauth:refresh:{user_id}",
        timeout=30,
        blocking_timeout=15,
    )
