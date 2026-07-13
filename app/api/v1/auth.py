"""Auth endpoints: register, login, refresh, Google OAuth, GitHub OAuth.

All routes are rate-limited to 60 req/min per IP.

Firestore collections used:
  users/{user_id}  — email, hashed_password, created_at, updated_at
"""
import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
import uuid
from datetime import UTC, datetime
from urllib.parse import urlencode

import firebase_admin.auth
import httpx
from fastapi import APIRouter, Query, Request
from fastapi.responses import RedirectResponse
from google.api_core.exceptions import ResourceExhausted

from app.core.exceptions import ConflictError, RateLimitedError, UnauthorizedError
from app.core.login_guard import (
    check_not_locked_out,
    clear_failed_attempts,
    record_failed_attempt,
)
from app.core.rate_limit import limiter
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_refresh_token,
    hash_password,
    verify_password,
)
from app.deps import DBSession, RedisDep, SettingsDep
from app.schemas.auth import (
    GoogleAuthRequest,
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _raise_for_firestore_quota(exc: Exception, *, context: str) -> None:
    """Translate transient Firestore quota errors into a client-visible 429."""
    if isinstance(exc, ResourceExhausted):
        raise RateLimitedError(
            f"The authentication service is temporarily unavailable while {context}. Please try again shortly."
        ) from exc

    message = str(exc).lower()
    if "resource exhausted" in message or "quota exceeded" in message or "quota" in message and "exceeded" in message:
        raise RateLimitedError(
            f"The authentication service is temporarily unavailable while {context}. Please try again shortly."
        ) from exc


@router.post("/register", response_model=TokenResponse, status_code=201)
@limiter.limit("60/minute")
async def register(
    request: Request,
    body: RegisterRequest,
    db: DBSession,
    settings: SettingsDep,
) -> TokenResponse:
    """Create a new user account; return access + refresh tokens."""
    try:
        logger.debug("Register request for email: %s", body.email)
        
        # Check if email already exists
        try:
            existing = await db.collection("users").where("email", "==", body.email).limit(1).get()
            if existing:
                logger.warning("Registration attempt with existing email: %s", body.email)
                raise ConflictError(f"'{body.email}' is already registered")
        except ConflictError:
            raise
        except Exception as e:
            _raise_for_firestore_quota(e, context="checking whether the email already exists")
            logger.error("Database query error during email check: %s", e, exc_info=True)
            raise

        # Generate user ID and timestamp
        user_id = str(uuid.uuid4())
        now = datetime.now(UTC)
        
        # Hash password
        try:
            hashed_pwd = hash_password(body.password)
        except Exception as e:
            logger.error("Password hashing failed: %s", e, exc_info=True)
            raise

        # Write user to Firestore
        try:
            await db.collection("users").document(user_id).set({
                "email": body.email,
                "hashed_password": hashed_pwd,
                "tier": "free",
                "created_at": now,
                "updated_at": now,
            })
        except Exception as e:
            logger.error("Firestore write error: %s", e, exc_info=True)
            raise

        # Generate tokens
        try:
            access_token = create_access_token(user_id, settings)
            refresh_token = create_refresh_token(user_id, settings)
        except Exception as e:
            logger.error("Token generation failed: %s", e, exc_info=True)
            raise

        logger.info("User registered successfully: id=%s, email=%s", user_id, body.email)
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
        )
    
    except ConflictError:
        raise
    except Exception as e:
        logger.error("Unexpected error during registration for %s: %s", body.email, e, exc_info=True)
        raise


@router.post("/login", response_model=TokenResponse)
@limiter.limit("60/minute")
async def login(
    request: Request,
    body: LoginRequest,
    db: DBSession,
    redis: RedisDep,
    settings: SettingsDep,
) -> TokenResponse:
    """Authenticate with email + password; return access + refresh tokens."""
    await check_not_locked_out(redis, body.email)

    try:
        docs = await db.collection("users").where("email", "==", body.email).limit(1).get()
    except Exception as exc:
        _raise_for_firestore_quota(exc, context="looking up the user record")
        logger.error("Firestore lookup failure during login for %s: %s", body.email, exc, exc_info=True)
        raise

    user_doc = docs[0] if docs else None
    user_data = user_doc.to_dict() if user_doc else None

    # Always verify to avoid timing-based user enumeration
    if not user_data or not verify_password(body.password, user_data["hashed_password"]):
        await record_failed_attempt(redis, body.email)
        raise UnauthorizedError("Invalid email or password")

    await clear_failed_attempts(redis, body.email)
    logger.info("User logged in: id=%s", user_doc.id)
    return TokenResponse(
        access_token=create_access_token(user_doc.id, settings),
        refresh_token=create_refresh_token(user_doc.id, settings),
    )


@router.post("/refresh", response_model=TokenResponse)
@limiter.limit("60/minute")
async def refresh_tokens(
    request: Request,
    body: RefreshRequest,
    settings: SettingsDep,
) -> TokenResponse:
    """Exchange a valid refresh token for a new access + refresh pair."""
    subject = decode_refresh_token(body.refresh_token, settings)
    return TokenResponse(
        access_token=create_access_token(subject, settings),
        refresh_token=create_refresh_token(subject, settings),
    )


@router.post("/oauth", response_model=TokenResponse)
@router.post("/google", response_model=TokenResponse)  # legacy alias, pre-GitHub
@limiter.limit("60/minute")
async def oauth_auth(
    request: Request,
    body: GoogleAuthRequest,
    db: DBSession,
    settings: SettingsDep,
) -> TokenResponse:
    """Authenticate via a Firebase OAuth provider (Google, GitHub, …).

    Verifies the Firebase ID token and creates the user on first sign-in.
    Provider-agnostic: Firebase signs the token regardless of which provider
    the user picked in the popup.
    """
    try:
        decoded = firebase_admin.auth.verify_id_token(body.id_token)
    except Exception as exc:
        raise UnauthorizedError("Invalid sign-in token") from exc

    email: str | None = decoded.get("email")
    if not email:
        raise UnauthorizedError("This account has no email we can use")

    provider: str = decoded.get("firebase", {}).get("sign_in_provider", "unknown")
    try:
        docs = await db.collection("users").where("email", "==", email).limit(1).get()
    except Exception as exc:
        _raise_for_firestore_quota(exc, context="looking up the OAuth user record")
        logger.error("Firestore lookup failure during OAuth sign-in for %s: %s", email, exc, exc_info=True)
        raise

    if docs:
        # Only link into an existing account when the provider vouches for the
        # email — otherwise anyone could claim an unverified address on GitHub
        # and take over the matching Forgefy account.
        if decoded.get("email_verified") is False:
            raise UnauthorizedError(
                f"An account for {email} already exists. Verify that email with "
                "your sign-in provider first, or log in with your password."
            )
        user_id = docs[0].id
    else:
        user_id = str(uuid.uuid4())
        now = datetime.now(UTC)
        await db.collection("users").document(user_id).set({
            "email": email,
            "hashed_password": "",
            "tier": "free",
            "created_at": now,
            "updated_at": now,
        })
        logger.info("OAuth user created: id=%s email=%s provider=%s", user_id, email, provider)

    logger.info("OAuth user signed in: id=%s provider=%s", user_id, provider)
    return TokenResponse(
        access_token=create_access_token(user_id, settings),
        refresh_token=create_refresh_token(user_id, settings),
    )


# ---------------------------------------------------------------------------
# GitHub OAuth — link a personal GitHub account to the Forgefy user
# ---------------------------------------------------------------------------

def _make_state(user_id: str, secret: str) -> str:
    payload = base64.urlsafe_b64encode(
        json.dumps({"uid": user_id, "ts": int(time.time())}).encode()
    ).decode()
    sig = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()[:16]
    return f"{payload}.{sig}"


def _verify_state(state: str, secret: str) -> str | None:
    """Return user_id if state is valid and fresh, else None."""
    parts = state.rsplit(".", 1)
    if len(parts) != 2:
        return None
    payload_b64, sig = parts
    expected = hmac.new(secret.encode(), payload_b64.encode(), hashlib.sha256).hexdigest()[:16]
    if sig != expected:
        return None
    try:
        data = json.loads(base64.urlsafe_b64decode(payload_b64).decode())
    except Exception:
        return None
    if time.time() - data.get("ts", 0) > 600:
        return None
    return data.get("uid")


from app.deps import CurrentUser  # noqa: E402 — circular-safe here


@router.get("/github/authorize")
async def github_authorize(
    settings: SettingsDep,
    user: CurrentUser,
) -> dict:
    """Return the GitHub OAuth redirect URL for the current user."""
    if not settings.GITHUB_CLIENT_ID:
        return {"error": "GitHub OAuth not configured"}
    state = _make_state(str(user.id), settings.SECRET_KEY)
    params = urlencode({
        "client_id": settings.GITHUB_CLIENT_ID,
        "redirect_uri": f"{settings.PUBLIC_API_BASE_URL}/api/v1/auth/github/callback",
        "scope": "repo",
        "state": state,
    })
    return {"url": f"https://github.com/login/oauth/authorize?{params}"}


@router.get("/github/callback")
async def github_callback(
    code: str = Query(...),
    state: str = Query(...),
    db: DBSession = None,
    settings: SettingsDep = None,
) -> RedirectResponse:
    """Exchange code for GitHub token, store on user, redirect to frontend."""
    user_id = _verify_state(state, settings.SECRET_KEY)
    if not user_id:
        return RedirectResponse(f"{settings.FRONTEND_URL}/dashboard?github_error=invalid_state")

    try:
        async with httpx.AsyncClient() as http_client:
            resp = await http_client.post(
                "https://github.com/login/oauth/access_token",
                headers={"Accept": "application/json"},
                data={
                    "client_id": settings.GITHUB_CLIENT_ID,
                    "client_secret": settings.GITHUB_CLIENT_SECRET,
                    "code": code,
                },
                timeout=30,
            )
            token_data = resp.json()
            github_token = token_data.get("access_token")
            if not github_token:
                return RedirectResponse(f"{settings.FRONTEND_URL}/dashboard?github_error=exchange_failed")

            user_resp = await http_client.get(
                "https://api.github.com/user",
                headers={"Authorization": f"Bearer {github_token}", "Accept": "application/json"},
                timeout=15,
            )
            github_username: str = user_resp.json().get("login", "")

        await db.collection("users").document(user_id).update({
            "github_access_token": github_token,
            "github_username": github_username,
        })
        logger.info("GitHub linked user=%s username=%s", user_id, github_username)
    except Exception as exc:
        logger.error("GitHub OAuth callback error: %s", exc)
        return RedirectResponse(f"{settings.FRONTEND_URL}/dashboard?github_error=server_error")

    return RedirectResponse(f"{settings.FRONTEND_URL}/dashboard?github=connected")


@router.get("/github/status")
async def github_status(db: DBSession, user: CurrentUser) -> dict:
    """Return whether the current user has a linked GitHub account."""
    doc = await db.collection("users").document(str(user.id)).get()
    data = doc.to_dict() if doc.exists else {}
    return {
        "linked": bool(data.get("github_access_token")),
        "username": data.get("github_username"),
    }


# ---------------------------------------------------------------------------
# Supabase OAuth — link a personal Supabase account so Forgefy can provision
# a real database for the user's generated app (see app/api/v1/projects.py
# for the per-project "connect" step that uses this linked account).
# ---------------------------------------------------------------------------

_SUPABASE_PKCE_PREFIX = "supabase_oauth_verifier:"
_SUPABASE_PKCE_TTL = 600  # matches _verify_state's freshness window


def _make_pkce_pair() -> tuple[str, str]:
    """Return (code_verifier, code_challenge) per RFC 7636 (S256)."""
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


@router.get("/supabase/authorize")
async def supabase_authorize(
    settings: SettingsDep,
    redis: RedisDep,
    user: CurrentUser,
) -> dict:
    """Return the Supabase OAuth redirect URL for the current user."""
    if not settings.SUPABASE_CLIENT_ID:
        return {"error": "Supabase OAuth not configured"}

    state = _make_state(str(user.id), settings.SECRET_KEY)
    code_verifier, code_challenge = _make_pkce_pair()
    await redis.set(f"{_SUPABASE_PKCE_PREFIX}{state}", code_verifier, ex=_SUPABASE_PKCE_TTL)

    params = urlencode({
        "response_type": "code",
        "client_id": settings.SUPABASE_CLIENT_ID,
        "redirect_uri": f"{settings.PUBLIC_API_BASE_URL}/api/v1/auth/supabase/callback",
        "scope": "all",
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
    })
    return {"url": f"https://api.supabase.com/v1/oauth/authorize?{params}"}


@router.get("/supabase/callback")
async def supabase_callback(
    settings: SettingsDep,
    db: DBSession,
    redis: RedisDep,
    code: str = Query(...),
    state: str = Query(...),
) -> RedirectResponse:
    """Exchange code for Supabase tokens, store them (encrypted), redirect to the frontend."""
    from app.core.crypto import encrypt
    from app.integrations import supabase_management

    user_id = _verify_state(state, settings.SECRET_KEY)
    if not user_id:
        return RedirectResponse(f"{settings.FRONTEND_URL}/dashboard?supabase_error=invalid_state")

    redis_key = f"{_SUPABASE_PKCE_PREFIX}{state}"
    code_verifier = await redis.get(redis_key)
    await redis.delete(redis_key)
    if not code_verifier:
        return RedirectResponse(f"{settings.FRONTEND_URL}/dashboard?supabase_error=expired_state")

    try:
        token_data = await supabase_management.exchange_code_for_token(
            code=code,
            code_verifier=code_verifier,
            client_id=settings.SUPABASE_CLIENT_ID,
            client_secret=settings.SUPABASE_CLIENT_SECRET,
            redirect_uri=f"{settings.PUBLIC_API_BASE_URL}/api/v1/auth/supabase/callback",
        )
        await db.collection("users").document(user_id).update({
            "supabase_access_token": encrypt(token_data["access_token"]),
            "supabase_refresh_token": encrypt(token_data["refresh_token"]),
            "supabase_token_expires_at": time.time() + token_data.get("expires_in", 3600),
        })
        logger.info("Supabase account linked user=%s", user_id)
    except Exception as exc:
        logger.error("Supabase OAuth callback error: %s", exc)
        return RedirectResponse(f"{settings.FRONTEND_URL}/dashboard?supabase_error=server_error")

    return RedirectResponse(f"{settings.FRONTEND_URL}/dashboard?supabase=connected")


@router.get("/supabase/status")
async def supabase_status(db: DBSession, user: CurrentUser) -> dict:
    """Return whether the current user has a linked Supabase account."""
    doc = await db.collection("users").document(str(user.id)).get()
    data = doc.to_dict() if doc.exists else {}
    return {"linked": bool(data.get("supabase_access_token"))}


@router.get("/supabase/organizations")
async def supabase_organizations(db: DBSession, settings: SettingsDep, user: CurrentUser) -> list[dict]:
    """List the Supabase organizations available to the connected account."""
    from app.build.supabase_token import get_valid_supabase_token
    from app.integrations import supabase_management

    token = await get_valid_supabase_token(db, str(user.id), settings)
    if not token:
        raise UnauthorizedError("No Supabase account linked")
    return await supabase_management.list_organizations(token)


# ---------------------------------------------------------------------------
# Firebase OAuth — link a personal Google account so Forgefy can provision a
# real Firebase/Firestore project under the USER's own GCP account (see
# app/api/v1/projects.py for the per-project "connect" step that uses this
# linked account). Separate from Forgefy's own Firebase usage in app/db/firebase.py.
# ---------------------------------------------------------------------------

_FIREBASE_PKCE_PREFIX = "firebase_oauth_verifier:"
_FIREBASE_PKCE_TTL = 600  # matches _verify_state's freshness window


@router.get("/firebase/authorize")
async def firebase_authorize(
    settings: SettingsDep,
    redis: RedisDep,
    user: CurrentUser,
) -> dict:
    """Return the Google OAuth redirect URL for the current user."""
    if not settings.FIREBASE_OAUTH_CLIENT_ID:
        return {"error": "Firebase OAuth not configured"}

    state = _make_state(str(user.id), settings.SECRET_KEY)
    code_verifier, code_challenge = _make_pkce_pair()
    await redis.set(f"{_FIREBASE_PKCE_PREFIX}{state}", code_verifier, ex=_FIREBASE_PKCE_TTL)

    params = urlencode({
        "response_type": "code",
        "client_id": settings.FIREBASE_OAUTH_CLIENT_ID,
        "redirect_uri": f"{settings.PUBLIC_API_BASE_URL}/api/v1/auth/firebase/callback",
        "scope": "https://www.googleapis.com/auth/cloud-platform",
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
        "access_type": "offline",
        "prompt": "consent",
    })
    return {"url": f"https://accounts.google.com/o/oauth2/v2/auth?{params}"}


@router.get("/firebase/callback")
async def firebase_callback(
    settings: SettingsDep,
    db: DBSession,
    redis: RedisDep,
    code: str = Query(...),
    state: str = Query(...),
) -> RedirectResponse:
    """Exchange code for Google tokens, store them (encrypted), redirect to the frontend."""
    from app.core.crypto import encrypt
    from app.integrations import firebase_management

    user_id = _verify_state(state, settings.SECRET_KEY)
    if not user_id:
        return RedirectResponse(f"{settings.FRONTEND_URL}/dashboard?firebase_error=invalid_state")

    redis_key = f"{_FIREBASE_PKCE_PREFIX}{state}"
    code_verifier = await redis.get(redis_key)
    await redis.delete(redis_key)
    if not code_verifier:
        return RedirectResponse(f"{settings.FRONTEND_URL}/dashboard?firebase_error=expired_state")

    try:
        token_data = await firebase_management.exchange_code_for_token(
            code=code,
            code_verifier=code_verifier,
            client_id=settings.FIREBASE_OAUTH_CLIENT_ID,
            client_secret=settings.FIREBASE_OAUTH_CLIENT_SECRET,
            redirect_uri=f"{settings.PUBLIC_API_BASE_URL}/api/v1/auth/firebase/callback",
        )
        await db.collection("users").document(user_id).update({
            "firebase_access_token": encrypt(token_data["access_token"]),
            "firebase_refresh_token": encrypt(token_data["refresh_token"]),
            "firebase_token_expires_at": time.time() + token_data.get("expires_in", 3600),
        })
        logger.info("Firebase (Google) account linked user=%s", user_id)
    except Exception as exc:
        logger.error("Firebase OAuth callback error: %s", exc)
        return RedirectResponse(f"{settings.FRONTEND_URL}/dashboard?firebase_error=server_error")

    return RedirectResponse(f"{settings.FRONTEND_URL}/dashboard?firebase=connected")


@router.get("/firebase/status")
async def firebase_status(db: DBSession, user: CurrentUser) -> dict:
    """Return whether the current user has a linked Google account for Firebase."""
    doc = await db.collection("users").document(str(user.id)).get()
    data = doc.to_dict() if doc.exists else {}
    return {"linked": bool(data.get("firebase_access_token"))}
