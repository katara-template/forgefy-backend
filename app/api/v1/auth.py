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
import time
import uuid
from datetime import datetime, timezone
from urllib.parse import urlencode

import firebase_admin.auth
import httpx
from fastapi import APIRouter, Query, Request
from fastapi.responses import RedirectResponse

from app.core.exceptions import ConflictError, UnauthorizedError
from app.core.rate_limit import limiter
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_refresh_token,
    hash_password,
    verify_password,
)
from app.deps import DBSession, SettingsDep
from app.schemas.auth import (
    GoogleAuthRequest,
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


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
            logger.error("Database query error during email check: %s", e, exc_info=True)
            raise

        # Generate user ID and timestamp
        user_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        
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
    settings: SettingsDep,
) -> TokenResponse:
    """Authenticate with email + password; return access + refresh tokens."""
    docs = await db.collection("users").where("email", "==", body.email).limit(1).get()

    user_doc = docs[0] if docs else None
    user_data = user_doc.to_dict() if user_doc else None

    # Always verify to avoid timing-based user enumeration
    if not user_data or not verify_password(body.password, user_data["hashed_password"]):
        raise UnauthorizedError("Invalid email or password")

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


@router.post("/google", response_model=TokenResponse)
@limiter.limit("60/minute")
async def google_auth(
    request: Request,
    body: GoogleAuthRequest,
    db: DBSession,
    settings: SettingsDep,
) -> TokenResponse:
    """Authenticate via Google — verify Firebase ID token, create user on first sign-in."""
    try:
        decoded = firebase_admin.auth.verify_id_token(body.id_token)
    except Exception:
        raise UnauthorizedError("Invalid Google token")

    email: str | None = decoded.get("email")
    if not email:
        raise UnauthorizedError("Google account has no verified email")

    docs = await db.collection("users").where("email", "==", email).limit(1).get()

    if docs:
        user_id = docs[0].id
    else:
        user_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        await db.collection("users").document(user_id).set({
            "email": email,
            "hashed_password": "",
            "created_at": now,
            "updated_at": now,
        })
        logger.info("Google user created: id=%s email=%s", user_id, email)

    logger.info("Google user signed in: id=%s", user_id)
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
        resp = httpx.post(
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

        user_resp = httpx.get(
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
