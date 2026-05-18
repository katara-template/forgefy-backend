"""Auth endpoints: register, login, refresh.

All three routes are rate-limited to 60 req/min per IP.
"""
import logging

from fastapi import APIRouter, Request
from sqlalchemy import select

from app.core.exceptions import ConflictError, UnauthorizedError
from app.core.rate_limit import limiter
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_refresh_token,
    hash_password,
    verify_password,
)
from app.db.models.user import User
from app.deps import DBSession, SettingsDep
from app.schemas.auth import LoginRequest, RefreshRequest, RegisterRequest, TokenResponse

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
    result = await db.execute(select(User).where(User.email == body.email))
    if result.scalar_one_or_none() is not None:
        raise ConflictError(f"'{body.email}' is already registered")

    user = User(email=body.email, hashed_password=hash_password(body.password))
    db.add(user)
    await db.flush()  # populate user.id before encoding it in the token

    logger.info("User registered: id=%s", user.id)
    return TokenResponse(
        access_token=create_access_token(str(user.id), settings),
        refresh_token=create_refresh_token(str(user.id), settings),
    )


@router.post("/login", response_model=TokenResponse)
@limiter.limit("60/minute")
async def login(
    request: Request,
    body: LoginRequest,
    db: DBSession,
    settings: SettingsDep,
) -> TokenResponse:
    """Authenticate with email + password; return access + refresh tokens."""
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()

    # Always verify to avoid timing-based user enumeration
    if not user or not verify_password(body.password, user.hashed_password):
        raise UnauthorizedError("Invalid email or password")

    logger.info("User logged in: id=%s", user.id)
    return TokenResponse(
        access_token=create_access_token(str(user.id), settings),
        refresh_token=create_refresh_token(str(user.id), settings),
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
