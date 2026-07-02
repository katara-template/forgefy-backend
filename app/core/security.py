"""JWT token creation/verification and bcrypt password hashing.

Uses the `bcrypt` library directly — passlib 1.7.4 is incompatible with
bcrypt 4/5 (wrap-bug detection raises ValueError on long test passwords).
"""
from datetime import UTC, datetime, timedelta

import bcrypt as _bcrypt
from jose import JWTError, jwt

from app.config import Settings
from app.core.exceptions import UnauthorizedError

_BCRYPT_ROUNDS = 12


# ── Passwords ─────────────────────────────────────────────────────────────────


def hash_password(plain: str) -> str:
    """Return a bcrypt hash of plain (cost factor 12)."""
    hashed = _bcrypt.hashpw(plain.encode("utf-8"), _bcrypt.gensalt(rounds=_BCRYPT_ROUNDS))
    return hashed.decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if plain matches the stored bcrypt hash."""
    return _bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


# ── Tokens ────────────────────────────────────────────────────────────────────


def _make_token(subject: str, token_type: str, expire: datetime, settings: Settings) -> str:
    payload = {"sub": subject, "type": token_type, "exp": expire}
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def create_access_token(subject: str, settings: Settings) -> str:
    """Encode a short-lived access JWT for the given subject (user UUID)."""
    expire = datetime.now(UTC) + timedelta(
        minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    return _make_token(subject, "access", expire, settings)


def create_refresh_token(subject: str, settings: Settings) -> str:
    """Encode a long-lived refresh JWT for the given subject (user UUID)."""
    expire = datetime.now(UTC) + timedelta(
        days=settings.REFRESH_TOKEN_EXPIRE_DAYS
    )
    return _make_token(subject, "refresh", expire, settings)


def decode_token(
    token: str,
    settings: Settings,
    *,
    expected_type: str = "access",
) -> str:
    """Decode and validate a JWT; return the subject string.

    Raises UnauthorizedError on any validation failure so callers never
    need to handle JWTError directly.
    """
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
    except JWTError:
        raise UnauthorizedError("Invalid or expired token")

    if payload.get("type") != expected_type:
        raise UnauthorizedError(f"Expected '{expected_type}' token, got '{payload.get('type')}'")

    sub = payload.get("sub")
    if not sub:
        raise UnauthorizedError("Token payload is missing subject")
    return str(sub)


def decode_access_token(token: str, settings: Settings) -> str:
    """Decode an access token and return the user UUID string."""
    return decode_token(token, settings, expected_type="access")


def decode_refresh_token(token: str, settings: Settings) -> str:
    """Decode a refresh token and return the user UUID string."""
    return decode_token(token, settings, expected_type="refresh")
