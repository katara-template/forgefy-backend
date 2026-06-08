"""Shared FastAPI dependencies."""
import uuid
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import OAuth2PasswordBearer
from google.cloud.firestore import AsyncClient

from app.config import Settings, get_settings
from app.core.exceptions import UnauthorizedError
from app.core.security import decode_access_token
from app.db.models.user import User

# ── Settings ──────────────────────────────────────────────────────────────────

SettingsDep = Annotated[Settings, Depends(get_settings)]


# ── Firestore client ──────────────────────────────────────────────────────────


def get_db(request: Request) -> AsyncClient:
    """Return the app-level Firestore async client."""
    return request.app.state.firestore


DBSession = Annotated[AsyncClient, Depends(get_db)]


# ── Current user ──────────────────────────────────────────────────────────────

_oauth2 = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


async def get_current_user(
    token: Annotated[str, Depends(_oauth2)],
    db: DBSession,
    settings: SettingsDep,
) -> User:
    """Validate the Bearer JWT and return the corresponding User document."""
    user_id_str = decode_access_token(token, settings)
    try:
        uid = uuid.UUID(user_id_str)
    except ValueError:
        raise UnauthorizedError("Token subject is not a valid UUID")

    doc = await db.collection("users").document(str(uid)).get()
    if not doc.exists:
        raise UnauthorizedError("User not found")

    data = doc.to_dict()
    return User(
        id=uid,
        email=data["email"],
        hashed_password=data["hashed_password"],
        created_at=data["created_at"],
        updated_at=data["updated_at"],
        tier=data.get("tier", "free"),
    )


CurrentUser = Annotated[User, Depends(get_current_user)]
