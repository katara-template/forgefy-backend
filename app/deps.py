"""Shared FastAPI dependencies."""
import uuid
from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.core.exceptions import UnauthorizedError
from app.core.security import decode_access_token
from app.db.models.user import User

# ── Settings ──────────────────────────────────────────────────────────────────

SettingsDep = Annotated[Settings, Depends(get_settings)]


# ── Database session ──────────────────────────────────────────────────────────


async def get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """Yield an async DB session from the app-level session factory.

    Commits on clean exit; rolls back on any exception.
    """
    factory = request.app.state.db_session_factory
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


DBSession = Annotated[AsyncSession, Depends(get_db)]


# ── Current user ──────────────────────────────────────────────────────────────

_oauth2 = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


async def get_current_user(
    token: Annotated[str, Depends(_oauth2)],
    db: DBSession,
    settings: SettingsDep,
) -> User:
    """Validate the Bearer JWT and return the corresponding User row."""
    user_id_str = decode_access_token(token, settings)
    try:
        uid = uuid.UUID(user_id_str)
    except ValueError:
        raise UnauthorizedError("Token subject is not a valid UUID")

    result = await db.execute(select(User).where(User.id == uid))
    user = result.scalar_one_or_none()
    if user is None:
        raise UnauthorizedError("User not found")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]
