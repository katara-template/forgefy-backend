"""Shared FastAPI dependencies."""
from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings


# ── Settings ──────────────────────────────────────────────────────────────────

SettingsDep = Annotated[Settings, Depends(get_settings)]


# ── Database session ──────────────────────────────────────────────────────────

async def get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """Yield an async DB session from the app-level session factory."""
    factory = request.app.state.db_session_factory
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


DBSession = Annotated[AsyncSession, Depends(get_db)]

# current_user dependency is added in Step 3 (Auth)
