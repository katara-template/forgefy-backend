"""Pytest configuration and shared fixtures."""
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.deps import get_db
from app.main import app


@pytest.fixture
def mock_session() -> AsyncMock:
    """Pre-configured mock AsyncSession — customise execute() per test."""
    session = AsyncMock(spec=AsyncSession)
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.flush = AsyncMock()
    return session


@pytest.fixture
async def client(mock_session: AsyncMock) -> AsyncGenerator[AsyncClient, None]:
    """AsyncClient with:
    - DB session dependency overridden by mock_session
    - build_engine / build_session_factory patched so lifespan never
      attempts a real PostgreSQL connection
    """

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield mock_session

    app.dependency_overrides[get_db] = override_get_db

    with (
        patch("app.main.build_engine") as mock_build_engine,
        patch("app.main.build_session_factory"),
    ):
        mock_engine = AsyncMock()
        mock_build_engine.return_value = mock_engine

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            yield ac

    app.dependency_overrides.clear()
