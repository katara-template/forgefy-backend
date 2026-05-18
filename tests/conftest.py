"""Pytest configuration and shared fixtures."""
import uuid
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.user import User
from app.deps import get_current_user, get_db
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
def test_user() -> User:
    """A minimal User object for auth-protected endpoint tests."""
    return User(
        id=uuid.uuid4(),
        email="testuser@example.com",
        hashed_password="irrelevant-in-tests",
    )


@pytest.fixture
async def client(mock_session: AsyncMock) -> AsyncGenerator[AsyncClient, None]:
    """Unauthenticated AsyncClient with DB mocked and lifespan engine patched."""

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield mock_session

    app.dependency_overrides[get_db] = override_get_db

    with (
        patch("app.main.build_engine") as mock_build_engine,
        patch("app.main.build_session_factory"),
    ):
        mock_build_engine.return_value = AsyncMock()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            yield ac

    app.dependency_overrides.clear()


@pytest.fixture
async def auth_client(
    client: AsyncClient, test_user: User
) -> AsyncGenerator[AsyncClient, None]:
    """AsyncClient with CurrentUser dependency short-circuited to test_user.

    The get_current_user override bypasses JWT validation and the DB lookup,
    so tests don't need a valid token or a mock user query.
    """
    app.dependency_overrides[get_current_user] = lambda: test_user
    yield client
    # get_db override and engine patch are cleaned up by the client fixture
