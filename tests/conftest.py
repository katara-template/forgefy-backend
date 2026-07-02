"""Pytest configuration and shared fixtures."""
import os
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

# Must run before `app.main` is imported — it reads SENTRY_DSN at import time
# and, if set, starts sending real events over the network during test runs.
os.environ["SENTRY_DSN"] = ""

import pytest
from httpx import ASGITransport, AsyncClient

from app.db.models.user import User
from app.deps import get_current_user, get_db, get_redis
from app.main import app

_ASYNC_TERMINALS = ("get", "set", "update", "delete", "add", "stream")
_CHAIN_METHODS = ("collection", "document", "where", "limit", "order_by")


def wire_firestore_chain(node: MagicMock, depth: int = 4) -> None:
    """Recursively configure a MagicMock to look like a Firestore AsyncClient.

    `.collection()`/`.document()`/`.where()`/`.limit()`/`.order_by()` build a
    query synchronously (as in the real client); the terminal calls
    (`.get()`, `.set()`, `.update()`, `.delete()`, `.add()`, `.stream()`) are
    awaited, so they're configured as AsyncMock. Depth covers subcollection
    chains like `sessions/{id}/events`.
    """
    for name in _ASYNC_TERMINALS:
        setattr(node, name, AsyncMock())
    if depth <= 0:
        return
    for name in _CHAIN_METHODS:
        wire_firestore_chain(getattr(node, name).return_value, depth - 1)


def make_doc_snapshot(data: dict | None, doc_id: str = "doc-id") -> MagicMock:
    """Build a Firestore DocumentSnapshot-like mock: .exists / .to_dict() / .id."""
    snap = MagicMock()
    snap.exists = data is not None
    snap.to_dict.return_value = data
    snap.id = doc_id
    return snap


@pytest.fixture
def mock_db() -> MagicMock:
    """Firestore AsyncClient mock. Configure per-test, e.g.:

        mock_db.collection.return_value.document.return_value.get.return_value = snapshot
        mock_db.collection.return_value.where.return_value.limit.return_value.get.return_value = [snapshot]
    """
    db = MagicMock()
    wire_firestore_chain(db)
    return db


@pytest.fixture
def mock_redis() -> AsyncMock:
    """Async Redis client mock, covering the methods app code actually awaits."""
    redis = AsyncMock()
    redis.get.return_value = None
    redis.incr.return_value = 1
    return redis


@pytest.fixture
def test_user() -> User:
    """A minimal User object for auth-protected endpoint tests."""
    now = datetime.now(UTC)
    return User(
        id=uuid.uuid4(),
        email="testuser@example.com",
        hashed_password="irrelevant-in-tests",
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
async def client(
    mock_db: MagicMock, mock_redis: AsyncMock
) -> AsyncGenerator[AsyncClient, None]:
    """Unauthenticated AsyncClient with Firestore/Redis mocked and lifespan startup patched."""

    async def override_get_db() -> AsyncGenerator[MagicMock, None]:
        yield mock_db

    async def override_get_redis() -> AsyncGenerator[AsyncMock, None]:
        yield mock_redis

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_redis] = override_get_redis

    with (
        patch("app.main.init_firebase", return_value=mock_db),
        patch("app.main.aioredis.from_url", return_value=mock_redis),
    ):
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
    # get_db/get_redis overrides and the firebase/redis patches are cleaned
    # up by the client fixture.
