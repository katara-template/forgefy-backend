"""Auth endpoint tests: register / login / refresh."""
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import AsyncClient

from app.config import get_settings
from app.core.security import create_access_token, create_refresh_token, hash_password
from app.db.models.user import User

settings = get_settings()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_user(
    email: str = "test@example.com",
    password: str = "password123",
) -> User:
    """Return an unsaved User ORM object with a hashed password."""
    return User(
        id=uuid.uuid4(),
        email=email,
        hashed_password=hash_password(password),
    )


def _mock_result(value: object) -> MagicMock:
    """Return a mock SQLAlchemy result whose scalar_one_or_none() returns value."""
    result = MagicMock()
    result.scalar_one_or_none.return_value = value
    return result


# ── Register ──────────────────────────────────────────────────────────────────


class TestRegister:
    async def test_success_returns_tokens(
        self, client: AsyncClient, mock_session: AsyncMock
    ) -> None:
        mock_session.execute.return_value = _mock_result(None)

        resp = await client.post(
            "/api/v1/auth/register",
            json={"email": "new@example.com", "password": "password123"},
        )

        assert resp.status_code == 201
        body = resp.json()
        assert body["token_type"] == "bearer"
        assert "access_token" in body
        assert "refresh_token" in body

    async def test_duplicate_email_returns_409(
        self, client: AsyncClient, mock_session: AsyncMock
    ) -> None:
        mock_session.execute.return_value = _mock_result(_make_user())

        resp = await client.post(
            "/api/v1/auth/register",
            json={"email": "taken@example.com", "password": "password123"},
        )

        assert resp.status_code == 409
        assert resp.headers["content-type"] == "application/problem+json"

    async def test_short_password_returns_422(
        self, client: AsyncClient, mock_session: AsyncMock
    ) -> None:
        resp = await client.post(
            "/api/v1/auth/register",
            json={"email": "new@example.com", "password": "short"},
        )

        assert resp.status_code == 422

    async def test_invalid_email_returns_422(
        self, client: AsyncClient, mock_session: AsyncMock
    ) -> None:
        resp = await client.post(
            "/api/v1/auth/register",
            json={"email": "not-an-email", "password": "password123"},
        )

        assert resp.status_code == 422


# ── Login ─────────────────────────────────────────────────────────────────────


class TestLogin:
    async def test_success_returns_tokens(
        self, client: AsyncClient, mock_session: AsyncMock
    ) -> None:
        user = _make_user(password="correct_password")
        mock_session.execute.return_value = _mock_result(user)

        resp = await client.post(
            "/api/v1/auth/login",
            json={"email": "test@example.com", "password": "correct_password"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["token_type"] == "bearer"
        assert "access_token" in body

    async def test_wrong_password_returns_401(
        self, client: AsyncClient, mock_session: AsyncMock
    ) -> None:
        user = _make_user(password="correct_password")
        mock_session.execute.return_value = _mock_result(user)

        resp = await client.post(
            "/api/v1/auth/login",
            json={"email": "test@example.com", "password": "wrong_password"},
        )

        assert resp.status_code == 401

    async def test_unknown_email_returns_401(
        self, client: AsyncClient, mock_session: AsyncMock
    ) -> None:
        mock_session.execute.return_value = _mock_result(None)

        resp = await client.post(
            "/api/v1/auth/login",
            json={"email": "nobody@example.com", "password": "password123"},
        )

        assert resp.status_code == 401


# ── Refresh ───────────────────────────────────────────────────────────────────


class TestRefresh:
    async def test_success_returns_new_tokens(self, client: AsyncClient) -> None:
        refresh_token = create_refresh_token(str(uuid.uuid4()), settings)

        resp = await client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh_token},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert "access_token" in body
        assert "refresh_token" in body

    async def test_garbage_token_returns_401(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": "not.a.real.token"},
        )

        assert resp.status_code == 401

    async def test_access_token_rejected_as_refresh(self, client: AsyncClient) -> None:
        """Passing an access token where a refresh token is expected must fail."""
        access_token = create_access_token(str(uuid.uuid4()), settings)

        resp = await client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": access_token},
        )

        assert resp.status_code == 401
