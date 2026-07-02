"""Auth endpoint tests: register / login / refresh."""
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

from httpx import AsyncClient

from app.config import get_settings
from app.core.security import create_access_token, create_refresh_token, hash_password
from tests.conftest import make_doc_snapshot

settings = get_settings()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _user_doc(
    email: str = "test@example.com",
    password: str = "password123",
    doc_id: str | None = None,
) -> MagicMock:
    """A Firestore user document snapshot with a hashed password."""
    return make_doc_snapshot(
        {
            "email": email,
            "hashed_password": hash_password(password),
            "tier": "free",
            "is_admin": False,
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
        },
        doc_id=doc_id or str(uuid.uuid4()),
    )


def _set_email_query_result(mock_db: MagicMock, docs: list) -> None:
    """Configure db.collection("users").where("email", ...).limit(1).get()."""
    mock_db.collection.return_value.where.return_value.limit.return_value.get.return_value = docs


# ── Register ──────────────────────────────────────────────────────────────────


class TestRegister:
    async def test_success_returns_tokens(
        self, client: AsyncClient, mock_db: MagicMock
    ) -> None:
        _set_email_query_result(mock_db, [])

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
        self, client: AsyncClient, mock_db: MagicMock
    ) -> None:
        _set_email_query_result(mock_db, [_user_doc(email="taken@example.com")])

        resp = await client.post(
            "/api/v1/auth/register",
            json={"email": "taken@example.com", "password": "password123"},
        )

        assert resp.status_code == 409
        assert resp.headers["content-type"] == "application/problem+json"

    async def test_short_password_returns_422(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/auth/register",
            json={"email": "new@example.com", "password": "short"},
        )

        assert resp.status_code == 422

    async def test_invalid_email_returns_422(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/auth/register",
            json={"email": "not-an-email", "password": "password123"},
        )

        assert resp.status_code == 422


# ── Login ─────────────────────────────────────────────────────────────────────


class TestLogin:
    async def test_success_returns_tokens(
        self, client: AsyncClient, mock_db: MagicMock
    ) -> None:
        doc = _user_doc(email="test@example.com", password="correct_password")
        _set_email_query_result(mock_db, [doc])

        resp = await client.post(
            "/api/v1/auth/login",
            json={"email": "test@example.com", "password": "correct_password"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["token_type"] == "bearer"
        assert "access_token" in body

    async def test_wrong_password_returns_401(
        self, client: AsyncClient, mock_db: MagicMock
    ) -> None:
        doc = _user_doc(email="test@example.com", password="correct_password")
        _set_email_query_result(mock_db, [doc])

        resp = await client.post(
            "/api/v1/auth/login",
            json={"email": "test@example.com", "password": "wrong_password"},
        )

        assert resp.status_code == 401

    async def test_unknown_email_returns_401(
        self, client: AsyncClient, mock_db: MagicMock
    ) -> None:
        _set_email_query_result(mock_db, [])

        resp = await client.post(
            "/api/v1/auth/login",
            json={"email": "nobody@example.com", "password": "password123"},
        )

        assert resp.status_code == 401

    async def test_locked_out_account_returns_429(
        self, client: AsyncClient, mock_db: MagicMock, mock_redis: AsyncMock
    ) -> None:
        doc = _user_doc(email="test@example.com", password="correct_password")
        _set_email_query_result(mock_db, [doc])
        mock_redis.get.return_value = "5"  # at the lockout threshold

        resp = await client.post(
            "/api/v1/auth/login",
            json={"email": "test@example.com", "password": "correct_password"},
        )

        assert resp.status_code == 429


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
