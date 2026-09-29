"""Auth endpoint tests: register / login / refresh.

Credentials live in Firebase Auth: register creates the Firebase user
server-side, login verifies the password via the Identity Toolkit REST API
(mocked here through the _firebase_password_signin helper).
"""
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.api_core.exceptions import ResourceExhausted
from httpx import AsyncClient

from app.config import Settings, get_settings
from app.core.security import create_access_token, create_refresh_token, hash_password
from app.main import app
from tests.conftest import make_doc_snapshot

settings = get_settings()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _user_doc(
    email: str = "test@example.com",
    doc_id: str | None = None,
    *,
    legacy_password: str | None = None,
    migrated: bool = True,
) -> MagicMock:
    """A Firestore user document snapshot.

    migrated=True → bound to Firebase (firebase_uid == doc id, no local hash).
    legacy_password → pre-migration doc carrying a bcrypt hash instead.
    """
    doc_id = doc_id or str(uuid.uuid4())
    data = {
        "email": email,
        "hashed_password": hash_password(legacy_password) if legacy_password else "",
        "tier": "free",
        "is_admin": False,
        "created_at": datetime.now(UTC),
        "updated_at": datetime.now(UTC),
    }
    if migrated:
        data["firebase_uid"] = doc_id
    snap = make_doc_snapshot(data, doc_id=doc_id)
    snap.reference.set = AsyncMock()  # login stamps firebase_uid via doc.reference
    return snap


def _set_email_query_result(mock_db: MagicMock, docs: list) -> None:
    """Configure db.collection("users").where(...).limit(1).get()."""
    mock_db.collection.return_value.where.return_value.limit.return_value.get.return_value = docs


@pytest.fixture
def firebase_settings():
    """Real Settings with the Firebase Web API key set, so login is 'configured'."""
    app.dependency_overrides[get_settings] = lambda: Settings(FIREBASE_WEB_API_KEY="test-key")
    yield
    app.dependency_overrides.pop(get_settings, None)


@pytest.fixture
def omnisend_settings():
    """Real Settings with the Omnisend API key set, so contact sync is 'configured'."""
    app.dependency_overrides[get_settings] = lambda: Settings(OMNISEND_API_KEY="omni-test-key")
    yield
    app.dependency_overrides.pop(get_settings, None)


# ── Register ──────────────────────────────────────────────────────────────────


class TestRegister:
    async def test_success_creates_firebase_user_and_returns_tokens(
        self, client: AsyncClient, mock_db: MagicMock
    ) -> None:
        _set_email_query_result(mock_db, [])

        with patch("firebase_admin.auth.create_user") as mock_create:
            resp = await client.post(
                "/api/v1/auth/register",
                json={"email": "new@example.com", "password": "password123"},
            )

        assert resp.status_code == 201
        body = resp.json()
        assert body["token_type"] == "bearer"
        assert "access_token" in body
        assert "refresh_token" in body

        # Credential goes to Firebase; the Firestore doc stores no password.
        assert mock_create.call_args.kwargs["email"] == "new@example.com"
        assert mock_create.call_args.kwargs["password"] == "password123"
        doc = mock_db.collection.return_value.document.return_value.set.call_args[0][0]
        assert doc["hashed_password"] == ""
        assert doc["firebase_uid"] == mock_create.call_args.kwargs["uid"]

    async def test_duplicate_email_returns_409(
        self, client: AsyncClient, mock_db: MagicMock
    ) -> None:
        _set_email_query_result(mock_db, [_user_doc(email="taken@example.com")])

        with patch("firebase_admin.auth.create_user") as mock_create:
            resp = await client.post(
                "/api/v1/auth/register",
                json={"email": "taken@example.com", "password": "password123"},
            )

        assert resp.status_code == 409
        assert resp.headers["content-type"] == "application/problem+json"
        mock_create.assert_not_called()

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

    async def test_marks_new_user_as_requiring_verification_and_sends_email(
        self, client: AsyncClient, mock_db: MagicMock, firebase_settings
    ) -> None:
        _set_email_query_result(mock_db, [])

        with (
            patch("firebase_admin.auth.create_user"),
            patch(
                "app.api.v1.auth._firebase_password_signin_response",
                new=AsyncMock(return_value={"localId": "uid-123", "idToken": "id-token-abc"}),
            ),
            patch(
                "app.api.v1.auth._send_verification_email", new=AsyncMock()
            ) as mock_send,
        ):
            resp = await client.post(
                "/api/v1/auth/register",
                json={"email": "new@example.com", "password": "password123"},
            )

        assert resp.status_code == 201
        doc = mock_db.collection.return_value.document.return_value.set.call_args[0][0]
        assert doc["requires_email_verification"] is True
        mock_send.assert_awaited_once_with("test-key", "id-token-abc")

    async def test_register_succeeds_even_if_verification_email_fails(
        self, client: AsyncClient, mock_db: MagicMock, firebase_settings
    ) -> None:
        _set_email_query_result(mock_db, [])

        with (
            patch("firebase_admin.auth.create_user"),
            patch(
                "app.api.v1.auth._firebase_password_signin_response",
                new=AsyncMock(side_effect=RuntimeError("network down")),
            ),
        ):
            resp = await client.post(
                "/api/v1/auth/register",
                json={"email": "new@example.com", "password": "password123"},
            )

        assert resp.status_code == 201
        assert "access_token" in resp.json()

    async def test_syncs_opted_in_user_as_subscribed_to_omnisend(
        self, client: AsyncClient, mock_db: MagicMock, omnisend_settings
    ) -> None:
        _set_email_query_result(mock_db, [])

        with (
            patch("firebase_admin.auth.create_user"),
            patch("app.integrations.omnisend.upsert_contact", new=AsyncMock()) as mock_sync,
        ):
            resp = await client.post(
                "/api/v1/auth/register",
                json={"email": "new@example.com", "password": "password123", "marketing_opt_in": True},
            )

        assert resp.status_code == 201
        mock_sync.assert_awaited_once_with(
            "omni-test-key", email="new@example.com", subscribed=True, tags=["forgefy-app"]
        )

    async def test_syncs_non_opted_in_user_as_nonsubscribed_to_omnisend(
        self, client: AsyncClient, mock_db: MagicMock, omnisend_settings
    ) -> None:
        _set_email_query_result(mock_db, [])

        with (
            patch("firebase_admin.auth.create_user"),
            patch("app.integrations.omnisend.upsert_contact", new=AsyncMock()) as mock_sync,
        ):
            resp = await client.post(
                "/api/v1/auth/register",
                json={"email": "new@example.com", "password": "password123"},
            )

        assert resp.status_code == 201
        doc = mock_db.collection.return_value.document.return_value.set.call_args[0][0]
        assert doc["marketing_opt_in"] is False
        mock_sync.assert_awaited_once_with(
            "omni-test-key", email="new@example.com", subscribed=False, tags=["forgefy-app"]
        )

    async def test_skips_omnisend_sync_when_not_configured(
        self, client: AsyncClient, mock_db: MagicMock
    ) -> None:
        _set_email_query_result(mock_db, [])
        app.dependency_overrides[get_settings] = lambda: Settings(OMNISEND_API_KEY="")

        try:
            with (
                patch("firebase_admin.auth.create_user"),
                patch("app.integrations.omnisend.upsert_contact", new=AsyncMock()) as mock_sync,
            ):
                resp = await client.post(
                    "/api/v1/auth/register",
                    json={"email": "new@example.com", "password": "password123", "marketing_opt_in": True},
                )
        finally:
            app.dependency_overrides.pop(get_settings, None)

        assert resp.status_code == 201
        mock_sync.assert_not_called()

    async def test_register_succeeds_even_if_omnisend_sync_fails(
        self, client: AsyncClient, mock_db: MagicMock, omnisend_settings
    ) -> None:
        _set_email_query_result(mock_db, [])

        with (
            patch("firebase_admin.auth.create_user"),
            patch(
                "app.integrations.omnisend.upsert_contact",
                new=AsyncMock(side_effect=RuntimeError("omnisend down")),
            ),
        ):
            resp = await client.post(
                "/api/v1/auth/register",
                json={"email": "new@example.com", "password": "password123", "marketing_opt_in": True},
            )

        assert resp.status_code == 201
        assert "access_token" in resp.json()


# ── Login ─────────────────────────────────────────────────────────────────────


class TestLogin:
    async def test_success_returns_tokens(
        self, client: AsyncClient, mock_db: MagicMock, firebase_settings
    ) -> None:
        doc = _user_doc(email="test@example.com")
        _set_email_query_result(mock_db, [doc])

        with patch(
            "app.api.v1.auth._firebase_password_signin",
            new=AsyncMock(return_value=doc.id),
        ):
            resp = await client.post(
                "/api/v1/auth/login",
                json={"email": "test@example.com", "password": "correct_password"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["token_type"] == "bearer"
        assert "access_token" in body

    async def test_wrong_password_returns_401(
        self, client: AsyncClient, mock_db: MagicMock, firebase_settings
    ) -> None:
        doc = _user_doc(email="test@example.com")
        _set_email_query_result(mock_db, [doc])

        with patch(
            "app.api.v1.auth._firebase_password_signin",
            new=AsyncMock(return_value=None),
        ):
            resp = await client.post(
                "/api/v1/auth/login",
                json={"email": "test@example.com", "password": "wrong_password"},
            )

        assert resp.status_code == 401

    async def test_unknown_email_returns_401(
        self, client: AsyncClient, mock_db: MagicMock, firebase_settings
    ) -> None:
        _set_email_query_result(mock_db, [])

        with patch(
            "app.api.v1.auth._firebase_password_signin",
            new=AsyncMock(return_value=None),
        ):
            resp = await client.post(
                "/api/v1/auth/login",
                json={"email": "nobody@example.com", "password": "password123"},
            )

        assert resp.status_code == 401

    async def test_legacy_user_is_lazily_migrated(
        self, client: AsyncClient, mock_db: MagicMock, firebase_settings
    ) -> None:
        """Pre-Firebase account (bcrypt hash, no firebase_uid): correct password
        still logs in and creates the Firebase credential on the fly."""
        doc = _user_doc(email="old@example.com", legacy_password="correct_password", migrated=False)
        _set_email_query_result(mock_db, [doc])

        with (
            patch(
                "app.api.v1.auth._firebase_password_signin",
                new=AsyncMock(return_value=None),  # not in Firebase yet
            ),
            patch("firebase_admin.auth.create_user") as mock_create,
        ):
            resp = await client.post(
                "/api/v1/auth/login",
                json={"email": "old@example.com", "password": "correct_password"},
            )

        assert resp.status_code == 200
        assert mock_create.call_args.kwargs["uid"] == doc.id
        assert mock_create.call_args.kwargs["password"] == "correct_password"
        doc.reference.set.assert_awaited_with({"firebase_uid": doc.id}, merge=True)

    async def test_legacy_user_wrong_password_returns_401(
        self, client: AsyncClient, mock_db: MagicMock, firebase_settings
    ) -> None:
        doc = _user_doc(email="old@example.com", legacy_password="correct_password", migrated=False)
        _set_email_query_result(mock_db, [doc])

        with (
            patch(
                "app.api.v1.auth._firebase_password_signin",
                new=AsyncMock(return_value=None),
            ),
            patch("firebase_admin.auth.create_user") as mock_create,
        ):
            resp = await client.post(
                "/api/v1/auth/login",
                json={"email": "old@example.com", "password": "wrong_password"},
            )

        assert resp.status_code == 401
        mock_create.assert_not_called()

    async def test_locked_out_account_returns_429(
        self, client: AsyncClient, mock_db: MagicMock, mock_redis: AsyncMock, firebase_settings
    ) -> None:
        from app.core.login_guard import _MAX_ATTEMPTS

        doc = _user_doc(email="test@example.com")
        _set_email_query_result(mock_db, [doc])
        mock_redis.get.return_value = str(_MAX_ATTEMPTS)  # at the lockout threshold

        resp = await client.post(
            "/api/v1/auth/login",
            json={"email": "test@example.com", "password": "correct_password"},
        )

        assert resp.status_code == 429

    async def test_login_survives_redis_outage(
        self, client: AsyncClient, mock_db: MagicMock, mock_redis: AsyncMock, firebase_settings
    ) -> None:
        """An unreachable Redis must not break authentication.

        Regression: a deleted Redis instance let ConnectionError escape
        check_not_locked_out, turning every login into a 500.
        """
        from redis.exceptions import ConnectionError as RedisConnectionError

        doc = _user_doc(email="test@example.com")
        _set_email_query_result(mock_db, [doc])
        outage = RedisConnectionError("Error -2 connecting to redis:6379. Name or service not known.")
        mock_redis.get.side_effect = outage
        mock_redis.delete.side_effect = outage

        with patch(
            "app.api.v1.auth._firebase_password_signin",
            new=AsyncMock(return_value=doc.id),
        ):
            resp = await client.post(
                "/api/v1/auth/login",
                json={"email": "test@example.com", "password": "correct_password"},
            )

        assert resp.status_code == 200
        assert "access_token" in resp.json()

    async def test_bad_credentials_still_401_during_redis_outage(
        self, client: AsyncClient, mock_db: MagicMock, mock_redis: AsyncMock, firebase_settings
    ) -> None:
        """Failing open on the counter must not turn a rejection into a 500."""
        from redis.exceptions import ConnectionError as RedisConnectionError

        doc = _user_doc(email="test@example.com")
        _set_email_query_result(mock_db, [doc])
        outage = RedisConnectionError("Name or service not known.")
        mock_redis.get.side_effect = outage
        mock_redis.incr.side_effect = outage

        with patch(
            "app.api.v1.auth._firebase_password_signin",
            new=AsyncMock(return_value=None),
        ):
            resp = await client.post(
                "/api/v1/auth/login",
                json={"email": "test@example.com", "password": "wrong_password"},
            )

        assert resp.status_code == 401

    async def test_firestore_quota_error_returns_429(
        self, client: AsyncClient, mock_db: MagicMock, mock_redis: AsyncMock, firebase_settings
    ) -> None:
        query = mock_db.collection.return_value.where.return_value.limit.return_value
        query.get.side_effect = ResourceExhausted("429 Quota exceeded")

        resp = await client.post(
            "/api/v1/auth/login",
            json={"email": "test@example.com", "password": "password123"},
        )

        assert resp.status_code == 429
        assert resp.headers["content-type"] == "application/problem+json"

    async def test_not_configured_returns_422(
        self, client: AsyncClient, mock_db: MagicMock
    ) -> None:
        app.dependency_overrides[get_settings] = lambda: Settings(FIREBASE_WEB_API_KEY="")
        try:
            resp = await client.post(
                "/api/v1/auth/login",
                json={"email": "test@example.com", "password": "password123"},
            )
            assert resp.status_code == 422
        finally:
            app.dependency_overrides.pop(get_settings, None)


# ── Email verification ───────────────────────────────────────────────────────


class TestEmailVerification:
    def _unverified_doc(self, email: str = "test@example.com") -> MagicMock:
        doc_id = str(uuid.uuid4())
        data = {
            "email": email,
            "hashed_password": "",
            "firebase_uid": doc_id,
            "tier": "free",
            "requires_email_verification": True,
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
        }
        snap = make_doc_snapshot(data, doc_id=doc_id)
        snap.reference.set = AsyncMock()
        return snap

    async def test_login_blocked_when_flagged_and_not_verified(
        self, client: AsyncClient, mock_db: MagicMock, firebase_settings
    ) -> None:
        doc = self._unverified_doc()
        _set_email_query_result(mock_db, [doc])
        fake_user_record = MagicMock(email_verified=False)

        with (
            patch(
                "app.api.v1.auth._firebase_password_signin",
                new=AsyncMock(return_value=doc.id),
            ),
            patch("firebase_admin.auth.get_user", return_value=fake_user_record),
        ):
            resp = await client.post(
                "/api/v1/auth/login",
                json={"email": doc.to_dict()["email"], "password": "correct_password"},
            )

        assert resp.status_code == 403
        assert "verify" in resp.json()["detail"].lower()

    async def test_login_succeeds_and_clears_flag_once_verified(
        self, client: AsyncClient, mock_db: MagicMock, firebase_settings
    ) -> None:
        doc = self._unverified_doc()
        _set_email_query_result(mock_db, [doc])
        fake_user_record = MagicMock(email_verified=True)

        with (
            patch(
                "app.api.v1.auth._firebase_password_signin",
                new=AsyncMock(return_value=doc.id),
            ),
            patch("firebase_admin.auth.get_user", return_value=fake_user_record),
        ):
            resp = await client.post(
                "/api/v1/auth/login",
                json={"email": doc.to_dict()["email"], "password": "correct_password"},
            )

        assert resp.status_code == 200
        assert "access_token" in resp.json()
        mock_db.collection.return_value.document.return_value.update.assert_any_call(
            {"requires_email_verification": False}
        )

    async def test_legacy_user_without_flag_is_not_blocked(
        self, client: AsyncClient, mock_db: MagicMock, firebase_settings
    ) -> None:
        """Existing accounts predating this feature have no
        requires_email_verification field and must be grandfathered in."""
        doc = _user_doc(email="grandfathered@example.com")
        _set_email_query_result(mock_db, [doc])

        with (
            patch(
                "app.api.v1.auth._firebase_password_signin",
                new=AsyncMock(return_value=doc.id),
            ),
            patch("firebase_admin.auth.get_user") as mock_get_user,
        ):
            resp = await client.post(
                "/api/v1/auth/login",
                json={"email": "grandfathered@example.com", "password": "correct_password"},
            )

        assert resp.status_code == 200
        mock_get_user.assert_not_called()

    async def test_resend_verification_sends_email_on_valid_credentials(
        self, client: AsyncClient, firebase_settings
    ) -> None:
        with (
            patch(
                "app.api.v1.auth._firebase_password_signin_response",
                new=AsyncMock(return_value={"localId": "uid-123", "idToken": "id-token-xyz"}),
            ),
            patch("app.api.v1.auth._send_verification_email", new=AsyncMock()) as mock_send,
        ):
            resp = await client.post(
                "/api/v1/auth/resend-verification",
                json={"email": "test@example.com", "password": "correct_password"},
            )

        assert resp.status_code == 200
        assert "sent" in resp.json()["message"].lower()
        mock_send.assert_awaited_once_with("test-key", "id-token-xyz")

    async def test_resend_verification_rejects_wrong_password(
        self, client: AsyncClient, firebase_settings
    ) -> None:
        with patch(
            "app.api.v1.auth._firebase_password_signin_response",
            new=AsyncMock(return_value=None),
        ):
            resp = await client.post(
                "/api/v1/auth/resend-verification",
                json={"email": "test@example.com", "password": "wrong_password"},
            )

        assert resp.status_code == 401

    async def test_resend_verification_not_configured_returns_422(self, client: AsyncClient) -> None:
        app.dependency_overrides[get_settings] = lambda: Settings(FIREBASE_WEB_API_KEY="")
        try:
            resp = await client.post(
                "/api/v1/auth/resend-verification",
                json={"email": "test@example.com", "password": "whatever123"},
            )
            assert resp.status_code == 422
        finally:
            app.dependency_overrides.pop(get_settings, None)


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
