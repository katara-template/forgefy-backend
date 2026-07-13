"""Tests for the provider-neutral Firebase OAuth sign-in endpoint."""
import uuid
from unittest.mock import MagicMock, patch

from httpx import AsyncClient

from tests.conftest import make_doc_snapshot


def _decoded_token(**overrides) -> dict:
    base = {
        "email": "dev@example.com",
        "email_verified": True,
        "firebase": {"sign_in_provider": "github.com"},
    }
    base.update(overrides)
    return base


def _no_existing_user(mock_db: MagicMock) -> None:
    mock_db.collection.return_value.where.return_value.limit.return_value.get.return_value = []


def _existing_user(mock_db: MagicMock, doc_id: str) -> None:
    snap = make_doc_snapshot({"email": "dev@example.com"}, doc_id=doc_id)
    mock_db.collection.return_value.where.return_value.limit.return_value.get.return_value = [snap]


class TestOAuthSignIn:
    async def test_invalid_token_returns_401(self, client: AsyncClient) -> None:
        with patch("firebase_admin.auth.verify_id_token", side_effect=ValueError("bad")):
            resp = await client.post("/api/v1/auth/oauth", json={"id_token": "junk"})
        assert resp.status_code == 401

    async def test_new_user_is_created_and_gets_tokens(
        self, client: AsyncClient, mock_db: MagicMock
    ) -> None:
        _no_existing_user(mock_db)
        with patch("firebase_admin.auth.verify_id_token", return_value=_decoded_token()):
            resp = await client.post("/api/v1/auth/oauth", json={"id_token": "t"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["access_token"] and body["refresh_token"]

        created = mock_db.collection.return_value.document.return_value.set.call_args[0][0]
        assert created["email"] == "dev@example.com"
        assert created["hashed_password"] == ""

    async def test_existing_user_signs_into_same_account(
        self, client: AsyncClient, mock_db: MagicMock
    ) -> None:
        _existing_user(mock_db, doc_id=str(uuid.uuid4()))
        with patch("firebase_admin.auth.verify_id_token", return_value=_decoded_token()):
            resp = await client.post("/api/v1/auth/oauth", json={"id_token": "t"})

        assert resp.status_code == 200
        # No new user document is written on sign-in to an existing account.
        mock_db.collection.return_value.document.return_value.set.assert_not_called()

    async def test_unverified_email_cannot_claim_existing_account(
        self, client: AsyncClient, mock_db: MagicMock
    ) -> None:
        _existing_user(mock_db, doc_id=str(uuid.uuid4()))
        decoded = _decoded_token(email_verified=False)
        with patch("firebase_admin.auth.verify_id_token", return_value=decoded):
            resp = await client.post("/api/v1/auth/oauth", json={"id_token": "t"})
        assert resp.status_code == 401

    async def test_missing_email_returns_401(
        self, client: AsyncClient, mock_db: MagicMock
    ) -> None:
        decoded = _decoded_token()
        del decoded["email"]
        with patch("firebase_admin.auth.verify_id_token", return_value=decoded):
            resp = await client.post("/api/v1/auth/oauth", json={"id_token": "t"})
        assert resp.status_code == 401

    async def test_legacy_google_route_still_works(
        self, client: AsyncClient, mock_db: MagicMock
    ) -> None:
        _no_existing_user(mock_db)
        decoded = _decoded_token(firebase={"sign_in_provider": "google.com"})
        with patch("firebase_admin.auth.verify_id_token", return_value=decoded):
            resp = await client.post("/api/v1/auth/google", json={"id_token": "t"})
        assert resp.status_code == 200
