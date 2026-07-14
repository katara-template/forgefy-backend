"""Tests for the provider-neutral Firebase sign-in endpoint (/auth/oauth).

The endpoint looks users up twice: first by firebase_uid (stable binding),
then by email (one-time linking of pre-Firebase accounts). The query mock's
side_effect list maps to those calls in order.
"""
import uuid
from unittest.mock import MagicMock, patch

from httpx import AsyncClient

from tests.conftest import make_doc_snapshot

_FB_UID = "firebase-uid-1"


def _decoded_token(**overrides) -> dict:
    base = {
        "uid": _FB_UID,
        "email": "dev@example.com",
        "email_verified": True,
        "firebase": {"sign_in_provider": "github.com"},
    }
    base.update(overrides)
    return base


def _set_query_results(mock_db: MagicMock, *results: list) -> None:
    """Queue results for successive users-collection queries (uid, then email)."""
    mock_db.collection.return_value.where.return_value.limit.return_value.get.side_effect = list(
        results
    )


def _user_snap(doc_id: str | None = None) -> MagicMock:
    return make_doc_snapshot({"email": "dev@example.com"}, doc_id=doc_id or str(uuid.uuid4()))


class TestOAuthSignIn:
    async def test_invalid_token_returns_401(self, client: AsyncClient) -> None:
        with patch("firebase_admin.auth.verify_id_token", side_effect=ValueError("bad")):
            resp = await client.post("/api/v1/auth/oauth", json={"id_token": "junk"})
        assert resp.status_code == 401

    async def test_new_user_is_created_and_gets_tokens(
        self, client: AsyncClient, mock_db: MagicMock
    ) -> None:
        _set_query_results(mock_db, [], [])  # no uid match, no email match
        with patch("firebase_admin.auth.verify_id_token", return_value=_decoded_token()):
            resp = await client.post("/api/v1/auth/oauth", json={"id_token": "t"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["access_token"] and body["refresh_token"]

        created = mock_db.collection.return_value.document.return_value.set.call_args[0][0]
        assert created["email"] == "dev@example.com"
        assert created["hashed_password"] == ""
        assert created["firebase_uid"] == _FB_UID

    async def test_bound_user_signs_into_same_account(
        self, client: AsyncClient, mock_db: MagicMock
    ) -> None:
        _set_query_results(mock_db, [_user_snap()])  # uid match on first query
        with patch("firebase_admin.auth.verify_id_token", return_value=_decoded_token()):
            resp = await client.post("/api/v1/auth/oauth", json={"id_token": "t"})

        assert resp.status_code == 200
        # Bound accounts need no doc writes on sign-in.
        mock_db.collection.return_value.document.return_value.set.assert_not_called()
        mock_db.collection.return_value.document.return_value.update.assert_not_called()

    async def test_email_match_links_firebase_uid(
        self, client: AsyncClient, mock_db: MagicMock
    ) -> None:
        _set_query_results(mock_db, [], [_user_snap()])  # no uid match, email match
        with patch("firebase_admin.auth.verify_id_token", return_value=_decoded_token()):
            resp = await client.post("/api/v1/auth/oauth", json={"id_token": "t"})

        assert resp.status_code == 200
        linked = mock_db.collection.return_value.document.return_value.update.call_args[0][0]
        assert linked["firebase_uid"] == _FB_UID

    async def test_unverified_email_cannot_claim_existing_account(
        self, client: AsyncClient, mock_db: MagicMock
    ) -> None:
        _set_query_results(mock_db, [], [_user_snap()])
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
        _set_query_results(mock_db, [], [])
        decoded = _decoded_token(firebase={"sign_in_provider": "google.com"})
        with patch("firebase_admin.auth.verify_id_token", return_value=decoded):
            resp = await client.post("/api/v1/auth/google", json={"id_token": "t"})
        assert resp.status_code == 200
