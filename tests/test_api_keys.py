"""Tests for API key auth: core helpers, the get_api_key dependency, and /keys endpoints."""
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import AsyncClient

from app.core.api_keys import (
    KEY_PREFIX,
    display_prefix,
    generate_api_key,
    hash_api_key,
    looks_like_api_key,
)
from app.core.exceptions import UnauthorizedError
from app.deps import get_api_key
from tests.conftest import make_doc_snapshot

# ── Core helpers ──────────────────────────────────────────────────────────────


class TestKeyHelpers:
    def test_generate_has_prefix_and_entropy(self) -> None:
        key = generate_api_key()
        assert key.startswith(KEY_PREFIX)
        assert len(key) > len(KEY_PREFIX) + 40  # 32 bytes url-safe ≈ 43 chars

    def test_generate_is_unique(self) -> None:
        assert generate_api_key() != generate_api_key()

    def test_hash_is_deterministic_sha256_hex(self) -> None:
        key = generate_api_key()
        digest = hash_api_key(key)
        assert digest == hash_api_key(key)
        assert len(digest) == 64
        int(digest, 16)  # valid hex

    def test_display_prefix_is_short(self) -> None:
        key = generate_api_key()
        prefix = display_prefix(key)
        assert key.startswith(prefix)
        assert len(prefix) == 12

    def test_looks_like_api_key(self) -> None:
        assert looks_like_api_key(generate_api_key())
        assert not looks_like_api_key("eyJhbGciOiJIUzI1NiJ9.jwt.token")


# ── get_api_key dependency ────────────────────────────────────────────────────


def _key_doc(
    *,
    owner: str,
    key: str,
    revoked_at: datetime | None = None,
    doc_id: str | None = None,
) -> MagicMock:
    """A Firestore snapshot for an api_keys/{id} doc, with an awaitable .reference.set."""
    snap = make_doc_snapshot(
        {
            "owner_user_id": owner,
            "name": "test key",
            "prefix": display_prefix(key),
            "key_hash": hash_api_key(key),
            "created_at": datetime.now(UTC),
            "last_used_at": None,
            "revoked_at": revoked_at,
        },
        doc_id=doc_id or str(uuid.uuid4()),
    )
    snap.reference.set = AsyncMock()
    return snap


def _db_returning(docs: list) -> MagicMock:
    """A Firestore mock whose api_keys hash-lookup query returns `docs`."""
    db = MagicMock()
    db.collection.return_value.where.return_value.limit.return_value.get = AsyncMock(
        return_value=docs
    )
    return db


class TestGetApiKeyDependency:
    async def test_missing_header_rejected(self) -> None:
        with pytest.raises(UnauthorizedError):
            await get_api_key(_db_returning([]), authorization=None)

    async def test_non_bearer_header_rejected(self) -> None:
        with pytest.raises(UnauthorizedError):
            await get_api_key(_db_returning([]), authorization="Basic abc123")

    async def test_jwt_shaped_token_rejected_without_lookup(self) -> None:
        db = _db_returning([])
        with pytest.raises(UnauthorizedError):
            await get_api_key(db, authorization="Bearer eyJhbGciOiJIUzI1NiJ9.x.y")
        db.collection.assert_not_called()

    async def test_unknown_key_rejected(self) -> None:
        with pytest.raises(UnauthorizedError):
            await get_api_key(_db_returning([]), authorization=f"Bearer {generate_api_key()}")

    async def test_revoked_key_rejected(self) -> None:
        key = generate_api_key()
        owner = str(uuid.uuid4())
        doc = _key_doc(owner=owner, key=key, revoked_at=datetime.now(UTC))
        with pytest.raises(UnauthorizedError):
            await get_api_key(_db_returning([doc]), authorization=f"Bearer {key}")

    async def test_valid_key_resolves(self) -> None:
        key = generate_api_key()
        owner = str(uuid.uuid4())
        doc = _key_doc(owner=owner, key=key)
        api_key = await get_api_key(_db_returning([doc]), authorization=f"Bearer {key}")
        assert str(api_key.owner_user_id) == owner
        assert api_key.key_hash == hash_api_key(key)
        assert not api_key.revoked

    async def test_valid_key_stamps_last_used(self) -> None:
        key = generate_api_key()
        doc = _key_doc(owner=str(uuid.uuid4()), key=key)
        await get_api_key(_db_returning([doc]), authorization=f"Bearer {key}")
        doc.reference.set.assert_awaited_once()
        assert "last_used_at" in doc.reference.set.await_args[0][0]

    async def test_second_call_served_from_cache(self) -> None:
        key = generate_api_key()
        doc = _key_doc(owner=str(uuid.uuid4()), key=key)
        db = _db_returning([doc])
        first = await get_api_key(db, authorization=f"Bearer {key}")
        second = await get_api_key(db, authorization=f"Bearer {key}")
        assert first is second
        db.collection.return_value.where.return_value.limit.return_value.get.assert_awaited_once()


# ── /api/v1/keys endpoints ────────────────────────────────────────────────────


def _stored_key_doc(
    *,
    owner: str,
    name: str = "k",
    created_at: datetime | None = None,
    revoked_at: datetime | None = None,
    doc_id: str | None = None,
) -> MagicMock:
    key = generate_api_key()
    return make_doc_snapshot(
        {
            "owner_user_id": owner,
            "name": name,
            "prefix": display_prefix(key),
            "key_hash": hash_api_key(key),
            "created_at": created_at or datetime.now(UTC),
            "last_used_at": None,
            "revoked_at": revoked_at,
        },
        doc_id=doc_id or str(uuid.uuid4()),
    )


class TestCreateApiKey:
    async def test_create_returns_full_key_once(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        mock_db.collection.return_value.where.return_value.get.return_value = []
        resp = await auth_client.post("/api/v1/keys", json={"name": "CI pipeline"})
        assert resp.status_code == 201
        body = resp.json()
        assert body["key"].startswith(KEY_PREFIX)
        assert body["prefix"] == body["key"][:12]
        assert body["name"] == "CI pipeline"

    async def test_create_stores_hash_not_raw_key(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        mock_db.collection.return_value.where.return_value.get.return_value = []
        resp = await auth_client.post("/api/v1/keys", json={"name": "k"})
        body = resp.json()
        stored = mock_db.collection.return_value.document.return_value.set.await_args[0][0]
        assert stored["key_hash"] == hash_api_key(body["key"])
        assert stored["owner_user_id"] == str(test_user.id)
        assert body["key"] not in str(stored)

    async def test_create_blocked_at_active_key_limit(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        existing = [_stored_key_doc(owner=str(test_user.id)) for _ in range(10)]
        mock_db.collection.return_value.where.return_value.get.return_value = existing
        resp = await auth_client.post("/api/v1/keys", json={"name": "one too many"})
        assert resp.status_code == 422

    async def test_revoked_keys_dont_count_toward_limit(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        existing = [
            _stored_key_doc(owner=str(test_user.id), revoked_at=datetime.now(UTC))
            for _ in range(10)
        ]
        mock_db.collection.return_value.where.return_value.get.return_value = existing
        resp = await auth_client.post("/api/v1/keys", json={"name": "fresh"})
        assert resp.status_code == 201

    async def test_empty_name_rejected(
        self, auth_client: AsyncClient, mock_db: MagicMock
    ) -> None:
        resp = await auth_client.post("/api/v1/keys", json={"name": "   "})
        assert resp.status_code == 422

    async def test_requires_auth(self, client: AsyncClient) -> None:
        resp = await client.post("/api/v1/keys", json={"name": "k"})
        assert resp.status_code == 401


class TestListApiKeys:
    async def test_list_newest_first_without_hashes(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        now = datetime.now(UTC)
        older = _stored_key_doc(owner=str(test_user.id), name="older", created_at=now - timedelta(days=1))
        newer = _stored_key_doc(owner=str(test_user.id), name="newer", created_at=now)
        mock_db.collection.return_value.where.return_value.get.return_value = [older, newer]
        resp = await auth_client.get("/api/v1/keys")
        assert resp.status_code == 200
        body = resp.json()
        assert [k["name"] for k in body] == ["newer", "older"]
        assert all("key_hash" not in k and "key" not in k for k in body)

    async def test_empty_list(
        self, auth_client: AsyncClient, mock_db: MagicMock
    ) -> None:
        mock_db.collection.return_value.where.return_value.get.return_value = []
        resp = await auth_client.get("/api/v1/keys")
        assert resp.status_code == 200
        assert resp.json() == []


class TestRevokeApiKey:
    async def test_revoke_own_key(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        key_id = str(uuid.uuid4())
        doc = _stored_key_doc(owner=str(test_user.id), doc_id=key_id)
        doc_ref = mock_db.collection.return_value.document.return_value
        doc_ref.get.return_value = doc
        resp = await auth_client.delete(f"/api/v1/keys/{key_id}")
        assert resp.status_code == 204
        stored = doc_ref.set.await_args[0][0]
        assert stored["revoked_at"] is not None

    async def test_revoke_is_idempotent(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        key_id = str(uuid.uuid4())
        doc = _stored_key_doc(
            owner=str(test_user.id), doc_id=key_id, revoked_at=datetime.now(UTC)
        )
        doc_ref = mock_db.collection.return_value.document.return_value
        doc_ref.get.return_value = doc
        resp = await auth_client.delete(f"/api/v1/keys/{key_id}")
        assert resp.status_code == 204
        doc_ref.set.assert_not_awaited()

    async def test_revoke_missing_key_404(
        self, auth_client: AsyncClient, mock_db: MagicMock
    ) -> None:
        doc_ref = mock_db.collection.return_value.document.return_value
        doc_ref.get.return_value = make_doc_snapshot(None)
        resp = await auth_client.delete(f"/api/v1/keys/{uuid.uuid4()}")
        assert resp.status_code == 404

    async def test_revoke_other_users_key_404(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        key_id = str(uuid.uuid4())
        doc = _stored_key_doc(owner=str(uuid.uuid4()), doc_id=key_id)  # different owner
        doc_ref = mock_db.collection.return_value.document.return_value
        doc_ref.get.return_value = doc
        resp = await auth_client.delete(f"/api/v1/keys/{key_id}")
        assert resp.status_code == 404
        doc_ref.set.assert_not_awaited()
