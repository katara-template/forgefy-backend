"""Tests for per-user build-model resolution and the /account/build-model endpoint."""
from unittest.mock import MagicMock

import pytest
from httpx import AsyncClient

from app.core.build_model import get_effective_build_model, get_user_build_model
from tests.conftest import make_doc_snapshot, wire_firestore_chain


def _routed_db(users_data: dict | None, system_data: dict | None) -> MagicMock:
    """A Firestore mock whose users/{id} and system/config docs differ."""
    db = MagicMock()
    wire_firestore_chain(db)

    users_node = MagicMock()
    wire_firestore_chain(users_node)
    users_node.document.return_value.get.return_value = make_doc_snapshot(users_data)

    system_node = MagicMock()
    wire_firestore_chain(system_node)
    system_node.document.return_value.get.return_value = make_doc_snapshot(system_data)

    def _route(name: str):
        return {"users": users_node, "system": system_node}.get(name, MagicMock())

    db.collection.side_effect = _route
    return db


class TestGetUserBuildModel:
    async def test_returns_none_when_no_user_id(self) -> None:
        assert await get_user_build_model(MagicMock(), "") is None

    async def test_returns_none_when_doc_missing(self) -> None:
        db = _routed_db(users_data=None, system_data=None)
        assert await get_user_build_model(db, "user-1") is None

    async def test_returns_none_when_field_unset(self) -> None:
        db = _routed_db(users_data={"email": "a@b.com"}, system_data=None)
        assert await get_user_build_model(db, "user-1") is None

    async def test_returns_the_users_override(self) -> None:
        db = _routed_db(users_data={"build_model": "claude"}, system_data=None)
        assert await get_user_build_model(db, "user-1") == "claude"


class TestGetEffectiveBuildModel:
    async def test_user_override_wins(self) -> None:
        settings = MagicMock(BUILD_MODEL="gemini")
        db = _routed_db(users_data={"build_model": "claude"}, system_data={"build_model": "gpt"})
        assert await get_effective_build_model(db, settings, user_id="user-1") == "claude"

    async def test_falls_back_to_system_config(self) -> None:
        settings = MagicMock(BUILD_MODEL="gemini")
        db = _routed_db(users_data=None, system_data={"build_model": "gpt"})
        assert await get_effective_build_model(db, settings, user_id="user-1") == "gpt"

    async def test_falls_back_to_env_default(self) -> None:
        settings = MagicMock(BUILD_MODEL="gemini")
        db = _routed_db(users_data=None, system_data=None)
        assert await get_effective_build_model(db, settings, user_id="user-1") == "gemini"

    async def test_no_user_id_skips_user_lookup(self) -> None:
        settings = MagicMock(BUILD_MODEL="gemini")
        db = _routed_db(users_data={"build_model": "claude"}, system_data={"build_model": "gpt"})
        assert await get_effective_build_model(db, settings, user_id=None) == "gpt"


class TestBuildModelEndpoint:
    async def test_get_defaults_to_platform_setting(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        db = _routed_db(users_data=None, system_data=None)
        mock_db.collection.side_effect = db.collection.side_effect
        resp = await auth_client.get("/api/v1/account/build-model")
        assert resp.status_code == 200
        body = resp.json()
        assert body["is_custom"] is False

    async def test_get_reflects_own_override(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        db = _routed_db(users_data={"build_model": "claude"}, system_data=None)
        mock_db.collection.side_effect = db.collection.side_effect
        resp = await auth_client.get("/api/v1/account/build-model")
        assert resp.status_code == 200
        body = resp.json()
        assert body == {"model": "claude", "is_custom": True}

    async def test_patch_rejects_invalid_model(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        resp = await auth_client.patch(
            "/api/v1/account/build-model", json={"model": "not-a-real-model"}
        )
        assert resp.status_code == 422

    async def test_patch_persists_the_choice(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        db = _routed_db(users_data=None, system_data=None)
        mock_db.collection.side_effect = db.collection.side_effect
        resp = await auth_client.patch("/api/v1/account/build-model", json={"model": "gpt"})
        assert resp.status_code == 200
        assert resp.json() == {"model": "gpt", "is_custom": True}
        db.collection.side_effect("users").document.return_value.set.assert_awaited_once_with(
            {"build_model": "gpt"}, merge=True
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
