"""API tests for the admin-curated build-model catalogue endpoints.

Covers the admin CRUD surface (``/api/v1/admin/build-models``) that the dashboard
uses to *add* models, and the user-facing listing
(``/api/v1/account/build-models``) the model picker reads.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from httpx import AsyncClient

from app.config import get_settings
from app.core.build_model import DEFAULT_BUILD_MODELS
from app.deps import get_current_user
from app.main import app
from tests.conftest import make_doc_snapshot


def _wire_config(mock_db: MagicMock, data: dict | None) -> MagicMock:
    """Point the system/config read at ``data`` (None = document missing)."""
    doc = make_doc_snapshot(data)
    mock_db.collection.return_value.document.return_value.get.return_value = doc
    return mock_db


def _default_keys() -> set[str]:
    return {m["model"] for m in DEFAULT_BUILD_MODELS}


class TestAdminListBuildModels:
    async def test_requires_admin(self, auth_client: AsyncClient) -> None:
        resp = await auth_client.get("/api/v1/admin/build-models")
        assert resp.status_code == 403

    async def test_requires_auth(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/admin/build-models")
        assert resp.status_code == 401

    async def test_falls_back_to_shipped_defaults(
        self, client: AsyncClient, mock_db: MagicMock, admin_user
    ) -> None:
        app.dependency_overrides[get_current_user] = lambda: admin_user
        try:
            _wire_config(mock_db, None)  # no catalogue curated yet
            resp = await client.get("/api/v1/admin/build-models")
        finally:
            del app.dependency_overrides[get_current_user]

        assert resp.status_code == 200
        models = resp.json()
        assert {m["model"] for m in models} == _default_keys()
        # Metadata the picker renders is present.
        assert all({"model", "label", "provider", "sub"} <= set(m) for m in models)

    async def test_returns_curated_catalogue(
        self, client: AsyncClient, mock_db: MagicMock, admin_user
    ) -> None:
        app.dependency_overrides[get_current_user] = lambda: admin_user
        try:
            _wire_config(
                mock_db,
                {"build_models": [{"model": "deepseek", "label": "DeepSeek", "provider": "DeepSeek", "sub": ""}]},
            )
            resp = await client.get("/api/v1/admin/build-models")
        finally:
            del app.dependency_overrides[get_current_user]

        assert resp.status_code == 200
        assert resp.json() == [
            {"model": "deepseek", "label": "DeepSeek", "provider": "DeepSeek", "sub": ""}
        ]


class TestAdminAddBuildModel:
    async def test_requires_admin(self, auth_client: AsyncClient) -> None:
        resp = await auth_client.post("/api/v1/admin/build-models", json={"model": "x"})
        assert resp.status_code == 403

    async def test_adds_model_to_catalogue(
        self, client: AsyncClient, mock_db: MagicMock, admin_user
    ) -> None:
        app.dependency_overrides[get_current_user] = lambda: admin_user
        try:
            _wire_config(mock_db, {"build_models": [{"model": "claude", "label": "Claude"}]})
            resp = await client.post(
                "/api/v1/admin/build-models",
                json={"model": "deepseek", "label": "DeepSeek", "provider": "DeepSeek"},
            )
        finally:
            del app.dependency_overrides[get_current_user]

        assert resp.status_code == 200
        models = resp.json()
        assert [m["model"] for m in models] == ["claude", "deepseek"]
        # Persisted with merge=True on the shared system/config doc.
        args, kwargs = mock_db.collection.return_value.document.return_value.set.call_args
        assert args[0]["build_models"][-1]["model"] == "deepseek"
        assert kwargs["merge"] is True

    async def test_rejects_duplicate(
        self, client: AsyncClient, mock_db: MagicMock, admin_user
    ) -> None:
        app.dependency_overrides[get_current_user] = lambda: admin_user
        try:
            _wire_config(mock_db, {"build_models": [{"model": "claude", "label": "Claude"}]})
            resp = await client.post("/api/v1/admin/build-models", json={"model": "claude"})
        finally:
            del app.dependency_overrides[get_current_user]

        assert resp.status_code == 422
        assert "already" in resp.json()["detail"]

    async def test_rejects_empty_name(
        self, client: AsyncClient, mock_db: MagicMock, admin_user
    ) -> None:
        app.dependency_overrides[get_current_user] = lambda: admin_user
        try:
            _wire_config(mock_db, {"build_models": [{"model": "claude"}]})
            resp = await client.post("/api/v1/admin/build-models", json={"model": "   "})
        finally:
            del app.dependency_overrides[get_current_user]

        assert resp.status_code == 422
class TestAdminRemoveBuildModel:
    async def test_requires_admin(self, auth_client: AsyncClient) -> None:
        resp = await auth_client.delete("/api/v1/admin/build-models/claude")
        assert resp.status_code == 403

    async def test_removes_model(
        self, client: AsyncClient, mock_db: MagicMock, admin_user
    ) -> None:
        app.dependency_overrides[get_current_user] = lambda: admin_user
        try:
            _wire_config(
                mock_db,
                {
                    "build_model": "gemini",
                    "build_models": [{"model": "gemini"}, {"model": "claude"}],
                },
            )
            resp = await client.delete("/api/v1/admin/build-models/claude")
        finally:
            del app.dependency_overrides[get_current_user]

        assert resp.status_code == 200
        assert [m["model"] for m in resp.json()] == ["gemini"]

    async def test_refuses_to_remove_the_active_model(
        self, client: AsyncClient, mock_db: MagicMock, admin_user
    ) -> None:
        app.dependency_overrides[get_current_user] = lambda: admin_user
        try:
            _wire_config(
                mock_db,
                {
                    "build_model": "claude",
                    "build_models": [{"model": "gemini"}, {"model": "claude"}],
                },
            )
            resp = await client.delete("/api/v1/admin/build-models/claude")
        finally:
            del app.dependency_overrides[get_current_user]

        assert resp.status_code == 422
        assert "active" in resp.json()["detail"]

    async def test_unknown_model_is_404(
        self, client: AsyncClient, mock_db: MagicMock, admin_user
    ) -> None:
        app.dependency_overrides[get_current_user] = lambda: admin_user
        try:
            _wire_config(mock_db, {"build_models": [{"model": "claude"}]})
            resp = await client.delete("/api/v1/admin/build-models/nope")
        finally:
            del app.dependency_overrides[get_current_user]

        assert resp.status_code == 404


class TestAccountListBuildModels:
    async def test_requires_auth(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/account/build-models")
        assert resp.status_code == 401

    async def test_returns_the_same_catalogue_to_users(
        self, auth_client: AsyncClient, mock_db: MagicMock
    ) -> None:
        _wire_config(
            mock_db,
            {"build_models": [{"model": "gemini", "label": "Gemini", "provider": "Google", "sub": "fast"}]},
        )
        resp = await auth_client.get("/api/v1/account/build-models")
        assert resp.status_code == 200
        assert resp.json() == [
            {"model": "gemini", "label": "Gemini", "provider": "Google", "sub": "fast"}
        ]

    async def test_falls_back_to_defaults_when_uncurated(
        self, auth_client: AsyncClient, mock_db: MagicMock
    ) -> None:
        _wire_config(mock_db, None)
        resp = await auth_client.get("/api/v1/account/build-models")
        assert resp.status_code == 200
        assert {m["model"] for m in resp.json()} == _default_keys()


class TestAdminDeepSeekModels:
    """The portal lists DeepSeek's own models so an operator can switch builds
    without editing .env — these cover the listing, not the catalogue mutation
    (that is the existing POST above)."""

    async def test_requires_admin(self, auth_client: AsyncClient) -> None:
        resp = await auth_client.get("/api/v1/admin/deepseek-models")
        assert resp.status_code == 403

    async def test_requires_auth(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/admin/deepseek-models")
        assert resp.status_code == 401

    async def test_reports_not_configured_without_a_key(
        self, client: AsyncClient, mock_db: MagicMock, admin_user
    ) -> None:
        app.dependency_overrides[get_current_user] = lambda: admin_user
        app.dependency_overrides[get_settings] = lambda: MagicMock(DEEPSEEK_API_KEY="  ")
        try:
            _wire_config(mock_db, None)
            resp = await client.get("/api/v1/admin/deepseek-models")
        finally:
            del app.dependency_overrides[get_current_user]
            del app.dependency_overrides[get_settings]

        assert resp.status_code == 200
        body = resp.json()
        # A blank key must be reported, not treated as a live (failing) lookup.
        assert body["configured"] is False
        assert body["models"] == []
        assert "DEEPSEEK_API_KEY" in body["detail"]

    async def test_lists_models_against_the_configured_key(
        self, client: AsyncClient, mock_db: MagicMock, admin_user, monkeypatch
    ) -> None:
        app.dependency_overrides[get_current_user] = lambda: admin_user
        app.dependency_overrides[get_settings] = lambda: MagicMock(DEEPSEEK_API_KEY="ds-key")
        try:
            _wire_config(mock_db, {"build_models": [{"model": "deepseek:deepseek-v4-pro"}]})

            class _Resp:
                status_code = 200

                @staticmethod
                def json():
                    return {"data": [
                        {"id": "deepseek-flash", "name": "DeepSeek-V4.1-Flash",
                         "context_window": 1048576, "max_output_tokens": 393216,
                         "input_modalities": ["text", "image"]},
                        {"id": "deepseek-v4-pro", "name": "DeepSeek-V4-Pro",
                         "context_window": 1048576, "max_output_tokens": 393216,
                         "input_modalities": ["text"]},
                    ]}

            seen: dict = {}

            def _fake_get(url, **kw):
                seen["url"] = url
                seen["auth"] = (kw.get("headers") or {}).get("Authorization")
                return _Resp()

            import httpx

            monkeypatch.setattr(httpx, "get", _fake_get)
            resp = await client.get("/api/v1/admin/deepseek-models")
        finally:
            del app.dependency_overrides[get_current_user]
            del app.dependency_overrides[get_settings]

        assert resp.status_code == 200
        body = resp.json()
        assert body["configured"] is True
        # It must talk to DeepSeek directly with this deployment's own key.
        assert seen["url"] == "https://api.deepseek.com/models"
        assert seen["auth"] == "Bearer ds-key"

        by_id = {m["id"]: m for m in body["models"]}
        assert by_id["deepseek-flash"]["supports_vision"] is True
        assert by_id["deepseek-v4-pro"]["supports_vision"] is False
        # Only the model already in the catalogue is flagged as such.
        assert by_id["deepseek-flash"]["in_catalogue"] is False
        assert by_id["deepseek-v4-pro"]["in_catalogue"] is True

    async def test_upstream_failure_is_reported_not_raised(
        self, client: AsyncClient, mock_db: MagicMock, admin_user, monkeypatch
    ) -> None:
        app.dependency_overrides[get_current_user] = lambda: admin_user
        app.dependency_overrides[get_settings] = lambda: MagicMock(DEEPSEEK_API_KEY="ds-key")
        try:
            _wire_config(mock_db, None)

            class _Resp:
                status_code = 401

            import httpx

            def _fake_get(url, **kw):
                raise httpx.ConnectError("boom")

            monkeypatch.setattr(httpx, "get", _fake_get)
            resp = await client.get("/api/v1/admin/deepseek-models")
        finally:
            del app.dependency_overrides[get_current_user]
            del app.dependency_overrides[get_settings]

        # A dead/unreachable provider must not 500 the admin page.
        assert resp.status_code == 200
        body = resp.json()
        assert body["models"] == []
        assert body["detail"]


class TestDynamicValidation:
    """A model an admin added is assignable — the whole point of the catalogue."""

    async def test_per_user_override_accepts_a_curated_model(
        self, auth_client: AsyncClient, mock_db: MagicMock
    ) -> None:
        _wire_config(mock_db, {"build_models": [{"model": "deepseek"}]})
        resp = await auth_client.patch(
            "/api/v1/account/build-model", json={"model": "deepseek"}
        )
        assert resp.status_code == 200
        assert resp.json() == {"model": "deepseek", "is_custom": True}

    async def test_admin_can_activate_a_curated_model(
        self, client: AsyncClient, mock_db: MagicMock, admin_user
    ) -> None:
        app.dependency_overrides[get_current_user] = lambda: admin_user
        try:
            _wire_config(mock_db, {"build_models": [{"model": "deepseek"}]})
            resp = await client.patch("/api/v1/admin/build-model", json={"model": "deepseek"})
        finally:
            del app.dependency_overrides[get_current_user]

        assert resp.status_code == 200
        assert resp.json() == {"model": "deepseek"}

    async def test_admin_cannot_activate_an_unlisted_model(
        self, client: AsyncClient, mock_db: MagicMock, admin_user
    ) -> None:
        app.dependency_overrides[get_current_user] = lambda: admin_user
        try:
            _wire_config(mock_db, {"build_models": [{"model": "gemini"}]})
            resp = await client.patch("/api/v1/admin/build-model", json={"model": "nope"})
        finally:
            del app.dependency_overrides[get_current_user]

        assert resp.status_code == 422