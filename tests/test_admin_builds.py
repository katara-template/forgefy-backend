"""Tests for GET /api/v1/admin/builds — name/description/artifact_url only.

Deliberately excludes any meeting/session linkage, framework, status, or
timestamp from the response — see app/api/v1/admin.py's list_builds.
"""
import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock

from httpx import AsyncClient

from app.deps import get_current_user
from app.main import app
from tests.conftest import make_doc_snapshot, wire_firestore_chain


def _project_doc(**overrides) -> MagicMock:
    data = {
        "app_name": "StockFlow",
        "artifact_url": "https://cdn.example.com/stockflow.apk",
        "blueprint_id": None,
        "created_at": datetime.now(UTC),
    }
    data.update(overrides)
    return make_doc_snapshot(data, doc_id=str(uuid.uuid4()))


class TestListBuilds:
    async def test_no_auth_returns_401(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/admin/builds")
        assert resp.status_code == 401

    async def test_non_admin_returns_403(self, auth_client: AsyncClient) -> None:
        resp = await auth_client.get("/api/v1/admin/builds")
        assert resp.status_code == 403

    async def test_returns_name_description_artifact_only(
        self, client: AsyncClient, mock_db: MagicMock, admin_user
    ) -> None:
        app.dependency_overrides[get_current_user] = lambda: admin_user
        try:
            bp_id = str(uuid.uuid4())
            projects_col = MagicMock()
            wire_firestore_chain(projects_col)
            projects_col.order_by.return_value.limit.return_value.get.return_value = [
                _project_doc(app_name="StockFlow", blueprint_id=bp_id),
            ]

            blueprints_col = MagicMock()
            wire_firestore_chain(blueprints_col)
            blueprints_col.document.return_value.get.return_value = make_doc_snapshot(
                {"json_output": {"app_description": "An inventory tracking app"}},
                doc_id=bp_id,
            )

            mock_db.collection.side_effect = lambda name: {
                "projects": projects_col,
                "blueprints": blueprints_col,
            }[name]

            resp = await client.get("/api/v1/admin/builds")
        finally:
            del app.dependency_overrides[get_current_user]

        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        assert body[0] == {
            "name": "StockFlow",
            "description": "An inventory tracking app",
            "artifact_url": "https://cdn.example.com/stockflow.apk",
        }
        # No meeting/session/framework/status/timestamp fields anywhere.
        assert "session_id" not in body[0]
        assert "framework" not in body[0]
        assert "status" not in body[0]
        assert "meeting_id" not in body[0]
        assert "meetingTitle" not in body[0]

    async def test_no_blueprint_id_gives_empty_description(
        self, client: AsyncClient, mock_db: MagicMock, admin_user
    ) -> None:
        app.dependency_overrides[get_current_user] = lambda: admin_user
        try:
            projects_col = MagicMock()
            wire_firestore_chain(projects_col)
            projects_col.order_by.return_value.limit.return_value.get.return_value = [
                _project_doc(app_name="NoBlueprintApp", blueprint_id=None, artifact_url=None),
            ]

            mock_db.collection.side_effect = lambda name: {"projects": projects_col}[name]

            resp = await client.get("/api/v1/admin/builds")
        finally:
            del app.dependency_overrides[get_current_user]

        assert resp.status_code == 200
        assert resp.json()[0] == {
            "name": "NoBlueprintApp",
            "description": "",
            "artifact_url": None,
        }
