"""Tests for the database consent gate endpoints: skip-database and wire-database.

See app/api/v1/blueprints.py's approve_blueprint (withholds the initial build when
a database looks needed) and app/api/v1/projects.py's connect_* endpoints (the
paired "prompt_wire_in" signal) for the surrounding flow these two endpoints complete.
"""
import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from httpx import AsyncClient


def _project_doc(owner_id: uuid.UUID, **overrides) -> MagicMock:
    from tests.conftest import make_doc_snapshot

    now = datetime.now(UTC)
    data = {
        "owner_id": str(owner_id),
        "app_name": "stockflow-manager",
        "template_key": "next",
        "repo_full_name": "",
        "github_url": "",
        "session_id": str(uuid.uuid4()),
        "is_updating": False,
        "created_at": now,
        "updated_at": now,
    }
    data.update(overrides)
    return make_doc_snapshot(data, doc_id=str(uuid.uuid4()))


class TestSkipDatabase:
    async def test_no_auth_returns_401(self, client: AsyncClient) -> None:
        resp = await client.post(f"/api/v1/projects/{uuid.uuid4()}/skip-database")
        assert resp.status_code == 401

    async def test_not_owner_returns_403(
        self, auth_client: AsyncClient, mock_db: MagicMock
    ) -> None:
        other_owner = uuid.uuid4()
        mock_db.collection.return_value.document.return_value.get.return_value = (
            _project_doc(other_owner, db_decision_pending=True)
        )
        resp = await auth_client.post(f"/api/v1/projects/{uuid.uuid4()}/skip-database")
        assert resp.status_code == 403

    async def test_no_pending_decision_returns_422(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        mock_db.collection.return_value.document.return_value.get.return_value = (
            _project_doc(test_user.id, db_decision_pending=False)
        )
        resp = await auth_client.post(f"/api/v1/projects/{uuid.uuid4()}/skip-database")
        assert resp.status_code == 422

    async def test_success_clears_flag_and_dispatches_build(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        session_id = str(uuid.uuid4())
        mock_db.collection.return_value.document.return_value.get.return_value = (
            _project_doc(
                test_user.id,
                db_decision_pending=True,
                db_decision_reason="Tracks inventory.",
                session_id=session_id,
            )
        )
        with patch("app.workers.build_worker.run_build.apply_async") as mock_dispatch:
            resp = await auth_client.post(f"/api/v1/projects/{uuid.uuid4()}/skip-database")

        assert resp.status_code == 200
        assert resp.json() == {"status": "queued"}
        mock_dispatch.assert_called_once()
        assert mock_dispatch.call_args.kwargs["args"][0] == session_id

        payload = mock_db.collection.return_value.document.return_value.update.call_args[0][0]
        assert payload["db_decision_pending"] is False
        assert payload["db_decision_reason"] is None
        assert payload["is_updating"] is True


class TestWireDatabase:
    async def test_no_auth_returns_401(self, client: AsyncClient) -> None:
        resp = await client.post(f"/api/v1/projects/{uuid.uuid4()}/wire-database")
        assert resp.status_code == 401

    async def test_not_owner_returns_403(
        self, auth_client: AsyncClient, mock_db: MagicMock
    ) -> None:
        other_owner = uuid.uuid4()
        mock_db.collection.return_value.document.return_value.get.return_value = (
            _project_doc(other_owner, supabase_project_ref="abc")
        )
        resp = await auth_client.post(f"/api/v1/projects/{uuid.uuid4()}/wire-database")
        assert resp.status_code == 403

    async def test_no_database_connected_returns_422(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        mock_db.collection.return_value.document.return_value.get.return_value = (
            _project_doc(test_user.id)
        )
        resp = await auth_client.post(f"/api/v1/projects/{uuid.uuid4()}/wire-database")
        assert resp.status_code == 422

    async def test_already_updating_returns_422(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        mock_db.collection.return_value.document.return_value.get.return_value = (
            _project_doc(test_user.id, supabase_project_ref="abc", is_updating=True)
        )
        resp = await auth_client.post(f"/api/v1/projects/{uuid.uuid4()}/wire-database")
        assert resp.status_code == 422

    async def test_success_queues_update_naming_supabase(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        mock_db.collection.return_value.document.return_value.get.return_value = (
            _project_doc(test_user.id, supabase_project_ref="abc")
        )
        with patch("app.workers.update_worker.apply_update.apply_async") as mock_dispatch:
            resp = await auth_client.post(f"/api/v1/projects/{uuid.uuid4()}/wire-database")

        assert resp.status_code == 200
        assert resp.json() == {"status": "queued"}
        mock_dispatch.assert_called_once()
        prompt = mock_dispatch.call_args.kwargs["args"][1]
        assert "Supabase" in prompt

    async def test_success_queues_update_naming_neon(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        mock_db.collection.return_value.document.return_value.get.return_value = (
            _project_doc(test_user.id, neon_project_id="neon-1")
        )
        with patch("app.workers.update_worker.apply_update.apply_async") as mock_dispatch:
            resp = await auth_client.post(f"/api/v1/projects/{uuid.uuid4()}/wire-database")

        assert resp.status_code == 200
        prompt = mock_dispatch.call_args.kwargs["args"][1]
        assert "Neon" in prompt

    async def test_success_queues_update_naming_firebase(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        mock_db.collection.return_value.document.return_value.get.return_value = (
            _project_doc(test_user.id, firebase_project_id="fg-1")
        )
        with patch("app.workers.update_worker.apply_update.apply_async") as mock_dispatch:
            resp = await auth_client.post(f"/api/v1/projects/{uuid.uuid4()}/wire-database")

        assert resp.status_code == 200
        prompt = mock_dispatch.call_args.kwargs["args"][1]
        assert "Firebase" in prompt
