"""Project-level Supabase provisioning endpoint tests."""
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import AsyncClient

from app.core.crypto import decrypt


def _project_doc(owner_id: uuid.UUID, **overrides) -> MagicMock:
    from tests.conftest import make_doc_snapshot

    now = datetime.now(UTC)
    data = {
        "owner_id": str(owner_id),
        "app_name": "stockflow-manager",
        "template_key": "next",
        "repo_full_name": "acme/stockflow-manager",
        "github_url": "https://github.com/acme/stockflow-manager",
        "created_at": now,
        "updated_at": now,
    }
    data.update(overrides)
    return make_doc_snapshot(data, doc_id=str(uuid.uuid4()))


class TestConnectSupabase:
    async def test_no_auth_returns_401(self, client: AsyncClient) -> None:
        resp = await client.post(
            f"/api/v1/projects/{uuid.uuid4()}/supabase/connect",
            json={"organization_id": "org-1"},
        )
        assert resp.status_code == 401

    async def test_not_owner_returns_403(
        self, auth_client: AsyncClient, mock_db: MagicMock
    ) -> None:
        other_owner = uuid.uuid4()
        mock_db.collection.return_value.document.return_value.get.return_value = (
            _project_doc(other_owner)
        )
        resp = await auth_client.post(
            f"/api/v1/projects/{uuid.uuid4()}/supabase/connect",
            json={"organization_id": "org-1"},
        )
        assert resp.status_code == 403

    async def test_no_linked_supabase_account_returns_422(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        mock_db.collection.return_value.document.return_value.get.return_value = (
            _project_doc(test_user.id)
        )
        with patch(
            "app.build.supabase_token.get_valid_supabase_token",
            new=AsyncMock(return_value=None),
        ):
            resp = await auth_client.post(
                f"/api/v1/projects/{uuid.uuid4()}/supabase/connect",
                json={"organization_id": "org-1"},
            )
        assert resp.status_code == 422

    async def test_success_provisions_and_stores_fields(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        mock_db.collection.return_value.document.return_value.get.return_value = (
            _project_doc(test_user.id)
        )
        with (
            patch(
                "app.build.supabase_token.get_valid_supabase_token",
                new=AsyncMock(return_value="sb-access-token"),
            ),
            patch(
                "app.integrations.supabase_management.create_project",
                new=AsyncMock(return_value={"id": "abcxyzproj"}),
            ),
            patch(
                "app.integrations.supabase_management.get_api_keys",
                new=AsyncMock(return_value=[
                    {"name": "anon", "api_key": "anon-key-value"},
                    {"name": "service_role", "api_key": "service-role-key-value"},
                ]),
            ),
        ):
            resp = await auth_client.post(
                f"/api/v1/projects/{uuid.uuid4()}/supabase/connect",
                json={"organization_id": "org-1"},
            )

        assert resp.status_code == 200
        # Already-built project (repo_full_name set, no pending decision) — connecting
        # only provisions; wiring the code needs a separate explicit confirmation.
        assert resp.json() == {
            "status": "connected", "project_ref": "abcxyzproj", "prompt_wire_in": True,
        }

        payload = mock_db.collection.return_value.document.return_value.update.call_args[0][0]
        assert payload["supabase_project_ref"] == "abcxyzproj"
        assert payload["supabase_url"] == "https://abcxyzproj.supabase.co"
        assert payload["supabase_anon_key"] == "anon-key-value"
        # The anon key is meant to be public; the DB password must never be.
        assert payload["supabase_db_pass"] != payload["supabase_anon_key"]
        assert "service-role-key-value" not in payload.values()
        decrypted_pass = decrypt(payload["supabase_db_pass"])
        assert decrypted_pass and decrypted_pass not in payload["supabase_db_pass"]

    async def test_pending_db_decision_dispatches_withheld_build(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        """Connecting while an initial build was withheld pending this exact
        decision must clear the flag and finally dispatch the build."""
        mock_db.collection.return_value.document.return_value.get.return_value = (
            _project_doc(
                test_user.id,
                repo_full_name="",
                github_url="",
                db_decision_pending=True,
                db_decision_reason="Tracks inventory.",
                session_id=str(uuid.uuid4()),
            )
        )
        with (
            patch(
                "app.build.supabase_token.get_valid_supabase_token",
                new=AsyncMock(return_value="sb-access-token"),
            ),
            patch(
                "app.integrations.supabase_management.create_project",
                new=AsyncMock(return_value={"id": "abcxyzproj"}),
            ),
            patch(
                "app.integrations.supabase_management.get_api_keys",
                new=AsyncMock(return_value=[{"name": "anon", "api_key": "anon-key-value"}]),
            ),
            patch("app.workers.build_worker.run_build.apply_async") as mock_dispatch,
        ):
            resp = await auth_client.post(
                f"/api/v1/projects/{uuid.uuid4()}/supabase/connect",
                json={"organization_id": "org-1"},
            )

        assert resp.status_code == 200
        assert resp.json()["build_queued"] is True
        mock_dispatch.assert_called_once()
