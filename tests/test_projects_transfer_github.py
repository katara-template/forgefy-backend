"""Tests for POST /api/v1/projects/{id}/transfer-github."""
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import AsyncClient

from tests.conftest import make_doc_snapshot


def _project_doc(owner_id: uuid.UUID, **overrides) -> MagicMock:
    now = datetime.now(UTC)
    data = {
        "owner_id": str(owner_id),
        "app_name": "stockflow-manager",
        "template_key": "next",
        "repo_full_name": "forgefy/stockflow-manager",
        "github_url": "https://github.com/forgefy/stockflow-manager",
        "repo_owner": "platform",
        "is_updating": False,
        "created_at": now,
        "updated_at": now,
    }
    data.update(overrides)
    return make_doc_snapshot(data, doc_id=str(uuid.uuid4()))


class TestTransferGitHub:
    async def test_no_auth_returns_401(self, client: AsyncClient) -> None:
        resp = await client.post(f"/api/v1/projects/{uuid.uuid4()}/transfer-github")
        assert resp.status_code == 401

    async def test_success_forks_repo_and_updates_project(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        project_id = uuid.uuid4()
        mock_db.collection.return_value.document.return_value.get.return_value = (
            _project_doc(test_user.id, repo_full_name="forgefy/stockflow-manager")
        )

        async def fake_get_token(*args, **kwargs):
            return "ghp_test"

        class FakeAsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, url: str, headers: dict, json: dict):
                assert url.endswith("/repos/forgefy/stockflow-manager/forks")
                return SimpleNamespace(
                    status_code=202,
                    json=lambda: {"full_name": "octocat/stockflow-manager", "html_url": "https://github.com/octocat/stockflow-manager"},
                )

        with (
            patch("app.build.github_token.get_valid_github_token", new=AsyncMock(side_effect=fake_get_token)),
            patch("app.api.v1.projects.httpx.AsyncClient", FakeAsyncClient),
        ):
            resp = await auth_client.post(f"/api/v1/projects/{project_id}/transfer-github")

        assert resp.status_code == 200
        payload = resp.json()
        assert payload["repo_full_name"] == "octocat/stockflow-manager"
        assert payload["github_url"] == "https://github.com/octocat/stockflow-manager"
        assert payload["repo_owner"] == "user"

        update_payload = mock_db.collection.return_value.document.return_value.update.call_args[0][0]
        assert update_payload["repo_full_name"] == "octocat/stockflow-manager"
        assert update_payload["github_url"] == "https://github.com/octocat/stockflow-manager"
        assert update_payload["repo_owner"] == "user"
