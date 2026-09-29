"""Tests for coding-agent selection on the projects API."""
import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from httpx import AsyncClient

from tests.conftest import make_doc_snapshot

_PATCH = "app.workers.update_worker.apply_update.apply_async"


def _project_doc(owner_id, **overrides):
    now = datetime.now(UTC)
    data = {
        "owner_id": str(owner_id),
        "app_name": "stockflow-manager",
        "template_key": "next",
        "repo_full_name": "acme/stockflow-manager",
        "github_url": "https://github.com/acme/stockflow-manager",
        "is_updating": False,
        "created_at": now,
        "updated_at": now,
        "agent": "forgefy",
    }
    data.update(overrides)
    return make_doc_snapshot(data, doc_id=str(uuid.uuid4()))


class TestAgentOnUpdate:
    async def test_no_auth_returns_401(self, client):
        resp = await client.post(
            f"/api/v1/projects/{uuid.uuid4()}/update",
            json={"prompt": "add dark mode", "agent": "claude_code"},
        )
        assert resp.status_code == 401

    async def test_valid_agent_is_persisted(self, auth_client, mock_db, test_user):
        mock_db.collection.return_value.document.return_value.get.return_value = _project_doc(test_user.id)
        with patch(_PATCH):
            resp = await auth_client.post(
                f"/api/v1/projects/{uuid.uuid4()}/update",
                json={"prompt": "add dark mode", "agent": "claude_code"},
            )
        assert resp.status_code == 200
        updates = mock_db.collection.return_value.document.return_value.update.call_args[0][0]
        assert updates["agent"] == "claude_code"

    async def test_invalid_agent_is_rejected(self, auth_client, mock_db, test_user):
        mock_db.collection.return_value.document.return_value.get.return_value = _project_doc(test_user.id)
        resp = await auth_client.post(
            f"/api/v1/projects/{uuid.uuid4()}/update",
            json={"prompt": "add dark mode", "agent": "not-an-agent"},
        )
        assert resp.status_code == 422
        assert "not-an-agent" in resp.json()["detail"]

    async def test_omitted_agent_keeps_current_selection(self, auth_client, mock_db, test_user):
        """Existing clients that never send `agent` keep working unchanged."""
        mock_db.collection.return_value.document.return_value.get.return_value = _project_doc(test_user.id, agent="claude_code")
        with patch(_PATCH):
            resp = await auth_client.post(
                f"/api/v1/projects/{uuid.uuid4()}/update", json={"prompt": "add dark mode"}
            )
        assert resp.status_code == 200
        update_calls = mock_db.collection.return_value.document.return_value.update.call_args_list
        assert all("agent" not in (c.args[0] if c.args else {}) for c in update_calls)

    async def test_project_out_includes_agent(self, auth_client, mock_db, test_user):
        mock_db.collection.return_value.document.return_value.get.return_value = _project_doc(test_user.id, agent="claude_code")
        resp = await auth_client.get(f"/api/v1/projects/{uuid.uuid4()}")
        assert resp.status_code == 200
        assert resp.json()["agent"] == "claude_code"


class TestPatchProject:
    async def test_patch_persists_agent_without_build(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        """PATCH persists agent and returns 200 — without dispatching a build."""
        first = _project_doc(test_user.id)
        second = _project_doc(test_user.id, agent="claude_code")
        mock_db.collection.return_value.document.return_value.get.side_effect = [first, second]
        resp = await auth_client.patch(
            f"/api/v1/projects/{uuid.uuid4()}", json={"agent": "claude_code"}
        )
        assert resp.status_code == 200
        assert resp.json()["agent"] == "claude_code"
        # The agent selection was written to the project document.
        updates = mock_db.collection.return_value.document.return_value.update.call_args[0][0]
        assert updates["agent"] == "claude_code"

    async def test_patch_rejects_unknown_agent(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        mock_db.collection.return_value.document.return_value.get.return_value = (
            _project_doc(test_user.id)
        )
        resp = await auth_client.patch(
            f"/api/v1/projects/{uuid.uuid4()}", json={"agent": "bogus"}
        )
        assert resp.status_code == 422

    async def test_patch_no_body_leaves_agent_unchanged(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        mock_db.collection.return_value.document.return_value.get.return_value = (
            _project_doc(test_user.id, agent="claude_code")
        )
        resp = await auth_client.patch(f"/api/v1/projects/{uuid.uuid4()}", json={})
        assert resp.status_code == 200
        assert resp.json()["agent"] == "claude_code"

    async def test_patch_requires_auth(self, client: AsyncClient) -> None:
        resp = await client.patch(
            f"/api/v1/projects/{uuid.uuid4()}", json={"agent": "claude_code"}
        )
        assert resp.status_code == 401

    async def test_patch_persists_claude_code_model(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        mock_db.collection.return_value.document.return_value.get.return_value = (
            _project_doc(test_user.id)
        )
        resp = await auth_client.patch(
            f"/api/v1/projects/{uuid.uuid4()}", json={"claude_code_model": "deepseek-coder"}
        )
        assert resp.status_code == 200
        updates = mock_db.collection.return_value.document.return_value.update.call_args[0][0]
        assert updates["claude_code_model"] == "deepseek-coder"

    async def test_patch_rejects_invalid_backend(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        mock_db.collection.return_value.document.return_value.get.return_value = (
            _project_doc(test_user.id)
        )
        resp = await auth_client.patch(
            f"/api/v1/projects/{uuid.uuid4()}", json={"claude_code_backend": "bogus"}
        )
        assert resp.status_code == 422


class TestAgentOnChat:
    async def test_valid_agent_is_persisted_before_dispatch(self, auth_client, mock_db, test_user):
        mock_db.collection.return_value.document.return_value.get.return_value = _project_doc(test_user.id)
        body = {"message": "x" * 700, "agent": "claude_code"}
        with patch(_PATCH):
            resp = await auth_client.post(f"/api/v1/projects/{uuid.uuid4()}/chat", json=body)
        assert resp.status_code == 200
        assert resp.json()["update_queued"] is True
        updates = mock_db.collection.return_value.document.return_value.update.call_args[0][0]
        assert updates["agent"] == "claude_code"

    async def test_invalid_agent_is_rejected(self, auth_client, mock_db, test_user):
        mock_db.collection.return_value.document.return_value.get.return_value = _project_doc(test_user.id)
        resp = await auth_client.post(
            f"/api/v1/projects/{uuid.uuid4()}/chat",
            json={"message": "hello", "agent": "bogus"},
        )
        assert resp.status_code == 422


class TestAgentsEndpoint:
    async def test_lists_both_agents_with_availability(self, auth_client):
        resp = await auth_client.get("/api/v1/agents")
        assert resp.status_code == 200
        agents = {a["key"]: a for a in resp.json()}
        assert set(agents) == {"forgefy", "claude_code"}
        for payload in agents.values():
            assert set(payload) == {"key", "display_name", "available", "status", "label"}
            assert "detail" not in payload

    async def test_available_without_auth(self, client):
        """The agent catalogue is a public availability listing."""
        resp = await client.get("/api/v1/agents")
        assert resp.status_code == 200
