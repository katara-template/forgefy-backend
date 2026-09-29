        mock_dispatch = patch("app.workers.update_worker.apply_update.applyasync").start()
        try:
            assert resp.status_code == 200
            mock_dispatch.assert_called_once()
            updates = mock_db.collection.return_value.document.return_value.update.call_args[0][0]
            assert updates["agent"] == "claude_code"
        finally:
            patch.stopall()

    async def test_invalid_agent_is_rejected(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        mock_db.collection.return_value.document.return_value.get.return_value = (
            _project_doc(test_user.id)
        )
        resp = await auth_client.post(
            f"/api/v1/projects/{uuid.uuid4()}/update",
            json={"prompt": "add dark mode", "agent": "not-an-agent"},
        )
        assert resp.status_code == 422
        assert "not-an-agent" in resp.json()["detail"]

    async def test_omitted_agent_keeps_the_current_selection(
        self, auth_client, mock_db, test_user
    ) -> None:
        """Existing API clients that never send `agent` keep working unchanged."""
        mock_db.collection.return_value.document.return_value.get.return_value = (
            _project_doc(test_user.id, agent="claude_code")
        )
        with patch("app.workers.update_worker.apply_update.applyasync"):
            resp = await auth_client.post(
                f"/api/v1/projects/{uuid.uuid4()}/update", json={"prompt": "add dark mode"}
            )
        assert resp.status_code == 200
        update_calls = mock_db.collection.return_value.document.return_value.update.call_args_list
        assert all("agent" not in (c.args[0] if c.args else {}) for c in update_calls)

    async def test_project_out_includes_agent(self, auth_client, mock_db, test_user) -> None:
        mock_db.collection.return_value.document.return_value.get.return_value = (
            _project_doc(test_user.id, agent="claude_code")
        )
        resp = await auth_client.get(f"/api/v1/projects/{uuid.uuid4()}")
        assert resp.status_code == 200
        assert resp.json()["agent"] == "claude_code"


class TestAgentOnChat:
    async def test_valid_agent_is_persisted_before_dispatch(
        self, auth_client, mock_db, test_user
    ) -> None:
        project_id = uuid.uuid4()
        mock_db.collection.return_value.document.return_value.get.return_value = (
            _project_doc(test_user.id)
        )
        body = {"message": "x" * 700, "agent": "claude_code"}
        with patch("app.workers.update_worker.apply_update.applyasync"):
            resp = await auth_client.post(f"/api/v1/projects/{project_id}/chat", json=body)
        assert resp.status_code == 200
        assert resp.json()["update_queued"] is True
        updates = mock_db.collection.return_value.document.return_value.update.call_args[0][0]
        assert updates["agent"] == "claude_code"

    async def test_invalid_agent_is_rejected(self, auth_client, mock_db, test_user) -> None:
        mock_db.collection.return_value.document.return_value.get.return_value = (
            _project_doc(test_user.id)
        )
        resp = await auth_client.post(
            f"/api/v1/projects/{uuid.uuid4()}/chat",
            json={"message": "hello", "agent": "bogus"},
        )
        assert resp.status_code == 422


class TestAgentsEndpoint:
    async def test_lists_both_agents_with_availability(self, auth_client) -> None:
        resp = await auth_client.get("/api/v1/agents")
        assert resp.status_code == 200
        agents = {a["key"]: a for a in resp.json()}
        assert set(agents) == {"forgefy", "claude_code"}
        for payload in agents.values():
            assert set(payload) == {"key", "display_name", "available", "status", "label"}
            assert "detail" not in payload

    async def test_available_without_auth(self, client) -> None:
        """The agent catalogue is a public availability listing."""
        resp = await client.get("/api/v1/agents")
        assert resp.status_code == 200


