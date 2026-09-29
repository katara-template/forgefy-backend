"""Tests for the CLI-facing OpenAI-compatible endpoint (app/api/v1/cli.py)."""
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import AsyncClient

from app.ai.openrouter import ASSISTANT, OpenRouterError, code_models, resolve_models
from app.config import get_settings
from app.core.usage import QuotaOutcome
from app.main import app

_MODEL = code_models()[0]


def _router_settings(**overrides) -> MagicMock:
    base = dict(
        OPENROUTER_API_KEY="test-router-key",
        OPENROUTER_TIMEOUT=30,
        OPENROUTER_SITE_URL="https://forgefy.app",
        OPENROUTER_APP_NAME="Forgefy",
    )
    base.update(overrides)
    return MagicMock(**base)


def _fake_response(content: str = "hi", tokens: int = 42) -> dict:
    return {
        "id": "gen-1",
        "model": _MODEL,
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}
        ],
        "usage": {"prompt_tokens": tokens - 10, "completion_tokens": 10, "total_tokens": tokens},
    }


class TestListModels:
    async def test_requires_api_key(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/cli/models")
        assert resp.status_code == 401

    async def test_returns_allowed_models(self, api_client: AsyncClient) -> None:
        resp = await api_client.get("/api/v1/cli/models")
        assert resp.status_code == 200
        ids = {m["id"] for m in resp.json()["data"]}
        assert ids == set(code_models()) | set(resolve_models(ASSISTANT))


class TestChatCompletions:
    async def test_requires_api_key(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/cli/chat/completions",
            json={"model": _MODEL, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 401

    async def test_rejects_unknown_model(self, api_client: AsyncClient) -> None:
        app.dependency_overrides[get_settings] = lambda: _router_settings()
        resp = await api_client.post(
            "/api/v1/cli/chat/completions",
            json={"model": "not-a-real-model", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 422

    async def test_rejects_streaming(self, api_client: AsyncClient) -> None:
        app.dependency_overrides[get_settings] = lambda: _router_settings()
        resp = await api_client.post(
            "/api/v1/cli/chat/completions",
            json={"model": _MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        assert resp.status_code == 422

    async def test_unconfigured_server_returns_502(self, api_client: AsyncClient) -> None:
        app.dependency_overrides[get_settings] = lambda: _router_settings(OPENROUTER_API_KEY="")
        resp = await api_client.post(
            "/api/v1/cli/chat/completions",
            json={"model": _MODEL, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 502

    async def test_quota_block_returns_402(self, api_client: AsyncClient) -> None:
        app.dependency_overrides[get_settings] = lambda: _router_settings()
        blocked = QuotaOutcome("block", "You're out of tokens.", None, "free")
        with patch("app.api.v1.cli.evaluate_quota", AsyncMock(return_value=blocked)):
            resp = await api_client.post(
                "/api/v1/cli/chat/completions",
                json={"model": _MODEL, "messages": [{"role": "user", "content": "hi"}]},
            )
        assert resp.status_code == 402

    async def test_happy_path_proxies_and_records_usage(
        self, api_client: AsyncClient, mock_db: MagicMock
    ) -> None:
        app.dependency_overrides[get_settings] = lambda: _router_settings()
        fake = _fake_response("Here's the fix.", tokens=123)

        with (
            patch("app.api.v1.cli.post_chat", return_value=fake) as mock_post,
            patch("app.api.v1.cli.record_usage", AsyncMock()) as mock_record,
        ):
            resp = await api_client.post(
                "/api/v1/cli/chat/completions",
                json={
                    "model": _MODEL,
                    "messages": [
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "hi"},
                    ],
                },
            )

        assert resp.status_code == 200
        assert resp.json() == fake
        mock_post.assert_called_once()
        assert mock_post.call_args.args[0] == _MODEL
        mock_record.assert_awaited_once()
        assert mock_record.call_args.args[2] == 123

    async def test_forwards_tools_for_the_edit_loop(self, api_client: AsyncClient) -> None:
        app.dependency_overrides[get_settings] = lambda: _router_settings()
        fake = _fake_response(tokens=0)  # no content, a pure tool call
        fake["choices"][0]["message"] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
            ],
        }
        fake["choices"][0]["finish_reason"] = "tool_calls"

        with (
            patch("app.api.v1.cli.post_chat", return_value=fake) as mock_post,
            patch("app.api.v1.cli.record_usage", AsyncMock()),
        ):
            resp = await api_client.post(
                "/api/v1/cli/chat/completions",
                json={
                    "model": _MODEL,
                    "messages": [{"role": "user", "content": "read config.py"}],
                    "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {}}}],
                    "tool_choice": "auto",
                },
            )

        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "read_file"
        _, kwargs = mock_post.call_args
        assert kwargs["tools"] == [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]

    async def test_upstream_failure_returns_502(self, api_client: AsyncClient) -> None:
        app.dependency_overrides[get_settings] = lambda: _router_settings()
        with patch(
            "app.api.v1.cli.post_chat", side_effect=OpenRouterError("model unavailable")
        ):
            resp = await api_client.post(
                "/api/v1/cli/chat/completions",
                json={"model": _MODEL, "messages": [{"role": "user", "content": "hi"}]},
            )
        assert resp.status_code == 502
