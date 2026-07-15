from pathlib import Path
from types import SimpleNamespace

from app.workers import build_worker


def test_run_agent_fix_uses_openrouter_when_configured(monkeypatch, tmp_path: Path) -> None:
    settings = SimpleNamespace(
        BUILD_MODEL="Qwen3",
        OPENROUTER_API_KEY="test-openrouter-key",
        OPENROUTER_MODEL="qwen/qwen3-coder:free",
        OLLAMA_URL="http://ollama:11434",
        OLLAMA_MODEL="qwen3:8b",
        OLLAMA_TIMEOUT=300,
    )
    called: dict[str, object] = {}

    def fake_run_fix_agent_openrouter(*, workspace, prompt, app_name, template_key, log_fn):
        # The OpenRouter fix agent reads the key + CODE model chain from settings
        # internally (see _openrouter_loop), so the worker passes neither here.
        called["prompt"] = prompt
        called["template_key"] = template_key
        return "done", 0

    def fake_run_fix_agent_ollama(*args, **kwargs):
        raise AssertionError("Qwen3 should not use Ollama when OpenRouter is configured")

    monkeypatch.setattr(build_worker, "get_settings", lambda: settings)
    monkeypatch.setattr("app.ai.qwen.using_openrouter", lambda: True)
    monkeypatch.setattr("app.build.build_agent.run_fix_agent_openrouter", fake_run_fix_agent_openrouter)
    monkeypatch.setattr("app.build.build_agent.run_fix_agent_ollama", fake_run_fix_agent_ollama)

    summary = build_worker._run_agent_fix(
        workspace_path=tmp_path,
        error_msg="boom",
        template_key="flutter",
        app_name="demo",
        blueprint={},
        settings=settings,
        log_fn=None,
    )

    assert summary == "done"
    assert called["template_key"] == "flutter"
    assert "boom" in str(called["prompt"])
