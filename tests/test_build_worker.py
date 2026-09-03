from pathlib import Path
from types import SimpleNamespace

from app.workers import build_worker


def test_run_agent_fix_delegates_to_the_unified_run_fix_agent(monkeypatch, tmp_path: Path) -> None:
    """Since Part L the worker does not route per provider — build_agent's
    adapter factory resolves BUILD_MODEL; the worker just calls run_fix_agent."""
    settings = SimpleNamespace(BUILD_MODEL="Qwen3")
    called: dict[str, object] = {}

    def fake_run_fix_agent(**kwargs):
        called.update(kwargs)
        return "done", 0

    monkeypatch.setattr("app.build.build_agent.run_fix_agent", fake_run_fix_agent)

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
    assert called["app_name"] == "demo"
