"""Security tests for the agent's workspace sandbox and subprocess environment.

Run:
    venv/Scripts/python -m pytest tests/build/test_agent_tools_security.py -v
"""
from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.build.agent_tools import _safe, execute_tool
from app.build.subprocess_env import build_subprocess_env

# ── path sandbox ──────────────────────────────────────────────────────────────


class TestPathSandbox:
    def test_sibling_directory_sharing_a_name_prefix_is_rejected(self, tmp_path):
        """The bug the string-prefix check had: /builds/proj admitted /builds/proj-evil."""
        workspace = tmp_path / "proj"
        workspace.mkdir()
        evil = tmp_path / "proj-evil"
        evil.mkdir()
        (evil / "secret.txt").write_text("stolen", encoding="utf-8")

        with pytest.raises(ValueError, match="Path escape"):
            _safe(workspace, "../proj-evil/secret.txt")

    def test_sibling_prefix_escape_is_refused_through_the_tool_dispatcher(self, tmp_path):
        workspace = tmp_path / "proj"
        workspace.mkdir()
        evil = tmp_path / "proj-evil"
        evil.mkdir()
        (evil / "secret.txt").write_text("stolen", encoding="utf-8")

        result = execute_tool(
            "read_file", {"path": "../proj-evil/secret.txt"}, workspace,
        )

        assert "stolen" not in result
        assert "ERROR" in result

    def test_parent_traversal_is_rejected(self, tmp_path):
        workspace = tmp_path / "proj"
        workspace.mkdir()

        with pytest.raises(ValueError, match="Path escape"):
            _safe(workspace, "../../etc/passwd")

    def test_absolute_path_outside_the_workspace_is_rejected(self, tmp_path):
        workspace = tmp_path / "proj"
        workspace.mkdir()
        outside = tmp_path / "elsewhere.txt"
        outside.write_text("nope", encoding="utf-8")

        with pytest.raises(ValueError, match="Path escape"):
            _safe(workspace, str(outside))

    def test_ordinary_nested_paths_still_resolve(self, tmp_path):
        workspace = tmp_path / "proj"
        (workspace / "lib" / "src").mkdir(parents=True)

        resolved = _safe(workspace, "lib/src/main.dart")

        assert resolved == (workspace / "lib" / "src" / "main.dart").resolve()

    def test_the_workspace_root_itself_is_allowed(self, tmp_path):
        workspace = tmp_path / "proj"
        workspace.mkdir()

        assert _safe(workspace, ".") == workspace.resolve()


# ── subprocess environment ────────────────────────────────────────────────────


_SECRETS = {
    "ANTHROPIC_API_KEY": "sk-ant-secret",
    "GITHUB_TOKEN": "ghp_secret",
    "FIREBASE_CREDENTIALS": '{"private_key":"secret"}',
    "SUPABASE_SERVICE_KEY": "svc-secret",
    "DATABASE_URL": "postgres://user:pw@host/db",
    "FAL_API_KEY": "fal-secret",
}


class TestSubprocessEnv:
    def test_secrets_are_not_passed_to_generated_code(self, monkeypatch):
        for key, value in _SECRETS.items():
            monkeypatch.setenv(key, value)

        env = build_subprocess_env()

        for key in _SECRETS:
            assert key not in env, f"{key} must not reach a model-authored npm script"

    def test_toolchain_variables_survive(self, monkeypatch):
        monkeypatch.setenv("JAVA_HOME", "/opt/java")
        monkeypatch.setenv("ANDROID_HOME", "/opt/android")
        monkeypatch.setenv("PUB_CACHE", "/cache/pub")

        env = build_subprocess_env()

        assert env["JAVA_HOME"] == "/opt/java"
        assert env["ANDROID_HOME"] == "/opt/android"
        assert env["PUB_CACHE"] == "/cache/pub"

    def test_path_is_always_present(self, monkeypatch):
        monkeypatch.delenv("PATH", raising=False)

        assert build_subprocess_env()["PATH"]

    def test_extra_values_are_applied_last(self):
        env = build_subprocess_env({"GIT_TERMINAL_PROMPT": "0"})

        assert env["GIT_TERMINAL_PROMPT"] == "0"

    def test_proxy_settings_survive_so_builds_work_behind_a_proxy(self, monkeypatch):
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy:8080")

        assert build_subprocess_env()["HTTPS_PROXY"] == "http://proxy:8080"


# ── the tools that spawn processes must use the scrubbed env ──────────────────


def _completed(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = stderr
    return proc


class TestToolsUseTheScrubbedEnv:
    @pytest.fixture(autouse=True)
    def _set_secret(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")

    def test_run_tests_scrubs_the_environment(self, tmp_path):
        (tmp_path / "package.json").write_text("{}", encoding="utf-8")

        with (
            patch("app.build.agent_tools.shutil.which", return_value="/usr/bin/npm"),
            patch(
                "app.build.agent_tools.subprocess.run", return_value=_completed(0, "ok"),
            ) as run,
        ):
            execute_tool("run_tests", {}, tmp_path)

        assert "ANTHROPIC_API_KEY" not in run.call_args.kwargs["env"]

    def test_analyze_code_scrubs_the_environment_for_dart(self, tmp_path):
        (tmp_path / "pubspec.yaml").write_text("name: app\n", encoding="utf-8")

        with (
            patch("app.build.agent_tools.shutil.which", return_value="/usr/bin/dart"),
            patch(
                "app.build.agent_tools.subprocess.run", return_value=_completed(0, ""),
            ) as run,
        ):
            execute_tool("analyze_code", {}, tmp_path)

        assert "ANTHROPIC_API_KEY" not in run.call_args.kwargs["env"]

    def test_analyze_code_scrubs_the_environment_for_typescript(self, tmp_path):
        (tmp_path / "tsconfig.json").write_text("{}", encoding="utf-8")

        with (
            patch("app.build.agent_tools.shutil.which", return_value="/usr/bin/npx"),
            patch(
                "app.build.agent_tools.subprocess.run", return_value=_completed(0, ""),
            ) as run,
        ):
            execute_tool("analyze_code", {}, tmp_path)

        assert "ANTHROPIC_API_KEY" not in run.call_args.kwargs["env"]

    def test_workspace_run_helper_scrubs_the_environment(self, tmp_path):
        from app.build.workspace import _run

        with patch(
            "app.build.workspace.subprocess.run", return_value=_completed(0, "ok"),
        ) as run:
            _run(["npm", "install"], cwd=tmp_path)

        env = run.call_args.kwargs["env"]
        assert "ANTHROPIC_API_KEY" not in env
        assert env["GIT_TERMINAL_PROMPT"] == "0", "git must still never prompt"


class TestNpmIgnoreScripts:
    def test_lifecycle_scripts_run_by_default(self, monkeypatch):
        from app.build import workspace as ws

        monkeypatch.setattr(
            ws, "get_settings", lambda: SimpleNamespace(NPM_IGNORE_SCRIPTS=False),
            raising=False,
        )
        with patch("app.config.get_settings", return_value=SimpleNamespace(NPM_IGNORE_SCRIPTS=False)):
            assert "--ignore-scripts" not in ws._npm_install_args()

    def test_lifecycle_scripts_can_be_refused(self):
        from app.build import workspace as ws

        with patch("app.config.get_settings", return_value=SimpleNamespace(NPM_IGNORE_SCRIPTS=True)):
            args = ws._npm_install_args()

        assert args[:3] == ["npm", "install", "--legacy-peer-deps"]
        assert "--ignore-scripts" in args


def test_real_subprocess_does_not_see_the_key(tmp_path, monkeypatch):
    """End-to-end: spawn a real process and confirm the secret is absent."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "import os; print(os.environ.get('ANTHROPIC_API_KEY', 'ABSENT'))"],
        capture_output=True,
        text=True,
        env=build_subprocess_env(),
        timeout=60,
    )

    assert result.stdout.strip() == "ABSENT"
