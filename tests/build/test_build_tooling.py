"""Tests for the run_tests tool and the pre-agent dependency install.

Run:
    venv/Scripts/python -m pytest tests/build/test_build_tooling.py -v

No real toolchain is invoked — subprocess and the workspace _run helper are mocked.
"""
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

from app.build.agent_tools import TOOLS, _run_tests
from app.build.workspace import install_dependencies_at


def _completed(returncode: int, stdout: str = "", stderr: str = "") -> MagicMock:
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = stderr
    return proc


# ── run_tests tool ────────────────────────────────────────────────────────────


class TestRunTestsTool:
    def test_tool_is_registered_with_an_empty_schema(self):
        tool = next((t for t in TOOLS if t["name"] == "run_tests"), None)

        assert tool is not None, "run_tests must be exposed or no agent can run a suite"
        assert tool["input_schema"]["properties"] == {}

    def test_unknown_project_type_is_reported_not_crashed(self, tmp_path):
        assert "No recognisable project type" in _run_tests(tmp_path)

    def test_flutter_project_runs_flutter_test(self, tmp_path):
        (tmp_path / "pubspec.yaml").write_text("name: app\n", encoding="utf-8")

        with (
            patch("app.build.agent_tools.shutil.which", return_value="/usr/bin/flutter"),
            patch("app.build.agent_tools.subprocess.run", return_value=_completed(0, "All tests passed!")) as run,
        ):
            result = _run_tests(tmp_path)

        assert run.call_args.args[0][:2] == ["/usr/bin/flutter", "test"]
        assert "Tests passed" in result

    def test_node_project_runs_npm_test(self, tmp_path):
        (tmp_path / "package.json").write_text("{}", encoding="utf-8")

        with (
            patch("app.build.agent_tools.shutil.which", return_value="/usr/bin/npm"),
            patch("app.build.agent_tools.subprocess.run", return_value=_completed(0, "ok")) as run,
        ):
            _run_tests(tmp_path)

        cmd = run.call_args.args[0]
        assert cmd[:2] == ["/usr/bin/npm", "test"]
        # --if-present so a template with no test script doesn't fail the phase
        assert "--if-present" in cmd

    def test_failure_is_surfaced_as_failure(self, tmp_path):
        (tmp_path / "package.json").write_text("{}", encoding="utf-8")

        with (
            patch("app.build.agent_tools.shutil.which", return_value="/usr/bin/npm"),
            patch("app.build.agent_tools.subprocess.run", return_value=_completed(1, "", "2 failing")),
        ):
            result = _run_tests(tmp_path)

        assert result.startswith("Tests FAILED:")
        assert "2 failing" in result

    def test_missing_toolchain_does_not_raise(self, tmp_path):
        (tmp_path / "pubspec.yaml").write_text("name: app\n", encoding="utf-8")

        with patch("app.build.agent_tools.shutil.which", return_value=None):
            assert "flutter not found" in _run_tests(tmp_path)

    def test_timeout_is_reported_not_raised(self, tmp_path):
        (tmp_path / "package.json").write_text("{}", encoding="utf-8")

        with (
            patch("app.build.agent_tools.shutil.which", return_value="/usr/bin/npm"),
            patch(
                "app.build.agent_tools.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="npm test", timeout=300),
            ),
        ):
            assert "timed out" in _run_tests(tmp_path)

    def test_clean_run_with_no_suite_is_not_a_failure(self, tmp_path):
        (tmp_path / "package.json").write_text("{}", encoding="utf-8")

        with (
            patch("app.build.agent_tools.shutil.which", return_value="/usr/bin/npm"),
            patch("app.build.agent_tools.subprocess.run", return_value=_completed(0, "", "")),
        ):
            assert "no test suite found" in _run_tests(tmp_path)


# ── pre-agent dependency install ──────────────────────────────────────────────


class TestInstallDependencies:
    def test_flutter_runs_pub_get(self, tmp_path):
        with patch("app.build.workspace._run") as run:
            assert install_dependencies_at(tmp_path, "flutter") is True

        assert run.call_args.args[0] == ["flutter", "pub", "get"]

    def test_next_runs_npm_install(self, tmp_path):
        with (
            patch("app.build.workspace._run") as run,
            patch("app.build.workspace._find_package_root", return_value=tmp_path),
        ):
            assert install_dependencies_at(tmp_path, "next") is True

        assert run.call_args.args[0] == ["npm", "install", "--legacy-peer-deps"]

    def test_unknown_template_returns_false(self, tmp_path):
        with patch("app.build.workspace._run") as run:
            assert install_dependencies_at(tmp_path, "cobol") is False

        run.assert_not_called()

    def test_failure_is_non_fatal_and_warns_the_user(self, tmp_path):
        """A failed install must not kill the build — but the user needs to know
        that analysis errors from here on may not be real."""
        logged: list[tuple[str, str]] = []

        with patch("app.build.workspace._run", side_effect=RuntimeError("network down")):
            result = install_dependencies_at(
                tmp_path, "flutter", log_fn=lambda lvl, msg: logged.append((lvl, msg))
            )

        assert result is False
        assert any(lvl == "warning" for lvl, _ in logged)
        assert any("not real" in msg for _, msg in logged)
