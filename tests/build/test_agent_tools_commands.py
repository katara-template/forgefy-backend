"""Tests for run_command, the background job tools, and todo_write.

Run:
    venv/Scripts/python -m pytest tests/build/test_agent_tools_commands.py -v

Real processes are spawned, using the current Python interpreter (which is on the
allowlist), so the job lifecycle is exercised rather than mocked.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from app.build import jobs
from app.build.agent_tools import _ALLOWED_COMMANDS, TOOLS, execute_tool


@pytest.fixture(autouse=True)
def _no_surviving_jobs(tmp_path):
    """Never let a test leak a live child process into the next one."""
    yield
    jobs.kill_every_job()


def _run(name, inputs, workspace, log_fn=None):
    return execute_tool(name, inputs, workspace, log_fn)


# ── registration ──────────────────────────────────────────────────────────────


class TestRegistration:
    @pytest.mark.parametrize(
        "name", ["run_command", "job_start", "job_output", "job_kill", "todo_write"],
    )
    def test_tool_is_registered(self, name):
        tool = next((t for t in TOOLS if t["name"] == name), None)

        assert tool is not None, f"{name} is not exposed to the model"
        assert tool["description"].strip()

    def test_command_is_an_array_in_the_schema(self):
        for name in ("run_command", "job_start"):
            tool = next(t for t in TOOLS if t["name"] == name)
            assert tool["input_schema"]["properties"]["command"]["type"] == "array"

    def test_convenience_wrappers_are_still_available(self):
        """run_command supplements analyze_code / run_tests, it does not replace them."""
        names = {t["name"] for t in TOOLS}

        assert {"analyze_code", "run_tests"} <= names


# ── run_command ───────────────────────────────────────────────────────────────


class TestRunCommand:
    def test_runs_an_allowlisted_command(self, tmp_path):
        result = _run(
            "run_command",
            {"command": [sys.executable, "-c", "print('hello from python')"]},
            tmp_path,
        )

        assert result.startswith("OK:")
        assert "hello from python" in result

    def test_non_zero_exit_is_reported_with_output(self, tmp_path):
        result = _run(
            "run_command",
            {"command": [sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"]},
            tmp_path,
        )

        assert "FAILED (exit 3)" in result
        assert "boom" in result

    def test_disallowed_program_is_rejected(self, tmp_path):
        result = _run("run_command", {"command": ["curl", "https://example.com"]}, tmp_path)

        assert "ERROR" in result
        assert "not an allowed command" in result

    def test_shell_string_is_rejected_with_guidance(self, tmp_path):
        result = _run("run_command", {"command": "npm run lint"}, tmp_path)

        assert "ERROR" in result
        assert "array" in result

    def test_shell_metacharacters_cannot_smuggle_a_second_program(self, tmp_path):
        """No shell means '&&' is just an argument, never a command separator."""
        marker = tmp_path / "pwned.txt"
        result = _run(
            "run_command",
            {"command": [sys.executable, "-c", "print('ok')", "&&", f"touch {marker}"]},
            tmp_path,
        )

        assert not marker.exists()
        assert "ok" in result

    def test_empty_command_is_rejected(self, tmp_path):
        assert "ERROR" in _run("run_command", {"command": []}, tmp_path)

    def test_non_string_arguments_are_rejected(self, tmp_path):
        result = _run("run_command", {"command": ["python", 42]}, tmp_path)

        assert "ERROR" in result and "string" in result

    def test_timeout_is_enforced_and_suggests_job_start(self, tmp_path):
        result = _run(
            "run_command",
            {
                "command": [sys.executable, "-c", "import time; time.sleep(30)"],
                "timeout_seconds": 2,
            },
            tmp_path,
        )

        assert "timed out" in result
        assert "job_start" in result

    def test_timeout_is_capped_at_the_maximum(self, tmp_path):
        from app.build.agent_tools import _RUN_MAX_TIMEOUT

        # A huge request must be clamped, not honoured.
        assert _RUN_MAX_TIMEOUT == 600
        result = _run(
            "run_command",
            {"command": [sys.executable, "-c", "print(1)"], "timeout_seconds": 99999},
            tmp_path,
        )

        assert result.startswith("OK:")

    def test_runs_inside_the_workspace(self, tmp_path):
        result = _run(
            "run_command",
            {"command": [sys.executable, "-c", "import os; print(os.getcwd())"]},
            tmp_path,
        )

        assert str(tmp_path.resolve()) in result.replace("\\\\", "\\")

    def test_secrets_are_not_visible_to_the_command(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")

        result = _run(
            "run_command",
            {
                "command": [
                    sys.executable, "-c",
                    "import os; print(os.environ.get('ANTHROPIC_API_KEY', 'ABSENT'))",
                ],
            },
            tmp_path,
        )

        assert "ABSENT" in result
        assert "sk-ant-secret" not in result

    def test_output_is_tail_capped(self, tmp_path):
        from app.build.agent_tools import _RUN_OUTPUT_LIMIT

        result = _run(
            "run_command",
            {"command": [sys.executable, "-c", "print('x' * 40000)"]},
            tmp_path,
        )

        assert len(result) < _RUN_OUTPUT_LIMIT + 500
        assert "earlier output trimmed" in result

    def test_allowlist_covers_the_documented_toolchain(self):
        assert {"npm", "npx", "node", "pnpm", "yarn", "dart", "flutter",
                "git", "python", "pytest"} == set(_ALLOWED_COMMANDS)


# ── background jobs ───────────────────────────────────────────────────────────


def _wait_for(predicate, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return False


class TestBackgroundJobs:
    def test_job_start_returns_immediately_with_an_id(self, tmp_path):
        started = time.time()
        result = _run(
            "job_start",
            {
                "command": [sys.executable, "-c", "import time; time.sleep(10)"],
                "description": "slow thing",
            },
            tmp_path,
        )

        assert "Started job" in result
        assert time.time() - started < 5, "job_start blocked instead of detaching"

    def test_job_output_reports_progress_then_completion(self, tmp_path):
        result = _run(
            "job_start",
            {"command": [sys.executable, "-c", "print('working'); print('done')"]},
            tmp_path,
        )
        job_id = result.split("Started job ")[1].split(":")[0]

        assert _wait_for(
            lambda: "completed" in _run("job_output", {"job_id": job_id}, tmp_path)
        ), "job never reported completion"
        output = _run("job_output", {"job_id": job_id}, tmp_path)
        assert "working" in output and "done" in output

    def test_failed_job_reports_its_exit_code(self, tmp_path):
        result = _run(
            "job_start",
            {"command": [sys.executable, "-c", "import sys; sys.exit(2)"]},
            tmp_path,
        )
        job_id = result.split("Started job ")[1].split(":")[0]

        assert _wait_for(
            lambda: "failed" in _run("job_output", {"job_id": job_id}, tmp_path)
        )
        assert "exit 2" in _run("job_output", {"job_id": job_id}, tmp_path)

    def test_job_kill_stops_a_running_job(self, tmp_path):
        result = _run(
            "job_start",
            {"command": [sys.executable, "-c", "import time; time.sleep(60)"]},
            tmp_path,
        )
        job_id = result.split("Started job ")[1].split(":")[0]
        job = jobs.get_job(tmp_path, job_id)
        assert job.running

        killed = _run("job_kill", {"job_id": job_id}, tmp_path)

        assert "killed" in killed
        assert _wait_for(lambda: not job.running), "process survived job_kill"

    def test_unknown_job_id_is_an_error_not_a_crash(self, tmp_path):
        assert "ERROR" in _run("job_output", {"job_id": "nope"}, tmp_path)
        assert "ERROR" in _run("job_kill", {"job_id": "nope"}, tmp_path)

    def test_jobs_are_scoped_to_their_workspace(self, tmp_path):
        one = tmp_path / "one"
        two = tmp_path / "two"
        one.mkdir()
        two.mkdir()
        result = _run(
            "job_start", {"command": [sys.executable, "-c", "import time; time.sleep(5)"]}, one,
        )
        job_id = result.split("Started job ")[1].split(":")[0]

        assert "ERROR" in _run("job_output", {"job_id": job_id}, two)
        assert "job" in _run("job_output", {"job_id": job_id}, one)

    def test_job_start_rejects_a_disallowed_program(self, tmp_path):
        result = _run("job_start", {"command": ["curl", "http://example.com"]}, tmp_path)

        assert "ERROR" in result and "not an allowed command" in result

    def test_concurrent_job_count_is_bounded(self, tmp_path):
        for _ in range(jobs._MAX_JOBS_PER_WORKSPACE):
            _run(
                "job_start",
                {"command": [sys.executable, "-c", "import time; time.sleep(30)"]},
                tmp_path,
            )

        result = _run(
            "job_start",
            {"command": [sys.executable, "-c", "import time; time.sleep(30)"]},
            tmp_path,
        )

        assert "ERROR" in result
        assert "already running" in result

    def test_kill_all_jobs_stops_every_survivor(self, tmp_path):
        for _ in range(3):
            _run(
                "job_start",
                {"command": [sys.executable, "-c", "import time; time.sleep(60)"]},
                tmp_path,
            )
        live = [j for j in jobs.list_jobs(tmp_path) if j.running]
        assert len(live) == 3

        killed = jobs.kill_all_jobs(tmp_path)

        assert killed == 3
        assert _wait_for(lambda: all(not j.running for j in live))

    def test_workspace_cleanup_kills_surviving_jobs(self, tmp_path):
        """A cancelled build must not leave an npm install running."""
        from app.build.workspace import _kill_workspace_jobs

        _run(
            "job_start",
            {"command": [sys.executable, "-c", "import time; time.sleep(60)"]},
            tmp_path,
        )
        job = jobs.list_jobs(tmp_path)[0]
        assert job.running

        _kill_workspace_jobs(tmp_path)

        assert _wait_for(lambda: not job.running), "cleanup left an orphaned process"

    def test_job_output_does_not_block_on_a_running_job(self, tmp_path):
        result = _run(
            "job_start",
            {"command": [sys.executable, "-c", "import time; time.sleep(30)"]},
            tmp_path,
        )
        job_id = result.split("Started job ")[1].split(":")[0]

        started = time.time()
        output = _run("job_output", {"job_id": job_id}, tmp_path)

        assert time.time() - started < 5
        assert "running" in output


# ── todo_write ────────────────────────────────────────────────────────────────


class TestTodoWrite:
    def test_emits_a_structured_todo_event(self, tmp_path):
        events: list[tuple[str, str]] = []
        todos = [
            {"content": "Add settings page", "status": "in_progress",
             "active_form": "Adding settings page"},
            {"content": "Wire the route", "status": "pending",
             "active_form": "Wiring the route"},
        ]

        result = _run("todo_write", {"todos": todos}, tmp_path,
                      lambda k, m: events.append((k, m)))

        assert "0/2 complete" in result
        kinds = [k for k, _ in events]
        assert "todo" in kinds, "the WS feed cannot render a checklist without the event"
        payload = json.loads(next(m for k, m in events if k == "todo"))
        assert payload[0]["content"] == "Add settings page"
        assert payload[0]["status"] == "in_progress"

    def test_completed_count_is_reported(self, tmp_path):
        todos = [
            {"content": "a", "status": "completed", "active_form": "doing a"},
            {"content": "b", "status": "completed", "active_form": "doing b"},
            {"content": "c", "status": "pending", "active_form": "doing c"},
        ]

        assert "2/3 complete" in _run("todo_write", {"todos": todos}, tmp_path)

    def test_invalid_status_is_rejected(self, tmp_path):
        todos = [{"content": "a", "status": "wip", "active_form": "doing a"}]

        result = _run("todo_write", {"todos": todos}, tmp_path)

        assert "ERROR" in result and "status must be one of" in result

    def test_empty_list_is_rejected(self, tmp_path):
        assert "ERROR" in _run("todo_write", {"todos": []}, tmp_path)

    def test_blank_content_is_rejected(self, tmp_path):
        todos = [{"content": "  ", "status": "pending", "active_form": "x"}]

        assert "ERROR" in _run("todo_write", {"todos": todos}, tmp_path)

    def test_multiple_in_progress_is_flagged_but_accepted(self, tmp_path):
        todos = [
            {"content": "a", "status": "in_progress", "active_form": "doing a"},
            {"content": "b", "status": "in_progress", "active_form": "doing b"},
        ]

        result = _run("todo_write", {"todos": todos}, tmp_path)

        assert "one at a time" in result

    def test_list_replaces_the_previous_one(self, tmp_path):
        from app.build.agent_tools import _TODOS

        _run("todo_write", {"todos": [
            {"content": "old", "status": "pending", "active_form": "old"},
        ]}, tmp_path)
        _run("todo_write", {"todos": [
            {"content": "new", "status": "completed", "active_form": "new"},
        ]}, tmp_path)

        stored = _TODOS[str(Path(tmp_path).resolve())]
        assert [t["content"] for t in stored] == ["new"]
