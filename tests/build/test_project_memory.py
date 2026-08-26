"""Tests for per-project MEMORY.md.

Run:
    venv/Scripts/python -m pytest tests/build/test_project_memory.py -v

The contract: writes are asynchronous and silent, reads are logged, and the
memory reduces file exploration without ever standing in for reading a file
the agent is about to edit.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.build import build_agent, project_memory
from app.build.project_memory import (
    MEMORY_FILENAME,
    flush_project_memory,
    read_project_memory,
    update_project_memory_async,
    write_project_memory,
)

from .test_ollama_stream import _call, _chunk, _run  # reuse the stream harness


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.tsx").write_text("export function App() { return null; }\n")
    return tmp_path


class TestWriteAndRead:
    def test_round_trip(self, workspace):
        write_project_memory(workspace, "Add a home page", "DONE: built it", ["src/page.tsx"])
        text = read_project_memory(workspace)
        assert "Add a home page" in text
        assert "DONE: built it" in text
        assert "src/page.tsx" in text

    def test_missing_memory_reads_as_empty(self, workspace):
        assert read_project_memory(workspace) == ""

    def test_entries_accumulate_newest_first(self, workspace):
        write_project_memory(workspace, "First change", "did first", [])
        write_project_memory(workspace, "Second change", "did second", [])
        text = read_project_memory(workspace)
        assert text.index("Second change") < text.index("First change")

    def test_change_log_is_bounded(self, workspace):
        for i in range(project_memory._MAX_ENTRIES + 6):
            write_project_memory(workspace, f"change {i}", f"summary {i}", [])
        text = read_project_memory(workspace)
        assert text.count("### ") <= project_memory._MAX_ENTRIES
        assert "change 0" not in text, "oldest entry should have been dropped"

    def test_includes_a_file_map_for_locating_code(self, workspace):
        write_project_memory(workspace, "x", "y", [])
        text = read_project_memory(workspace)
        assert "src/app.tsx" in text, "map should tell the agent where code lives"

    def test_read_is_capped(self, workspace):
        (workspace / MEMORY_FILENAME).write_text("x" * 50_000)
        assert len(read_project_memory(workspace)) < 20_000

    def test_unreadable_memory_never_raises(self, workspace):
        (workspace / MEMORY_FILENAME).mkdir()  # a directory, not a file
        assert read_project_memory(workspace) == ""

    def test_write_is_atomic_leaving_no_temp_file(self, workspace):
        write_project_memory(workspace, "x", "y", [])
        assert not list(workspace.glob("*.tmp"))


class TestAsyncUpdate:
    def test_update_runs_off_thread_and_completes(self, workspace):
        thread = update_project_memory_async(workspace, "Async change", "done", ["a.ts"])
        assert thread is not None
        thread.join(timeout=10)
        assert "Async change" in read_project_memory(workspace)

    def test_flush_waits_for_pending_writes(self, workspace):
        update_project_memory_async(workspace, "Flushed change", "done", [])
        flush_project_memory(timeout=10)
        assert (workspace / MEMORY_FILENAME).exists(), "flush must guarantee it is on disk"

    def test_a_failing_write_never_propagates(self, workspace, monkeypatch):
        """Memory is an optimisation; losing it must not fail a build."""
        monkeypatch.setattr(
            project_memory, "write_project_memory",
            lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")),
        )
        thread = update_project_memory_async(workspace, "x", "y", [])
        thread.join(timeout=10)  # must not raise


class TestAgentIntegration:
    def test_existing_memory_is_read_logged_and_injected(self, monkeypatch, workspace):
        write_project_memory(workspace, "Earlier work", "built the nav", ["src/Nav.tsx"])
        seen: list[tuple[str, str]] = []
        turns = [[_chunk(content="DONE"), _chunk(done=True)]]
        _, _, poster = _run(
            monkeypatch, turns, workspace, log_fn=lambda lvl, m: seen.append((lvl, m)),
        )

        # Reading IS logged.
        assert any(MEMORY_FILENAME in m for _, m in seen), "memory read was not logged"
        # …and actually reaches the model.
        blob = json.dumps(poster.payloads[0]["messages"])
        assert "Earlier work" in blob and "PROJECT MEMORY" in blob

    def test_no_memory_means_no_log_line(self, monkeypatch, workspace):
        seen: list[tuple[str, str]] = []
        turns = [[_chunk(content="DONE"), _chunk(done=True)]]
        _run(monkeypatch, turns, workspace, log_fn=lambda lvl, m: seen.append((lvl, m)))
        assert not any(MEMORY_FILENAME in m for _, m in seen)

    def test_writes_are_recorded_but_never_logged(self, monkeypatch, workspace):
        """The update must be invisible to the user — that was the requirement."""
        seen: list[tuple[str, str]] = []
        captured: dict = {}

        def fake_async(ws, request, summary, files):
            captured.update(workspace=ws, request=request, summary=summary, files=files)
            return None

        monkeypatch.setattr(build_agent, "update_project_memory_async", fake_async)
        turns = [
            [_chunk(tool_calls=[_call("write_file", {"path": "src/new.tsx", "content": "x"})]),
             _chunk(done=True)],
            [_chunk(content="DONE"), _chunk(done=True)],
        ]
        _run(monkeypatch, turns, workspace, log_fn=lambda lvl, m: seen.append((lvl, m)))

        assert captured["files"] == ["src/new.tsx"], "written files not handed to memory"
        assert not any("memory" in m.lower() for _, m in seen), \
            "memory update leaked into the user-facing log"

    def test_no_memory_update_when_nothing_was_written(self, monkeypatch, workspace):
        calls: list = []
        monkeypatch.setattr(
            build_agent, "update_project_memory_async",
            lambda *a, **k: calls.append(a) or None,
        )
        turns = [[_chunk(content="Just looking around. DONE"), _chunk(done=True)]]
        _run(monkeypatch, turns, workspace, log_fn=None)
        assert not calls, "a read-only run should not rewrite memory"
