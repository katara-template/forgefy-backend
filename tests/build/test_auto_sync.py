"""Tests for WorkspaceAutoSync.

Run:
    venv/Scripts/python -m pytest tests/build/test_auto_sync.py -v

The point of this component is that work survives an ending its caller never
handles — exhausted tokens, an OOM kill, a timeout. So the tests care mostly
about what happens when things go wrong: a failing push must not escape, and a
checkpoint must never run git at the same time as the caller's own sync.
"""
from __future__ import annotations

import threading
import time

from app.build.auto_sync import WorkspaceAutoSync


class FakeWorkspace:
    """Records sync calls; optionally fails or blocks to expose races."""

    def __init__(self, *, dirty: bool = True, fail: bool = False, delay: float = 0.0):
        self.dirty = dirty
        self.fail = fail
        self.delay = delay
        self.messages: list[str] = []
        self.concurrent = False
        self._inside = 0
        self._lock = threading.Lock()

    def sync_to_github(self, commit_message: str, push_url: str) -> bool:
        with self._lock:
            self._inside += 1
            if self._inside > 1:
                self.concurrent = True
        try:
            if self.delay:
                time.sleep(self.delay)
            self.messages.append(commit_message)
            if self.fail:
                raise RuntimeError("push rejected")
            return self.dirty
        finally:
            with self._lock:
                self._inside -= 1


class TestSyncNow:
    def test_pushes_and_reports_success(self):
        ws = FakeWorkspace()
        s = WorkspaceAutoSync(ws, "https://x/y.git")
        assert s.sync_now("feat: thing") is True
        assert ws.messages == ["feat: thing"]
        assert s.pushed_anything is True

    def test_clean_tree_is_not_a_push(self):
        s = WorkspaceAutoSync(FakeWorkspace(dirty=False), "https://x/y.git")
        assert s.sync_now("noop") is False
        assert s.pushed_anything is False

    def test_a_failing_push_never_raises(self):
        """A broken checkpoint must not take down the run it is protecting."""
        s = WorkspaceAutoSync(FakeWorkspace(fail=True), "https://x/y.git")
        assert s.sync_now("feat: thing") is False
        assert s.pushed_anything is False


class TestPeriodicCheckpointing:
    def test_checkpoints_while_running(self):
        ws = FakeWorkspace()
        s = WorkspaceAutoSync(ws, "https://x/y.git", interval=0.05)
        s.start()
        time.sleep(0.35)
        s.stop()
        assert s.checkpoints >= 2, f"expected repeated checkpoints, got {s.checkpoints}"
        assert all("auto-save" in m for m in ws.messages)

    def test_checkpoint_commits_are_labelled(self):
        """History must distinguish a mid-run save from a completed update."""
        ws = FakeWorkspace()
        s = WorkspaceAutoSync(ws, "https://x/y.git", interval=0.05, label="update")
        s.start()
        time.sleep(0.15)
        s.stop()
        assert ws.messages and ws.messages[0].startswith("chore(update):")

    def test_stop_ends_the_loop(self):
        ws = FakeWorkspace()
        s = WorkspaceAutoSync(ws, "https://x/y.git", interval=0.05)
        s.start()
        time.sleep(0.15)
        s.stop()
        count = len(ws.messages)
        time.sleep(0.2)
        assert len(ws.messages) == count, "checkpoints continued after stop()"

    def test_start_is_idempotent(self):
        s = WorkspaceAutoSync(FakeWorkspace(), "https://x/y.git", interval=0.05)
        s.start()
        s.start()
        s.stop()

    def test_final_sync_runs_before_stop_returns(self):
        """The caller deletes the workspace right after — the push must be done."""
        ws = FakeWorkspace()
        s = WorkspaceAutoSync(ws, "https://x/y.git", interval=10)
        s.start()
        assert s.stop("feat: done") is True
        assert ws.messages[-1] == "feat: done"


class TestConcurrencySafety:
    def test_checkpoint_never_races_the_callers_sync(self):
        """Two concurrent git operations on one workspace would corrupt state."""
        ws = FakeWorkspace(delay=0.05)
        s = WorkspaceAutoSync(ws, "https://x/y.git", interval=0.01)
        s.start()
        for _ in range(6):
            s.sync_now("feat: from caller")
        s.stop()
        assert not ws.concurrent, "checkpoint overlapped the caller's sync"


class TestContextManager:
    def test_stops_on_normal_exit(self):
        ws = FakeWorkspace()
        with WorkspaceAutoSync(ws, "https://x/y.git", interval=0.05):
            time.sleep(0.12)
        count = len(ws.messages)
        time.sleep(0.15)
        assert len(ws.messages) == count

    def test_exception_propagates_and_loop_stops(self):
        ws = FakeWorkspace()
        s = WorkspaceAutoSync(ws, "https://x/y.git", interval=0.05)
        try:
            with s:
                time.sleep(0.12)
                raise ValueError("agent blew up")
        except ValueError:
            pass
        else:
            raise AssertionError("exception was swallowed")
        count = len(ws.messages)
        time.sleep(0.15)
        assert len(ws.messages) == count, "loop kept running after an exception"
        # Work done before the failure was already checkpointed.
        assert s.pushed_anything is True
