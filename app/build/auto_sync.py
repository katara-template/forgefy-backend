"""Periodic workspace → GitHub sync, independent of whatever is using the workspace.

An agent run can end in ways its caller never gets to handle: the token budget
runs out, the container is OOM-killed, the task times out, the process dies. In
every one of those the workspace is discarded, and the code written up to that
point goes with it — the user paid for tokens and has nothing to show.

This syncer removes the dependency on a clean shutdown. Start it and it commits
and pushes on its own schedule, so the remote is never more than one interval
behind whatever is on disk. Nothing has to call it at the right moment, and it
knows nothing about builds or updates — hand it a workspace and a push URL.

Design constraints:
  * It must never break a build. Every failure is logged and swallowed; a
    checkpoint that cannot push is strictly less bad than a crashed worker.
  * It must never race the caller's own sync. All git work goes through one
    lock, so a checkpoint and a final commit cannot interleave.
  * Checkpoint commits are labelled, so history distinguishes "saved mid-run"
    from "this update completed".
"""
from __future__ import annotations

import logging
import threading
from typing import Protocol

logger = logging.getLogger(__name__)

# Long enough that a normal short update finishes without a checkpoint, short
# enough that a killed worker loses little. Agent phases run minutes each.
DEFAULT_INTERVAL_SECONDS = 90


class SyncableWorkspace(Protocol):
    """The one method this needs — keeps it usable with any workspace type."""

    def sync_to_github(self, commit_message: str, push_url: str) -> bool: ...


class WorkspaceAutoSync:
    """Commits and pushes a workspace on a timer until stopped.

    Usable as a context manager, which is the intended form: the final sync then
    happens on exit whether the body succeeded or raised.

        with WorkspaceAutoSync(workspace, push_url) as syncer:
            run_the_agent(...)
        pushed = syncer.pushed_anything
    """

    def __init__(
        self,
        workspace: SyncableWorkspace,
        push_url: str,
        *,
        interval: float = DEFAULT_INTERVAL_SECONDS,
        label: str = "checkpoint",
    ) -> None:
        self._workspace = workspace
        self._push_url = push_url
        self._interval = interval
        self._label = label

        # Guards git: a checkpoint and the final sync must never run at once.
        self._git_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self.checkpoints = 0
        self.pushed_anything = False

    # ── Public API ───────────────────────────────────────────────────────────

    def sync_now(self, commit_message: str) -> bool:
        """Commit and push immediately. Returns True if anything was pushed.

        Safe to call from any thread and at any time; never raises. A no-op when
        the working tree is clean, so repeated calls are cheap.
        """
        with self._git_lock:
            try:
                pushed = bool(self._workspace.sync_to_github(commit_message, self._push_url))
            except Exception:
                # A checkpoint is best-effort. Losing one costs at most an
                # interval of work; raising here would cost the whole run.
                logger.warning("auto-sync failed: %s", commit_message, exc_info=True)
                return False

        if pushed:
            self.pushed_anything = True
        return pushed

    def start(self) -> None:
        """Begin checkpointing in the background. Idempotent."""
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="workspace-auto-sync", daemon=True
        )
        self._thread.start()

    def stop(self, final_message: str | None = None) -> bool:
        """Stop checkpointing and optionally take one final sync.

        The final sync runs on the caller's thread, so it is complete before
        this returns — the caller can delete the workspace immediately after.
        """
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            # Bounded: the loop wakes on the event, so this returns promptly
            # unless a git operation is genuinely in flight.
            thread.join(timeout=self._interval)

        if final_message is None:
            return False
        return self.sync_now(final_message)

    # ── Context manager ──────────────────────────────────────────────────────

    def __enter__(self) -> WorkspaceAutoSync:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # Stop the timer but do NOT take a final sync here: the caller decides
        # the commit message, and on the failure path wants it marked as partial.
        # Returning False lets any exception propagate untouched.
        self.stop()
        return False

    # ── Internals ────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        """Sync every `interval` seconds until stopped.

        Silent to the user by design: checkpoints are background data-safety,
        not work the user asked for. Operators can follow them in the worker
        log; the build feed stays about the agent's actual work.
        """
        while not self._stop.wait(self._interval):
            self.checkpoints += 1
            message = f"chore({self._label}): auto-save #{self.checkpoints}"
            if self.sync_now(message):
                logger.info("auto-sync: %s pushed", message)
