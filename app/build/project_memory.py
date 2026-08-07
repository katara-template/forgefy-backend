"""Per-project MEMORY.md — what the build agent already knows about a workspace.

Written after every build or update so the next run starts with a map of the
codebase and a record of recent changes instead of rediscovering both by reading
files one at a time.

This REDUCES exploration, it does not replace it. The map records where things
live and what changed, never what the code currently says, so an agent about to
edit a file still reads that file. Treating the memory as a substitute for
reading would make it a source of stale-code bugs.

Updates run on a background thread and are deliberately not logged: they happen
after the user already has their result, so surfacing them would be noise about
work they did not ask for. Reads ARE logged — which files the agent consulted is
answerable history, and a silent read looks like the agent guessing.

The update is pure file I/O and regex scanning: no LLM call, so it costs nothing,
cannot hallucinate, and cannot fail in a way that affects the build.
"""
from __future__ import annotations

import contextlib
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

MEMORY_FILENAME = "MEMORY.md"

# Bounds. This file is injected into the agent's context on every run, so it has
# to stay small enough that reading it is cheaper than the reads it saves.
_MAX_ENTRIES = 12          # change-log entries kept before the oldest is dropped
_MAX_SUMMARY_CHARS = 500   # per entry
_MAX_REQUEST_CHARS = 200   # per entry
_MAX_FILES_PER_ENTRY = 12
_MAX_MEMORY_CHARS = 14000  # hard cap on what we inject

_HEADER = "# Project Memory"
_CHANGELOG_HEADING = "## Recent changes"
_MAP_HEADING = "## File map"

_PREAMBLE = (
    "_Maintained automatically by the Forgefy build agent._\n\n"
    "Use this to find WHERE things live instead of listing and reading files to\n"
    "discover the layout. It is a map, not a mirror: it does not contain current\n"
    "file contents, so still read any file you are about to edit.\n"
)


def _first_line(text: str, limit: int) -> str:
    """Collapse text to a single bounded line for a change-log entry."""
    flat = " ".join((text or "").split())
    return flat[:limit] + ("…" if len(flat) > limit else "")


def read_project_memory(workspace: Path) -> str:
    """Return the workspace's MEMORY.md, bounded, or '' when absent/unreadable."""
    try:
        path = Path(workspace) / MEMORY_FILENAME
        if not path.is_file():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception:  # unreadable memory must never break a build
        return ""
    if len(text) > _MAX_MEMORY_CHARS:
        text = text[:_MAX_MEMORY_CHARS] + "\n…[memory truncated]"
    return text


def _existing_entries(path: Path) -> list[str]:
    """Parse the previous change-log entries out of an existing MEMORY.md."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    if _CHANGELOG_HEADING not in text:
        return []
    body = text.split(_CHANGELOG_HEADING, 1)[1]
    body = body.split(_MAP_HEADING, 1)[0]
    # Entries start with "### " — keep them whole, newest first.
    chunks = [c.strip() for c in body.split("\n### ") if c.strip()]
    return [c if c.startswith("### ") else f"### {c}" for c in chunks]


def _render(entries: list[str], file_map: str) -> str:
    parts = [_HEADER, "", _PREAMBLE, "", _CHANGELOG_HEADING, ""]
    parts.extend(e.rstrip() + "\n" for e in entries[:_MAX_ENTRIES])
    parts.extend(["", _MAP_HEADING, "", "Each line is `path: symbols defined there`.", "", file_map, ""])
    return "\n".join(parts)


def write_project_memory(
    workspace: Path,
    request: str,
    summary: str,
    files_written: list[str] | None = None,
) -> None:
    """Merge this run into MEMORY.md. Synchronous — see the async wrapper below."""
    workspace = Path(workspace)
    path = workspace / MEMORY_FILENAME

    files = sorted(dict.fromkeys(files_written or []))
    shown = files[:_MAX_FILES_PER_ENTRY]
    more = len(files) - len(shown)
    files_line = ", ".join(f"`{f}`" for f in shown) + (f" (+{more} more)" if more > 0 else "")

    entry = (
        f"### {datetime.now(UTC):%Y-%m-%d} · {_first_line(request, _MAX_REQUEST_CHARS)}\n"
        f"{_first_line(summary, _MAX_SUMMARY_CHARS)}\n"
    )
    if files_line:
        entry += f"\nFiles changed: {files_line}\n"

    entries = [entry] + _existing_entries(path)

    # Rebuild the map from the workspace as it stands now, reusing the same
    # symbol extraction the agent's project map uses.
    try:
        from app.build.build_agent import _build_project_map
        file_map = _build_project_map(workspace)
    except Exception:
        logger.debug("project map unavailable for MEMORY.md", exc_info=True)
        file_map = "(map unavailable)"

    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(_render(entries, file_map), encoding="utf-8")
    tmp.replace(path)  # atomic — a crashed write must not corrupt memory


_pending: list[threading.Thread] = []
_pending_lock = threading.Lock()


def flush_project_memory(timeout: float = 5.0) -> None:
    """Wait for in-flight memory writes to land. Called before a git commit.

    The write is asynchronous so it never delays the user's result, but it still
    has to be on disk before the workspace is staged — an uncommitted MEMORY.md
    is lost the moment the workspace is re-cloned, which would silently defeat
    the whole feature.
    """
    with _pending_lock:
        threads, _pending[:] = list(_pending), []
    for thread in threads:
        # A stuck writer must never block a commit.
        with contextlib.suppress(Exception):
            thread.join(timeout=timeout)


def update_project_memory_async(
    workspace: Path,
    request: str,
    summary: str,
    files_written: list[str] | None = None,
) -> threading.Thread | None:
    """Update MEMORY.md off the critical path. Never raises, never logs to the user.

    Returns the thread so callers (and tests) can join it; production callers
    deliberately do not, which is the point — the user's result is already sent.
    """
    def _run() -> None:
        try:
            write_project_memory(workspace, request, summary, files_written)
        except Exception:
            # Memory is an optimisation. Losing an update costs one run's worth
            # of extra file reads, which is never worth failing a build over.
            logger.warning("MEMORY.md update failed for %s", workspace, exc_info=True)

    try:
        # Not a daemon: the write is milliseconds of local I/O, and we would
        # rather the worker wait than lose the update at process exit.
        thread = threading.Thread(target=_run, name="project-memory", daemon=False)
        with _pending_lock:
            _pending.append(thread)
        thread.start()
        return thread
    except Exception:
        logger.warning("could not start MEMORY.md update thread", exc_info=True)
        return None


def memory_context_block(memory: str) -> str:
    """Wrap memory for injection into the agent's context."""
    return (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"PROJECT MEMORY ({MEMORY_FILENAME}) — what previous runs established\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Use this to locate code instead of listing directories and opening files\n"
        "to find your way around. It records where things are and what recent runs\n"
        "changed — it does NOT contain current file contents, so read any file you\n"
        "are about to modify.\n\n"
        + memory.strip() + "\n"
        + "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    )
