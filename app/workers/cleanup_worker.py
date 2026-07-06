"""Workspace cleanup — daily Celery Beat task to remove stale workspace directories.

Strategy
--------
* Build workspaces  (``{session_id}/``)  — deleted after BUILD_MAX_AGE_DAYS (7 days).
* Edit workspaces   (``edit_{project_id}/``) — deleted after EDIT_MAX_AGE_DAYS (3 days);
  they are re-cloned from GitHub automatically on the next update.
* Hard cap: if total workspace disk usage exceeds MAX_TOTAL_GB (5 GB), the oldest
  directories are removed first until usage is below the cap.
"""
from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

WORKSPACE_ROOT = Path("/tmp/forgefy_workspaces")
BUILD_MAX_AGE_DAYS = 7
EDIT_MAX_AGE_DAYS = 3
MAX_TOTAL_GB = 5.0


def _dir_size(path: Path) -> int:
    """Return total bytes used by all files under path."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


@celery_app.task(name="app.workers.cleanup_worker.cleanup_workspaces")
def cleanup_workspaces() -> dict:
    """Remove stale workspace directories and enforce the disk cap."""
    if not WORKSPACE_ROOT.exists():
        return {"removed": 0, "bytes_freed": 0}

    now = time.time()
    removed = 0
    bytes_freed = 0

    # Collect all workspace dirs with metadata
    entries: list[tuple[float, Path, int]] = []
    for p in WORKSPACE_ROOT.iterdir():
        if not p.is_dir():
            continue
        try:
            mtime = p.stat().st_mtime
            size = _dir_size(p)
            entries.append((mtime, p, size))
        except Exception as exc:
            logger.warning("Could not stat workspace %s: %s", p, exc)

    # 1. Age-based removal
    for mtime, path, size in list(entries):
        is_edit = path.name.startswith("edit_")
        max_age = (EDIT_MAX_AGE_DAYS if is_edit else BUILD_MAX_AGE_DAYS) * 86400
        if now - mtime > max_age:
            try:
                shutil.rmtree(path)
                removed += 1
                bytes_freed += size
                entries.remove((mtime, path, size))
                logger.info(
                    "Removed stale workspace %s (%.1f MB, %.1f days old)",
                    path.name, size / 1_048_576, (now - mtime) / 86400,
                )
            except Exception as exc:
                logger.warning("Failed to remove workspace %s: %s", path, exc)

    # 2. Disk cap — evict oldest first if still over limit
    total_bytes = sum(s for _, _, s in entries)
    cap_bytes = int(MAX_TOTAL_GB * 1_073_741_824)

    if total_bytes > cap_bytes:
        entries.sort(key=lambda x: x[0])  # oldest first
        for _mtime, path, size in entries:
            if total_bytes <= cap_bytes:
                break
            try:
                shutil.rmtree(path)
                total_bytes -= size
                bytes_freed += size
                removed += 1
                logger.info(
                    "Disk cap eviction: %s (%.1f MB)",
                    path.name, size / 1_048_576,
                )
            except Exception as exc:
                logger.warning("Failed to evict workspace %s: %s", path, exc)

    logger.info(
        "Workspace cleanup done: removed=%d freed=%.1f MB remaining_total=%.1f MB",
        removed,
        bytes_freed / 1_048_576,
        total_bytes / 1_048_576,
    )
    return {"removed": removed, "bytes_freed": bytes_freed}
