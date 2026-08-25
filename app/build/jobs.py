"""Background job control for the build agent.

A long `npm install` or test run costs the model nothing to wait for except
wall-clock — but it holds the agent loop open the whole time. These helpers start
such a command detached, buffer its output to a temp file, and let the agent poll
it between other work.

Jobs are registered per workspace so a finished or cancelled build can kill every
survivor: an orphaned `npm install` keeps a workspace directory locked and a CPU
busy long after the build that started it has gone.
"""
from __future__ import annotations

import atexit
import contextlib
import logging
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

from app.build.subprocess_env import build_subprocess_env

logger = logging.getLogger(__name__)

_MAX_JOBS_PER_WORKSPACE = 5
_TAIL_DEFAULT = 50


@dataclass
class Job:
    job_id: str
    command: list[str]
    description: str
    process: subprocess.Popen[bytes]
    out_path: Path
    out_handle: IO[bytes] | None = field(repr=False, default=None)

    @property
    def running(self) -> bool:
        return self.process.poll() is None

    @property
    def status(self) -> str:
        code = self.process.poll()
        if code is None:
            return "running"
        return "completed" if code == 0 else f"failed (exit {code})"


# workspace path -> {job_id: Job}
_REGISTRY: dict[str, dict[str, Job]] = {}
_LOCK = threading.Lock()


def _key(workspace: Path) -> str:
    return str(workspace.resolve())


def start_job(
    workspace: Path, command: list[str], description: str = "",
) -> tuple[str, Job] | tuple[None, str]:
    """Spawn `command` detached. Returns (job_id, Job) or (None, error message)."""
    key = _key(workspace)
    with _LOCK:
        existing = _REGISTRY.setdefault(key, {})
        live = sum(1 for j in existing.values() if j.running)
        if live >= _MAX_JOBS_PER_WORKSPACE:
            return None, (
                f"ERROR: {live} background jobs are already running. "
                "Wait for one to finish with job_output, or stop it with job_kill."
            )

    handle = tempfile.NamedTemporaryFile(  # noqa: SIM115 — closed in _reap/kill
        prefix="forgefy_job_", suffix=".log", delete=False,
    )
    out_path = Path(handle.name)
    try:
        process = subprocess.Popen(
            command,
            cwd=workspace,
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            shell=False,
            env=build_subprocess_env(),
            # A new process group is what makes job_kill able to take down the
            # whole tree: npm spawns children, and killing only npm leaves them.
            # Each flag is a no-op on the platform it does not apply to.
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            start_new_session=True,
        )
    except (OSError, ValueError) as exc:
        handle.close()
        out_path.unlink(missing_ok=True)
        return None, f"ERROR: could not start {command[0]!r}: {exc}"

    job = Job(
        job_id=uuid.uuid4().hex[:8],
        command=list(command),
        description=description,
        process=process,
        out_path=out_path,
        out_handle=handle,
    )
    with _LOCK:
        _REGISTRY.setdefault(key, {})[job.job_id] = job
    return job.job_id, job


def get_job(workspace: Path, job_id: str) -> Job | None:
    with _LOCK:
        return _REGISTRY.get(_key(workspace), {}).get(job_id)


def list_jobs(workspace: Path) -> list[Job]:
    with _LOCK:
        return list(_REGISTRY.get(_key(workspace), {}).values())


def read_output(job: Job, tail_lines: int = _TAIL_DEFAULT) -> str:
    """Read what the job has produced so far, without blocking on it."""
    try:
        text = job.out_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"ERROR: could not read job output: {exc}"
    lines = text.splitlines()
    shown = lines[-tail_lines:] if tail_lines > 0 else lines
    head = f"[job {job.job_id} · {job.status}]"
    if not shown:
        return f"{head} no output yet."
    omitted = len(lines) - len(shown)
    prefix = f"…[{omitted:,} earlier lines omitted]\n" if omitted > 0 else ""
    return f"{head}\n{prefix}" + "\n".join(shown)


def kill_job(job: Job) -> str:
    """Terminate a job's whole process tree. Safe to call on a finished job."""
    if not job.running:
        return f"Job {job.job_id} had already finished ({job.status})."
    try:
        job.process.kill()  # SIGKILL / TerminateProcess on the group leader
        job.process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        return f"ERROR: job {job.job_id} did not die within 10 s."
    except OSError as exc:
        return f"ERROR: could not kill job {job.job_id}: {exc}"
    finally:
        _close(job)
    return f"Job {job.job_id} killed."


def _close(job: Job) -> None:
    handle = job.out_handle
    if handle is not None and not getattr(handle, "closed", True):
        with contextlib.suppress(OSError):
            handle.close()


def _discard_log(job: Job) -> None:
    """Delete a job's output file, tolerating Windows' lingering handles.

    On Windows the just-killed child can still hold the inherited stdout handle
    for a moment, and unlink raises PermissionError. A leftover file in the OS
    temp directory is harmless; a raised exception during build teardown is not.
    """
    for _ in range(3):
        try:
            job.out_path.unlink(missing_ok=True)
            return
        except PermissionError:
            time.sleep(0.1)
        except OSError:
            break
    logger.debug("Left job log %s on disk — still locked", job.out_path)


def kill_all_jobs(workspace: Path) -> int:
    """Kill every job for one workspace. Returns how many were still running."""
    with _LOCK:
        jobs = list(_REGISTRY.pop(_key(workspace), {}).values())
    killed = 0
    for job in jobs:
        if job.running:
            kill_job(job)
            killed += 1
        else:
            _close(job)
        _discard_log(job)
    if killed:
        logger.info("Killed %d surviving background job(s) for %s", killed, workspace)
    return killed


def kill_every_job() -> int:
    """Kill all jobs in every workspace — the process-exit safety net."""
    with _LOCK:
        keys = list(_REGISTRY)
    return sum(kill_all_jobs(Path(k)) for k in keys)


atexit.register(kill_every_job)
