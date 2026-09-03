"""File-system tools for the build agent — sandboxed to the workspace root."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.build.subprocess_env import build_subprocess_env

logger = logging.getLogger(__name__)

# Build-log callback: (event_type, message). None when nothing is listening.
LogFn = Callable[[str, str], None] | None

_READ_DEFAULT_LIMIT = 2000      # lines returned by read_file when not asked otherwise
_GREP_DEFAULT_HEAD = 100        # result lines returned by grep when not asked otherwise
_GREP_MAX_FILES = 2000          # bound on the pure-Python fallback walk
_GLOB_MAX_RESULTS = 200

_RUN_DEFAULT_TIMEOUT = 120
_RUN_MAX_TIMEOUT = 600
_RUN_OUTPUT_LIMIT = 8000        # chars of tail kept from a command's output

# Programs run_command and job_start may launch. The agent picks the arguments,
# so the allowlist is the only thing standing between a model-authored string and
# an arbitrary binary; commands are never passed through a shell, so there is no
# pipe, redirect or substitution to smuggle a second program in with.
_ALLOWED_COMMANDS = frozenset({
    "npm", "npx", "node", "pnpm", "yarn", "dart", "flutter", "git", "python", "pytest",
})

TOOLS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": (
            "Read a file inside the workspace. Lines come back numbered, so you can "
            f"quote them straight into edit_file. Returns the first {_READ_DEFAULT_LIMIT} "
            "lines unless you pass offset/limit."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to the file"},
                "offset": {
                    "type": "integer",
                    "description": "1-based line number to start from (default: 1)",
                },
                "limit": {
                    "type": "integer",
                    "description": f"Maximum lines to return (default: {_READ_DEFAULT_LIMIT})",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Write a file inside the workspace, creating it or replacing it entirely. "
            "Use this for NEW files and for deliberate full rewrites only. "
            "If the file already exists and you are changing part of it, use edit_file "
            "instead — rewriting a whole file to change a few lines wastes output tokens "
            "and is how large files get mangled."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to the file"},
                "content": {"type": "string", "description": "Full content to write"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": (
            "Replace an exact string in an existing file. This is the preferred way to "
            "change a file that already exists — it costs a fraction of a full rewrite "
            "and cannot truncate the parts you did not touch.\n"
            "old_string must appear EXACTLY ONCE unless replace_all is true. If it is "
            "ambiguous, include more surrounding lines to make it unique. "
            "old_string must match the file byte for byte, including indentation — read "
            "the file first if you are unsure. Returns a diff of what changed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to the file"},
                "old_string": {
                    "type": "string",
                    "description": "Exact text to replace, including indentation",
                },
                "new_string": {
                    "type": "string",
                    "description": "Replacement text (empty string deletes the match)",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace every occurrence instead of requiring exactly one",
                },
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    {
        "name": "grep",
        "description": (
            "Search file CONTENTS across the workspace with a regular expression. "
            "Use this to find where something is defined or used instead of reading "
            "files one by one."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regular expression to search for"},
                "path": {
                    "type": "string",
                    "description": "Relative directory or file to search (default: whole workspace)",
                },
                "glob": {
                    "type": "string",
                    "description": "Only search files matching this glob, e.g. '*.tsx'",
                },
                "case_insensitive": {"type": "boolean", "description": "Ignore case"},
                "output_mode": {
                    "type": "string",
                    "enum": ["content", "files_with_matches", "count"],
                    "description": (
                        "'content' shows matching lines (default), 'files_with_matches' "
                        "lists file paths, 'count' shows per-file match counts"
                    ),
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Lines of context around each match (content mode only)",
                },
                "head_limit": {
                    "type": "integer",
                    "description": "Maximum result lines to return (default 100)",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "glob",
        "description": (
            "Find files by NAME pattern, e.g. 'lib/**/*.dart' or '**/*.test.tsx'. "
            "Returns the most recently modified matches first. Use grep to search "
            "file contents instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern, e.g. 'src/**/*.ts'. '**' matches nested dirs.",
                },
                "path": {
                    "type": "string",
                    "description": "Relative directory to search from (default: workspace root)",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "list_files",
        "description": "List files and directories inside a workspace directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative directory path (use '.' for workspace root)",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "create_directory",
        "description": "Create a directory (and any missing parents) inside the workspace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path of the directory to create"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "delete_file",
        "description": "Delete a file or directory tree inside the workspace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to delete"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "move_file",
        "description": "Move or rename a file or directory inside the workspace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Current relative path"},
                "destination": {"type": "string", "description": "New relative path"},
            },
            "required": ["source", "destination"],
        },
    },
    {
        "name": "generate_image",
        "description": (
            "Generate an image using AI (FLUX.1) and save it to the app's assets folder. "
            "Use this for backgrounds, hero images, illustrations, icons, onboarding artwork, "
            "and any other visual asset the app needs. "
            "The saved path is returned — reference it in your code immediately after. "
            "For Flutter: also declare the assets/ directory in pubspec.yaml. "
            "For Next.js: reference as /images/<filename>. "
            "For React Native: use require('./assets/images/<filename>')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "Detailed visual description of the image. Include style, colours, "
                        "mood, and content. Example: 'A clean minimalist fitness app hero image "
                        "with a person running at sunrise, soft orange gradient background, "
                        "modern flat illustration style'"
                    ),
                },
                "filename": {
                    "type": "string",
                    "description": "Filename to save as, e.g. 'hero_background.png' or 'onboarding_1.png'",
                },
            },
            "required": ["prompt", "filename"],
        },
    },
    {
        "name": "run_command",
        "description": (
            "Run a project command and wait for it to finish. Use this for anything "
            "analyze_code and run_tests do not cover — installing a package, running a "
            "linter, a one-off script, a git query.\n"
            "The command is an ARRAY of arguments, not a shell string: no pipes, no "
            "redirection, no globbing. Only these programs are allowed: "
            f"{', '.join(sorted(_ALLOWED_COMMANDS))}.\n"
            "For anything that takes more than a minute or two, prefer job_start so you "
            "can keep working while it runs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Argument list, e.g. ['npm', 'run', 'lint']",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": (
                        f"Seconds to wait (default {_RUN_DEFAULT_TIMEOUT}, "
                        f"max {_RUN_MAX_TIMEOUT})"
                    ),
                },
                "description": {
                    "type": "string",
                    "description": "Short note on why you are running this, shown in the build log",
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "job_start",
        "description": (
            "Start a long-running command in the BACKGROUND and get a job id back "
            "immediately. Use this for npm install, a full test run, or a build — then "
            "carry on with other work and check on it later with job_output. "
            "Same argument-array rules and allowlist as run_command."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Argument list, e.g. ['npm', 'install']",
                },
                "description": {
                    "type": "string",
                    "description": "Short note on what this job is for",
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "job_output",
        "description": (
            "Read the output a background job has produced so far, without blocking. "
            "Tells you whether the job is still running, completed or failed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "Id returned by job_start"},
                "tail_lines": {
                    "type": "integer",
                    "description": "How many trailing lines to return (default 50)",
                },
            },
            "required": ["job_id"],
        },
    },
    {
        "name": "job_kill",
        "description": "Stop a background job and its child processes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "Id returned by job_start"},
            },
            "required": ["job_id"],
        },
    },
    {
        "name": "todo_write",
        "description": (
            "Record your task list for this build so the user can see progress. "
            "Send the WHOLE list every time — it replaces the previous one. "
            "Keep exactly one task 'in_progress' at a time and mark each one "
            "'completed' as soon as it is done, not in a batch at the end."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "description": "The complete task list, in order",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The task, imperative: 'Add the settings page'",
                            },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                            },
                            "active_form": {
                                "type": "string",
                                "description": "Present continuous: 'Adding the settings page'",
                            },
                        },
                        "required": ["content", "status", "active_form"],
                    },
                },
            },
            "required": ["todos"],
        },
    },
    {
        "name": "analyze_code",
        "description": (
            "Run the real static analyzer on the workspace to find type errors, missing imports, "
            "undefined identifiers, and other issues that will cause a compilation failure. "
            "Flutter → runs 'dart analyze'. Next.js / React Native → runs 'tsc --noEmit'. "
            "Call this BEFORE reporting your validation result so real compiler errors are caught. "
            "If it returns errors, fix them with write_file before finishing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "run_tests",
        "description": (
            "Run the project's test suite and return the results. "
            "Flutter → 'flutter test'. Next.js / React Native → 'npm test'. "
            "Use this after writing tests to confirm they actually pass, and to check "
            "that existing tests still pass after a change. A test that has never been "
            "run is not evidence of anything. "
            "Returns the runner's output, including which tests failed and why."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "generate_video",
        "description": (
            "Generate a short AI video (Kling Video) and save it to the app's assets folder. "
            "Use this for splash screen animations, onboarding loops, or background videos. "
            "Generation takes 30–90 seconds. The saved path is returned — reference it in your code. "
            "For Flutter: use the video_player package and declare assets/ in pubspec.yaml. "
            "For Next.js: use a <video> tag with src='/videos/<filename>'. "
            "For React Native: use expo-av or react-native-video."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "Detailed description of the video. Include motion, style, and mood. "
                        "Example: 'A smooth looping abstract blue gradient animation with "
                        "gentle flowing particles, suitable for a meditation app background'"
                    ),
                },
                "filename": {
                    "type": "string",
                    "description": "Filename to save as, e.g. 'splash.mp4' or 'onboarding_bg.mp4'",
                },
            },
            "required": ["prompt", "filename"],
        },
    },
]


def _safe(workspace: Path, rel: str) -> Path:
    """Resolve a workspace-relative path; raise ValueError on path-escape attempt.

    Compared with ``is_relative_to`` rather than a string prefix: a workspace at
    /builds/proj shares a string prefix with the sibling /builds/proj-evil, so the
    prefix test let a path escape into any directory whose name merely started
    with the workspace's.
    """
    root = workspace.resolve()
    resolved = (root / rel).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Path escape attempt: {rel!r}")
    return resolved


# Directories that are never worth searching: vendored code and build output
# dwarf the project's own source and would fill every result page.
_SEARCH_SKIP_DIRS = frozenset({
    ".git", "node_modules", ".next", "dist", "build", ".dart_tool", ".expo",
    "ios/Pods", "__pycache__", ".venv", "venv", "coverage", ".turbo", ".gradle",
})
# Binary-ish files a regex search can only produce noise from.
_SEARCH_SKIP_EXTS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".svg", ".mp4", ".mov",
    ".woff", ".woff2", ".ttf", ".otf", ".zip", ".gz", ".jar", ".apk", ".pdf",
    ".lock", ".map",
})

def _is_searchable(path: Path) -> bool:
    if path.suffix.lower() in _SEARCH_SKIP_EXTS:
        return False
    return not any(part in _SEARCH_SKIP_DIRS for part in path.parts)


def _read_file(path: Path, offset: Any = None, limit: Any = None) -> str:
    """Return file content with 1-based line numbers, windowed by offset/limit.

    Line numbers are what make edit_file usable: without them the model has to
    reconstruct where it is in the file from context and routinely picks an
    old_string that does not match.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    total = len(lines)

    start = max(0, int(offset) - 1) if offset else 0
    count = int(limit) if limit else _READ_DEFAULT_LIMIT
    window = lines[start:start + count]

    if not window:
        if total == 0:
            return "(file is empty)"
        return f"ERROR: offset {offset} is past the end of the file ({total} lines)."

    width = len(str(start + len(window)))
    body = "\n".join(
        f"{start + i + 1:>{width}}\t{line}" for i, line in enumerate(window)
    )

    shown_to = start + len(window)
    if start > 0 or shown_to < total:
        header = f"[lines {start + 1}-{shown_to} of {total}]\n"
        footer = ""
        if shown_to < total:
            footer = (
                f"\n…[{total - shown_to:,} more lines. Call read_file again with "
                f"offset={shown_to + 1} to continue.]"
            )
        return header + body + footer
    return body


def _edit_file(workspace: Path, path: str, old: str, new: str, replace_all: bool) -> str:
    """Replace an exact string in a file and return a unified diff of the change."""
    import difflib

    p = _safe(workspace, path)
    if not p.exists():
        return f"ERROR: {path} not found. Use write_file to create a new file."
    if p.is_dir():
        return f"ERROR: {path} is a directory."

    original = p.read_text(encoding="utf-8", errors="replace")
    occurrences = original.count(old)

    if occurrences == 0:
        return (
            f"ERROR: old_string was not found in {path}. It must match the file exactly, "
            "including indentation and line breaks. Read the file and copy the text "
            "you want to replace directly from it."
        )
    if occurrences > 1 and not replace_all:
        return (
            f"ERROR: old_string appears {occurrences} times in {path}, so this edit is "
            "ambiguous. Add the surrounding lines to old_string to identify the one you "
            "mean, or pass replace_all=true to change all of them."
        )
    if old == new:
        return f"ERROR: old_string and new_string are identical — nothing to change in {path}."

    updated = original.replace(old, new) if replace_all else original.replace(old, new, 1)
    p.write_text(updated, encoding="utf-8")

    diff = "\n".join(
        difflib.unified_diff(
            original.splitlines(), updated.splitlines(),
            fromfile=path, tofile=path, lineterm="", n=2,
        )
    )
    replaced = occurrences if replace_all else 1
    # The diff is the receipt: it is short, and it lets the model confirm it
    # changed what it meant to without spending a read_file on the whole file.
    return f"OK: replaced {replaced} occurrence(s) in {path}\n{_truncate(diff, 4000)}"


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n…[diff truncated]"


def _grep_argv(rg: str, inputs: dict[str, Any], target: Path) -> list[str]:
    """Build the ripgrep argv. Never goes through a shell."""
    mode = inputs.get("output_mode") or "content"
    argv = [rg, "--no-messages", "--color", "never"]
    if inputs.get("case_insensitive"):
        argv.append("-i")
    if mode == "files_with_matches":
        argv.append("--files-with-matches")
    elif mode == "count":
        argv.append("--count")
    else:
        argv += ["--line-number", "--no-heading", "--with-filename"]
        if context := inputs.get("context_lines"):
            argv += ["--context", str(int(context))]
    if glob_pat := inputs.get("glob"):
        argv += ["--glob", str(glob_pat)]
    for skip in ("node_modules", ".next", "dist", "build", ".dart_tool", ".git"):
        argv += ["--glob", f"!{skip}/**"]
    argv += ["--regexp", str(inputs["pattern"]), str(target)]
    return argv


def _grep_python(
    pattern: str, root: Path, inputs: dict[str, Any], workspace: Path,
) -> list[str]:
    """Bounded pure-Python fallback for when ripgrep is not installed."""
    import fnmatch
    import re as _re

    flags = _re.IGNORECASE if inputs.get("case_insensitive") else 0
    try:
        regex = _re.compile(pattern, flags)
    except _re.error as exc:
        return [f"ERROR: invalid regular expression: {exc}"]

    mode = inputs.get("output_mode") or "content"
    glob_pat = inputs.get("glob")
    context = int(inputs.get("context_lines") or 0)
    out: list[str] = []
    scanned = 0

    candidates = [root] if root.is_file() else sorted(root.rglob("*"))
    for file in candidates:
        if scanned >= _GREP_MAX_FILES:
            out.append(f"…[stopped after scanning {_GREP_MAX_FILES} files]")
            break
        if not file.is_file() or not _is_searchable(file):
            continue
        if glob_pat and not fnmatch.fnmatch(file.name, str(glob_pat)):
            continue
        scanned += 1
        try:
            lines = file.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue

        hits = [i for i, line in enumerate(lines) if regex.search(line)]
        if not hits:
            continue
        try:
            rel = file.relative_to(workspace)
        except ValueError:
            rel = file

        if mode == "files_with_matches":
            out.append(str(rel))
        elif mode == "count":
            out.append(f"{rel}:{len(hits)}")
        else:
            for i in hits:
                low = max(0, i - context)
                high = min(len(lines), i + context + 1)
                for j in range(low, high):
                    sep = ":" if j == i else "-"
                    out.append(f"{rel}{sep}{j + 1}{sep}{lines[j]}")
    return out


def _grep(workspace: Path, inputs: dict[str, Any]) -> str:
    pattern = inputs.get("pattern")
    if not pattern:
        return "ERROR: grep requires a 'pattern'."
    target = _safe(workspace, inputs.get("path") or ".")
    if not target.exists():
        return f"ERROR: {inputs.get('path')} not found"

    head = int(inputs.get("head_limit") or _GREP_DEFAULT_HEAD)
    rg = shutil.which("rg") or shutil.which("rg.exe")

    if rg:
        try:
            proc = subprocess.run(
                _grep_argv(rg, inputs, target),
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=60,
                env=build_subprocess_env(),
            )
            # rg exits 1 when there are simply no matches — not an error.
            if proc.returncode not in (0, 1):
                return f"ERROR: ripgrep failed: {(proc.stderr or '').strip()[:400]}"
            lines = proc.stdout.splitlines()
        except subprocess.TimeoutExpired:
            return "ERROR: search timed out after 60 s — narrow the pattern or path."
        except Exception as exc:  # noqa: BLE001 — fall back rather than fail the build
            logger.warning("ripgrep unavailable (%s); using the Python fallback", exc)
            lines = _grep_python(str(pattern), target, inputs, workspace)
    else:
        lines = _grep_python(str(pattern), target, inputs, workspace)

    if not lines:
        return f"No matches for {pattern!r}."
    if len(lines) > head:
        shown = "\n".join(lines[:head])
        return (
            f"{shown}\n…[{len(lines) - head:,} more result lines. Narrow the pattern, "
            "pass a 'path' or 'glob', or raise head_limit.]"
        )
    return "\n".join(lines)


def _glob(workspace: Path, inputs: dict[str, Any]) -> str:
    pattern = inputs.get("pattern")
    if not pattern:
        return "ERROR: glob requires a 'pattern'."
    root = _safe(workspace, inputs.get("path") or ".")
    if not root.exists():
        return f"ERROR: {inputs.get('path')} not found"
    if not root.is_dir():
        return f"ERROR: {inputs.get('path')} is a file, not a directory."

    try:
        matches = [p for p in root.glob(str(pattern)) if p.is_file() and _is_searchable(p)]
    except (ValueError, NotImplementedError) as exc:
        return f"ERROR: invalid glob pattern {pattern!r}: {exc}"

    if not matches:
        return f"No files match {pattern!r}."

    # Newest first: in a build, the file just written is nearly always the one
    # the next question is about.
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    rendered = [str(p.relative_to(workspace)) for p in matches[:_GLOB_MAX_RESULTS]]
    if len(matches) > _GLOB_MAX_RESULTS:
        rendered.append(
            f"…[{len(matches) - _GLOB_MAX_RESULTS:,} more matches not shown — "
            "narrow the pattern.]"
        )
    return "\n".join(rendered)


_FINDING_SEVERITIES = ("critical", "high", "medium", "low")

# Last structured report per workspace, written by report_findings and consumed
# by the phase sequence. A store rather than a return value because the phase
# runner's contract is (summary, tokens) across five backends, and the report has
# to survive the loop's own return path.
_REPORTS: dict[str, dict[str, Any]] = {}

REPORT_FINDINGS_TOOL: dict[str, Any] = {
    "name": "report_findings",
    "description": (
        "Report the RESULT of your review. You MUST call this exactly once, as the "
        "last thing you do, before your final message.\n"
        "status='clean' means you found nothing that needs fixing. "
        "status='issues_found' means there is at least one real problem — list each "
        "one in findings.\n"
        "Only report problems you actually verified. If you fixed an issue yourself "
        "during this phase, it is no longer a finding: report the end state, not the "
        "history. What you report here decides whether an expensive fix pass runs, so "
        "an inaccurate 'issues_found' wastes a great deal of work."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["clean", "issues_found"],
                "description": "'clean' if nothing needs fixing, else 'issues_found'",
            },
            "findings": {
                "type": "array",
                "description": "One entry per real problem. Empty when status is 'clean'.",
                "items": {
                    "type": "object",
                    "properties": {
                        "severity": {
                            "type": "string",
                            "enum": list(_FINDING_SEVERITIES),
                        },
                        "file": {
                            "type": "string",
                            "description": "Workspace-relative path the problem is in",
                        },
                        "line": {
                            "type": "integer",
                            "description": "1-based line number, or 0 if not line-specific",
                        },
                        "summary": {
                            "type": "string",
                            "description": "What is wrong, in one sentence, specific enough to fix",
                        },
                    },
                    "required": ["severity", "file", "summary"],
                },
            },
            "summary": {
                "type": "string",
                "description": "One or two sentences describing the overall result",
            },
        },
        "required": ["status", "summary"],
    },
}


def _report_findings(workspace: Path, inputs: dict[str, Any], log_fn: LogFn = None) -> str:
    status = str(inputs.get("status", "")).strip()
    if status not in ("clean", "issues_found"):
        return "ERROR: status must be exactly 'clean' or 'issues_found'."

    raw = inputs.get("findings") or []
    if not isinstance(raw, list):
        return "ERROR: 'findings' must be an array of objects."

    findings: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            return "ERROR: every finding must be an object with severity, file and summary."
        severity = str(entry.get("severity", "")).strip().lower()
        if severity not in _FINDING_SEVERITIES:
            return f"ERROR: severity must be one of {', '.join(_FINDING_SEVERITIES)}."
        text = str(entry.get("summary", "")).strip()
        if not text:
            return "ERROR: every finding needs a non-empty 'summary'."
        try:
            line = int(entry.get("line") or 0)
        except (TypeError, ValueError):
            line = 0
        findings.append({
            "severity": severity,
            "file": str(entry.get("file", "")).strip(),
            "line": line,
            "summary": text,
        })

    # A 'clean' status with findings attached is contradictory, and resolving it
    # the safe way (trust the findings) is how a real problem gets fixed rather
    # than dropped on a mislabelled report.
    if status == "clean" and findings:
        status = "issues_found"

    report = {
        "status": status,
        "findings": findings,
        "summary": str(inputs.get("summary", "")).strip(),
        "reported": True,
    }
    _REPORTS[str(workspace.resolve())] = report

    if log_fn:
        # A structured event so the dashboard can group findings by severity
        # instead of rendering one flat sentence. Same contract as the "todo"
        # event above: json, parsed by the client. The prose line below stays
        # for log readers and for clients with no renderer for this type.
        import json as _json

        log_fn("findings", _json.dumps(report))

        if status == "clean":
            log_fn("info", "Review result: clean — no issues found.")
        else:
            worst = min(
                (_FINDING_SEVERITIES.index(f["severity"]) for f in findings),
                default=len(_FINDING_SEVERITIES) - 1,
            )
            log_fn(
                "info",
                f"Review result: {len(findings)} issue(s) found "
                f"(worst: {_FINDING_SEVERITIES[worst]}).",
            )
    return (
        f"Report recorded: {status}"
        + (f" with {len(findings)} finding(s)." if findings else ".")
        + " Now write your final message."
    )


def missing_report(phase: str) -> dict[str, Any]:
    """The report to assume when a phase ended without calling report_findings.

    Treated as issues_found on purpose: a phase that did not report did not
    finish its workflow, and silently passing that as clean is how a broken build
    reaches the user signed off.
    """
    return {
        "status": "issues_found",
        "findings": [{
            "severity": "high",
            "file": "",
            "line": 0,
            "summary": (
                f"The {phase} phase ended without calling report_findings, so its "
                "result is unknown and cannot be treated as a pass."
            ),
        }],
        "summary": f"{phase} phase did not report a result.",
        "reported": False,
    }


def take_report(workspace: Path) -> dict[str, Any] | None:
    """Pop the report a phase filed, if any."""
    return _REPORTS.pop(str(workspace.resolve()), None)


def reset_report(workspace: Path) -> None:
    """Drop any stale report before a phase runs."""
    _REPORTS.pop(str(workspace.resolve()), None)


def _resolve_command(command: Any) -> tuple[list[str], str]:
    """Validate a model-supplied argv. Returns (argv, error) — one of them empty."""
    if isinstance(command, str):
        return [], (
            "ERROR: 'command' must be an array of arguments, not a string. "
            'Use ["npm", "run", "lint"] rather than "npm run lint". '
            "Shell features (pipes, &&, >, globs) are not available."
        )
    if not isinstance(command, list) or not command:
        return [], "ERROR: 'command' must be a non-empty array of strings."
    if not all(isinstance(part, str) for part in command):
        return [], "ERROR: every element of 'command' must be a string."

    program = Path(command[0]).name.lower()
    for suffix in (".exe", ".cmd", ".bat"):
        program = program.removesuffix(suffix)
    if program not in _ALLOWED_COMMANDS:
        return [], (
            f"ERROR: {command[0]!r} is not an allowed command. "
            f"Allowed: {', '.join(sorted(_ALLOWED_COMMANDS))}."
        )

    resolved = shutil.which(program) or shutil.which(f"{program}.cmd") or shutil.which(
        f"{program}.exe"
    )
    if not resolved:
        return [], f"ERROR: {program!r} is not installed on this machine."
    return [resolved, *command[1:]], ""


def _run_command(workspace: Path, inputs: dict[str, Any], log_fn: LogFn = None) -> str:
    argv, error = _resolve_command(inputs.get("command"))
    if error:
        return error

    timeout = int(inputs.get("timeout_seconds") or _RUN_DEFAULT_TIMEOUT)
    timeout = max(1, min(timeout, _RUN_MAX_TIMEOUT))
    printable = " ".join(str(a) for a in inputs["command"])

    try:
        result = subprocess.run(
            argv,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
            env=build_subprocess_env(),
        )
    except subprocess.TimeoutExpired:
        return (
            f"ERROR: `{printable}` timed out after {timeout} s. "
            "If it genuinely needs longer, start it with job_start instead."
        )
    except Exception as exc:  # noqa: BLE001 — hand it back for self-correction
        return f"ERROR: `{printable}` failed to start: {exc}"

    output = (result.stdout + result.stderr).strip()
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    if not output:
        return f"{status}: `{printable}` produced no output."
    # Tail, not head: a compiler or test runner puts the verdict at the end.
    if len(output) > _RUN_OUTPUT_LIMIT:
        output = f"…[earlier output trimmed]\n{output[-_RUN_OUTPUT_LIMIT:]}"
    return f"{status}: `{printable}`\n{output}"


def _job_start(workspace: Path, inputs: dict[str, Any], log_fn: LogFn = None) -> str:
    from app.build.jobs import start_job

    argv, error = _resolve_command(inputs.get("command"))
    if error:
        return error

    description = str(inputs.get("description") or "").strip()
    job_id, job_or_error = start_job(workspace, argv, description)
    if job_id is None:
        return str(job_or_error)

    printable = " ".join(str(a) for a in inputs["command"])
    return (
        f"Started job {job_id}: `{printable}`. It is running in the background — "
        f"carry on with other work and call job_output('{job_id}') to check on it."
    )


def _job_output(workspace: Path, inputs: dict[str, Any]) -> str:
    from app.build.jobs import get_job, read_output

    job_id = str(inputs.get("job_id") or "")
    job = get_job(workspace, job_id)
    if job is None:
        return f"ERROR: no background job with id {job_id!r} in this workspace."
    tail = int(inputs.get("tail_lines") or 50)
    return read_output(job, tail)


def _job_kill(workspace: Path, inputs: dict[str, Any]) -> str:
    from app.build.jobs import get_job, kill_job

    job_id = str(inputs.get("job_id") or "")
    job = get_job(workspace, job_id)
    if job is None:
        return f"ERROR: no background job with id {job_id!r} in this workspace."
    return kill_job(job)


_TODO_STATUSES = ("pending", "in_progress", "completed")
# Per-workspace task list, so the agent can be reminded of what it wrote.
_TODOS: dict[str, list[dict[str, str]]] = {}


def _todo_write(workspace: Path, inputs: dict[str, Any], log_fn: LogFn = None) -> str:
    todos = inputs.get("todos")
    if not isinstance(todos, list) or not todos:
        return "ERROR: 'todos' must be a non-empty array of task objects."

    cleaned: list[dict[str, str]] = []
    for entry in todos:
        if not isinstance(entry, dict):
            return "ERROR: every todo must be an object with content, status and active_form."
        status = str(entry.get("status", "")).strip()
        content = str(entry.get("content", "")).strip()
        if not content:
            return "ERROR: every todo needs a non-empty 'content'."
        if status not in _TODO_STATUSES:
            return f"ERROR: status must be one of {', '.join(_TODO_STATUSES)} — got {status!r}."
        cleaned.append({
            "content": content,
            "status": status,
            "active_form": str(entry.get("active_form", "") or content).strip(),
        })

    in_progress = [t for t in cleaned if t["status"] == "in_progress"]
    _TODOS[str(workspace.resolve())] = cleaned

    if log_fn:
        # A structured event so the WS feed can render a live checklist instead of
        # an opaque spinner. json, not prose: the dashboard parses this.
        import json as _json

        log_fn("todo", _json.dumps(cleaned))

    done = sum(1 for t in cleaned if t["status"] == "completed")
    summary = f"Task list updated: {done}/{len(cleaned)} complete."
    if len(in_progress) > 1:
        summary += " Note: more than one task is in_progress — work on one at a time."
    return summary


def _detect_asset_dir(workspace: Path, media_type: str) -> Path:
    """Return (and create) the correct assets directory for the template type."""
    subdir = "images" if media_type == "image" else "videos"
    if (workspace / "pubspec.yaml").exists():
        # Flutter
        d = workspace / "assets" / subdir
    elif (workspace / "app.json").exists() or (workspace / "app.config.js").exists():
        # React Native / Expo
        d = workspace / "assets" / subdir
    else:
        # Next.js / generic web
        d = workspace / "public" / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ensure_extension(filename: str, default_ext: str) -> str:
    return filename if "." in filename else f"{filename}{default_ext}"


def _placeholder_png() -> bytes:
    """Return a minimal valid 1×1 transparent PNG (no external deps)."""
    import struct
    import zlib
    def chunk(tag: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    idat = chunk(b"IDAT", zlib.compress(b"\x00\xcc\xcc\xcc"))  # grey pixel
    iend = chunk(b"IEND", b"")
    return signature + ihdr + idat + iend


def _save_placeholder(asset_dir: Path, filename: str, media_type: str) -> str:
    """Save a placeholder asset so code references compile even when generation fails."""
    out_path = asset_dir / filename
    if media_type == "image":
        out_path.write_bytes(_placeholder_png())
    else:
        out_path.write_bytes(b"")  # empty mp4 placeholder — won't play but won't break build
    return str(out_path.relative_to(asset_dir.parent.parent) if asset_dir.parent.parent.exists() else out_path)


def _run_analysis(workspace: Path, log_fn=None) -> str:
    """Run dart analyze or tsc --noEmit and return the output."""
    is_flutter = (workspace / "pubspec.yaml").exists()
    is_ts = (workspace / "tsconfig.json").exists()

    if is_flutter:
        dart = shutil.which("dart") or shutil.which("dart.exe")
        if not dart:
            return "dart not found on PATH — static analysis skipped."
        try:
            result = subprocess.run(
                [dart, "analyze", "--no-fatal-infos"],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=120,
                env=build_subprocess_env(),
            )
            output = (result.stdout + result.stderr).strip()
            if result.returncode == 0:
                return "dart analyze: no issues found."
            # Trim to the most useful tail
            return ("dart analyze errors:\n" + output)[-4000:]
        except subprocess.TimeoutExpired:
            return "dart analyze timed out after 120 s."
        except Exception as exc:
            return f"dart analyze failed: {exc}"

    if is_ts:
        npx = shutil.which("npx") or shutil.which("npx.cmd")
        if not npx:
            return "npx not found on PATH — TypeScript analysis skipped."
        try:
            result = subprocess.run(
                [npx, "tsc", "--noEmit", "--pretty", "false"],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=120,
                env=build_subprocess_env(),
            )
            output = (result.stdout + result.stderr).strip()
            if result.returncode == 0:
                return "tsc --noEmit: no type errors found."
            return ("TypeScript errors:\n" + output)[-4000:]
        except subprocess.TimeoutExpired:
            return "tsc --noEmit timed out after 120 s."
        except Exception as exc:
            return f"tsc --noEmit failed: {exc}"

    return "No recognisable project type in workspace (no pubspec.yaml or tsconfig.json)."


def _run_tests(workspace: Path, log_fn=None) -> str:
    """Run the project's test suite and return the runner output."""
    is_flutter = (workspace / "pubspec.yaml").exists()
    is_node = (workspace / "package.json").exists()

    if is_flutter:
        flutter = shutil.which("flutter") or shutil.which("flutter.bat")
        if not flutter:
            return "flutter not found on PATH — tests skipped."
        cmd = [flutter, "test", "--reporter", "compact"]
    elif is_node:
        npm = shutil.which("npm") or shutil.which("npm.cmd")
        if not npm:
            return "npm not found on PATH — tests skipped."
        # --if-present so a template without a test script reports cleanly instead
        # of failing the phase with npm's "missing script" error.
        cmd = [npm, "test", "--if-present", "--", "--watch=false"]
    else:
        return "No recognisable project type in workspace (no pubspec.yaml or package.json)."

    try:
        result = subprocess.run(
            cmd,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=300,
            env=build_subprocess_env(),
        )
    except subprocess.TimeoutExpired:
        return (
            "Test run timed out after 300 s. A hanging test usually means a watch mode "
            "or an unawaited async call — check the test setup rather than the app code."
        )
    except Exception as exc:
        return f"Test run failed to start: {exc}"

    output = (result.stdout + result.stderr).strip()
    if result.returncode == 0:
        if not output:
            return "Tests: no test suite found in this project."
        return ("Tests passed.\n" + output)[-4000:]
    return ("Tests FAILED:\n" + output)[-4000:]


def _generate_image(
    prompt: str,
    filename: str,
    workspace: Path,
    log_fn=None,
) -> str:
    asset_dir = _detect_asset_dir(workspace, "image")
    filename = _ensure_extension(filename, ".png")
    out_path = asset_dir / filename
    is_flutter = (workspace / "pubspec.yaml").exists()
    flutter_note = " — declare 'assets/images/' under flutter > assets in pubspec.yaml" if is_flutter else ""

    try:
        import fal_client
        import httpx

        from app.config import get_settings

        settings = get_settings()
        if not settings.FAL_API_KEY:
            msg = "Image generation skipped — FAL_API_KEY not configured. A placeholder was saved instead."
            if log_fn:
                log_fn("warning", msg)
            out_path.write_bytes(_placeholder_png())
            rel = out_path.relative_to(workspace)
            return f"PLACEHOLDER: {rel} (no FAL key){flutter_note}. Reference this path in code — replace with real asset later."

        os.environ["FAL_KEY"] = settings.FAL_API_KEY

        result = fal_client.run(
            "fal-ai/flux/schnell",
            arguments={
                "prompt": prompt,
                "num_images": 1,
                "image_size": "landscape_4_3",
                "num_inference_steps": 4,
            },
        )
        image_url: str = result["images"][0]["url"]

        resp = httpx.get(image_url, timeout=60, follow_redirects=True)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)

        rel = out_path.relative_to(workspace)
        return f"OK: image saved to {rel}{flutter_note}"

    except Exception as exc:
        raw = str(exc)
        # Detect billing / balance issues and surface them clearly
        if "exhausted balance" in raw.lower() or "locked" in raw.lower() or "billing" in raw.lower():
            user_msg = "Image generation unavailable — fal.ai account balance is exhausted. Top up at fal.ai/dashboard/billing."
        elif "unauthorized" in raw.lower() or "invalid" in raw.lower():
            user_msg = "Image generation unavailable — FAL_API_KEY is invalid. Check your .env file."
        else:
            user_msg = f"Image generation failed — saving placeholder instead. ({raw[:120]})"

        if log_fn:
            log_fn("warning", user_msg)

        # Always write a placeholder so asset references in code don't break the build
        try:
            out_path.write_bytes(_placeholder_png())
            rel = out_path.relative_to(workspace)
            return f"PLACEHOLDER: {rel} ({user_msg}){flutter_note}. Use this path in code — the image will be grey until generation works."
        except Exception:
            return f"ERROR: {user_msg}"


def _generate_video(
    prompt: str,
    filename: str,
    workspace: Path,
    log_fn=None,
) -> str:
    asset_dir = _detect_asset_dir(workspace, "video")
    filename = _ensure_extension(filename, ".mp4")
    out_path = asset_dir / filename
    is_flutter = (workspace / "pubspec.yaml").exists()
    flutter_note = " — declare 'assets/videos/' under flutter > assets in pubspec.yaml" if is_flutter else ""

    try:
        import fal_client
        import httpx

        from app.config import get_settings

        settings = get_settings()
        if not settings.FAL_API_KEY:
            msg = "Video generation skipped — FAL_API_KEY not configured. A placeholder was saved."
            if log_fn:
                log_fn("warning", msg)
            out_path.write_bytes(b"")
            rel = out_path.relative_to(workspace)
            return f"PLACEHOLDER: {rel} (no FAL key){flutter_note}. Reference this path in code."

        os.environ["FAL_KEY"] = settings.FAL_API_KEY

        result = fal_client.run(
            "fal-ai/kling-video/v1.6/standard/text-to-video",
            arguments={
                "prompt": prompt,
                "duration": "5",
                "aspect_ratio": "16:9",
            },
        )
        video_url: str = result["video"]["url"]

        resp = httpx.get(video_url, timeout=300, follow_redirects=True)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)

        rel = out_path.relative_to(workspace)
        return f"OK: video saved to {rel}{flutter_note}"

    except Exception as exc:
        raw = str(exc)
        if "exhausted balance" in raw.lower() or "locked" in raw.lower() or "billing" in raw.lower():
            user_msg = "Video generation unavailable — fal.ai account balance is exhausted. Top up at fal.ai/dashboard/billing."
        elif "unauthorized" in raw.lower() or "invalid" in raw.lower():
            user_msg = "Video generation unavailable — FAL_API_KEY is invalid."
        else:
            user_msg = f"Video generation failed — saving placeholder instead. ({raw[:120]})"

        if log_fn:
            log_fn("warning", user_msg)

        try:
            out_path.write_bytes(b"")
            rel = out_path.relative_to(workspace)
            return f"PLACEHOLDER: {rel} ({user_msg}){flutter_note}. Use this path in code."
        except Exception:
            return f"ERROR: {user_msg}"


def execute_tool(
    name: str,
    inputs: dict[str, Any],
    workspace: Path,
    log_fn=None,
) -> str:
    """Dispatch a tool call and return the result as a string.

    Any exception from a tool — a wrong path type, a missing argument, an I/O
    error — is converted into an "ERROR: …" string rather than raised, so one bad
    tool call is fed back to the model to self-correct instead of crashing the
    whole build loop. Applies to every backend (Claude/Gemini/OpenAI/Ollama/
    OpenRouter), all of which call execute_tool and treat the return as context.
    """
    try:
        return _dispatch_tool(name, inputs, workspace, log_fn)
    except KeyError as exc:
        return f"ERROR: tool {name!r} is missing required argument {exc}"
    except Exception as exc:  # noqa: BLE001 — surface to the model, never crash the build
        logger.warning("Tool %r failed on input %s: %s", name, inputs, exc)
        return f"ERROR: tool {name!r} failed: {exc}"


def _dispatch_tool(
    name: str,
    inputs: dict[str, Any],
    workspace: Path,
    log_fn=None,
) -> str:
    match name:
        case "read_file":
            p = _safe(workspace, inputs["path"])
            if not p.exists():
                return f"ERROR: {inputs['path']} not found"
            if p.is_dir():
                return f"ERROR: {inputs['path']} is a directory. Use list_files to list it."
            return _read_file(p, inputs.get("offset"), inputs.get("limit"))

        case "edit_file":
            return _edit_file(
                workspace,
                inputs["path"],
                inputs["old_string"],
                inputs["new_string"],
                bool(inputs.get("replace_all")),
            )

        case "grep":
            return _grep(workspace, inputs)

        case "glob":
            return _glob(workspace, inputs)

        case "write_file":
            p = _safe(workspace, inputs["path"])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(inputs["content"], encoding="utf-8")
            return f"OK: wrote {len(inputs['content'])} chars to {inputs['path']}"

        case "list_files":
            p = _safe(workspace, inputs["path"])
            if not p.exists():
                return f"ERROR: {inputs['path']} not found"
            if not p.is_dir():
                # The model sometimes points list_files at a file (e.g. a route
                # file). os.listdir would raise NotADirectoryError — steer it to
                # read_file instead of crashing the build loop.
                return f"ERROR: {inputs['path']} is a file, not a directory. Use read_file to read it."
            entries = sorted(os.listdir(p))
            if not entries:
                return "(empty)"
            lines = []
            for entry in entries:
                suffix = "/" if (p / entry).is_dir() else ""
                lines.append(f"{entry}{suffix}")
            return "\n".join(lines)

        case "create_directory":
            p = _safe(workspace, inputs["path"])
            p.mkdir(parents=True, exist_ok=True)
            return f"OK: created {inputs['path']}"

        case "delete_file":
            p = _safe(workspace, inputs["path"])
            if not p.exists():
                return f"ERROR: {inputs['path']} not found"
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
            return f"OK: deleted {inputs['path']}"

        case "move_file":
            src = _safe(workspace, inputs["source"])
            dst = _safe(workspace, inputs["destination"])
            if not src.exists():
                return f"ERROR: {inputs['source']} not found"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            return f"OK: moved {inputs['source']} → {inputs['destination']}"

        case "run_command":
            return _run_command(workspace, inputs, log_fn)

        case "job_start":
            return _job_start(workspace, inputs, log_fn)

        case "job_output":
            return _job_output(workspace, inputs)

        case "job_kill":
            return _job_kill(workspace, inputs)

        case "todo_write":
            return _todo_write(workspace, inputs, log_fn)

        case "report_findings":
            return _report_findings(workspace, inputs, log_fn)

        case "analyze_code":
            return _run_analysis(workspace, log_fn)

        case "run_tests":
            return _run_tests(workspace, log_fn)

        case "generate_image":
            return _generate_image(inputs["prompt"], inputs["filename"], workspace, log_fn)

        case "generate_video":
            return _generate_video(inputs["prompt"], inputs["filename"], workspace, log_fn)

        case _:
            return f"ERROR: unknown tool {name!r}"
