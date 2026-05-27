"""File-system tools for the build agent — sandboxed to the workspace root."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

TOOLS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": "Read the full text content of a file inside the workspace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to the file"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write (create or overwrite) a file inside the workspace.",
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
]


def _safe(workspace: Path, rel: str) -> Path:
    """Resolve a workspace-relative path; raise ValueError on path-escape attempt."""
    resolved = (workspace / rel).resolve()
    if not str(resolved).startswith(str(workspace.resolve())):
        raise ValueError(f"Path escape attempt: {rel!r}")
    return resolved


def execute_tool(name: str, inputs: dict[str, Any], workspace: Path) -> str:
    """Dispatch a tool call and return the result as a string."""
    match name:
        case "read_file":
            p = _safe(workspace, inputs["path"])
            if not p.exists():
                return f"ERROR: {inputs['path']} not found"
            return p.read_text(encoding="utf-8", errors="replace")

        case "write_file":
            p = _safe(workspace, inputs["path"])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(inputs["content"], encoding="utf-8")
            return f"OK: wrote {len(inputs['content'])} chars to {inputs['path']}"

        case "list_files":
            p = _safe(workspace, inputs["path"])
            if not p.exists():
                return f"ERROR: {inputs['path']} not found"
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

        case _:
            return f"ERROR: unknown tool {name!r}"
