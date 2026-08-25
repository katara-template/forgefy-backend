"""Unfinished-work tracking, so an interrupted run can be continued later.

A run can stop with work outstanding: the step limit hits, the token budget runs
out, the worker dies. The code written so far reaches GitHub (app/build/auto_sync.py)
and MEMORY.md records which files changed — but nothing records WHAT was being
attempted or how far it got. So the next run starts from the user's original
sentence with no idea a previous attempt exists, and tends to redo finished work.

This stores that record on the project document and turns it back into prompt
context. It is deliberately small and stringly-typed: the record crosses a
Firestore document, written by one worker and read by another possibly days
later, so it must not depend on any in-process state.

It complements rather than replaces the other two continuity sources:
  * MEMORY.md   — what the codebase looks like now
  * chat history — what has been asked before
  * this        — what was in flight when everything stopped
"""
from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

# Field name on the project document.
FIELD = "unfinished_work"

# Why a run stopped, and how to say it to the model.
_REASONS: dict[str, str] = {
    "step_limit": "it reached its maximum number of steps",
    "quota": "the token budget for the account ran out",
    "error": "it hit an unexpected error",
    "stopped": "the user stopped it",
}

# A bare "continue" carries no task of its own — the task is whatever was
# interrupted. Kept tight on purpose: "continue the checkout flow" is a real
# instruction and must NOT be swallowed and replaced by an older request.
_CONTINUE_PHRASES = re.compile(
    r"^(continue|resume|keep going|carry on|go on|finish( it| that)?|"
    r"continue( it| that)?|pick up where you left off)[.!]?$",
    re.I,
)

# Long summaries are truncated: this is injected into every subsequent prompt.
_MAX_PROGRESS_CHARS = 1200
_MAX_REQUEST_CHARS = 500


def is_continue_command(prompt: str) -> bool:
    """True when the prompt is a bare 'continue' with no task of its own."""
    return bool(_CONTINUE_PHRASES.match((prompt or "").strip()))


def record(request: str, reason: str, progress: str = "", kind: str = "update") -> dict[str, Any]:
    """Build the project-document patch marking work as unfinished.

    ``kind`` tells the continue path how to resume: ``"build"`` records were an
    initial build interrupted mid-run (resume by re-entering the code-writing
    worker against the partially built repo); ``"update"`` were ordinary chat
    updates.
    """
    return {
        FIELD: {
            "request": (request or "")[:_MAX_REQUEST_CHARS],
            "reason": reason if reason in _REASONS else "error",
            "progress": (progress or "")[:_MAX_PROGRESS_CHARS],
            "kind": kind if kind in ("build", "update") else "update",
            "recorded_at": datetime.now(UTC).isoformat(),
        }
    }


def build_request(blueprint: dict[str, Any], app_name: str) -> str:
    """The original task text for an interrupted BUILD, rebuilt from its blueprint.

    Stored the moment a build starts so that even a hard worker death leaves a
    resumable record — the next device only needs this plus the GitHub repo
    auto-sync kept current.
    """
    description = str((blueprint or {}).get("app_description") or "").strip()
    features = [str(f.get("name", "")).strip()
                for f in (blueprint or {}).get("features", [])[:10]]
    features = [f for f in features if f]
    parts = [f"Finish building the '{app_name}' application."]
    if description:
        parts.append(f"App description: {description}")
    if features:
        parts.append("Blueprint features to have working: " + ", ".join(features) + ".")
    return " ".join(parts)[:_MAX_REQUEST_CHARS]


def clear() -> dict[str, Any]:
    """Patch clearing the record — a completed run has nothing outstanding."""
    return {FIELD: None}


def pending(project: dict[str, Any]) -> dict[str, Any] | None:
    """Return the unfinished-work record on a project, if any is usable."""
    raw = (project or {}).get(FIELD)
    if not isinstance(raw, dict) or not raw.get("request"):
        return None
    return raw


def resolve_request(prompt: str, unfinished: dict[str, Any] | None) -> str:
    """The task to actually run.

    A bare "continue" means "the thing that was interrupted", so the original
    request is substituted. Anything else is taken at face value — the user gets
    to change direction, and the resume context below still tells the agent what
    came before.
    """
    if unfinished and is_continue_command(prompt):
        return str(unfinished["request"])
    return prompt


def resume_context(unfinished: dict[str, Any] | None) -> str:
    """Prompt block describing the interrupted run, or '' when there is none."""
    if not unfinished:
        return ""

    reason = _REASONS.get(str(unfinished.get("reason")), _REASONS["error"])
    progress = str(unfinished.get("progress") or "").strip()
    progress_block = (
        f"\nWhat that run reported doing before stopping:\n{progress}\n"
        if progress
        else "\nThat run stopped before reporting any progress.\n"
    )

    return (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "CONTINUING AN INTERRUPTED RUN\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "A previous attempt on this project did not finish because "
        f"{reason}.\n\n"
        f"What was originally asked for:\n{unfinished['request']}\n"
        + progress_block
        + "\nThe partial work from that run is ALREADY in this workspace — it was\n"
        "committed before the run ended. Read the relevant files and continue from\n"
        "what is there. Do NOT start over, and do NOT redo what is listed above as\n"
        "already done; finish what remains.\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    )
