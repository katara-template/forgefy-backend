"""Shared coding-agent contract for Forgefy build agents.

Both Forgefy's own agent and the Claude Code agent implement this interface so
the rest of the build pipeline (Build Manager / workers) does not care which
agent performs the coding work. The only difference between agents is *who*
implements the build inside the project workspace.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# A log publisher / callback produced by app.build.build_logger.make_log_publisher.
LogFn = Callable[[str, str], None]
# A callable that returns True once the user has requested cancellation.
CancelFn = Callable[[], bool]


@dataclass
class AgentHealth:
    """Availability probe result for a coding agent.

    ``available`` drives the UI/backend "is this agent usable?" answer.
    ``status`` is one of ``available``, ``not_configured``, ``error``.
    ``detail`` is a human-readable, secret-free explanation (shown to admins,
    never to end users wholesale).
    """

    available: bool = False
    status: str = "not_configured"
    detail: str = ""
    # Optional short ("Available" / "Not configured" / "Error") label for UIs.
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            if self.available:
                self.label = "Available"
            elif self.status == "error":
                self.label = "Error"
            else:
                self.label = "Not configured"


@dataclass
class AgentResult:
    """Normalised result of one agent execution.

    ``outcome`` mirrors the Forgefy build state machine so the worker does not
    need agent-specific understanding: ``success``, ``failed``, ``cancelled``,
    or ``timeout``.
    """

    summary: str = ""
    tokens_used: int = 0
    outcome: str = "success"  # success | failed | cancelled | timeout
    error: str = ""
    exit_code: int | None = None
    detail: dict[str, Any] = field(default_factory=dict)


class CodingAgent(ABC):
    """Abstract coding agent — the contract every agent must honour.

    Adapt the names/types to the actual codebase (these mirror the existing
    ``run_build_agent`` / ``run_update_agent`` / ``run_fix_agent`` signatures so
    the Forgefy agent can be wrapped without changing its behaviour).
    """

    #: Stable registry key, e.g. "forgefy" or "claude_code".
    key: str = ""
    #: Human-friendly name for UIs.
    display_name: str = ""

    @abstractmethod
    def health_check(self) -> AgentHealth:
        """Return whether this agent is configured and usable, without a full build."""

    # ``build_model`` and ``claude_code_model`` are agent-specific hints carried
    # through the shared signature so the worker can call every agent the same
    # way. Each agent uses the one that applies to it and ignores the other:
    # ForgefyAgent honours ``build_model`` (its provider is BUILD_MODEL);
    # ClaudeCodeAgent honours ``claude_code_model`` (its own provider config).

    def run_build(
        self,
        *,
        workspace: Path,
        blueprint: dict[str, Any],
        app_name: str,
        template_key: str,
        log_fn: LogFn | None = None,
        cancel_fn: CancelFn | None = None,
        claude_code_model: str | None = None,
    ) -> tuple[str, int]:
        """Implement a fresh build from the blueprint. Returns (summary, tokens)."""
        raise NotImplementedError

    def run_update(
        self,
        *,
        workspace: Path,
        prompt: str,
        blueprint: dict[str, Any],
        app_name: str,
        template_key: str,
        log_fn: LogFn | None = None,
        cancel_fn: CancelFn | None = None,
        build_model: str | None = None,
        claude_code_model: str | None = None,
    ) -> tuple[str, int]:
        """Apply a prompt-driven change to an existing workspace. Returns (summary, tokens)."""
        raise NotImplementedError

    def run_fix(
        self,
        *,
        workspace: Path,
        prompt: str,
        app_name: str,
        template_key: str,
        log_fn: LogFn | None = None,
        cancel_fn: CancelFn | None = None,
        claude_code_model: str | None = None,
    ) -> tuple[str, int]:
        """Fix a compile/test error in-place. Returns (summary, tokens)."""
        raise NotImplementedError