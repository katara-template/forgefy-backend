"""Coding agent registry — the single place Forgefy resolves *which* agent codes.

Forgefy supports two coding agents that share one build pipeline:

    forgefy      → ForgefyAgent     (the original agent, unchanged)
    claude_code  → ClaudeCodeAgent  (runs the Claude Code CLI headlessly)

The rest of Forgefy calls ``get_coding_agent(selected)`` (or ``resolve_agent`` to
safely fall back) and never cares which agent is selected. A project that omits
the field keeps using ``forgefy`` — no migration required.
"""
from __future__ import annotations

from pathlib import Path

from app.build.agents.base import AgentHealth, CodingAgent
from app.build.agents.claude_code import ClaudeCodeAgent


class ForgefyAgent(CodingAgent):
    """The original Forgefy agent, adapted to the shared ``CodingAgent`` contract.

    Thin wrapper over the existing, unchanged internal loop entry points
    (``run_build_agent`` / ``run_update_agent`` / ``run_fix_agent``): existing
    projects keep working without migration. The agent's model selection
    (``BUILD_MODEL``) is independent of agent selection.
    """

    key = "forgefy"
    display_name = "Forgefy Agent"

    def health_check(self) -> AgentHealth:
        from app.build.build_agent import _build_model
        from app.config import get_settings

        s = get_settings()
        model = _build_model()
        configured = {
            "claude": bool(s.ANTHROPIC_API_KEY),
            "gemini": bool(s.GEMINI_API_KEY),
            "gpt": bool(s.OPENAI_API_KEY),
            "qwen3": bool(s.OLLAMA_URL) or bool(s.OPENROUTER_API_KEY),
        }
        if model in configured and configured[model]:
            return AgentHealth(available=True, status="available",
                               detail=f"Backend {model!r} is configured.")
        if any(configured.values()):
            return AgentHealth(available=True, status="available",
                               detail="A provider backend is configured.")
        return AgentHealth(available=False, status="not_configured",
                           detail="No build provider (BUILD_MODEL) is configured in .env.")

    def run_build(
        self,
        *,
        workspace: Path,
        blueprint: dict,
        app_name: str,
        template_key: str,
        log_fn=None,
        cancel_fn=None,
        claude_code_model: str | None = None,  # ignored — Forgefy's model is BUILD_MODEL
    ) -> tuple[str, int]:
        from app.build.build_agent import run_build_agent

        return run_build_agent(
            workspace=workspace,
            blueprint=blueprint,
            app_name=app_name,
            template_key=template_key,
            log_fn=log_fn,
            cancel_fn=cancel_fn,
        )

    def run_update(
        self,
        *,
        workspace: Path,
        prompt: str,
        blueprint: dict,
        app_name: str,
        template_key: str,
        log_fn=None,
        cancel_fn=None,
        build_model: str | None = None,
        claude_code_model: str | None = None,  # ignored — Forgefy's model is BUILD_MODEL
    ) -> tuple[str, int]:
        from app.build.build_agent import run_update_agent

        return run_update_agent(
            workspace=workspace,
            prompt=prompt,
            blueprint=blueprint,
            app_name=app_name,
            template_key=template_key,
            log_fn=log_fn,
            cancel_fn=cancel_fn,
            build_model=build_model,
        )

    def run_fix(
        self,
        *,
        workspace: Path,
        prompt: str,
        app_name: str,
        template_key: str,
        log_fn=None,
        cancel_fn=None,
        claude_code_model: str | None = None,  # ignored — Forgefy's model is BUILD_MODEL
    ) -> tuple[str, int]:
        from app.build.build_agent import run_fix_agent

        return run_fix_agent(
            workspace=workspace,
            prompt=prompt,
            app_name=app_name,
            template_key=template_key,
            log_fn=log_fn,
            cancel_fn=cancel_fn,
        )


# The canonical set of supported coding agents. Keep in sync with the frontend.
SUPPORTED_AGENTS: tuple[str, ...] = ("forgefy", "claude_code")

DEFAULT_AGENT: str = "forgefy"

__all__ = [
    "AgentHealth",
    "CodingAgent",
    "ForgefyAgent",
    "SUPPORTED_AGENTS",
    "DEFAULT_AGENT",
    "get_coding_agent",
    "resolve_agent",
    "list_agents",
    "agent_health",
]


def get_coding_agent(agent: str | None) -> CodingAgent:
    """Return the CodingAgent instance for ``agent``.

    Raises ``ValueError`` for an unknown agent so callers can surface a proper
    validation error (API layer) instead of silently mis-shuffling builds.
    """
    normalized = (agent or DEFAULT_AGENT).strip().lower()
    if normalized == "forgefy":
        return ForgefyAgent()
    if normalized == "claude_code":
        return ClaudeCodeAgent()
    raise ValueError(f"Unknown coding agent {agent!r}. Supported agents: {', '.join(SUPPORTED_AGENTS)}")


def resolve_agent(agent: str | None) -> str:
    """Return a validated agent key, defaulting to ``forgefy``.

    Unlike ``get_coding_agent`` this does not raise: it returns ``forgefy`` for
    anything unrecognised, preserving backwards compatibility for old projects.
    """
    normalized = (agent or "").strip().lower()
    return normalized if normalized in SUPPORTED_AGENTS else DEFAULT_AGENT


def list_agents() -> list[CodingAgent]:
    """Return every registered agent (useful for availability UIs)."""
    return [ForgefyAgent(), ClaudeCodeAgent()]


def agent_health(agent: str | None) -> AgentHealth:
    """Health-check one agent; ``None``/unknown falls back to Forgefy Agent."""
    try:
        return get_coding_agent(agent).health_check()
    except Exception as exc:  # noqa: BLE001 — a probe must never crash the API
        return AgentHealth(available=False, status="error", detail=str(exc)[:200])


def all_agents_health() -> list[dict]:
    """Health-check every agent, returning UI-ready dicts (no secrets)."""
    out = []
    for key in SUPPORTED_AGENTS:
        health = agent_health(key)
        out.append(
            {
                "key": key,
                "display_name": get_coding_agent(key).display_name,
                "available": health.available,
                "status": health.status,
                "label": health.label,
            }
        )
    return out