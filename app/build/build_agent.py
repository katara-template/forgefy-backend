"""Build agent — Claude with file-system tools that implements a blueprint."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import anthropic

from app.build.agent_tools import TOOLS, execute_tool
from app.build.build_logger import tool_message

logger = logging.getLogger(__name__)

_MAX_ITERATIONS = 60

_BUILD_SYSTEM = """You are the Forgefy Build Agent.
Your task: implement a complete, working application from a product blueprint by modifying the files in the workspace provided to you via tools.

Instructions:
1. Start with `list_files` on '.' to see the template structure.
2. Read key config files (pubspec.yaml, package.json, app.json, etc.) to understand the template.
3. Update the app name and description everywhere it appears.
4. Implement every feature listed in the blueprint — write real, working code; no placeholders or TODOs.
5. Create all screens, components, and logic the features require.
6. When the implementation is complete, write a short summary starting with the word DONE.

Use the file tools freely. Write production-quality code."""


def run_build_agent(
    workspace: Path,
    blueprint: dict[str, Any],
    app_name: str,
    template_key: str,
    api_key: str,
    model: str,
    log_fn: Callable[[str, str], None] | None = None,
) -> str:
    """Run the build agent tool loop; return a summary string."""
    client = anthropic.Anthropic(api_key=api_key)
    user_msg = (
        f"App name: {app_name}\n"
        f"Template: {template_key}\n\n"
        f"Blueprint:\n{json.dumps(blueprint, indent=2)}\n\n"
        "Implement this application now. Start by exploring the workspace, then implement all features. "
        "Finish by writing a summary starting with DONE."
    )
    return _loop(client, model, _BUILD_SYSTEM, workspace, user_msg, log_fn)


_UPDATE_SYSTEM = """You are the Forgefy Update Agent making changes to an existing application.
Apply only what the user asked for. Do not rewrite working code unnecessarily.
Read the relevant files first, make the changes, then write a short summary starting with DONE."""


def run_update_agent(
    workspace: Path,
    prompt: str,
    blueprint: dict[str, Any],
    app_name: str,
    api_key: str,
    model: str,
    log_fn: Callable[[str, str], None] | None = None,
) -> str:
    """Run the update agent; return a summary string."""
    client = anthropic.Anthropic(api_key=api_key)
    user_msg = (
        f"App name: {app_name}\n"
        f"Existing blueprint context:\n{json.dumps(blueprint, indent=2)}\n\n"
        f"User's update request: {prompt}\n\n"
        "Apply this change now. Read the relevant files first, make the changes, "
        "then write a summary starting with DONE."
    )
    return _loop(client, model, _UPDATE_SYSTEM, workspace, user_msg, log_fn)


def _loop(
    client: anthropic.Anthropic,
    model: str,
    system: str,
    workspace: Path,
    initial_user_msg: str,
    log_fn: Callable[[str, str], None] | None = None,
) -> str:
    messages: list[dict[str, Any]] = [{"role": "user", "content": initial_user_msg}]

    for _ in range(_MAX_ITERATIONS):
        if log_fn:
            log_fn("thinking", "Thinking…")

        response = client.messages.create(
            model=model,
            max_tokens=8096,
            system=system,
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        tool_results: list[dict[str, Any]] = []
        last_text = ""

        for block in response.content:
            if block.type == "text":
                last_text = block.text
                logger.debug("Agent text: %s", block.text[:120])
                if log_fn and block.text.strip():
                    # Show first 120 chars of the agent's thinking
                    preview = block.text.strip()[:120]
                    if len(block.text.strip()) > 120:
                        preview += "…"
                    log_fn("text", preview)
                if "DONE" in block.text.upper():
                    if log_fn:
                        log_fn("done", "Agent finished.")
                    return block.text
            elif block.type == "tool_use":
                msg = tool_message(block.name, block.input)
                logger.debug("Tool %s → %s", block.name, msg)
                if log_fn:
                    log_fn("tool", msg)
                result = execute_tool(block.name, block.input, workspace)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": result}
                )

        if response.stop_reason == "end_turn":
            if log_fn:
                log_fn("done", "Agent finished.")
            return last_text or "Done."

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

    if log_fn:
        log_fn("error", "Agent reached iteration limit.")
    return "Agent reached iteration limit."
