"""Claude Code coding agent — a native Forgefy agent backed by the Claude Agent SDK.

Claude Code runs through the official Python Agent SDK (``claude-agent-sdk``),
which owns the bundled CLI process, wire protocol and lifecycle. The agent
operates directly on the same project workspace the Forgefy agent uses, and its
streamed output is converted into Forgefy's canonical build events (published
through the same Redis log publisher) so the frontend never needs to know which
agent produced them.

Design notes
------------
* The SDK's ``query()`` async generator yields typed messages; we consume the
  ``StreamEvent`` deltas (the SDK's parsed form of ``--output-format stream-json``)
  and the final ``ResultMessage``. No hand-rolled subprocess or JSON parsing.
* Provider configuration (Anthropic directly, or an Anthropic-compatible
  endpoint such as Ollama) is injected through ``ClaudeAgentOptions.env`` — a
  whitelisted dict built on the existing scrubbed ``build_subprocess_env``, so
  Forgefy secrets are not handed to the agent and no global ``os.environ``
  mutation is needed.
* Process safety: an overall timeout, an idle timeout, exit-outcome taxonomy
  (success / failed / cancelled / timeout) and cancellation are enforced here;
  closing/cancelling the SDK generator terminates the CLI process, so no
  orphaned Claude Code survives a cancelled build.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import platform
import queue
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.build.agents.base import AgentHealth, CodingAgent, LogFn, CancelFn
from app.build.subprocess_env import build_subprocess_env

logger = logging.getLogger(__name__)

# Tool names that write files — used to surface file_written-style events and a
# human-friendly tool label.
_WRITE_TOOLS = frozenset({"Write", "Edit", "MultiEdit", "NotebookEdit", "StrReplaceEdit"})
_READ_TOOLS = frozenset({"Read", "ReadBlock", "Glob", "List"})


class ClaudeCodeError(RuntimeError):
    """A Claude Code run failed for a non-secret reason."""


class ClaudeCodeCancelled(RuntimeError):
    """The run was cancelled by the user before it finished."""


class ClaudeCodeTimeout(ClaudeCodeError):
    """The run exceeded its wall-clock or idle budget."""


@dataclass
class _Outcome:
    summary: str = ""
    tokens: int = 0
    state: str = "success"  # success | failed | cancelled | timeout | idle_timeout
    error: str = ""


def _bundled_cli() -> str | None:
    """Path of the CLI binary bundled inside the claude-agent-sdk wheel, if any.

    Returns ``None`` when the SDK isn't installed so ``resolve_cli`` can fall
    through to a PATH lookup instead of raising ImportError at call time.
    """
    try:
        import claude_agent_sdk
    except ImportError:
        return None

    cli_name = "claude.exe" if platform.system() == "Windows" else "claude"
    bundled = Path(claude_agent_sdk.__file__).parent / "_bundled" / cli_name
    return str(bundled) if bundled.is_file() else None


def resolve_cli(settings) -> str | None:
    """Resolve the Claude Code CLI the SDK will run.

    Order: an explicit CLAUDE_CODE_CLI_PATH, the bundled SDK CLI, then
    CLAUDE_CODE_COMMAND / ``claude`` on PATH. ``None`` means not installed.
    """
    explicit = (getattr(settings, "CLAUDE_CODE_CLI_PATH", "") or "").strip()
    if explicit and Path(explicit).is_file():
        return explicit
    bundled = _bundled_cli()
    if bundled:
        return bundled
    command = (getattr(settings, "CLAUDE_CODE_COMMAND", "") or "claude").strip()
    if command and os.path.isabs(command) and Path(command).is_file():
        return command
    return shutil.which(command or "claude")


def _claude_env(settings, claude_code_model: str | None = None) -> dict[str, str]:
    """Build the whitelisted environment dict handed to the SDK.

    Starts from ``build_subprocess_env`` (toolchain vars only, no secrets), then
    injects exactly the provider configuration Claude Code needs:
      * Anthropic-compatible base URL / auth token (supports an Ollama endpoint
        exposing the Anthropic-compatible API, e.g. ``ANTHROPIC_AUTH_TOKEN=ollama``
        + ``ANTHROPIC_BASE_URL=http://localhost:11434``).
      * An API key when the operator configured one for Claude Code (falls back
        to the platform ANTHROPIC_API_KEY; a missing key simply means Claude Code
        must resolve auth another way, e.g. a pre-provisioned keychain).
    ``claude_code_model`` overrides the model env var when the operator set a
    per-project model (multi-model picker).
    """
    extra: dict[str, str] = {}
    api_key = (getattr(settings, "CLAUDE_CODE_API_KEY", "") or "").strip() or (
        getattr(settings, "ANTHROPIC_API_KEY", "") or ""
    ).strip()
    if api_key:
        extra["ANTHROPIC_API_KEY"] = api_key
    base_url = (getattr(settings, "CLAUDE_CODE_BASE_URL", "") or "").strip()
    if base_url:
        extra["ANTHROPIC_BASE_URL"] = base_url
    auth_token = (getattr(settings, "CLAUDE_CODE_AUTH_TOKEN", "") or "").strip()
    if auth_token:
        extra["ANTHROPIC_AUTH_TOKEN"] = auth_token
    model = claude_code_model or (getattr(settings, "CLAUDE_CODE_MODEL", "") or "").strip()
    if model:
        extra["ANTHROPIC_MODEL"] = model
    return build_subprocess_env(extra)


def _build_options(
    *,
    settings,
    workspace: Path,
    prompt: str,
    system_extra: str = "",
    claude_code_model: str | None = None,
) -> Any:
    """Build ``ClaudeAgentOptions`` for one headless build run.

    ``options.env`` is the whitelisted provider/toolchain dict (never the whole
    worker environment); ``cwd`` and ``add_dirs`` pin Claude Code to this
    project's workspace so it cannot wander into other projects.
    """
    from claude_agent_sdk import ClaudeAgentOptions

    allowed = (getattr(settings, "CLAUDE_CODE_ALLOWED_TOOLS", "") or "").strip()
    options_kwargs: dict[str, Any] = {
        "cwd": str(workspace.resolve()),
        "add_dirs": [str(workspace.resolve())],
        "include_partial_messages": True,
        "env": _claude_env(settings, claude_code_model=claude_code_model),
    }
    cli_path = resolve_cli(settings)
    if cli_path:
        options_kwargs["cli_path"] = cli_path
    permission_mode = (getattr(settings, "CLAUDE_CODE_PERMISSION_MODE", "") or "").strip()
    if permission_mode:
        options_kwargs["permission_mode"] = permission_mode
    if allowed:
        options_kwargs["allowed_tools"] = [t.strip() for t in allowed.split(",") if t.strip()]
    model = claude_code_model or (getattr(settings, "CLAUDE_CODE_MODEL", "") or "").strip()
    if model:
        options_kwargs["model"] = model
    max_turns = getattr(settings, "CLAUDE_CODE_MAX_TURNS", 0) or 0
    if max_turns:
        options_kwargs["max_turns"] = int(max_turns)
    # Claude Code's own system prompt, with the Forgefy build context appended.
    if system_extra:
        options_kwargs["system_prompt"] = {
            "type": "preset",
            "preset": "claude_code",
            "append": system_extra,
        }
    return ClaudeAgentOptions(**options_kwargs)


class _StreamParser:
    """Convert Claude Code SDK stream events into Forgefy build events.

    The SDK yields ``StreamEvent`` objects whose ``.event`` is the raw
    stream-json wire dict (``content_block_start/delta/stop``, ``message_start``
    …) — the parsed form of ``--output-format stream-json``. This mapper turns
    those into the canonical Forgefy log events (``text``, ``thinking``,
    ``tool``, ``file_written``) consumed by the existing WebSocket feed.
    """

    _FLUSH_CHARS = 160

    def __init__(self, log_fn: LogFn | None) -> None:
        self.log_fn = log_fn
        self.text_buf: list[str] = []
        self.thinking_buf: list[str] = []
        # content index -> {"name", "input_json", "input"}
        self._tools: dict[int, dict[str, Any]] = {}
        self.result_text: list[str] = []
        self.result_meta: dict[str, Any] = {}
        self.error_text: str = ""
        self.is_error: bool = False
        self.subtype: str = ""
        self.finished: bool = False

    def _emit(self, event_type: str, message: str) -> None:
        if self.log_fn is not None:
            try:
                self.log_fn(event_type, message)
            except Exception as exc:  # noqa: BLE001 — a bad UI callback must not kill the build
                logger.warning("log_fn failed for %s: %s", event_type, exc)

    def _flush_text(self, force: bool = False) -> None:
        if self.text_buf:
            chunk = "".join(self.text_buf)
            self.text_buf = []
            if chunk:
                self._emit("text", chunk)
        if force:
            self._flush_thinking()

    def _flush_thinking(self) -> None:
        buf = "".join(self.thinking_buf)
        self.thinking_buf = []
        if buf:
            self._emit("thinking", buf)

    def _maybe_flush_text(self) -> None:
        s = "".join(self.text_buf)
        if len(s) >= self._FLUSH_CHARS or s.endswith("\n") or s.endswith(". ") or s.endswith("? ") or s.endswith("! "):
            self._flush_text(force=True)

    def _tool_label(self, name: str, inputs: dict) -> str:
        name = name or "tool"
        path = inputs.get("file_path") or inputs.get("path") or inputs.get("notebook_path") or inputs.get("pattern") or inputs.get("dir_path") or ""
        if name in _WRITE_TOOLS:
            return f"Writing file · {path}" if path else "Writing file …"
        if name in _READ_TOOLS:
            return f"Reading · {path}" if path else "Reading files …"
        if name == "Bash":
            d = inputs.get("command") or inputs.get("description") or ""
            return f"Running `{d[:80]}`" if d else "Running command"
        if name == "TodoWrite":
            return "Updating task list"
        return f"Using {name}"

    def handle_stream_event(self, event: dict[str, Any]) -> None:
        """Consume one raw stream-json wire event (``StreamEvent.event``)."""
        if not isinstance(event, dict):
            return
        etype = event.get("type")
        if etype == "message_start":
            self._flush_text(force=True)
            return
        if etype == "content_block_start":
            block = event.get("content_block") or {}
            if block.get("type") == "tool_use":
                self._tools[event.get("index")] = {
                    "name": block.get("name") or "",
                    "input_json": "",
                    "input": {},
                }
            return
        if etype == "content_block_delta":
            delta = event.get("delta") or {}
            dtype = delta.get("type")
            idx = event.get("index")
            if dtype == "text_delta":
                text = delta.get("text", "")
                if text:
                    self.text_buf.append(text)
                    self._maybe_flush_text()
            elif dtype == "thinking_delta":
                thinking = delta.get("thinking", "")
                if thinking:
                    self.thinking_buf.append(thinking)
                    if len("".join(self.thinking_buf)) >= self._FLUSH_CHARS:
                        self._flush_thinking()
            elif dtype == "input_json_delta":
                tool = self._tools.get(idx)
                if tool is not None:
                    tool["input_json"] += delta.get("partial_json", "")
            return
        if etype == "content_block_stop":
            tool = self._tools.pop(event.get("index"), None)
            if tool is not None:
                try:
                    tool["input"] = json.loads(tool["input_json"]) if tool.get("input_json") else {}
                except (json.JSONDecodeError, ValueError):
                    tool["input"] = {}
                name = tool.get("name") or ""
                self._emit("tool", self._tool_label(name, tool["input"]))
                if name in _WRITE_TOOLS and tool["input"].get("file_path"):
                    self._emit("file_written", f"Wrote {tool['input']['file_path']}")
            return
        # ping / message_stop / message_delta / system — ignored.

    def handle_result(self, msg: Any) -> None:
        """Consume the SDK's final ``ResultMessage``."""
        self.finished = True
        self.subtype = getattr(msg, "subtype", "") or ""
        self.is_error = bool(getattr(msg, "is_error", False))
        result = getattr(msg, "result", None)
        if isinstance(result, str) and result.strip():
            self.result_text.append(result)
        usage = getattr(msg, "usage", None)
        if isinstance(usage, dict):
            self.result_meta = usage
        if self.is_error:
            errors = getattr(msg, "errors", None) or []
            self.error_text = " ".join(str(e) for e in errors) if errors else ""

    def token_count(self) -> int:
        m = self.result_meta
        keys = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
        return int(sum(int(m.get(k) or 0) for k in keys))

    def close(self) -> None:
        self._flush_text(force=True)


async def _drive(
    *,
    options: Any,
    prompt: str,
    parser: _StreamParser,
    cancel_fn: CancelFn | None,
    timeout: float,
    idle_timeout: float,
) -> _Outcome:
    """Drive the SDK ``query()`` generator with timeout/idle/cancel enforcement.

    The SDK owns the CLI process: cancelling the consumer task unwinds the
    generator, and the SDK's cleanup terminates the bundled Claude Code process —
    so a cancelled or timed-out build cannot leave an orphan behind.
    """
    from claude_agent_sdk import query

    gen = query(prompt=prompt, options=options)
    state = {"last": time.monotonic()}

    async def _consume() -> None:
        async for msg in gen:
            state["last"] = time.monotonic()
            kind = type(msg).__name__
            if kind == "StreamEvent":
                event = getattr(msg, "event", None)
                if isinstance(event, dict):
                    parser.handle_stream_event(event)
            elif kind == "ResultMessage":
                parser.handle_result(msg)
                return  # the run is over
            # AssistantMessage / SystemMessage / UserMessage: the deltas above
            # already carry text/thinking/tool content; complete messages would
            # only duplicate them.

    consumer = asyncio.create_task(_consume())
    start = time.monotonic()
    trip: str | None = None
    try:
        while not consumer.done():
            if cancel_fn is not None and cancel_fn():
                trip = "cancelled"
                break
            now = time.monotonic()
            if now - start > timeout:
                trip = "timeout"
                logger.warning("Claude Code overall timeout after %.0fs", timeout)
                break
            if now - state["last"] > idle_timeout:
                trip = "idle_timeout"
                logger.warning("Claude Code idle timeout after %.0fs", idle_timeout)
                break
            await asyncio.sleep(0.5)
    finally:
        if trip is not None:
            # Cancelling the iteration makes the SDK tear the CLI process down.
            consumer.cancel()
            try:
                await consumer
            except BaseException:  # noqa: BLE001 — teardown only
                pass
        else:
            # Let a natural finish (or its exception) surface unchanged.
            try:
                await consumer
            except asyncio.CancelledError:
                trip = "cancelled"
        # Close the generator so the SDK releases the CLI process promptly
        # instead of waiting for garbage collection after the loop is gone.
        with contextlib.suppress(Exception):
            await gen.aclose()
        parser.close()

    if trip is not None:
        return _Outcome(state=trip, error="Claude Code stopped by user." if trip == "cancelled" else "Claude Code timed out.")

    if not parser.finished:
        # The generator ended without a ResultMessage — treat as failure.
        return _Outcome(state="failed", error="Claude Code ended without a final result.")

    summary = "\n".join(t for t in parser.result_text if t).strip()
    if parser.is_error or parser.subtype.endswith("error"):
        return _Outcome(
            state="failed",
            tokens=parser.token_count(),
            summary=summary,
            error=parser.error_text or "Claude Code reported an error result.",
        )
    return _Outcome(state="success", tokens=parser.token_count(), summary=summary)


# ---------------------------------------------------------------------------
# Force-cancel registry (worker teardown safety net)
# ---------------------------------------------------------------------------

# workspace path -> set of Events that force the in-flight run to stop. The SDK
# owns the CLI process and the driver's cancel check honours these, so worker
# teardown can stop a wedged run without touching SDK internals.
_FORCE_STOP: dict[str, set[threading.Event]] = {}
_FORCE_LOCK = threading.Lock()


def _pop_force_events(key: str) -> set[threading.Event]:
    with _FORCE_LOCK:
        return _FORCE_STOP.pop(key, set())


def kill_workspace_claude(workspace: Path) -> int:
    """Force-stop every in-flight Claude Code run for one workspace."""
    events = _pop_force_events(str(Path(workspace).resolve()))
    for ev in events:
        ev.set()
    if events:
        logger.info("Requested stop for %d Claude Code run(s) for %s", len(events), workspace)
    return len(events)


def kill_all_claude() -> int:
    """Force-stop every in-flight Claude Code run (process-exit safety net)."""
    with _FORCE_LOCK:
        keys = list(_FORCE_STOP)
    return sum(len(_pop_force_events(k)) for k in keys)


def _map_sdk_error(exc: BaseException) -> ClaudeCodeError:
    """Translate an SDK exception into a clear, secret-free Forgefy error."""
    from claude_agent_sdk import CLIConnectionError, CLINotFoundError, ProcessError, ResultError

    if isinstance(exc, CLINotFoundError):
        return ClaudeCodeError(
            "The Claude Code CLI could not be found. Install it (or ship the "
            "claude-agent-sdk wheel with its bundled CLI) and check CLAUDE_CODE_ENABLED."
        )
    # ResultError subclasses ProcessError (an error result, not a CLI crash), so
    # it must be checked first to preserve the provider's message.
    if isinstance(exc, ResultError):
        return ClaudeCodeError(f"Claude Code returned an error result: {str(exc)[:500]}")
    if isinstance(exc, ProcessError):
        code = getattr(exc, "exit_code", None)
        stderr = (getattr(exc, "stderr", "") or "").strip()
        detail = f" (exit code {code})" if code is not None else ""
        snippet = f": {stderr[:500]}" if stderr else ""
        return ClaudeCodeError(f"Claude Code exited unexpectedly{detail}{snippet}")
    if isinstance(exc, CLIConnectionError):
        return ClaudeCodeError(
            "Could not start a Claude Code session. Check the CLI installation and provider configuration."
        )
    return ClaudeCodeError(f"Claude Code could not complete the build: {str(exc)[:500]}")


def run_claude(
    *,
    workspace: Path,
    prompt: str,
    log_fn: LogFn | None = None,
    cancel_fn: CancelFn | None = None,
    settings=None,
    system_extra: str = "",
    timeout: float | None = None,
    idle_timeout: float | None = None,
    claude_code_model: str | None = None,
) -> _Outcome:
    """Run one headless Claude Code build/update against ``workspace``.

    Returns a normalised ``_Outcome``. Raises ``ClaudeCodeCancelled`` on user
    cancel and ``ClaudeCodeTimeout`` on timeout; other failures raise
    ``ClaudeCodeError``. Streams progress to ``log_fn`` from the driver thread.
    ``claude_code_model`` overrides the model for this run (multi-model picker).
    """
    from app.config import get_settings

    settings = settings or get_settings()
    workspace = workspace.resolve()
    timeout = timeout if timeout is not None else float(getattr(settings, "CLAUDE_CODE_TIMEOUT", 1800) or 1800)
    idle_timeout = idle_timeout if idle_timeout is not None else float(getattr(settings, "CLAUDE_CODE_IDLE_TIMEOUT", 300) or 300)

    options = _build_options(
        settings=settings,
        workspace=workspace,
        prompt=prompt,
        system_extra=system_extra,
        claude_code_model=claude_code_model,
    )
    parser = _StreamParser(log_fn)
    force_stop = threading.Event()
    key = str(workspace)
    with _FORCE_LOCK:
        _FORCE_STOP.setdefault(key, set()).add(force_stop)

    def _combined_cancel() -> bool:
        if force_stop.is_set():
            return True
        try:
            return bool(cancel_fn()) if cancel_fn is not None else False
        except Exception:  # noqa: BLE001 — a broken cancel probe must not kill the build
            return False

    results: "queue.Queue[_Outcome | BaseException]" = queue.Queue()

    def _thread_main() -> None:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            outcome = loop.run_until_complete(
                _drive(
                    options=options,
                    prompt=prompt,
                    parser=parser,
                    cancel_fn=_combined_cancel,
                    timeout=timeout,
                    idle_timeout=idle_timeout,
                )
            )
            results.put(outcome)
        except BaseException as exc:  # noqa: BLE001 — forwarded to the worker thread
            results.put(exc)
        finally:
            asyncio.set_event_loop(None)
            loop.close()
            with _FORCE_LOCK:
                registry = _FORCE_STOP.get(key)
                if registry is not None:
                    registry.discard(force_stop)
                    if not registry:
                        _FORCE_STOP.pop(key, None)

    logger.info("Starting Claude Code agent (workspace=%s)", workspace)
    thread = threading.Thread(target=_thread_main, daemon=True, name="forgefy-claude-code")
    thread.start()
    try:
        outcome = results.get(timeout=timeout + 120)
    except queue.Empty:  # pragma: no cover — the driver enforces its own timeout
        force_stop.set()
        raise ClaudeCodeTimeout("Claude Code did not finish within its overall timeout.")

    if isinstance(outcome, BaseException):
        raise _map_sdk_error(outcome)

    if outcome.state == "cancelled":
        raise ClaudeCodeCancelled("Claude Code stopped by user.")
    if outcome.state in ("timeout", "idle_timeout"):
        raise ClaudeCodeTimeout(outcome.error or "Claude Code timed out.")
    if outcome.state == "failed":
        raise ClaudeCodeError(f"Claude Code could not complete the build: {outcome.error or 'unknown error'}")
    return outcome


def _build_system_prompt(app_name: str, template_key: str) -> str:
    """Forgefy build context appended to Claude Code's own system prompt."""
    return (
        f"You are implementing a {template_key} application named {app_name!r} inside the "
        "Forgefy build pipeline. Work directly in this project workspace using your tools — "
        "read existing files before writing, install dependencies, run the build/test commands, "
        "and only modify code that belongs to this project. Reply with a concise, user-friendly "
        "summary prefixed with 'DONE:' when finished."
    )


class ClaudeCodeAgent(CodingAgent):
    key = "claude_code"
    display_name = "Claude Code"

    def health_check(self) -> AgentHealth:
        from app.config import get_settings

        settings = get_settings()
        if not getattr(settings, "CLAUDE_CODE_ENABLED", True):
            return AgentHealth(available=False, status="not_configured",
                               detail="Claude Code is disabled (CLAUDE_CODE_ENABLED=false).")
        try:
            import claude_agent_sdk  # noqa: F401 — proves the SDK is importable
        except ImportError:
            return AgentHealth(available=False, status="not_configured",
                               detail="The claude-agent-sdk package is not installed.")
        cli = resolve_cli(settings)
        if not cli:
            return AgentHealth(available=False, status="not_configured",
                               detail="No Claude Code CLI found (neither the SDK-bundled binary nor CLAUDE_CODE_COMMAND on PATH).")
        return AgentHealth(available=True, status="available",
                           detail=f"Claude Code CLI resolved: {Path(cli).name}.")

    # ── build ────────────────────────────────────────────────────────────────

    def run_build(
        self,
        *,
        workspace: Path,
        blueprint: dict,
        app_name: str,
        template_key: str,
        log_fn: LogFn | None = None,
        cancel_fn: CancelFn | None = None,
        claude_code_model: str | None = None,
    ) -> tuple[str, int]:
        import json as _json

        from app.config import get_settings

        prompt = (
            f"App name: {app_name}\nTemplate: {template_key}\n\n"
            f"Blueprint:\n{_json.dumps(blueprint, indent=2)}\n\n"
            "Build this application now. Narrate each step as you go. When finished, "
            "write a user-friendly summary starting with DONE: that describes what was built "
            "— screens, features, and anything notable."
        )
        outcome = run_claude(
            workspace=workspace,
            prompt=prompt,
            log_fn=log_fn,
            cancel_fn=cancel_fn,
            settings=get_settings(),
            system_extra=_build_system_prompt(app_name, template_key),
            claude_code_model=claude_code_model,
        )
        return outcome.summary, outcome.tokens

    # ── update ───────────────────────────────────────────────────────────────

    def run_update(
        self,
        *,
        workspace: Path,
        prompt: str,
        blueprint: dict,
        app_name: str,
        template_key: str,
        log_fn: LogFn | None = None,
        cancel_fn: CancelFn | None = None,
        build_model: str | None = None,
        claude_code_model: str | None = None,
    ) -> tuple[str, int]:
        """``build_model`` is accepted for interface parity but is Forgefy-agent
        specific: Claude Code picks its model from its own provider configuration
        (CLAUDE_CODE_MODEL / ANTHROPIC_MODEL), independent of BUILD_MODEL.
        ``claude_code_model`` overrides the model for this run (multi-model picker).
        """
        from app.config import get_settings

        outcome = run_claude(
            workspace=workspace,
            prompt=prompt,
            log_fn=log_fn,
            cancel_fn=cancel_fn,
            settings=get_settings(),
            system_extra=_build_system_prompt(app_name, template_key),
            claude_code_model=claude_code_model,
        )
        return outcome.summary, outcome.tokens

    # ── fix ──────────────────────────────────────────────────────────────────

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
        from app.config import get_settings

        outcome = run_claude(
            workspace=workspace,
            prompt=prompt,
            log_fn=log_fn,
            cancel_fn=cancel_fn,
            settings=get_settings(),
            claude_code_model=claude_code_model,
        )
        return outcome.summary, outcome.tokens


import atexit  # noqa: E402

atexit.register(kill_all_claude)