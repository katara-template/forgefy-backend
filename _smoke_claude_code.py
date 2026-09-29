"""Dump raw SDK messages for one Claude Code run (debugging aid)."""
import asyncio
import tempfile
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, query

from app.build.agents.claude_code import _claude_env, resolve_cli
from app.config import get_settings


async def main() -> None:
    s = get_settings()
    ws = Path(tempfile.mkdtemp(prefix="forgefy_cc_dbg_"))
    opts = ClaudeAgentOptions(
        cwd=str(ws),
        add_dirs=[str(ws)],
        include_partial_messages=True,
        env=_claude_env(s),
        cli_path=resolve_cli(s),
        permission_mode="bypassPermissions",
        system_prompt={"type": "preset", "preset": "claude_code", "append": "Reply briefly."},
    )
    async for msg in query(prompt="Create hello.txt containing exactly: hi", options=opts):
        kind = type(msg).__name__
        if kind == "StreamEvent":
            ev = getattr(msg, "event", {})
            print("STREAM", ev.get("type"), str(ev.get("delta") or ev.get("content_block") or "")[:80])
        else:
            print("=====", kind, "=====")
            for f in ("subtype", "is_error", "num_turns", "stop_reason", "result", "usage", "errors"):
                print("   ", f, "=", repr(getattr(msg, f, "<none>"))[:160])


if __name__ == "__main__":
    asyncio.run(main())