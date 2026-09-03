"""Measure the prompt-caching saving on a representative build prompt.

    venv/Scripts/python scripts/measure_prompt_cache.py [template_key]

Sends the SAME byte-identical conversation twice — once in the pre-caching
request shape (plain `system` string, plain tool list) and once in the current
one (cached system block + cached tools + rolling message breakpoints) — so the
difference is attributable to caching alone rather than to two agents making
different choices.

Then runs a real `_loop` build over a scratch workspace and reports the
per-iteration cache split, which is the acceptance check: iteration 2 onward must
show a non-zero cache_read_input_tokens.

Costs a few cents of real API usage and needs a funded ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

import anthropic

from app.build.agent_tools import TOOLS
from app.build.build_agent import (
    _CACHED_TOOLS,
    _MAX_OUTPUT_TOKENS,
    _build_system,
    _cached_system,
    _mark_message_breakpoints,
)
from app.build.provider_loop import AnthropicAdapter, run_agent_loop
from app.config import get_settings

# Three fixed turns — enough to show the prefix being re-read on turns 2 and 3.
_TURNS = (
    "List the files in the project root, then tell me what framework this is.",
    "Now describe what you would change to add a settings page.",
    "Summarise your plan in two sentences.",
)


def _fmt(usage: Any) -> str:
    return (
        f"input={usage.input_tokens:>7,}  "
        f"cache_write={getattr(usage, 'cache_creation_input_tokens', 0) or 0:>7,}  "
        f"cache_read={getattr(usage, 'cache_read_input_tokens', 0) or 0:>7,}  "
        f"output={usage.output_tokens:>6,}"
    )


def _raw_input(usage: Any) -> int:
    return (
        usage.input_tokens
        + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
        + (getattr(usage, "cache_read_input_tokens", 0) or 0)
    )


def _billable(usage: Any) -> float:
    """Input cost in full-price-token units: writes bill 1.25x, reads 0.1x."""
    return (
        usage.input_tokens
        + 1.25 * (getattr(usage, "cache_creation_input_tokens", 0) or 0)
        + 0.10 * (getattr(usage, "cache_read_input_tokens", 0) or 0)
    )


def _replay(
    client: anthropic.Anthropic, model: str, system: str, cached: bool,
) -> tuple[int, float]:
    label = "WITH caching" if cached else "WITHOUT caching"
    print(f"-- {label} " + "-" * (56 - len(label)))
    messages: list[dict[str, Any]] = []
    raw_total = 0
    billable_total = 0.0

    for turn, text in enumerate(_TURNS, 1):
        messages.append({"role": "user", "content": [{"type": "text", "text": text}]})
        if cached:
            _mark_message_breakpoints(messages)
        resp = client.messages.create(
            model=model,
            max_tokens=1024,
            system=_cached_system(system) if cached else system,
            tools=_CACHED_TOOLS if cached else TOOLS,
            tool_choice={"type": "none"},
            messages=messages,
        )
        print(f"   turn {turn}: {_fmt(resp.usage)}")
        raw_total += _raw_input(resp.usage)
        billable_total += _billable(resp.usage)
        reply = "".join(b.text for b in resp.content if b.type == "text")
        messages.append({"role": "assistant", "content": [{"type": "text", "text": reply}]})

    print(f"   raw input tokens : {raw_total:,}")
    print(f"   billable units   : {billable_total:,.0f}\n")
    return raw_total, billable_total


class _UsageCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if message.startswith("anthropic usage:"):
            self.lines.append(message.removeprefix("anthropic usage: "))


def main() -> int:
    template = sys.argv[1] if len(sys.argv) > 1 else "next"
    settings = get_settings()
    if not settings.ANTHROPIC_API_KEY:
        print("ANTHROPIC_API_KEY is not set.")
        return 2

    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    model = settings.ANTHROPIC_MODEL
    system = _build_system(template)

    print(f"model           : {model}")
    print(f"template        : {template}")
    print(f"system prompt   : {len(system):,} chars")
    print(f"tools           : {len(TOOLS)}")
    print(f"max_tokens      : {_MAX_OUTPUT_TOKENS:,}\n")

    try:
        old_raw, old_bill = _replay(client, model, system, cached=False)
        new_raw, new_bill = _replay(client, model, system, cached=True)
    except anthropic.APIStatusError as exc:
        print(f"API call failed: {exc}")
        return 1

    print("=" * 64)
    print(f"raw input tokens   before={old_raw:,}  after={new_raw:,}")
    print(f"billable units     before={old_bill:,.0f}  after={new_bill:,.0f}")
    if old_bill:
        print(f"input cost change  {100 * (new_bill - old_bill) / old_bill:+.1f}%")
    print("=" * 64 + "\n")

    print("-- acceptance: real unified-loop build " + "-" * 24)
    capture = _UsageCapture()
    agent_log = logging.getLogger("app.build.build_agent")
    agent_log.setLevel(logging.INFO)
    agent_log.addHandler(capture)

    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        (workspace / "package.json").write_text(
            '{"name":"demo","version":"1.0.0"}', encoding="utf-8",
        )
        adapter = AnthropicAdapter(api_key="", model=model, client=client)
        summary, tokens = run_agent_loop(
            adapter, system=system,
            stable=(
                "Create src/lib/greet.ts exporting greet(name: string): string, "
                "then reply DONE: <one sentence>."
            ),
            workspace=workspace,
            max_iterations=6,
        )

    for i, line in enumerate(capture.lines, 1):
        print(f"   iter {i}: {line}")
    print(f"\nsummary      : {summary[:100]}")
    print(f"total tokens : {tokens:,}")

    later = capture.lines[1:]
    hit = any("cache_read=0 " not in ln and "cache_read=0" not in ln for ln in later)
    print(f"\nACCEPTANCE (non-zero cache_read after iteration 1): {'PASS' if hit else 'FAIL'}")
    return 0 if hit else 1


if __name__ == "__main__":
    raise SystemExit(main())
