"""Measure whether history trimming invalidates the Anthropic prompt cache.

    venv/Scripts/python scripts/measure_cache_trim.py [iterations]

A cache read happens when the rendered prefix up to a cache_control breakpoint is
byte-identical to one the API has already stored. That is a deterministic
property of the request we build, so it can be measured exactly here without
spending anything: this harness drives the real `_loop` against a scripted
client, hashes the prefix at every breakpoint on every iteration, and reports
which breakpoints would hit.

What it does NOT model: the 5-minute TTL, and the ~20-block lookback each
breakpoint walks to find a prior entry. Both can only reduce the hit rate below
what this reports, so treat these figures as the optimistic bound.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

from app.build.build_agent import _build_system
from app.build.provider_loop import AnthropicAdapter, run_agent_loop


def _digest(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]


def _serialise(block: Any) -> Any:
    """Render a content block the way the wire would see it."""
    if isinstance(block, dict):
        return {k: v for k, v in block.items() if k != "cache_control"}
    return {
        "type": getattr(block, "type", "?"),
        "text": getattr(block, "text", None),
        "id": getattr(block, "id", None),
        "name": getattr(block, "name", None),
        "input": getattr(block, "input", None),
    }


def _breakpoint_prefixes(kwargs: dict[str, Any]) -> list[tuple[str, str]]:
    """(label, prefix-hash) for each cache_control breakpoint, in render order."""
    out: list[tuple[str, str]] = []
    prefix: list[Any] = []

    for tool in kwargs["tools"]:
        prefix.append({k: v for k, v in tool.items() if k != "cache_control"})
        if "cache_control" in tool:
            out.append(("tools", _digest(prefix)))

    for block in kwargs["system"]:
        prefix.append(_serialise(block))
        if isinstance(block, dict) and "cache_control" in block:
            out.append(("system", _digest(prefix)))

    for idx, msg in enumerate(kwargs["messages"]):
        content = msg.get("content")
        blocks = content if isinstance(content, list) else [content]
        for block in blocks:
            prefix.append(_serialise(block))
            if isinstance(block, dict) and "cache_control" in block:
                out.append((f"msg[{idx}]", _digest(prefix)))
    return out


class _Stream:
    def __init__(self, message: Any) -> None:
        self._message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def __iter__(self):
        return iter(())

    def get_final_message(self) -> Any:
        return self._message


class _Client:
    """Answers every turn with one read_file call, so history grows steadily."""

    def __init__(self, requests: list[dict[str, Any]], stop_after: int) -> None:
        self.messages = self
        self._requests = requests
        self._stop_after = stop_after

    def stream(self, **kwargs: Any) -> _Stream:
        # Snapshot: the loop moves cache_control markers between turns by
        # mutating the very dicts it already sent, so storing the kwargs by
        # reference would make every recorded request show the final marker
        # positions rather than the ones actually transmitted.
        self._requests.append({
            "tools": [dict(t) for t in kwargs["tools"]],
            "system": [dict(b) if isinstance(b, dict) else b for b in kwargs["system"]],
            "messages": [
                {
                    "role": m["role"],
                    "content": (
                        [dict(b) if isinstance(b, dict) else b for b in m["content"]]
                        if isinstance(m.get("content"), list)
                        else m.get("content")
                    ),
                }
                for m in kwargs["messages"]
            ],
        })
        n = len(self._requests)
        usage = SimpleNamespace(
            input_tokens=100, output_tokens=20,
            cache_creation_input_tokens=0, cache_read_input_tokens=0,
        )
        if n >= self._stop_after:
            return _Stream(SimpleNamespace(
                content=[SimpleNamespace(type="text", text="DONE")],
                stop_reason="end_turn", usage=usage,
            ))
        block = SimpleNamespace(
            type="tool_use", id=f"t{n}", name="read_file", input={"path": f"f{n}.txt"},
        )
        return _Stream(SimpleNamespace(
            content=[block], stop_reason="tool_use", usage=usage,
        ))


def main() -> int:
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    requests: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []

    with TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        for i in range(iterations + 2):
            (workspace / f"f{i}.txt").write_text("x" * 400, encoding="utf-8")
        _ANTHROPIC_HISTORY_PAIRS = AnthropicAdapter.history_pairs
        adapter = AnthropicAdapter(
            api_key="", model="claude-sonnet-5", client=_Client(requests, iterations),
        )
        run_agent_loop(
            adapter,
            system=_build_system("next"),
            stable="build it",
            workspace=workspace,
            max_iterations=iterations + 5,
            cache_trace=trace,
        )

    print(f"history window : {_ANTHROPIC_HISTORY_PAIRS} pairs "
          f"({_ANTHROPIC_HISTORY_PAIRS * 2} messages)")
    print(f"iterations     : {len(requests)}\n")

    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    first_trim: int | None = None

    for i, kwargs in enumerate(requests):
        trimmed = trace[i]["trimmed"] if i < len(trace) else 0
        if trimmed and first_trim is None:
            first_trim = i
        hits = misses = 0
        detail = []
        for label, digest in _breakpoint_prefixes(kwargs):
            hit = digest in seen
            seen.add(digest)
            hits += hit
            misses += not hit
            detail.append(f"{label}={'HIT' if hit else 'miss'}")
        rows.append({"i": i, "trimmed": trimmed, "hits": hits,
                     "misses": misses, "detail": detail})

    print(f"{'iter':>4} {'trimmed':>8} {'hits':>5} {'miss':>5}  breakpoints")
    for r in rows:
        if r["i"] < 3 or r["trimmed"] or (first_trim and abs(r["i"] - first_trim) <= 2) \
                or r["i"] >= len(rows) - 2:
            print(f"{r['i']:>4} {r['trimmed']:>8} {r['hits']:>5} {r['misses']:>5}  "
                  f"{' '.join(r['detail'])}")

    def ratio(subset: list[dict[str, Any]]) -> str:
        h = sum(r["hits"] for r in subset)
        t = h + sum(r["misses"] for r in subset)
        return f"{h}/{t} ({100 * h / t:.0f}%)" if t else "n/a"

    print()
    if first_trim is None:
        print(f"No trim occurred in {len(requests)} iterations "
              f"(window holds {_ANTHROPIC_HISTORY_PAIRS * 2} messages).")
        print(f"overall breakpoint hit ratio : {ratio(rows)}")
        return 0

    before = [r for r in rows if r["i"] < first_trim]
    after = [r for r in rows if r["i"] >= first_trim]
    print(f"first trim at iteration {first_trim}")
    print(f"hit ratio BEFORE first trim : {ratio(before)}")
    print(f"hit ratio AFTER  first trim : {ratio(after)}")
    print()
    for kind in ("tools", "system"):
        subset = [
            (r, d) for r in after for d in r["detail"] if d.startswith(kind)
        ]
        hit = sum(1 for _, d in subset if d.endswith("HIT"))
        print(f"  {kind:<7} after trim: {hit}/{len(subset)} hit")
    msg_after = [d for r in after for d in r["detail"] if d.startswith("msg")]
    msg_hit = sum(1 for d in msg_after if d.endswith("HIT"))
    print(f"  {'msg':<7} after trim: {msg_hit}/{len(msg_after)} hit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
