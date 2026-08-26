"""Tests for _ollama_loop's handling of Ollama's streaming chat protocol.

Run:
    venv/Scripts/python -m pytest tests/build/test_ollama_stream.py -v

Ollama streams tool calls on an intermediate chunk (done=False) and closes with
a done chunk whose message is empty. Reading tool_calls off the done chunk
therefore drops every call, and the agent loops re-requesting the same tool
without ever writing a file. These tests pin that protocol shape.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.build import build_agent


def _chunk(content: str = "", tool_calls: list[dict] | None = None, done: bool = False) -> bytes:
    """One newline-delimited JSON chunk as Ollama emits it."""
    msg: dict = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    body: dict = {"message": msg, "done": done}
    if done:
        body.update(prompt_eval_count=100, eval_count=50)
    return json.dumps(body).encode()


def _call(name: str, args: dict) -> dict:
    return {"id": f"call_{name}", "function": {"index": 0, "name": name, "arguments": args}}


class _FakeResponse:
    """Minimal stand-in for a streaming requests.Response."""

    status_code = 200

    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_lines(self):
        yield from self._chunks

    def raise_for_status(self):
        return None


class _FakePoster:
    """Serves a scripted stream per turn and records the payloads sent."""

    def __init__(self, turns: list[list[bytes]]):
        self._turns = turns
        self.payloads: list[dict] = []

    def __call__(self, url, json=None, headers=None, timeout=None, stream=None):
        self.payloads.append(json)
        idx = min(len(self.payloads) - 1, len(self._turns) - 1)
        return _FakeResponse(self._turns[idx])


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


def _run(monkeypatch, turns: list[list[bytes]], workspace: Path, **kwargs):
    # _ollama_loop does `import requests as _req` at call time, so the patch has
    # to land on the requests module itself rather than a build_agent attribute.
    import requests

    poster = _FakePoster(turns)
    monkeypatch.setattr(requests, "post", poster)
    summary, tokens = build_agent._ollama_loop(
        "http://ollama:11434", "test-model", "system", workspace, "task",
        timeout=5, **kwargs,
    )
    return summary, tokens, poster


class TestStreamedToolCalls:
    def test_tool_call_on_intermediate_chunk_is_executed(self, monkeypatch, workspace):
        """The regression: tool_calls arrive before done and must not be dropped."""
        turns = [
            # Turn 1: tool call on a non-done chunk, empty done chunk after it.
            [
                _chunk(content="Writing the file."),
                _chunk(tool_calls=[_call("write_file", {"path": "app.py", "content": "x = 1\n"})]),
                _chunk(done=True),
            ],
            # Turn 2: agent finishes.
            [_chunk(content="DONE"), _chunk(done=True)],
        ]
        summary, tokens, poster = _run(monkeypatch, turns, workspace)

        assert (workspace / "app.py").read_text() == "x = 1\n", "tool call was dropped"
        assert "DONE" in summary
        assert tokens == 300, "token counts from both done chunks should accumulate"

        # The tool result must be fed back so the model can proceed.
        roles = [m["role"] for m in poster.payloads[1]["messages"]]
        assert "tool" in roles, "tool result was not sent back to the model"

    def test_tool_calls_on_done_chunk_still_work(self, monkeypatch, workspace):
        """Older/other servers may attach tool_calls to the done chunk."""
        turns = [
            [_chunk(tool_calls=[_call("write_file", {"path": "a.py", "content": "1"})], done=True)],
            [_chunk(content="DONE"), _chunk(done=True)],
        ]
        _run(monkeypatch, turns, workspace)
        assert (workspace / "a.py").exists()

    def test_multiple_tool_calls_across_chunks_all_execute(self, monkeypatch, workspace):
        turns = [
            [
                _chunk(tool_calls=[_call("write_file", {"path": "one.py", "content": "1"})]),
                _chunk(tool_calls=[_call("write_file", {"path": "two.py", "content": "2"})]),
                _chunk(done=True),
            ],
            [_chunk(content="DONE"), _chunk(done=True)],
        ]
        _run(monkeypatch, turns, workspace)
        assert (workspace / "one.py").exists() and (workspace / "two.py").exists()


class TestToolResultLinkage:
    def test_tool_result_carries_tool_name(self, monkeypatch, workspace):
        """Ollama identifies a result by tool_name; without it the model re-calls."""
        turns = [
            [_chunk(tool_calls=[_call("list_files", {"path": "."})]), _chunk(done=True)],
            [_chunk(content="DONE"), _chunk(done=True)],
        ]
        _, _, poster = _run(monkeypatch, turns, workspace)

        tool_msgs = [m for m in poster.payloads[1]["messages"] if m["role"] == "tool"]
        assert tool_msgs, "no tool result was sent back"
        assert tool_msgs[0].get("tool_name") == "list_files"


class TestExplorationBudget:
    def test_repeated_read_is_not_re_executed(self, monkeypatch, workspace):
        """The same read-only call twice runs once; the repeat gets a notice."""
        (workspace / "a.py").write_text("original\n")
        read = [_chunk(tool_calls=[_call("read_file", {"path": "a.py"})]), _chunk(done=True)]
        turns = [read, read, [_chunk(content="DONE"), _chunk(done=True)]]

        calls: list[tuple] = []
        real = build_agent.execute_tool
        monkeypatch.setattr(
            build_agent, "execute_tool",
            lambda n, i, w, log=None: (calls.append((n, i)), real(n, i, w, log))[1],
        )
        _, _, poster = _run(monkeypatch, turns, workspace)

        assert len(calls) == 1, f"read_file should execute once, ran {len(calls)}x"
        results = [m["content"] for m in poster.payloads[2]["messages"] if m["role"] == "tool"]
        assert any("already ran" in r for r in results), "repeat got no notice"

    def test_write_invalidates_the_read_cache(self, monkeypatch, workspace):
        """After a write the same read is legitimate again and must re-execute."""
        (workspace / "a.py").write_text("original\n")
        read = [_chunk(tool_calls=[_call("read_file", {"path": "a.py"})]), _chunk(done=True)]
        write = [
            _chunk(tool_calls=[_call("write_file", {"path": "a.py", "content": "changed\n"})]),
            _chunk(done=True),
        ]
        turns = [read, write, read, [_chunk(content="DONE"), _chunk(done=True)]]

        reads: list[str] = []
        real = build_agent.execute_tool

        def spy(name, args, ws, log=None):
            if name == "read_file":
                reads.append(args["path"])
            return real(name, args, ws, log)

        monkeypatch.setattr(build_agent, "execute_tool", spy)
        _run(monkeypatch, turns, workspace)
        assert len(reads) == 2, "read after a write must not be served from cache"

    def test_endless_exploration_is_interrupted(self, monkeypatch, workspace):
        """A model that only ever inspects gets pushed to write, then stopped."""
        explore = [_chunk(tool_calls=[_call("list_files", {"path": "."})]), _chunk(done=True)]
        _, _, poster = _run(monkeypatch, [explore], workspace, max_iterations=80)

        assert len(poster.payloads) < 80, (
            f"exploration ran {len(poster.payloads)} turns — budget did not engage"
        )
        pushes = [
            m for p in poster.payloads for m in p["messages"]
            if m["role"] == "user" and "without writing anything" in m.get("content", "")
        ]
        assert pushes, "agent was never pushed to stop exploring"


class TestResultTruncation:
    def test_truncation_tells_the_model_not_to_retry(self):
        """A bare '…[truncated]' invites a re-read that returns the same bytes."""
        out = build_agent._truncate_tool_result("x" * 5000, 1000)
        assert len(out) < 5000
        assert "5,000" in out and "1,000" in out, "should state what was withheld"
        assert "do not" in out.lower() and "again" in out.lower()

    def test_short_results_pass_through_untouched(self):
        assert build_agent._truncate_tool_result("hello", 1000) == "hello"

    def test_cloud_limit_fits_a_real_component_file(self):
        """1500 chars truncated mid-file, which is what stranded the fix agent."""
        assert build_agent._TOOL_RESULT_LIMIT_LARGE_CTX > 20_000
        assert build_agent._TOOL_RESULT_LIMIT_LARGE_CTX > build_agent._TOOL_RESULT_LIMIT_SMALL_CTX

    def test_large_file_is_not_truncated_on_cloud(self, monkeypatch, workspace):
        """A ~9KB file must reach the model whole when running against cloud."""
        body = "\n".join(f"// line {i} of a component file" for i in range(300))
        assert len(body) > 8000
        (workspace / "big.tsx").write_text(body)

        # _ollama_loop imports using_cloud at call time, so patch the source.
        monkeypatch.setattr("app.ai.ollama_http.using_cloud", lambda: True)
        turns = [
            [_chunk(tool_calls=[_call("read_file", {"path": "big.tsx"})]), _chunk(done=True)],
            [_chunk(content="DONE"), _chunk(done=True)],
        ]
        _, _, poster = _run(monkeypatch, turns, workspace)

        result = next(m["content"] for m in poster.payloads[1]["messages"] if m["role"] == "tool")
        assert "TRUNCATED" not in result, "cloud run truncated a 9KB file"
        assert "line 299" in result, "tail of the file never reached the model"


class TestNudgeBounding:
    def test_talking_agent_does_not_burn_every_iteration(self, monkeypatch, workspace):
        """A model that only ever talks gets bounded pushbacks, then we stop.

        Previously this nudged on every iteration, producing the observed
        'No files written yet' loop until max_iterations was exhausted.
        """
        chatty = [_chunk(content="I will list the files."), _chunk(done=True)]
        summary, _, poster = _run(
            monkeypatch, [chatty], workspace, max_iterations=20,
        )
        assert len(poster.payloads) <= build_agent._MAX_NUDGES + 1, (
            f"expected to stop after {build_agent._MAX_NUDGES} nudges, "
            f"made {len(poster.payloads)} requests"
        )
        assert summary
