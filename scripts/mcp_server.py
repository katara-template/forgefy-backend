"""Forgefy MCP server — exposes the developer extract API as MCP tools.

Lets MCP clients (Claude Code, Claude Desktop, Cursor, …) extract structured
product requirements from transcripts through your Forgefy account. Runs over
stdio; the only state is the API key in the environment.

You don't run this yourself — the MCP client spawns it over stdio. Setup:
    pip install "mcp[cli]" httpx

Claude Code:
    claude mcp add forgefy \
      -e FORGEFY_API_KEY=fgy_live_... \
      -e FORGEFY_API_URL=https://forgefy.com \
      -- python /path/to/mcp_server.py

Claude Desktop / Cursor: add the equivalent {command, args, env} entry to the
client's MCP settings JSON. Debug without an editor via:
    npx @modelcontextprotocol/inspector python scripts/mcp_server.py

Not imported by the app — this is a standalone client of the public API, so
`mcp` is deliberately not in requirements.txt.

NOTE: a copy of this file is served for download by the web app at
forgefy-meeting-to-app/public/mcp_server.py — re-copy it there after editing.
"""
from __future__ import annotations

import os
import sys

import httpx

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    sys.exit('The MCP SDK is required: pip install "mcp[cli]"')

API_URL = os.environ.get("FORGEFY_API_URL", "http://localhost:5000").rstrip("/")
API_KEY = os.environ.get("FORGEFY_API_KEY", "")
if not API_KEY:
    sys.exit("FORGEFY_API_KEY is required (create one in the Forgefy dashboard)")

mcp = FastMCP("forgefy")


def _request(method: str, path: str, headers: dict | None = None, **kwargs) -> dict:
    resp = httpx.request(
        method,
        f"{API_URL}/api/v1{path}",
        headers={"Authorization": f"Bearer {API_KEY}", **(headers or {})},
        timeout=180,
        **kwargs,
    )
    if resp.status_code >= 400:
        # RFC 7807 problem responses carry the useful message in `detail`.
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise RuntimeError(f"Forgefy API error {resp.status_code}: {detail}")
    return resp.json()


@mcp.tool()
def extract_requirements(
    transcript: str,
    extractors: list[str] | None = None,
    model_tier: str = "standard",
) -> dict:
    """Extract structured product requirements from a meeting transcript.

    Returns features, open questions, conflicting requirements, and action
    items as JSON. `extractors` limits which are run (any of: features,
    questions, conflicts, action_items) — omit for all four. Transcripts up
    to 50k characters; for longer ones use create_extract_job.
    """
    body: dict = {"transcript": transcript, "model_tier": model_tier}
    if extractors:
        body["extractors"] = extractors
    return _request("POST", "/extract", json=body)


@mcp.tool()
def create_extract_job(
    transcript: str,
    extractors: list[str] | None = None,
    model_tier: str = "standard",
    idempotency_key: str | None = None,
) -> dict:
    """Queue an async extraction job for a long transcript (up to 200k chars).

    Returns {job_id, status}. Poll with get_extract_job until status is
    "done" (result included) or "failed".
    """
    body: dict = {"transcript": transcript, "model_tier": model_tier}
    if extractors:
        body["extractors"] = extractors
    headers = {"Idempotency-Key": idempotency_key} if idempotency_key else {}
    return _request("POST", "/extract/jobs", json=body, headers=headers)


@mcp.tool()
def get_extract_job(job_id: str) -> dict:
    """Get the status of an async extraction job; includes `result` once done."""
    return _request("GET", f"/extract/jobs/{job_id}")


@mcp.tool()
def get_usage() -> dict:
    """Get this account's tier, monthly token budget, consumption, and reset date."""
    return _request("GET", "/usage")


if __name__ == "__main__":
    mcp.run()
