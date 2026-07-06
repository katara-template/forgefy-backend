"""Neon Management API client — embedded/project-per-user model.

Unlike Supabase, there is no per-user OAuth here: Forgefy provisions every
project under its own Neon account using a single platform-level API key
(NEON_API_KEY). All calls are async (used from FastAPI request handlers).
"""
from __future__ import annotations

import httpx

_API_BASE = "https://console.neon.tech/api/v2"


async def create_project(api_key: str, *, name: str) -> dict:
    """Provision a new Neon project. Response includes project id, default
    branch id, and a Postgres connection string (a real secret — never expose
    it to a generated client app, see app/build/workspace.py).
    """
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{_API_BASE}/projects",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"project": {"name": name}},
        )
        resp.raise_for_status()
        return resp.json()


async def enable_data_api(api_key: str, *, project_id: str, branch_id: str) -> dict:
    """Enable the REST Data API for a branch — the public-safe surface a
    generated client app can actually call (unlike the raw Postgres connection).
    """
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{_API_BASE}/projects/{project_id}/branches/{branch_id}/data_api",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        resp.raise_for_status()
        return resp.json()
