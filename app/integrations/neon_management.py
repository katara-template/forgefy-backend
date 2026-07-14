"""Neon Management API client — embedded/project-per-user model.

Unlike Supabase, there is no per-user OAuth here: Forgefy provisions every
project under its own Neon account using a single platform-level API key
(NEON_API_KEY). All calls are async (used from FastAPI request handlers).
"""
from __future__ import annotations

import httpx

from app.core.exceptions import ExternalServiceError

_API_BASE = "https://console.neon.tech/api/v2"


def _checked_json(resp: httpx.Response, action: str) -> dict:
    """Return the response JSON, or raise a domain error carrying Neon's
    message so the API surfaces *why* provisioning failed instead of an
    opaque 500."""
    if resp.status_code >= 400:
        raise ExternalServiceError(
            f"Neon API error while {action} (HTTP {resp.status_code}): {resp.text[:500]}"
        )
    return resp.json()


async def create_project(api_key: str, *, name: str, org_id: str | None = None) -> dict:
    """Provision a new Neon project. Response includes project id, default
    branch id, and a Postgres connection string (a real secret — never expose
    it to a generated client app, see app/build/workspace.py).

    org_id is required by Neon for organization-scoped API keys (found on the
    Neon console's organization settings page); personal keys may omit it.
    """
    project: dict = {"name": name}
    if org_id:
        project["org_id"] = org_id
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{_API_BASE}/projects",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"project": project},
        )
        return _checked_json(resp, "creating project")


async def enable_data_api(
    api_key: str, *, project_id: str, branch_id: str, database_name: str
) -> dict:
    """Enable the REST Data API for one database on a branch — the public-safe
    surface a generated client app can actually call (unlike the raw Postgres
    connection). Empty JSON body = no auth provider configured at enable time.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{_API_BASE}/projects/{project_id}/branches/{branch_id}/data-api/{database_name}",
            headers={"Authorization": f"Bearer {api_key}"},
            json={},
        )
        return _checked_json(resp, "enabling the Data API")
