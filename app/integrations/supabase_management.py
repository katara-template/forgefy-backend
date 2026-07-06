"""Supabase Management API client — OAuth token exchange/refresh, org listing,
project provisioning. All calls are async (this is used from FastAPI request
handlers, not Celery workers — see app/build/github_client.py for the sync
equivalent used in worker context).
"""
from __future__ import annotations

import httpx

_API_BASE = "https://api.supabase.com/v1"
_OAUTH_BASE = "https://api.supabase.com/v1/oauth"


async def exchange_code_for_token(
    *,
    code: str,
    code_verifier: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
) -> dict:
    """Exchange an OAuth authorization code (PKCE) for access/refresh tokens."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{_OAUTH_BASE}/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "code_verifier": code_verifier,
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uri": redirect_uri,
            },
        )
        resp.raise_for_status()
        return resp.json()


async def refresh_access_token(
    *, refresh_token: str, client_id: str, client_secret: str
) -> dict:
    """Exchange a refresh token for a new access/refresh token pair."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{_OAUTH_BASE}/token",
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": client_id,
                "client_secret": client_secret,
            },
        )
        resp.raise_for_status()
        return resp.json()


async def list_organizations(access_token: str) -> list[dict]:
    """List the Supabase organizations the connected account has access to."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{_API_BASE}/organizations",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        resp.raise_for_status()
        return resp.json()


async def create_project(
    access_token: str,
    *,
    organization_id: str,
    name: str,
    db_pass: str,
    region: str = "us-east-1",
    plan: str = "free",
) -> dict:
    """Provision a new Supabase project inside the given organization."""
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{_API_BASE}/projects",
            headers={"Authorization": f"Bearer {access_token}"},
            json={
                "name": name,
                "organization_id": organization_id,
                "db_pass": db_pass,
                "region": region,
                "plan": plan,
            },
        )
        resp.raise_for_status()
        return resp.json()


async def get_api_keys(access_token: str, project_ref: str) -> list[dict]:
    """Fetch the anon/service_role API keys for a provisioned project."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{_API_BASE}/projects/{project_ref}/api-keys",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        resp.raise_for_status()
        return resp.json()
