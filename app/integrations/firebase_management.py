"""Firebase/GCP Management API client — Google OAuth token exchange/refresh
plus project provisioning (GCP project → Firebase → web app → Firestore).

Unlike Supabase/Neon, Firebase has no dedicated "app builder" product: a real
per-user Firebase project only exists inside the connecting user's own GCP
account, so this is OAuth-based (like Supabase), and every creation step is an
asynchronous GCP long-running operation that must be polled before the next
step can run. All calls are async (used from FastAPI request handlers).

This is entirely separate from Forgefy's own Firebase usage in app/db/firebase.py,
which manages one global service-account-backed app for Forgefy's own Firestore
collections — nothing here touches that.
"""
from __future__ import annotations

import asyncio

import httpx

_TOKEN_URL = "https://oauth2.googleapis.com/token"
_RESOURCE_MANAGER_BASE = "https://cloudresourcemanager.googleapis.com/v3"
_FIREBASE_BASE = "https://firebase.googleapis.com/v1beta1"
_FIRESTORE_BASE = "https://firestore.googleapis.com/v1"

_OPERATION_TIMEOUT_S = 60.0
_OPERATION_POLL_INTERVAL_S = 2.0


async def exchange_code_for_token(
    *, code: str, code_verifier: str, client_id: str, client_secret: str, redirect_uri: str
) -> dict:
    """Exchange an OAuth authorization code (PKCE) for access/refresh tokens."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            _TOKEN_URL,
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


async def refresh_access_token(*, refresh_token: str, client_id: str, client_secret: str) -> dict:
    """Exchange a refresh token for a new access token."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            _TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": client_id,
                "client_secret": client_secret,
            },
        )
        resp.raise_for_status()
        return resp.json()


async def _poll_operation(client: httpx.AsyncClient, access_token: str, operation_name: str) -> dict:
    """Poll a GCP/Firebase long-running operation until done.

    `operation_name` is the resource name GCP returns (e.g.
    "operations/abc123" or "projects/x/operations/abc123") — always fetched
    relative to the Firebase Management API host, which serves both GCP
    Resource Manager and Firebase operations under /v1/{operation_name}.
    """
    deadline = asyncio.get_event_loop().time() + _OPERATION_TIMEOUT_S
    headers = {"Authorization": f"Bearer {access_token}"}
    while True:
        resp = await client.get(f"https://firebase.googleapis.com/v1/{operation_name}", headers=headers)
        resp.raise_for_status()
        op = resp.json()
        if op.get("done"):
            if "error" in op:
                raise RuntimeError(f"GCP operation {operation_name} failed: {op['error']}")
            return op.get("response", {})
        if asyncio.get_event_loop().time() >= deadline:
            raise TimeoutError(f"GCP operation {operation_name} did not complete within {_OPERATION_TIMEOUT_S}s")
        await asyncio.sleep(_OPERATION_POLL_INTERVAL_S)


async def create_gcp_project(access_token: str, *, project_id: str, display_name: str) -> dict:
    """Create a new standalone GCP project (no parent org/folder)."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{_RESOURCE_MANAGER_BASE}/projects",
            headers={"Authorization": f"Bearer {access_token}"},
            json={"projectId": project_id, "displayName": display_name},
        )
        resp.raise_for_status()
        operation = resp.json()
        return await _poll_operation(client, access_token, operation["name"])


async def add_firebase_to_project(access_token: str, *, project_id: str) -> dict:
    """Attach Firebase to an existing GCP project."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{_FIREBASE_BASE}/projects/{project_id}:addFirebase",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        resp.raise_for_status()
        operation = resp.json()
        return await _poll_operation(client, access_token, operation["name"])


async def create_web_app(access_token: str, *, project_id: str, display_name: str) -> dict:
    """Create a Firebase Web App under the project; returns the created app resource (has appId)."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{_FIREBASE_BASE}/projects/{project_id}/webApps",
            headers={"Authorization": f"Bearer {access_token}"},
            json={"displayName": display_name},
        )
        resp.raise_for_status()
        operation = resp.json()
        return await _poll_operation(client, access_token, operation["name"])


async def get_web_app_config(access_token: str, *, project_id: str, app_id: str) -> dict:
    """Fetch the client-safe SDK config (apiKey, authDomain, projectId, etc.)."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{_FIREBASE_BASE}/projects/{project_id}/webApps/{app_id}/config",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        resp.raise_for_status()
        return resp.json()


async def create_firestore_database(
    access_token: str, *, project_id: str, location_id: str = "nam5"
) -> dict:
    """Create the default Firestore database (Native mode) for the project."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{_FIRESTORE_BASE}/projects/{project_id}/databases",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"databaseId": "(default)"},
            json={"type": "FIRESTORE_NATIVE", "locationId": location_id},
        )
        resp.raise_for_status()
        operation = resp.json()
        return await _poll_operation(client, access_token, operation["name"])


async def provision_project(access_token: str, *, project_id: str, display_name: str) -> dict:
    """Run the full provisioning sequence and return the assembled web app config.

    Kept as a single orchestrating call (rather than have the endpoint sequence
    these itself) so app/api/v1/projects.py's connect_firebase stays a thin
    endpoint like the Supabase/Neon ones.
    """
    await create_gcp_project(access_token, project_id=project_id, display_name=display_name)
    await add_firebase_to_project(access_token, project_id=project_id)
    web_app = await create_web_app(access_token, project_id=project_id, display_name=display_name)
    config = await get_web_app_config(access_token, project_id=project_id, app_id=web_app["appId"])
    await create_firestore_database(access_token, project_id=project_id)
    return config
