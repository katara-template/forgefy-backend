"""Omnisend API client — contact sync for marketing email.

Omnisend has no "send this exact email now" transactional endpoint; instead
campaigns and automations (welcome series, product updates, etc.) are built in
the Omnisend dashboard and target contacts based on their subscription status
and tags synced here. All calls are async (used from FastAPI request handlers).
"""
from __future__ import annotations

import httpx

from app.core.exceptions import ExternalServiceError

_API_BASE = "https://api.omnisend.com/v3"


async def upsert_contact(
    api_key: str,
    *,
    email: str,
    subscribed: bool,
    tags: list[str] | None = None,
) -> dict:
    """Create or update an Omnisend contact by email.

    `subscribed` must reflect explicit marketing-email consent captured at
    signup — anyone without it syncs as "nonSubscribed" (visible in Omnisend
    for segmentation, but never sent marketing email) rather than defaulting
    to opted-in.
    """
    payload = {
        "identifiers": [
            {
                "type": "email",
                "id": email,
                "channels": {
                    "email": {"status": "subscribed" if subscribed else "nonSubscribed"}
                },
            }
        ],
    }
    if tags:
        payload["tags"] = tags

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            f"{_API_BASE}/contacts",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json=payload,
        )
    if resp.status_code >= 400:
        raise ExternalServiceError(
            f"Omnisend API error while syncing contact (HTTP {resp.status_code}): {resp.text[:500]}"
        )
    return resp.json()
