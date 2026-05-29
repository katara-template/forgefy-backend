"""GitHub token resolution — validates personal tokens and falls back to system token."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


async def get_valid_github_token(user_id: str, system_token: str) -> str:
    """Return the user's personal GitHub token if valid, otherwise the system token.

    If the stored personal token is invalid (revoked / expired), it is cleared
    from Firestore so the dashboard GitHub-connect banner reappears.
    """
    from app.db.firebase import get_firestore_client
    from app.build.github_client import GitHubClient

    db = get_firestore_client()
    doc = await db.collection("users").document(user_id).get()
    personal_token: str | None = (doc.to_dict() or {}).get("github_access_token") if doc.exists else None

    if not personal_token:
        return system_token

    if GitHubClient(personal_token).is_token_valid():
        return personal_token

    # Token is invalid — clear it so the user sees the "Connect GitHub" banner again
    logger.warning("Personal GitHub token invalid for user=%s — clearing and falling back", user_id)
    try:
        await db.collection("users").document(user_id).update({
            "github_access_token": None,
            "github_username": None,
        })
    except Exception as exc:
        logger.warning("Failed to clear invalid GitHub token user=%s: %s", user_id, exc)

    return system_token
