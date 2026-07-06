"""Supabase token resolution — decrypts the stored OAuth token, refreshing if expired."""
from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

_EXPIRY_BUFFER_S = 60  # refresh a little before actual expiry


async def get_valid_supabase_token(db, user_id: str, settings) -> str | None:
    """Return a valid (decrypted, refreshed-if-needed) Supabase access token for user_id.

    Returns None if the user has no linked Supabase account, or refresh fails
    (e.g. the connection was revoked — caller should treat this like "not linked").
    """
    from app.core.crypto import decrypt, encrypt
    from app.integrations import supabase_management

    doc = await db.collection("users").document(user_id).get()
    data = doc.to_dict() if doc.exists else {}
    encrypted_access = data.get("supabase_access_token")
    encrypted_refresh = data.get("supabase_refresh_token")
    expires_at = data.get("supabase_token_expires_at", 0)
    if not encrypted_access:
        return None

    if time.time() < expires_at - _EXPIRY_BUFFER_S:
        return decrypt(encrypted_access)

    if not encrypted_refresh:
        return None

    try:
        refreshed = await supabase_management.refresh_access_token(
            refresh_token=decrypt(encrypted_refresh),
            client_id=settings.SUPABASE_CLIENT_ID,
            client_secret=settings.SUPABASE_CLIENT_SECRET,
        )
    except Exception as exc:
        logger.warning("Supabase token refresh failed user=%s: %s", user_id, exc)
        return None

    new_access = refreshed["access_token"]
    new_refresh = refreshed.get("refresh_token") or decrypt(encrypted_refresh)
    await db.collection("users").document(user_id).update({
        "supabase_access_token": encrypt(new_access),
        "supabase_refresh_token": encrypt(new_refresh),
        "supabase_token_expires_at": time.time() + refreshed.get("expires_in", 3600),
    })
    return new_access
