"""Firebase (Google) token resolution — decrypts the stored OAuth token, refreshing if expired."""
from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

_EXPIRY_BUFFER_S = 60  # refresh a little before actual expiry


async def get_valid_firebase_token(db, user_id: str, settings) -> str | None:
    """Return a valid (decrypted, refreshed-if-needed) Google access token for user_id.

    Returns None if the user has no linked Google account, or refresh fails
    (e.g. the connection was revoked — caller should treat this like "not linked").
    """
    from app.core.crypto import decrypt, encrypt
    from app.integrations import firebase_management

    doc = await db.collection("users").document(user_id).get()
    data = doc.to_dict() if doc.exists else {}
    encrypted_access = data.get("firebase_access_token")
    encrypted_refresh = data.get("firebase_refresh_token")
    expires_at = data.get("firebase_token_expires_at", 0)
    if not encrypted_access:
        return None

    if time.time() < expires_at - _EXPIRY_BUFFER_S:
        return decrypt(encrypted_access)

    if not encrypted_refresh:
        return None

    try:
        refreshed = await firebase_management.refresh_access_token(
            refresh_token=decrypt(encrypted_refresh),
            client_id=settings.FIREBASE_OAUTH_CLIENT_ID,
            client_secret=settings.FIREBASE_OAUTH_CLIENT_SECRET,
        )
    except Exception as exc:
        logger.warning("Firebase token refresh failed user=%s: %s", user_id, exc)
        return None

    new_access = refreshed["access_token"]
    # Google's refresh grant does not always return a new refresh_token — keep the old one.
    new_refresh = refreshed.get("refresh_token") or decrypt(encrypted_refresh)
    await db.collection("users").document(user_id).update({
        "firebase_access_token": encrypt(new_access),
        "firebase_refresh_token": encrypt(new_refresh),
        "firebase_token_expires_at": time.time() + refreshed.get("expires_in", 3600),
    })
    return new_access
