"""API key generation and hashing.

Keys look like ``fgy_live_<43 url-safe chars>`` and are shown to the caller
exactly once at creation — only the SHA-256 hash is persisted.

SHA-256 (not bcrypt) is deliberate: generated keys carry 256 bits of entropy,
so offline brute-force against a leaked hash is not a realistic threat, and a
fast hash keeps per-request auth off the CPU (bcrypt at cost 12 burns ~250ms
per verification — see app/core/security.py — which is fine for logins but
not for every API call).
"""
from __future__ import annotations

import hashlib
import secrets

KEY_PREFIX = "fgy_live_"

# How much of the key is kept in plaintext for display in key lists
# ("fgy_live_k3J9…"). Enough to tell keys apart, far too little to guess the
# remaining ~37 chars of entropy.
_DISPLAY_PREFIX_LEN = 12


def generate_api_key() -> str:
    """Return a new API key: prefix + 32 random bytes, url-safe encoded."""
    return KEY_PREFIX + secrets.token_urlsafe(32)


def hash_api_key(key: str) -> str:
    """Return the hex SHA-256 digest under which the key is stored/looked up."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def display_prefix(key: str) -> str:
    """Return the short plaintext prefix stored for display in key lists."""
    return key[:_DISPLAY_PREFIX_LEN]


def looks_like_api_key(token: str) -> bool:
    """Cheap shape check so JWT bearer tokens are never hashed and looked up."""
    return token.startswith(KEY_PREFIX)


# Enough for any sane rotation scheme; a cap mostly guards against a runaway
# script (or a repeatedly-approved CLI login) minting keys in a loop. Shared
# by the dashboard's create_api_key (app/api/v1/keys.py) and the CLI device
# login's auto-minted key (app/api/v1/device_auth.py) — one account-wide cap.
MAX_ACTIVE_KEYS_PER_USER = 10


async def count_active_api_keys(db, owner_user_id: str) -> int:
    """Return how many of this user's API keys are not revoked."""
    docs = await db.collection("api_keys").where("owner_user_id", "==", owner_user_id).get()
    return sum(1 for d in docs if not (d.to_dict() or {}).get("revoked_at"))
