"""Symmetric encryption for secrets stored at rest (OAuth tokens, DB passwords).

Uses Fernet (AES-128-CBC + HMAC). The key is derived from SECRETS_ENCRYPTION_KEY
via SHA-256 so any string works as input, matching how SECRET_KEY is used
elsewhere in this app — no separate "generate a Fernet key" step required.
"""
from __future__ import annotations

import base64
import hashlib
from functools import lru_cache

from cryptography.fernet import Fernet

from app.config import get_settings


@lru_cache
def _fernet() -> Fernet:
    settings = get_settings()
    digest = hashlib.sha256(settings.SECRETS_ENCRYPTION_KEY.encode()).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def encrypt(plaintext: str) -> str:
    """Encrypt a secret for storage. Returns a URL-safe base64 token string."""
    return _fernet().encrypt(plaintext.encode()).decode()


def decrypt(ciphertext: str) -> str:
    """Decrypt a value previously returned by encrypt()."""
    return _fernet().decrypt(ciphertext.encode()).decode()
