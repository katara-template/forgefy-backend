"""Webhook signature verification."""
from __future__ import annotations

import hashlib
import hmac
import re

_HEX_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)


def verify_signature(payload: bytes | str, signature_header: str | None, secret: str) -> bool:
    """Verify a Forgefy webhook delivery.

    Deliveries are signed ``X-Forgefy-Signature: sha256=<hex>`` — an
    HMAC-SHA256 of the raw request body using the ``webhook_secret`` returned
    when the job was created. Always verify against the RAW body bytes,
    before any JSON parsing/re-serialization (which can reorder keys and
    break the signature).

    Comparison is constant-time.
    """
    if not signature_header or not secret:
        return False

    given = signature_header.removeprefix("sha256=")
    if not _HEX_RE.match(given):
        return False

    body = payload.encode("utf-8") if isinstance(payload, str) else payload
    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected.lower(), given.lower())
