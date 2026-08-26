"""Meeting URL parsing and Meeting SDK JWT minting.

Standard library only, deliberately: both the backend connector and the
dependency-free scripts/run_local.py rely on these, and a local smoke test of
the bot should not require the backend's virtualenv.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

_SDK_JWT_TTL_SECONDS = 24 * 60 * 60


def _b64url(raw: bytes) -> str:
    """Base64url without padding, per JWS."""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def build_sdk_jwt(client_id: str, client_secret: str, ttl_seconds: int = _SDK_JWT_TTL_SECONDS) -> str:
    """Mint the JWT the Meeting SDK authenticates with.

    Signed by the caller rather than inside the container so the SDK client
    secret never enters a process that joins untrusted meetings. The appKey /
    tokenExp claim names are fixed by Zoom.
    """
    issued = int(time.time())
    expires = issued + ttl_seconds

    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "appKey": client_id,
        "iat": issued,
        "exp": expires,
        "tokenExp": expires,
    }

    signing_input = (
        _b64url(json.dumps(header, separators=(",", ":")).encode())
        + "."
        + _b64url(json.dumps(payload, separators=(",", ":")).encode())
    )
    signature = hmac.new(
        client_secret.encode(), signing_input.encode(), hashlib.sha256
    ).digest()

    return f"{signing_input}.{_b64url(signature)}"


def parse_meeting_url(meeting_url: str) -> tuple[str, str | None]:
    """Extract (meeting_number, passcode) from a Zoom join URL or a bare ID.

    Handles the common shapes:
        https://zoom.us/j/1234567890?pwd=abc
        https://acme.zoom.us/w/1234567890?pwd=abc     (webinar)
        1234567890
        123 456 7890

    Personal-room links (/my/vanityname) carry no numeric ID and return ""; the
    caller is expected to reject those rather than guess.
    """
    from urllib.parse import parse_qs, urlparse

    raw = (meeting_url or "").strip()
    if not raw:
        return "", None

    # A bare meeting ID, possibly written with spaces or dashes.
    digits = "".join(c for c in raw if c.isdigit())
    if digits and digits == raw.replace(" ", "").replace("-", ""):
        return digits, None

    parsed = urlparse(raw)
    passcode = (parse_qs(parsed.query).get("pwd") or [None])[0]

    # The meeting ID is the path segment following /j/, /w/ or /s/.
    segments = [s for s in parsed.path.split("/") if s]
    for marker in ("j", "w", "s"):
        if marker in segments:
            index = segments.index(marker)
            if index + 1 < len(segments):
                candidate = "".join(c for c in segments[index + 1] if c.isdigit())
                if candidate:
                    return candidate, passcode

    # Fall back to any all-digit segment (covers /postattendee?confno=…).
    for segment in segments:
        if segment.isdigit():
            return segment, passcode

    return "", passcode
