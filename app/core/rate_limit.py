"""Shared slowapi Limiter singleton.

Both app/main.py (app.state.limiter registration) and route files
(@limiter.limit decorators) must reference the same instance.

Limits are stored in Redis so they're shared across uvicorn workers and
instances — per-process memory would multiply every limit by the number of
workers. If Redis is unreachable the limiter falls back to per-process memory
rather than failing requests.
"""
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import get_settings


def _storage_uri() -> str:
    """Build the limits storage URI from REDIS_URL.

    Strips the ssl_cert_reqs query param (same quirk handled in app/main.py)
    and adds short socket timeouts so an unreachable Redis degrades to the
    in-memory fallback quickly instead of stalling every request.
    """
    url = get_settings().REDIS_URL
    if not url:
        return ""  # slowapi's default — per-process in-memory storage
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    qs.pop("ssl_cert_reqs", None)
    qs["socket_connect_timeout"] = ["1"]
    qs["socket_timeout"] = ["1"]
    return urlunparse(parsed._replace(query=urlencode(qs, doseq=True)))


limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["300/minute"],
    storage_uri=_storage_uri(),
    in_memory_fallback_enabled=True,
)
