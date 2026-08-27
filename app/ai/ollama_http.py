"""Shared HTTP concerns for talking to Ollama — local daemon or Ollama Cloud.

Both speak the identical ``/api/chat`` protocol, so the only differences are the
host, a bearer token, and whether capping the context window makes sense.
Setting ``OLLAMA_API_KEY`` switches every Ollama call site to the hosted service
at https://ollama.com; leaving it blank keeps the original local behaviour.

Note this is a different axis from ``OPENROUTER_API_KEY`` (see app/ai/qwen.py):
OpenRouter replaces the Ollama protocol entirely, whereas this keeps it and
only changes where the request lands. OpenRouter still wins when both are set.
"""
from __future__ import annotations

import contextlib
import logging
from typing import Any

CLOUD_URL = "https://ollama.com"

logger = logging.getLogger(__name__)

# URLs that mean "the local daemon" — a cloud key alongside any of these is a
# leftover from the local setup rather than a deliberate choice.
_LOCAL_URLS = frozenset({
    "",
    "http://ollama:11434",
    "http://localhost:11434",
    "http://127.0.0.1:11434",
})


def _resolve(settings: Any | None):
    """Use caller-supplied settings when given, else the process settings.

    Call sites that already hold a Settings object pass it through so injected
    /test settings are honoured rather than silently bypassed.
    """
    if settings is not None:
        return settings
    from app.config import get_settings

    return get_settings()


def _api_key(settings: Any | None = None) -> str:
    return (getattr(_resolve(settings), "OLLAMA_API_KEY", "") or "").strip()


def using_cloud(settings: Any | None = None) -> bool:
    """True when Ollama calls should go to the hosted ollama.com service."""
    return bool(_api_key(settings))


def ollama_base_url(settings: Any | None = None) -> str:
    """Resolve the Ollama host, preferring cloud when an API key is configured.

    An API key paired with a Docker-internal URL can never work — that host does
    not exist in a cloud deployment — so the key wins and we point at cloud.
    An explicit non-local OLLAMA_URL is always respected (self-hosted, proxy).
    """
    settings = _resolve(settings)
    url = (getattr(settings, "OLLAMA_URL", "") or "").strip().rstrip("/")
    if using_cloud(settings) and url in _LOCAL_URLS:
        return CLOUD_URL
    return url or "http://ollama:11434"


def ollama_build_model(settings: Any | None = None) -> str:
    """Model for code generation — OLLAMA_BUILD_MODEL, else OLLAMA_MODEL.

    Kept separate because the two roles reward different things: synthesis wants
    thorough extraction, code generation wants fast decisive tool use. Mirrors
    the per-task routing the OpenRouter backend already does.
    """
    settings = _resolve(settings)
    build = (getattr(settings, "OLLAMA_BUILD_MODEL", "") or "").strip()
    return build or (getattr(settings, "OLLAMA_MODEL", "") or "").strip()


def ollama_headers(settings: Any | None = None) -> dict[str, str]:
    """Bearer auth for the hosted service; empty dict for a local daemon."""
    key = _api_key(settings)
    return {"Authorization": f"Bearer {key}"} if key else {}


def ollama_options(num_ctx: int, num_predict: int) -> dict[str, int]:
    """Generation options tuned for whichever backend is active.

    The num_ctx cap exists to stop a local model OOMing on small hardware.
    Cloud models are served with their full window (256K+) and no memory
    pressure on our side, so applying the cap there would silently truncate
    long transcripts for no benefit — drop it.
    """
    if using_cloud():
        return {"num_predict": num_predict}
    return {"num_ctx": num_ctx, "num_predict": num_predict}


def missing_model_hint(model: str) -> str:
    """Actionable message for an HTTP 404 (unknown model), per backend."""
    if using_cloud():
        return (
            f"Model '{model}' is not available on Ollama Cloud. Cloud tags end "
            f"in '-cloud' (e.g. 'qwen3.5:cloud') — see https://ollama.com/search?c=cloud"
        )
    return (
        f"Model '{model}' not found in Ollama. "
        f"Run 'docker compose exec ollama ollama pull {model}' to load it."
    )


def auth_error_hint(status_code: int) -> str | None:
    """Message for auth failures against the hosted service, else None."""
    if status_code not in (401, 403):
        return None
    if using_cloud():
        return (
            f"Ollama Cloud rejected the API key (HTTP {status_code}). Check "
            "OLLAMA_API_KEY — create one at https://ollama.com/settings/keys"
        )
    return (
        f"Ollama returned HTTP {status_code}. If this endpoint is the hosted "
        "service or an authenticating proxy, set OLLAMA_API_KEY."
    )


# ── Retrying /api/chat transport ─────────────────────────────────────────────
# Ollama Cloud rate-limits without warning (429), and a build phase fires
# requests back to back — a single transient response used to kill the whole
# task. Both the legacy loop and the provider-adapter path share this opener so
# there is exactly one backoff policy to reason about.

_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})
_MAX_RETRIES = 4
_BASE_DELAY_S = 5.0


def open_chat_stream(
    url: str,
    *,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: int,
    model: str,
    log_fn=None,
):
    """POST /api/chat with stream=True, retrying rate-limit/overload responses.

    Returns an open streamed ``requests`` response ready for ``iter_lines()``.
    Non-retryable failures raise immediately with an actionable hint; a
    retryable status is retried with exponential backoff, honouring any
    Retry-After header Ollama sends.
    """
    import time as _time

    import requests as _req

    delay = _BASE_DELAY_S
    last_exc: Exception | None = None
    told_user = False

    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = _req.post(
                url, json=payload, headers=headers, timeout=(30, timeout), stream=True,
            )
        except _req.exceptions.RequestException as exc:
            # A connection error is local/network, never someone else's load.
            last_exc = exc
            if log_fn and not told_user:
                told_user = True
                # User-facing wording stays provider-agnostic; the name and the
                # error live in the worker log only.
                log_fn("warning", "The AI service is briefly unreachable — checking the connection…")
            logger.warning("Ollama unreachable at %s: %s", url, exc)
        else:
            if resp.status_code == 404:
                raise RuntimeError(missing_model_hint(model))
            if hint := auth_error_hint(resp.status_code):
                raise RuntimeError(hint)
            if resp.status_code not in _RETRYABLE_STATUS:
                resp.raise_for_status()
                return resp

            # Retryable: prefer Ollama's own Retry-After over our guess.
            retry_after = resp.headers.get("retry-after")
            with contextlib.suppress(TypeError, ValueError):
                delay = max(delay, float(retry_after or 0))
            last_exc = RuntimeError(f"{resp.status_code} {resp.reason} for {url}")
            if log_fn and not told_user:
                told_user = True
                # Provider-agnostic on purpose: users should never see which
                # backend is serving their build.
                log_fn(
                    "thinking",
                    "The AI is in high demand right now — backing off briefly, hang tight…"
                    if resp.status_code == 429
                    else "High demand right now — taking a short pause…",
                )

        if attempt < _MAX_RETRIES:
            logger.info("Ollama retry %d/%d in %.0fs", attempt + 1, _MAX_RETRIES, delay)
            _time.sleep(delay)
            delay *= 2

    raise RuntimeError(f"Ollama agent request failed after {_MAX_RETRIES + 1} attempts: {last_exc}")
