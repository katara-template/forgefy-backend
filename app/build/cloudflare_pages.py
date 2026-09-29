"""Cloudflare Pages REST helpers — custom domains for published apps.

Deployments themselves go through ``wrangler`` (see
``app/workers/build_worker.py``). This module covers only the REST-API side that
wrangler does not expose: attaching a stable ``<slug>.<base-domain>`` to a Pages
project so a *published* app has a real URL instead of a ``*.pages.dev`` one.

Endpoint (verified against the Cloudflare API reference)::

    POST /accounts/{account_id}/pages/projects/{project_name}/domains
    {"name": "bun.forgefy.dev"}

Auth is the same bearer token wrangler uses; it needs the **Pages Write**
permission. When the base domain's zone is on the *same* account as the Pages
project, Cloudflare creates the CNAME record automatically. Otherwise the domain
is accepted but never resolves — which is why :func:`publish_domain` returns
``None`` when it cannot attach, letting the caller fall back to ``pages.dev``.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_API_BASE = "https://api.cloudflare.com/client/v4"

# Substrings that mean "somebody already owns this name" rather than a
# transient/auth failure — the only case where retrying a suffixed name helps.
_CONFLICT_MARKERS = ("already", "in use", "taken", "exists", "duplicate")


def slugify_name(name: str, *, max_len: int = 28) -> str:
    """Lowercase alphanumeric/hyphen slug.

    Capped at 28 chars so the result is valid both as a Cloudflare Pages project
    name and as a DNS label.
    """
    slug = re.sub(r"[^a-z0-9-]", "-", (name or "").lower())
    slug = re.sub(r"-+", "-", slug).strip("-")
    return (slug or "forgefy-app")[:max_len].rstrip("-") or "forgefy-app"


def _headers(api_token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }


def list_domains(account_id: str, project_name: str, api_token: str, *, timeout: int = 30) -> list[str]:
    """Return the custom domains currently attached to a Pages project.

    Best-effort: a lookup failure returns ``[]`` so a re-publish still proceeds
    to the attach call rather than aborting the whole publish.
    """
    import httpx

    url = f"{_API_BASE}/accounts/{account_id}/pages/projects/{project_name}/domains"
    try:
        resp = httpx.get(url, headers=_headers(api_token), timeout=timeout)
        data = resp.json()
        if not resp.is_success or not data.get("success"):
            logger.warning("Could not list Pages domains for %s: %s", project_name, resp.text[:300])
            return []
        return [d.get("name", "") for d in (data.get("result") or []) if d.get("name")]
    except Exception as exc:  # noqa: BLE001 — a listing failure must not block publishing
        logger.warning("Could not list Pages domains for %s: %s", project_name, exc)
        return []


def attach_domain(
    account_id: str, project_name: str, domain: str, api_token: str, *, timeout: int = 30
) -> tuple[bool, str]:
    """Attach ``domain`` to the Pages project. Returns ``(ok, detail)``."""
    import httpx

    url = f"{_API_BASE}/accounts/{account_id}/pages/projects/{project_name}/domains"
    try:
        resp = httpx.post(
            url,
            headers=_headers(api_token),
            json={"name": domain},
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001 — network faults are reported, not raised
        return False, str(exc)

    try:
        data = resp.json()
    except Exception:  # noqa: BLE001 — a non-JSON body (HTML error page) is still a failure
        return False, resp.text[:300]

    if resp.is_success and data.get("success"):
        return True, ""

    errors = data.get("errors") or []
    detail = "; ".join(str(e.get("message", e)) for e in errors) or resp.text[:300]
    return False, detail


def is_conflict(detail: str) -> bool:
    """True when ``detail`` means the name is already owned by someone else."""
    lowered = (detail or "").lower()
    return any(marker in lowered for marker in _CONFLICT_MARKERS)


def _candidates(slug: str, project_id: str) -> list[str]:
    """Preferred name first, then id-suffixed fallbacks for the collision case."""
    short = re.sub(r"[^a-z0-9]", "", (project_id or "").lower())[:6]
    names = [slug]
    if short:
        names.append(f"{slug}-{short}")
    return names


def publish_domain(
    app_name: str,
    project_id: str,
    base_domain: str,
    *,
    account_id: str,
    api_token: str,
    project_name: str | None = None,
) -> str | None:
    """Attach ``<slug>.<base_domain>`` to the app's Pages project.

    Returns the attached domain, or ``None`` when publishing has no base domain
    configured / credentials are missing / the name could not be attached (the
    caller then falls back to the project's ``*.pages.dev`` URL).

    Idempotent: re-publishing an app whose domain is already attached returns it
    without a second API call. A genuine collision (another app owns the name)
    retries once with a short project-id suffix.
    """
    base = (base_domain or "").strip().strip(".").lower()
    if not base or not account_id or not api_token:
        return None

    cf_project = project_name or slugify_name(app_name)
    slug = slugify_name(app_name)

    existing = list_domains(account_id, cf_project, api_token)
    for candidate in _candidates(slug, project_id):
        domain = f"{candidate}.{base}"
        if domain in existing:
            logger.info("Published domain already attached: %s → %s", domain, cf_project)
            return domain

        ok, detail = attach_domain(account_id, cf_project, domain, api_token)
        if ok:
            logger.info("Published domain attached: %s → %s", domain, cf_project)
            return domain
        if not is_conflict(detail):
            logger.warning("Could not attach published domain %s: %s", domain, detail)
            return None
        logger.info("Published domain %s is taken — trying the next candidate", domain)

    logger.warning("No free published domain available for %s under %s", app_name, base)
    return None

