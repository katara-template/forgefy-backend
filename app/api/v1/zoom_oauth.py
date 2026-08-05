"""Zoom account linking endpoints.

A host installs the Forgefy app against their Zoom account once; from then on
we can mint the per-meeting OBF and local-recording tokens the self-hosted bot
needs to join their meetings. See app/integrations/zoom_oauth.py for why this
became mandatory.

Mirrors the GitHub and Supabase linking flows in auth.py: `authorize` hands the
frontend a URL, `callback` completes the exchange and redirects back.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Query
from fastapi.responses import RedirectResponse

from app.deps import CurrentUser, DBSession, SettingsDep
from app.integrations import zoom_oauth

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/authorize")
async def zoom_authorize(settings: SettingsDep, user: CurrentUser) -> dict:
    """Return the Zoom consent URL for the current user."""
    # Reuses auth.py's signed state so the callback can prove which user began
    # the flow without a server-side session.
    from app.api.v1.auth import _make_state

    if not (settings.ZOOM_OAUTH_CLIENT_ID or settings.ZOOM_SDK_CLIENT_ID):
        return {"error": "Zoom OAuth not configured"}
    if not settings.PUBLIC_API_BASE_URL:
        return {"error": "PUBLIC_API_BASE_URL must be set so Zoom can reach the callback"}

    state = _make_state(str(user.id), settings.SECRET_KEY)
    return {"url": zoom_oauth.authorize_url(state, settings)}


@router.get("/callback")
async def zoom_callback(
    code: str = Query(default=""),
    state: str = Query(default=""),
    error: str = Query(default=""),
    db: DBSession = None,
    settings: SettingsDep = None,
) -> RedirectResponse:
    """Exchange the authorization code for tokens and store them."""
    from app.api.v1.auth import _verify_state

    dashboard = f"{settings.FRONTEND_URL}/dashboard"

    # The host can decline on Zoom's consent screen.
    if error:
        logger.info("Zoom OAuth declined: %s", error)
        return RedirectResponse(f"{dashboard}?zoom_error=declined")

    if not (code and state):
        return RedirectResponse(f"{dashboard}?zoom_error=missing_code")

    user_id = _verify_state(state, settings.SECRET_KEY)
    if not user_id:
        return RedirectResponse(f"{dashboard}?zoom_error=invalid_state")

    try:
        tokens = await zoom_oauth.exchange_code(code, settings)
        await zoom_oauth.store_tokens(db, user_id, tokens, settings)
        logger.info("Zoom account linked user=%s", user_id)
    except zoom_oauth.ZoomAuthError as exc:
        logger.error("Zoom OAuth exchange failed user=%s: %s", user_id, exc)
        return RedirectResponse(f"{dashboard}?zoom_error=exchange_failed")
    except Exception as exc:
        logger.error("Zoom OAuth callback error user=%s: %s", user_id, exc)
        return RedirectResponse(f"{dashboard}?zoom_error=server_error")

    return RedirectResponse(f"{dashboard}?zoom=connected")


@router.get("/status")
async def zoom_status(db: DBSession, user: CurrentUser) -> dict:
    """Whether the current user has a linked Zoom account, and which one."""
    return await zoom_oauth.is_linked(db, str(user.id))


@router.post("/disconnect")
async def zoom_disconnect(db: DBSession, user: CurrentUser) -> dict:
    """Forget the user's Zoom tokens.

    Their bot can still join meetings hosted on our own Zoom account, but not
    meetings hosted elsewhere.
    """
    await zoom_oauth.unlink(db, str(user.id))
    logger.info("Zoom account unlinked user=%s", user.id)
    return {"linked": False}
