"""Connector factory — returns the right connector for a given platform."""
from __future__ import annotations

from app.db.models.enums import Platform


def get_connector(platform: Platform, user_id: str | None = None):
    """Return an instantiated connector for the given platform.

    Meet and Teams always use the Recall.ai cloud bot service. Zoom follows
    ZOOM_BOT_PROVIDER: "recall" (default) or "self_hosted", which runs our own
    Meeting SDK container instead — see zoom-bot/README.md.
    Physical sessions have no bot (browser streams audio directly).

    user_id identifies whose Zoom OAuth grant to mint per-meeting tokens from.
    Only the self-hosted connector uses it; Recall handles that itself.
    """
    if platform == Platform.PHYSICAL:
        raise NotImplementedError("Physical sessions have no bot connector.")

    from app.config import get_settings

    settings = get_settings()

    if platform == Platform.ZOOM and settings.ZOOM_BOT_PROVIDER == "self_hosted":
        from app.connectors.zoom_selfhosted import ZoomSelfHostedConnector
        return ZoomSelfHostedConnector.from_settings(settings, user_id=user_id)

    from app.connectors.recall import RecallConnector
    if not settings.RECALL_API_KEY:
        raise RuntimeError(
            "RECALL_API_KEY is not configured. Set it in the backend .env to enable live meeting bots."
        )
    if not settings.PUBLIC_API_BASE_URL:
        raise RuntimeError(
            "PUBLIC_API_BASE_URL is not configured. Set it to the publicly reachable URL of this API "
            "so Recall.ai can deliver webhook events (e.g. https://yourapi.ngrok.io)."
        )

    return RecallConnector(
        api_key=settings.RECALL_API_KEY,
        region=settings.RECALL_REGION,
        redis_url=settings.REDIS_URL,
        webhook_base_url=settings.PUBLIC_API_BASE_URL,
        avatar_path=settings.RECALL_BOT_AVATAR_PATH,
    )
