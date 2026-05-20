"""Connector factory — returns the right connector for a given platform."""
from __future__ import annotations

from app.db.models.enums import Platform


def get_connector(platform: Platform):
    """Return an instantiated connector for the given platform.

    Meet, Zoom, and Teams use the Recall.ai cloud bot service.
    Physical sessions have no bot (browser streams audio directly).
    """
    if platform == Platform.PHYSICAL:
        raise NotImplementedError("Physical sessions have no bot connector.")

    from app.config import get_settings
    from app.connectors.recall import RecallConnector

    settings = get_settings()
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
    )
