"""Connector factory — returns the right connector for a given platform."""
from __future__ import annotations

from app.db.models.enums import Platform


def get_connector(platform: Platform):
    """Return an instantiated connector for platform.

    Raises NotImplementedError for platforms without a live implementation.
    """
    if platform == Platform.MEET:
        from app.connectors.meet import MeetConnector
        return MeetConnector()
    if platform == Platform.ZOOM:
        from app.connectors.zoom import ZoomConnector
        return ZoomConnector()
    if platform == Platform.TEAMS:
        from app.connectors.teams import TeamsConnector
        return TeamsConnector()
    # PHYSICAL sessions have no bot
    raise NotImplementedError(f"No connector available for platform '{platform.value}'")
