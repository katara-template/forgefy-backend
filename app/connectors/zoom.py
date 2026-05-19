"""Zoom connector — stub (not yet implemented)."""
from __future__ import annotations


class ZoomConnector:
    """Stub Zoom connector.  Raises NotImplementedError on all methods."""

    def join(self, meeting_url: str, session_id: str) -> None:
        raise NotImplementedError(
            "Zoom connector is not yet implemented. "
            "Use Platform.MEET for live sessions."
        )

    def leave(self) -> None:
        raise NotImplementedError("Zoom connector is not yet implemented.")
