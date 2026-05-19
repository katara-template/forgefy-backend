"""Connector base protocol — all platform connectors must satisfy this interface."""
from __future__ import annotations

from typing import Protocol


class MeetingConnector(Protocol):
    """Defines the contract every platform connector must implement."""

    def join(self, meeting_url: str, session_id: str) -> None:
        """Join a meeting and begin streaming audio.

        Implementations should:
        1. Launch a headless browser / SDK client pointed at meeting_url
        2. Capture audio and publish chunks to the meeting.audio Celery queue
        3. Block or run until the meeting ends or ``leave`` is called
        """
        ...

    def leave(self) -> None:
        """Gracefully leave the meeting and release all resources."""
        ...
