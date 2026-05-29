"""Sync Redis publisher for build/update log events.

Workers call make_log_publisher() to get a callable that streams
events to the channel  build:{project_id}:logs  via Redis pub/sub.
The WebSocket endpoint /ws/projects/{project_id}/logs subscribes to
this channel and forwards the events to the browser.
"""
from __future__ import annotations

import json
import logging
from typing import Callable

logger = logging.getLogger(__name__)

_TOOL_LABELS: dict[str, str] = {
    "read_file": "Reading",
    "write_file": "Writing",
    "list_files": "Listing",
    "create_directory": "Creating directory",
    "delete_file": "Deleting",
    "move_file": "Moving",
    "generate_image": "Generating image",
    "generate_video": "Generating video",
}


def tool_message(tool_name: str, inputs: dict) -> str:
    """Return a human-readable label for a tool call."""
    label = _TOOL_LABELS.get(tool_name, tool_name)
    path = inputs.get("path") or inputs.get("source") or inputs.get("filename") or ""
    return f"{label} `{path}`" if path else label


def make_log_publisher(project_id: str, redis_url: str) -> Callable[[str, str], None]:
    """Return a sync callable that publishes build log events.

    Event JSON:  { "type": "<event_type>", "message": "<text>" }
    Event types: started | info | thinking | tool | warning | error | done
    """
    channel = f"build:{project_id}:logs"
    _client: object = None

    def _get_client():
        nonlocal _client
        if _client is None:
            import redis
            _client = redis.from_url(redis_url, decode_responses=True)
        return _client

    def publish(event_type: str, message: str) -> None:
        try:
            _get_client().publish(channel, json.dumps({"type": event_type, "message": message}))
        except Exception as exc:
            logger.debug("build_logger publish error: %s", exc)

    return publish
