"""Sidecar configuration, read from the container environment."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SidecarConfig:
    session_id: str
    socket_path: str
    bot_binary: str

    deepgram_api_key: str
    deepgram_model: str
    language: str

    webhook_url: str
    webhook_secret: str

    @classmethod
    def from_env(cls) -> SidecarConfig:
        return cls(
            session_id=os.environ.get("FORGEFY_SESSION_ID", ""),
            socket_path=os.environ.get("FORGEFY_AUDIO_SOCKET", "/tmp/forgefy/audio.sock"),
            bot_binary=os.environ.get("FORGEFY_BOT_BINARY", "/opt/forgefy/bin/forgefy-zoom-bot"),
            deepgram_api_key=os.environ.get("DEEPGRAM_API_KEY", ""),
            deepgram_model=os.environ.get("DEEPGRAM_MODEL", "nova-3"),
            language=os.environ.get("DEEPGRAM_LANGUAGE", "en"),
            webhook_url=os.environ.get("FORGEFY_WEBHOOK_URL", ""),
            webhook_secret=os.environ.get("FORGEFY_WEBHOOK_SECRET", ""),
        )

    def validate(self) -> list[str]:
        """Return a list of problems; empty means good to start."""
        problems = []
        if not self.session_id:
            problems.append("FORGEFY_SESSION_ID is required")
        if not self.deepgram_api_key:
            problems.append("DEEPGRAM_API_KEY is required")
        if not self.webhook_url:
            problems.append("FORGEFY_WEBHOOK_URL is required")
        if not self.webhook_secret:
            problems.append("FORGEFY_WEBHOOK_SECRET is required")
        return problems
