"""GitHub REST API client — create repos and build authenticated push URLs."""
from __future__ import annotations

import logging
import re

import httpx

logger = logging.getLogger(__name__)


class GitHubClient:
    """Thin wrapper around GitHub REST API v3."""

    _BASE = "https://api.github.com"

    def __init__(self, token: str) -> None:
        self._token = token
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def create_repo(self, name: str, description: str = "", private: bool = True) -> dict:
        """Create a new GitHub repo; return the full API response dict."""
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"{self._BASE}/user/repos",
                headers=self._headers,
                json={
                    "name": name,
                    "description": description,
                    "private": private,
                    "auto_init": False,
                },
            )
            if resp.status_code not in (200, 201):
                raise RuntimeError(
                    f"GitHub create_repo failed {resp.status_code}: {resp.text}"
                )
            data = resp.json()
            logger.info("Created GitHub repo: %s", data["html_url"])
            return data

    def get_push_url(self, repo_full_name: str) -> str:
        """Return a token-authenticated HTTPS URL for pushing to the repo."""
        return f"https://{self._token}@github.com/{repo_full_name}.git"

    @staticmethod
    def slugify(name: str) -> str:
        """Convert an arbitrary string to a valid GitHub repo slug."""
        slug = re.sub(r"[^a-zA-Z0-9._-]", "-", name).strip("-").lower()
        return slug[:100] or "forgefy-app"
