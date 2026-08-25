"""GitHub REST API client — create repos and build authenticated push URLs."""
from __future__ import annotations

import logging
import re
import secrets

import httpx

logger = logging.getLogger(__name__)

# GitHub rejects a duplicate name with 422 and this message in the errors array:
#   {"field": "name", "message": "name already exists on this account"}
# Matched on the message rather than the status code — a 422 also covers invalid
# names and other validation failures, which must not be retried under a new name.
_NAME_TAKEN = re.compile(r"name already exists", re.I)

# Numeric suffixes first (predictable, human-readable), then random ones so a
# user with app, app-2 … app-9 does not burn every attempt walking the sequence.
_MAX_NUMERIC_SUFFIX = 9
_MAX_NAME_ATTEMPTS = 12
# GitHub caps repository names at 100 characters; leave room for the suffix.
_MAX_NAME_LEN = 100


def _is_name_taken(error_text: str) -> bool:
    """True when a create_repo failure was purely a name collision."""
    return bool(_NAME_TAKEN.search(error_text))


def _suffixed_name(base: str, attempt: int) -> str:
    """Name for the nth attempt: 1 → base, 2..9 → base-N, then base-<random>."""
    if attempt == 1:
        return base[:_MAX_NAME_LEN]

    suffix = (
        f"-{attempt}"
        if attempt <= _MAX_NUMERIC_SUFFIX
        # token_hex(2) gives 4 hex chars — short enough to stay readable, wide
        # enough (65k) that a collision on top of a collision is negligible.
        else f"-{secrets.token_hex(2)}"
    )
    return base[: _MAX_NAME_LEN - len(suffix)] + suffix


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

    def is_token_valid(self) -> bool:
        """Return True if the token can authenticate against GitHub API."""
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.get(f"{self._BASE}/user", headers=self._headers)
                return resp.status_code == 200
        except Exception:
            return False

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

    def create_repo_unique(
        self,
        name: str,
        description: str = "",
        private: bool = True,
        max_attempts: int = _MAX_NAME_ATTEMPTS,
    ) -> dict:
        """Create a repo, suffixing the name until one is free.

        Tries `name`, then `name-2`, `name-3`, … and finally `name-<random>`.
        A taken name is an ordinary outcome — a user rebuilding "todo app" twice
        should get a second repo, not a failed build they have to work around by
        renaming their app.

        Creation is attempted rather than checked first: asking "does this exist?"
        and then creating leaves a race between the two calls, and GitHub already
        answers definitively by rejecting the POST.
        """
        last_error: RuntimeError | None = None

        for attempt in range(1, max_attempts + 1):
            candidate = _suffixed_name(name, attempt)
            try:
                return self.create_repo(candidate, description, private)
            except RuntimeError as exc:
                if not _is_name_taken(str(exc)):
                    raise  # a real failure (auth, quota, invalid name) — do not mask it
                last_error = exc
                logger.info("Repo name '%s' taken — trying the next suffix", candidate)

        raise RuntimeError(
            f"Could not find a free GitHub repository name after {max_attempts} "
            f"attempts starting from '{name}': {last_error}"
        )

    def get_push_url(self, repo_full_name: str) -> str:
        """Return a token-authenticated HTTPS URL for pushing to the repo."""
        return f"https://{self._token}@github.com/{repo_full_name}.git"

    @staticmethod
    def slugify(name: str) -> str:
        """Convert an arbitrary string to a valid GitHub repo slug."""
        slug = re.sub(r"[^a-zA-Z0-9._-]", "-", name).strip("-").lower()
        return slug[:100] or "forgefy-app"
