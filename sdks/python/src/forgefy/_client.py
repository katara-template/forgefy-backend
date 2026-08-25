"""Forgefy API client."""
from __future__ import annotations

import os
import random
import time
import uuid
from typing import Any

import httpx

from .errors import (
    APIConnectionError,
    JobFailedError,
    JobTimeoutError,
    error_from_problem,
)

_USER_AGENT = "forgefy-sdk-py/0.1.0"


class Forgefy:
    """Client for the Forgefy Developer API.

    Usage::

        from forgefy import Forgefy

        client = Forgefy(api_key=os.environ["FORGEFY_API_KEY"],
                         base_url="https://your-forgefy-host")
        result = client.extract("We need Google login...", extractors=["features"])

    ``base_url`` falls back to ``$FORGEFY_API_URL``, then ``http://localhost:5000``.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        retry_delay: float = 0.5,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("Forgefy: api_key is required")
        self._api_key = api_key
        self._base_url = (
            base_url or os.environ.get("FORGEFY_API_URL") or "http://localhost:5000"
        ).rstrip("/")
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        # Injectable for tests (httpx.MockTransport); owned by the caller then.
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(timeout=timeout)
        self.jobs = Jobs(self)

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(
        self,
        transcript: str,
        *,
        extractors: list[str] | None = None,
        model_tier: str | None = None,
    ) -> dict:
        """Synchronously extract features / questions / conflicts / action items
        from a transcript (≤50k chars). For longer transcripts use ``jobs.create``.
        """
        return self._request(
            "POST", "/api/v1/extract",
            json_body=_extract_body(transcript, extractors, model_tier),
        )

    def usage(self) -> dict:
        """The key owner's tier, monthly token budget, consumption, and reset date."""
        return self._request("GET", "/api/v1/usage", retry_on_5xx=True)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Forgefy:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict | None = None,
        headers: dict[str, str] | None = None,
        retry_on_5xx: bool = False,
    ) -> Any:
        """One API call with retries.

        429 is always retried (with Retry-After honored). 5xx is retried only
        when the operation is safe to repeat — GETs and idempotent job
        creation; never the sync /extract POST, where a retry would bill
        tokens twice for work that may have partially run.
        """
        request_headers = {
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": _USER_AGENT,
            **(headers or {}),
        }

        attempt = 0
        while True:
            try:
                response = self._client.request(
                    method, f"{self._base_url}{path}",
                    json=json_body, headers=request_headers,
                )
            except httpx.HTTPError as exc:
                if attempt < self._max_retries:
                    time.sleep(self._backoff(attempt))
                    attempt += 1
                    continue
                raise APIConnectionError(
                    f"Could not reach the Forgefy API at {self._base_url}: {exc}"
                ) from exc

            if response.is_success:
                return None if response.status_code == 204 else response.json()

            retryable = response.status_code == 429 or (
                retry_on_5xx and response.status_code >= 500
            )
            if retryable and attempt < self._max_retries:
                retry_after = _parse_retry_after(response.headers.get("retry-after"))
                time.sleep(retry_after if retry_after is not None else self._backoff(attempt))
                attempt += 1
                continue

            try:
                problem = response.json()
            except ValueError:
                problem = {}
            raise error_from_problem(response.status_code, problem)

    def _backoff(self, attempt: int) -> float:
        # 1x, 2x, 4x… the base delay, with a little jitter.
        return self._retry_delay * (2**attempt) * (0.8 + random.random() * 0.4)


class Jobs:
    """Async extraction jobs (transcripts up to 200k chars)."""

    def __init__(self, client: Forgefy) -> None:
        self._client = client

    def create(
        self,
        transcript: str,
        *,
        extractors: list[str] | None = None,
        model_tier: str | None = None,
        webhook_url: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict:
        """Queue an async extraction job.

        An ``Idempotency-Key`` is generated automatically so network-level
        retries can never run the same job twice; pass your own to dedupe
        across processes (e.g. a stable import-batch id).
        """
        body = _extract_body(transcript, extractors, model_tier)
        if webhook_url is not None:
            body["webhook_url"] = webhook_url
        return self._client._request(
            "POST", "/api/v1/extract/jobs",
            json_body=body,
            headers={"Idempotency-Key": idempotency_key or str(uuid.uuid4())},
            retry_on_5xx=True,  # safe: the idempotency key dedupes on the server
        )

    def get(self, job_id: str) -> dict:
        """Current status of a job; ``result`` is set once status is "done"."""
        return self._client._request(
            "GET", f"/api/v1/extract/jobs/{job_id}", retry_on_5xx=True
        )

    def wait_for(
        self,
        job_id: str,
        *,
        poll_interval: float = 3.0,
        timeout: float = 600.0,
    ) -> dict:
        """Poll a job until it finishes.

        Returns the "done" status response, raises :class:`JobFailedError` on
        "failed" and :class:`JobTimeoutError` when ``timeout`` passes first.
        """
        deadline = time.monotonic() + timeout
        while True:
            job = self.get(job_id)
            if job.get("status") == "done":
                return job
            if job.get("status") == "failed":
                raise JobFailedError(job_id, job.get("error"))
            if time.monotonic() + poll_interval > deadline:
                raise JobTimeoutError(job_id, timeout)
            time.sleep(poll_interval)


def _extract_body(
    transcript: str, extractors: list[str] | None, model_tier: str | None
) -> dict:
    body: dict = {"transcript": transcript}
    if extractors is not None:
        body["extractors"] = extractors
    if model_tier is not None:
        body["model_tier"] = model_tier
    return body


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if seconds > 0 else None
