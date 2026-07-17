"""Typed errors mapped from the API's RFC 7807 problem+json responses."""
from __future__ import annotations


class ForgefyError(Exception):
    """Base for every SDK error."""

    def __init__(
        self,
        message: str,
        *,
        status: int = 0,
        detail: str | None = None,
        problem_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.detail = detail
        self.problem_type = problem_type


class AuthenticationError(ForgefyError):
    """401 — missing, malformed, or revoked API key."""


class QuotaExceededError(ForgefyError):
    """402 — monthly token budget exhausted (free tier). `detail` includes the reset date."""


class NotFoundError(ForgefyError):
    """404 — the resource doesn't exist or belongs to another account."""


class ValidationError(ForgefyError):
    """422 — invalid request (empty/oversized transcript, bad extractor name, …)."""


class RateLimitError(ForgefyError):
    """429 — over 60 req/min for this key, or the server is at sync capacity."""


class ServerError(ForgefyError):
    """5xx — extraction failed on every model, queue unreachable, or other server fault."""


class APIConnectionError(ForgefyError):
    """The request never got an HTTP response (DNS, refused, timeout)."""


class JobFailedError(ForgefyError):
    """jobs.wait_for: the job finished with status "failed"."""

    def __init__(self, job_id: str, detail: str | None) -> None:
        suffix = f": {detail}" if detail else ""
        super().__init__(f"Extract job {job_id} failed{suffix}", detail=detail)
        self.job_id = job_id


class JobTimeoutError(ForgefyError):
    """jobs.wait_for: the deadline passed before the job finished."""

    def __init__(self, job_id: str, timeout: float) -> None:
        super().__init__(f"Timed out after {timeout:g}s waiting for extract job {job_id}")
        self.job_id = job_id


_STATUS_TO_ERROR: dict[int, type[ForgefyError]] = {
    401: AuthenticationError,
    402: QuotaExceededError,
    404: NotFoundError,
    422: ValidationError,
    429: RateLimitError,
}


def error_from_problem(status: int, problem: dict) -> ForgefyError:
    """Build the right error class from an RFC 7807 body."""
    message = problem.get("detail") or problem.get("title") or f"Forgefy API error {status}"
    cls = _STATUS_TO_ERROR.get(status, ServerError if status >= 500 else ForgefyError)
    return cls(
        message,
        status=status,
        detail=problem.get("detail"),
        problem_type=problem.get("type"),
    )
