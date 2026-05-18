"""Custom exception hierarchy and RFC 7807 error handlers."""
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

_BASE_TYPE = "https://forgefy.dev/errors"


# ── Exception classes ─────────────────────────────────────────────────────────


class ForgefyError(Exception):
    """Base exception — all domain errors extend this."""

    status_code: int = 500
    title: str = "Internal Server Error"

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


class NotFoundError(ForgefyError):
    status_code = 404
    title = "Not Found"


class UnauthorizedError(ForgefyError):
    status_code = 401
    title = "Unauthorized"


class ForbiddenError(ForgefyError):
    status_code = 403
    title = "Forbidden"


class ConflictError(ForgefyError):
    status_code = 409
    title = "Conflict"


class ValidationError(ForgefyError):
    status_code = 422
    title = "Unprocessable Entity"


class InvalidStateTransition(ForgefyError):
    """Raised when a state machine transition is not permitted."""

    status_code = 422
    title = "Invalid State Transition"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _problem(exc: ForgefyError, request: Request) -> JSONResponse:
    """Build an RFC 7807 problem+json response."""
    slug = exc.__class__.__name__.lower()
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "type": f"{_BASE_TYPE}/{slug}",
            "title": exc.title,
            "status": exc.status_code,
            "detail": exc.detail,
            "instance": str(request.url),
        },
        headers={"Content-Type": "application/problem+json"},
    )


# ── Registration ──────────────────────────────────────────────────────────────


def register_exception_handlers(app: FastAPI) -> None:
    """Attach domain and catch-all exception handlers to the FastAPI app."""

    @app.exception_handler(ForgefyError)
    async def forgefy_handler(request: Request, exc: ForgefyError) -> JSONResponse:
        if exc.status_code >= 500:
            logger.error("Server error: %s", exc.detail, exc_info=exc)
        else:
            logger.warning("Client error %d: %s", exc.status_code, exc.detail)
        return _problem(exc, request)

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception on %s %s", request.method, request.url)
        return JSONResponse(
            status_code=500,
            content={
                "type": f"{_BASE_TYPE}/internalservererror",
                "title": "Internal Server Error",
                "status": 500,
                "detail": "An unexpected error occurred.",
                "instance": str(request.url),
            },
            headers={"Content-Type": "application/problem+json"},
        )
