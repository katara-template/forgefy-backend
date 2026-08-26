"""Custom exception hierarchy and RFC 7807 error handlers."""
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

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


class RateLimitedError(ForgefyError):
    status_code = 429
    title = "Too Many Requests"


class QuotaExceededError(ForgefyError):
    """Raised when a user has exhausted their tier's monthly token budget."""

    status_code = 402
    title = "Quota Exceeded"


class InvalidStateTransition(ForgefyError):
    """Raised when a state machine transition is not permitted."""

    status_code = 422
    title = "Invalid State Transition"


class ExternalServiceError(ForgefyError):
    """Raised when an upstream provider (Neon, Supabase, GitHub, …) rejects or
    fails a request Forgefy made on the user's behalf."""

    status_code = 502
    title = "Bad Gateway"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _cors_headers(request: Request) -> dict[str, str]:
    """Return CORS headers for the request origin by walking the middleware stack."""
    origin = request.headers.get("origin")
    if not origin:
        return {}
    # request.app is the bare FastAPI instance; the middleware instances live
    # on the stack built at first request (app.middleware_stack → .app → …).
    app = getattr(request.app, "middleware_stack", None)
    while app is not None:
        if isinstance(app, CORSMiddleware):
            if app.allow_all_origins or origin in app.allow_origins:
                allow = origin if not app.allow_all_origins else "*"
                headers = {
                    "Access-Control-Allow-Origin": allow,
                    "Vary": "Origin",
                }
                # Mirror the middleware rather than hardcoding "true" — under a
                # wildcard allowlist credentials are deliberately off, and error
                # responses must not advertise what normal responses refuse.
                if app.allow_credentials:
                    headers["Access-Control-Allow-Credentials"] = "true"
                return headers
            return {}
        app = getattr(app, "app", None)
    return {}


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
        headers={"Content-Type": "application/problem+json", **_cors_headers(request)},
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
            headers={"Content-Type": "application/problem+json", **_cors_headers(request)},
        )
