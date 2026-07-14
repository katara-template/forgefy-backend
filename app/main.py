"""FastAPI application factory."""
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import redis.asyncio as aioredis
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.api.v1.router import router as api_v1_router
from app.api.ws.build_logs import router as ws_build_logs_router
from app.api.ws.projects import router as ws_projects_router
from app.api.ws.sessions import router as ws_sessions_router
from app.api.ws.user_events import router as ws_user_events_router
from app.api.ws.voxa import router as ws_router
from app.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging
from app.core.rate_limit import limiter
from app.db.firebase import init_firebase
from app.deps import DBSession, RedisDep

configure_logging()
logger = logging.getLogger(__name__)

_settings = get_settings()
if _settings.SENTRY_DSN:
    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration

        sentry_sdk.init(
            dsn=_settings.SENTRY_DSN,
            environment=_settings.APP_ENV,
            integrations=[
                StarletteIntegration(transaction_style="endpoint"),
                FastApiIntegration(transaction_style="endpoint"),
            ],
            traces_sample_rate=1.0 if _settings.APP_ENV != "production" else 0.2,
            send_default_pii=False,
        )
    except ImportError:
        logger.warning("sentry-sdk not installed — error tracking disabled")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan — startup and shutdown hooks."""
    settings = get_settings()

    firestore_client = init_firebase()
    app.state.firestore = firestore_client

    # Absorb the first-call gRPC channel setup (~3s measured) at startup so it
    # doesn't land on the first user request after a deploy or scale-up.
    with suppress(Exception):
        await firestore_client.collection("system").document("healthcheck").get()

    # Strip any ssl_cert_reqs query param from rediss:// URLs — redis-py
    # handles TLS via the scheme; the query param causes issues on newer versions.
    redis_url = settings.REDIS_URL
    if isinstance(redis_url, str) and redis_url.startswith("rediss://"):
        parsed = urlparse(redis_url)
        qs = parse_qs(parsed.query)
        qs.pop("ssl_cert_reqs", None)
        redis_url = urlunparse(parsed._replace(query=urlencode(qs, doseq=True)))
        redis_client: aioredis.Redis = aioredis.from_url(
            redis_url, decode_responses=True, ssl_cert_reqs="none"
        )
    else:
        redis_client: aioredis.Redis = aioredis.from_url(redis_url, decode_responses=True)
    app.state.redis = redis_client

    logger.info("Forgefy backend starting up (env=%s)", settings.APP_ENV)
    yield
    logger.info("Forgefy backend shutting down")
    await redis_client.aclose()
    firestore_client.close()


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Forgefy Backend",
        description="Meeting Mode — AI orchestration backend",
        version="0.1.0",
        docs_url="/docs" if settings.APP_ENV != "production" else None,
        redoc_url="/redoc" if settings.APP_ENV != "production" else None,
        lifespan=lifespan,
    )

    # ── Request timing ─────────────────────────────────────────────────────────
    # Every response carries a Server-Timing header (visible per-request in the
    # browser devtools Network tab) so slow endpoints can be spotted without
    # extra tooling; anything over 1s is also logged server-side.
    @app.middleware("http")
    async def server_timing(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        response.headers["Server-Timing"] = f"app;dur={duration_ms:.1f}"
        if duration_ms > 1000:
            logger.warning(
                "slow request: %s %s took %.0f ms",
                request.method, request.url.path, duration_ms,
            )
        return response

    # ── Rate limiting (shared singleton from core.rate_limit) ─────────────────
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]
    app.add_middleware(SlowAPIMiddleware)

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Exception handlers ────────────────────────────────────────────────────
    register_exception_handlers(app)

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(api_v1_router, prefix="/api/v1")
    app.include_router(ws_router)
    app.include_router(ws_sessions_router)
    app.include_router(ws_projects_router)
    app.include_router(ws_build_logs_router)
    app.include_router(ws_user_events_router)
    # default route for sanity check    
    @app.get("/", tags=["ops"], summary="Sanity check")
    async def root() -> dict[str, str]:
        """Basic endpoint to verify the service is running."""
        return {"message": "Forgefy backend is up and running!"}
    
    # ── Ops endpoints ─────────────────────────────────────────────────────────
    # Orchestrators probe /health every few seconds; cache the dependency
    # checks briefly so probes don't consume Firestore read quota all day.
    # Kept on app.state so tests can reset it between cases.
    app.state.health_cache = {"at": 0.0, "checks": None}
    health_cache_ttl = 25.0

    @app.get("/health", tags=["ops"], summary="Liveness + dependency readiness probe")
    async def health(db: DBSession, redis: RedisDep) -> Response:
        """Return service status, including Redis and Firestore reachability.

        Returns 503 if either dependency is unreachable, so an orchestrator
        (Docker/Render health checks) can detect and recycle a container that's
        running but can't actually serve requests.
        """
        cache = app.state.health_cache
        checks: dict[str, str] | None = None
        if time.monotonic() - cache["at"] < health_cache_ttl:
            checks = cache["checks"]

        if checks is None:
            checks = {}
            try:
                await redis.ping()
                checks["redis"] = "ok"
            except Exception as exc:
                checks["redis"] = f"error: {exc}"

            try:
                await db.collection("system").document("healthcheck").get()
                checks["firestore"] = "ok"
            except Exception as exc:
                checks["firestore"] = f"error: {exc}"

            cache["at"] = time.monotonic()
            cache["checks"] = checks

        healthy = all(v == "ok" for v in checks.values())
        return JSONResponse(
            status_code=200 if healthy else 503,
            content={"status": "ok" if healthy else "degraded", "env": settings.APP_ENV, "checks": checks},
        )

    return app


app = create_app()
