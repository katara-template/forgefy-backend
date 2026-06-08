"""FastAPI application factory."""
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import redis.asyncio as aioredis
import ssl
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
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

    # Handle rediss:// URLs that may include a string `ssl_cert_reqs` query
    # param (e.g. `ssl_cert_reqs=CERT_NONE`). redis-py expects the
    # ssl_cert_reqs argument to be an `ssl` module constant (e.g. ssl.CERT_NONE).
    redis_url = settings.REDIS_URL
    ssl_cert_reqs = None
    ssl_context = None
    if isinstance(redis_url, str) and redis_url.startswith("rediss://"):
        # If the URL includes ssl_cert_reqs=..., remove it from the query
        # and map known string flags to the ssl module constants.
        parsed = urlparse(redis_url)
        qs = parse_qs(parsed.query)
        if "ssl_cert_reqs" in qs:
            val = qs.pop("ssl_cert_reqs")[0]
            mapping = {"CERT_NONE": ssl.CERT_NONE, "CERT_OPTIONAL": ssl.CERT_OPTIONAL, "CERT_REQUIRED": ssl.CERT_REQUIRED}
            ssl_cert_reqs = mapping.get(val, None)
            # Rebuild URL without ssl_cert_reqs
            new_query = urlencode(qs, doseq=True)
            parsed = parsed._replace(query=new_query)
            redis_url = urlunparse(parsed)
        # Build an SSLContext for the connection. Enforce TLSv1.2+ when available.
        ctx = ssl.create_default_context()
        # Map cert requirements into the context
        if ssl_cert_reqs is not None:
            if ssl_cert_reqs == ssl.CERT_NONE:
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
            else:
                ctx.verify_mode = ssl_cert_reqs
        # Prefer TLSv1.2+ where supported
        try:
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        except Exception:
            # older Python may not support TLSVersion
            pass
        ssl_context = ctx

    # Simplified: pass an SSLContext via `ssl_context` for TLS connections.
    # This is compatible with redis-py 4.x and 5.x.
    if isinstance(redis_url, str) and redis_url.startswith("rediss://"):
        # Create a permissive context when connecting to managed providers
        # that don't provide CA info in the runtime. This mirrors previous
        # behaviour where `ssl_cert_reqs=CERT_NONE` was appended.
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        redis_client: aioredis.Redis = aioredis.from_url(
            redis_url, 
            decode_responses=True, 
            ssl_cert_reqs="none",
            ssl_context=ctx
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
    @app.get("/health", tags=["ops"], summary="Liveness probe")
    async def health(request: Request) -> dict[str, str]:
        """Return service liveness status."""
        return {"status": "ok", "env": settings.APP_ENV}

    return app


app = create_app()
