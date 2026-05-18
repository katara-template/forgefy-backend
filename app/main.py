"""FastAPI application factory."""
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from app.api.v1.router import router as api_v1_router
from app.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging
from app.db.session import build_engine, build_session_factory

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan — startup and shutdown hooks."""
    settings = get_settings()
    engine = build_engine(
        settings.DATABASE_URL, echo=(settings.APP_ENV == "development")
    )
    app.state.db_engine = engine
    app.state.db_session_factory = build_session_factory(engine)
    logger.info("Forgefy backend starting up (env=%s)", settings.APP_ENV)
    # Redis async client will be wired here in Step 5
    yield
    logger.info("Forgefy backend shutting down")
    await engine.dispose()


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

    # ── Rate limiting ─────────────────────────────────────────────────────────
    limiter = Limiter(key_func=get_remote_address, default_limits=["300/minute"])
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

    # WebSocket gateway registered in Step 5
    # from app.api.ws.voxa import router as ws_router
    # app.include_router(ws_router)

    # ── Ops endpoints ─────────────────────────────────────────────────────────
    @app.get("/health", tags=["ops"], summary="Liveness probe")
    async def health(request: Request) -> dict[str, str]:
        """Return service liveness status."""
        return {"status": "ok", "env": settings.APP_ENV}

    return app


app = create_app()
