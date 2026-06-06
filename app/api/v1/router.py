"""Aggregated v1 API router."""
from fastapi import APIRouter

from app.api.v1 import auth, billing, blueprints, projects, sessions, webhooks

router = APIRouter()

router.include_router(auth.router, prefix="/auth", tags=["auth"])
router.include_router(sessions.router, prefix="/voxa/session", tags=["sessions"])
router.include_router(blueprints.router, prefix="/voxa/blueprint", tags=["blueprints"])
router.include_router(projects.router, prefix="/projects", tags=["projects"])
router.include_router(billing.router, prefix="/billing", tags=["billing"])
router.include_router(webhooks.router, prefix="/webhooks", tags=["webhooks"])
