"""Aggregated v1 API router."""
from fastapi import APIRouter

from app.api.v1 import auth, sessions

router = APIRouter()

router.include_router(auth.router, prefix="/auth", tags=["auth"])
router.include_router(sessions.router, prefix="/voxa/session", tags=["sessions"])

# Step 8 — Blueprints
# from app.api.v1 import blueprints
# router.include_router(blueprints.router, prefix="/voxa/blueprint", tags=["blueprints"])
