"""Admin endpoints — system-wide configuration (build model, etc.)."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.exceptions import ValidationError
from app.deps import CurrentUser, DBSession

router = APIRouter()

VALID_BUILD_MODELS = ("claude", "Qwen3", "gemini", "gpt")


class BuildModelResponse(BaseModel):
    model: str


class SetBuildModelRequest(BaseModel):
    model: str


@router.get("/build-model", response_model=BuildModelResponse)
async def get_build_model_setting(db: DBSession, user: CurrentUser) -> BuildModelResponse:
    """Return the active build model (Firestore override takes precedence over .env)."""
    from app.config import get_settings

    settings = get_settings()
    doc = await db.collection("system").document("config").get()
    model = (doc.to_dict() or {}).get("build_model") if doc.exists else None
    return BuildModelResponse(model=model or settings.BUILD_MODEL)


@router.patch("/build-model", response_model=BuildModelResponse)
async def set_build_model_setting(
    body: SetBuildModelRequest,
    db: DBSession,
    user: CurrentUser,
) -> BuildModelResponse:
    """Persist a new build model to Firestore (takes effect on the next build/update)."""
    if body.model not in VALID_BUILD_MODELS:
        raise ValidationError(f"Invalid model '{body.model}'. Choose from: {', '.join(VALID_BUILD_MODELS)}")

    await db.collection("system").document("config").set(
        {"build_model": body.model}, merge=True
    )
    return BuildModelResponse(model=body.model)
