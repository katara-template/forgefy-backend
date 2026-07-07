"""Self-service account preferences (current user only)."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.build_model import VALID_BUILD_MODELS, get_effective_build_model, get_user_build_model
from app.core.exceptions import ValidationError
from app.deps import CurrentUser, DBSession

router = APIRouter()


class BuildModelResponse(BaseModel):
    model: str
    is_custom: bool


class SetBuildModelRequest(BaseModel):
    model: str | None = None


@router.get("/build-model", response_model=BuildModelResponse)
async def get_my_build_model(db: DBSession, user: CurrentUser) -> BuildModelResponse:
    """Return the model this user's builds currently run on, and whether it's their own override."""
    from app.config import get_settings

    settings = get_settings()
    own_model = await get_user_build_model(db, str(user.id))
    effective = own_model or await get_effective_build_model(db, settings, user_id=str(user.id))
    return BuildModelResponse(model=effective, is_custom=bool(own_model))


@router.patch("/build-model", response_model=BuildModelResponse)
async def set_my_build_model(
    body: SetBuildModelRequest,
    db: DBSession,
    user: CurrentUser,
) -> BuildModelResponse:
    """Set (or clear, with model=null) this user's own build-model override."""
    from app.config import get_settings

    settings = get_settings()

    if body.model is not None and body.model not in VALID_BUILD_MODELS:
        raise ValidationError(f"Invalid model '{body.model}'. Choose from: {', '.join(VALID_BUILD_MODELS)}")

    await db.collection("users").document(str(user.id)).set(
        {"build_model": body.model}, merge=True
    )
    effective = body.model or await get_effective_build_model(db, settings, user_id=str(user.id))
    return BuildModelResponse(model=effective, is_custom=bool(body.model))
