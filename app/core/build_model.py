"""Runtime build-model resolution: per-user choice > Firestore system/config > .env BUILD_MODEL."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

VALID_BUILD_MODELS = ("claude", "Qwen3", "gemini", "gpt")


async def get_user_build_model(db, user_id: str) -> str | None:
    """Return the user's own build-model override, if they've set one."""
    if not user_id:
        return None
    try:
        doc = await db.collection("users").document(user_id).get()
        if doc.exists:
            return (doc.to_dict() or {}).get("build_model") or None
    except Exception as exc:
        logger.warning("Could not read user build model for user=%s: %s", user_id, exc)
    return None


async def get_effective_build_model(db, settings, user_id: str | None = None) -> str:
    """Return the build model to use: the user's own override, else the Firestore
    system/config override, else the .env default."""
    if user_id:
        user_model = await get_user_build_model(db, user_id)
        if user_model:
            return user_model

    try:
        doc = await db.collection("system").document("config").get()
        if doc.exists:
            model = (doc.to_dict() or {}).get("build_model")
            if model:
                return model
    except Exception as exc:
        logger.warning("Could not read build model from Firestore: %s — falling back to .env", exc)
    return settings.BUILD_MODEL
