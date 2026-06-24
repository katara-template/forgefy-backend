"""Runtime build-model resolution: Firestore system/config overrides .env BUILD_MODEL."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


async def get_effective_build_model(db, settings) -> str:
    """Return the build model to use, preferring the Firestore system/config value."""
    try:
        doc = await db.collection("system").document("config").get()
        if doc.exists:
            model = (doc.to_dict() or {}).get("build_model")
            if model:
                return model
    except Exception as exc:
        logger.warning("Could not read build model from Firestore: %s — falling back to .env", exc)
    return settings.BUILD_MODEL
