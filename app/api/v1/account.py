"""Self-service account preferences, data export, and deletion (current user only)."""
from __future__ import annotations

import logging
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.build_model import VALID_BUILD_MODELS, get_effective_build_model, get_user_build_model
from app.core.exceptions import ValidationError
from app.deps import CurrentUser, DBSession

logger = logging.getLogger(__name__)
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


# ---------------------------------------------------------------------------
# Data export + account deletion (CCPA/GDPR: right to know, right to delete)
# ---------------------------------------------------------------------------

# Never include stored credentials in an export — even the user's own.
_SECRET_KEY_MARKERS = ("token", "password", "secret", "credential", "api_key")


def _redact(doc: dict[str, Any]) -> dict[str, Any]:
    """Drop any field whose name suggests it holds a credential."""
    return {
        k: v for k, v in doc.items()
        if not any(marker in k.lower() for marker in _SECRET_KEY_MARKERS)
    }


@router.get("/export")
async def export_my_data(db: DBSession, user: CurrentUser) -> dict:
    """Return everything Forgefy stores about the current user as one JSON blob.

    Covers the profile, usage records, meeting sessions (with transcript
    text), blueprints, projects, and project chats. Stored credentials
    (OAuth tokens, password hash) are redacted.
    """
    uid = str(user.id)
    user_doc = await db.collection("users").document(uid).get()
    profile_raw = user_doc.to_dict() or {}

    usage_docs = await db.collection("users").document(uid).collection("usage").get()

    session_docs = await db.collection("sessions").where("user_id", "==", uid).get()
    sessions: list[dict[str, Any]] = []
    blueprints: list[dict[str, Any]] = []
    for sdoc in session_docs:
        sdata = _redact(sdoc.to_dict() or {})
        sdata["id"] = sdoc.id

        segment_docs = (
            await db.collection("sessions").document(sdoc.id).collection("events")
            .where("event_type", "==", "transcript.segment")
            .get()
        )
        segments = sorted(
            (d.to_dict() for d in segment_docs), key=lambda s: s.get("timestamp", "")
        )
        sdata["transcript"] = " ".join(
            s.get("payload", {}).get("text", "") for s in segments
        ).strip()
        sessions.append(sdata)

        bp_docs = await db.collection("blueprints").where("session_id", "==", sdoc.id).get()
        for bdoc in bp_docs:
            bdata = _redact(bdoc.to_dict() or {})
            bdata["id"] = bdoc.id
            blueprints.append(bdata)

    project_docs = await db.collection("projects").where("owner_id", "==", uid).get()
    projects: list[dict[str, Any]] = []
    project_chats: list[dict[str, Any]] = []
    for pdoc in project_docs:
        pdata = _redact(pdoc.to_dict() or {})
        pdata["id"] = pdoc.id
        projects.append(pdata)
        chat_doc = await db.collection("project_chats").document(pdoc.id).get()
        if chat_doc.exists:
            cdata = _redact(chat_doc.to_dict() or {})
            cdata["project_id"] = pdoc.id
            project_chats.append(cdata)

    logger.info("Data export generated for user=%s", uid)
    return {
        "exported_at": datetime.now(UTC),
        "profile": _redact(profile_raw),
        "connected_integrations": {
            "github": bool(profile_raw.get("github_access_token")),
            "supabase": bool(profile_raw.get("supabase_access_token")),
            "firebase": bool(profile_raw.get("firebase_access_token")),
        },
        "usage": [d.to_dict() | {"period": d.id} for d in usage_docs],
        "sessions": sessions,
        "blueprints": blueprints,
        "projects": projects,
        "project_chats": project_chats,
    }


class DeleteAccountRequest(BaseModel):
    confirm: str  # must be the literal string "DELETE"


@router.delete("")
async def delete_my_account(
    body: DeleteAccountRequest,
    db: DBSession,
    user: CurrentUser,
) -> dict:
    """Permanently delete the current user's account and all associated data.

    Removes sessions (with transcripts), blueprints, projects, project chats,
    usage records, the user document, and the Firebase Auth credential.
    GitHub repositories created for the user are NOT deleted — the code
    belongs to the user and lives in their own GitHub account.
    """
    if body.confirm != "DELETE":
        raise ValidationError('Pass {"confirm": "DELETE"} to permanently delete your account.')

    uid = str(user.id)

    # Refuse while a build is running — the worker would resurrect project docs.
    project_docs = await db.collection("projects").where("owner_id", "==", uid).get()
    if any((p.to_dict() or {}).get("is_updating") for p in project_docs):
        raise ValidationError("A build is in progress — stop it before deleting your account.")

    # Sessions + their events, and blueprints derived from them
    session_docs = await db.collection("sessions").where("user_id", "==", uid).get()
    for sdoc in session_docs:
        events_ref = db.collection("sessions").document(sdoc.id).collection("events")
        for edoc in await events_ref.get():
            await events_ref.document(edoc.id).delete()
        bp_docs = await db.collection("blueprints").where("session_id", "==", sdoc.id).get()
        for bdoc in bp_docs:
            await db.collection("blueprints").document(bdoc.id).delete()
        await db.collection("sessions").document(sdoc.id).delete()

    # Projects + chats (GitHub repos intentionally left untouched)
    for pdoc in project_docs:
        with suppress(Exception):
            await db.collection("project_chats").document(pdoc.id).delete()
        await db.collection("projects").document(pdoc.id).delete()

    # Usage subcollection, then the user doc itself
    usage_ref = db.collection("users").document(uid).collection("usage")
    for udoc in await usage_ref.get():
        await usage_ref.document(udoc.id).delete()

    user_doc = await db.collection("users").document(uid).get()
    firebase_uid = (user_doc.to_dict() or {}).get("firebase_uid") or uid
    await db.collection("users").document(uid).delete()

    # Finally the Firebase Auth credential — non-fatal if already gone.
    try:
        import anyio
        import firebase_admin.auth

        await anyio.to_thread.run_sync(firebase_admin.auth.delete_user, firebase_uid)
    except Exception as exc:
        logger.warning("Could not delete Firebase Auth credential user=%s: %s", uid, exc)

    logger.info(
        "Account deleted user=%s (sessions=%d, projects=%d)",
        uid, len(session_docs), len(project_docs),
    )
    return {"deleted": True}
