"""Blueprint endpoints: get and approve.

Firestore collections used:
  blueprints/{blueprint_id}  — session_id, json_output, approved, created_at
  sessions/{session_id}      — user_id, status, ...
"""
import logging
import uuid
from datetime import datetime

from fastapi import APIRouter
from google.cloud.firestore import AsyncClient

from pydantic import BaseModel

from app.core.exceptions import ForbiddenError, NotFoundError, ValidationError
from app.db.models.blueprint import Blueprint
from app.db.models.enums import SessionStatus
from app.deps import CurrentUser, DBSession
from app.modules.voxa.state_machine import MeetingStateMachine
from app.schemas.blueprint import BlueprintOut

logger = logging.getLogger(__name__)
router = APIRouter()


def _doc_to_blueprint(doc) -> Blueprint:
    data = doc.to_dict()
    return Blueprint(
        id=uuid.UUID(doc.id),
        session_id=uuid.UUID(data["session_id"]),
        json_output=data.get("json_output"),
        approved=data.get("approved", False),
        created_at=data["created_at"],
        repo_url=data.get("repo_url"),
        repo_name=data.get("repo_name"),
        build_summary=data.get("build_summary"),
        build_status=data.get("build_status"),
        artifact_url=data.get("artifact_url"),
        preview_url=data.get("preview_url"),
    )


async def _get_owned_blueprint(
    blueprint_id: uuid.UUID,
    user_id: uuid.UUID,
    db: AsyncClient,
) -> Blueprint:
    """Load blueprint; raise 404/403 if missing or session is unowned."""
    doc = await db.collection("blueprints").document(str(blueprint_id)).get()
    if not doc.exists:
        raise NotFoundError(f"Blueprint {blueprint_id} not found")

    blueprint = _doc_to_blueprint(doc)

    sess_doc = await db.collection("sessions").document(str(blueprint.session_id)).get()
    if not sess_doc.exists or uuid.UUID(sess_doc.to_dict()["user_id"]) != user_id:
        raise ForbiddenError("Access denied")

    return blueprint


@router.get("/session/{session_id}", response_model=BlueprintOut)
async def get_blueprint_by_session(
    session_id: uuid.UUID,
    db: DBSession,
    user: CurrentUser,
) -> BlueprintOut:
    """Return the latest blueprint for a session (ownership-checked)."""
    sess_doc = await db.collection("sessions").document(str(session_id)).get()
    if not sess_doc.exists or uuid.UUID(sess_doc.to_dict()["user_id"]) != user.id:
        raise NotFoundError(f"Session {session_id} not found")

    docs = (
        await db.collection("blueprints")
        .where("session_id", "==", str(session_id))
        .get()
    )
    if not docs:
        raise NotFoundError(f"No blueprint found for session {session_id}")

    latest = max(docs, key=lambda d: d.to_dict().get("created_at", ""))
    return BlueprintOut.model_validate(_doc_to_blueprint(latest))


@router.get("/{blueprint_id}", response_model=BlueprintOut)
async def get_blueprint(
    blueprint_id: uuid.UUID,
    db: DBSession,
    user: CurrentUser,
) -> BlueprintOut:
    """Retrieve a blueprint by ID."""
    blueprint = await _get_owned_blueprint(blueprint_id, user.id, db)
    return BlueprintOut.model_validate(blueprint)


class BlueprintPatchRequest(BaseModel):
    app_name: str | None = None
    template_key: str | None = None
    features: list[dict] | None = None


@router.patch("/{blueprint_id}", response_model=BlueprintOut)
async def patch_blueprint(
    blueprint_id: uuid.UUID,
    body: BlueprintPatchRequest,
    db: DBSession,
    user: CurrentUser,
) -> BlueprintOut:
    """Edit app_name, template, or features before approving a blueprint."""
    blueprint = await _get_owned_blueprint(blueprint_id, user.id, db)

    if blueprint.approved:
        raise ValidationError("Cannot edit an already-approved blueprint")

    current: dict = dict(blueprint.json_output or {})

    if body.app_name is not None:
        current["app_name"] = body.app_name
    if body.template_key is not None:
        current["template"] = body.template_key
    if body.features is not None:
        current["features"] = body.features

    await db.collection("blueprints").document(str(blueprint_id)).update({"json_output": current})
    blueprint.json_output = current
    return BlueprintOut.model_validate(blueprint)


@router.post("/{blueprint_id}/approve", response_model=BlueprintOut)
async def approve_blueprint(
    blueprint_id: uuid.UUID,
    db: DBSession,
    user: CurrentUser,
) -> BlueprintOut:
    """Approve a blueprint: transition session BLUEPRINT_READY → APPROVED."""
    blueprint = await _get_owned_blueprint(blueprint_id, user.id, db)

    sm = MeetingStateMachine(db)
    await sm.transition(
        blueprint.session_id,
        SessionStatus.APPROVED,
        extra_payload={"blueprint_id": str(blueprint_id), "approved_by": str(user.id)},
    )

    await db.collection("blueprints").document(str(blueprint_id)).update({"approved": True})
    blueprint.approved = True

    from app.workers.build_worker import run_build
    run_build.apply_async(args=[str(blueprint.session_id)], queue="build")
    logger.info("Build dispatched session=%s blueprint=%s", blueprint.session_id, blueprint_id)

    return BlueprintOut.model_validate(blueprint)
