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

from app.core.exceptions import ForbiddenError, NotFoundError
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

    logger.info(
        "TODO: dispatch build for session=%s blueprint=%s",
        blueprint.session_id,
        blueprint_id,
    )

    return BlueprintOut.model_validate(blueprint)
