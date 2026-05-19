"""Blueprint endpoints: get and approve."""
import logging
import uuid

from fastapi import APIRouter
from sqlalchemy import select

from app.core.exceptions import ForbiddenError, NotFoundError
from app.db.models.blueprint import Blueprint
from app.db.models.enums import SessionStatus
from app.db.models.meeting_session import MeetingSession
from app.deps import CurrentUser, DBSession
from app.modules.voxa.state_machine import MeetingStateMachine
from app.schemas.blueprint import BlueprintOut

logger = logging.getLogger(__name__)
router = APIRouter()


async def _get_owned_blueprint(
    blueprint_id: uuid.UUID,
    user_id: uuid.UUID,
    db: DBSession,
) -> Blueprint:
    """Load blueprint + session; raise 404/403 if missing or unowned."""
    result = await db.execute(
        select(Blueprint).where(Blueprint.id == blueprint_id)
    )
    blueprint = result.scalar_one_or_none()
    if blueprint is None:
        raise NotFoundError(f"Blueprint {blueprint_id} not found")

    sess_result = await db.execute(
        select(MeetingSession).where(MeetingSession.id == blueprint.session_id)
    )
    session = sess_result.scalar_one_or_none()
    if session is None or session.user_id != user_id:
        raise ForbiddenError("Access denied")
    return blueprint


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
    """Approve a blueprint: transition session BLUEPRINT_READY → APPROVED.

    The Forgefy build engine integration is stubbed — see the TODO below.
    """
    blueprint = await _get_owned_blueprint(blueprint_id, user.id, db)

    sm = MeetingStateMachine(db)
    await sm.transition(
        blueprint.session_id,
        SessionStatus.APPROVED,
        extra_payload={"blueprint_id": str(blueprint_id), "approved_by": str(user.id)},
    )

    blueprint.approved = True
    db.add(blueprint)

    # TODO: dispatch to Forgefy build engine when multi-platform app
    # generation is implemented.  For now we log the intent and leave the
    # session in APPROVED state until the build engine calls back.
    logger.info(
        "TODO: dispatch build for session=%s blueprint=%s",
        blueprint.session_id,
        blueprint_id,
    )

    return BlueprintOut.model_validate(blueprint)
