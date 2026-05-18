"""Session endpoints: create, join, end, get."""
import logging
import uuid

from fastapi import APIRouter

from app.deps import CurrentUser, DBSession
from app.modules.voxa.service import VoxaService
from app.schemas.session import (
    CreateSessionRequest,
    EndSessionRequest,
    JoinSessionRequest,
    SessionDetailOut,
    SessionEventOut,
    SessionOut,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/create", response_model=SessionOut, status_code=201)
async def create_session(
    body: CreateSessionRequest,
    db: DBSession,
    user: CurrentUser,
) -> SessionOut:
    """Create a new meeting session in WAITING state."""
    service = VoxaService(db)
    session = await service.create_session(user.id, body.platform, body.meeting_url)
    return SessionOut.model_validate(session)


@router.post("/join", response_model=SessionOut)
async def join_session(
    body: JoinSessionRequest,
    db: DBSession,
    user: CurrentUser,
) -> SessionOut:
    """Transition session to JOINING and dispatch the meeting bot."""
    service = VoxaService(db)
    session = await service.join_session(body.session_id, user.id)
    return SessionOut.model_validate(session)


@router.post("/end", response_model=SessionOut)
async def end_session(
    body: EndSessionRequest,
    db: DBSession,
    user: CurrentUser,
) -> SessionOut:
    """End the meeting and enqueue requirement extraction + blueprint generation."""
    service = VoxaService(db)
    session = await service.end_session(body.session_id, user.id)
    return SessionOut.model_validate(session)


@router.get("/{session_id}", response_model=SessionDetailOut)
async def get_session(
    session_id: uuid.UUID,
    db: DBSession,
    user: CurrentUser,
) -> SessionDetailOut:
    """Retrieve a session with its status and 20 most-recent events."""
    service = VoxaService(db)
    session, events = await service.get_session(session_id, user.id)
    return SessionDetailOut(
        **SessionOut.model_validate(session).model_dump(),
        recent_events=[SessionEventOut.model_validate(e) for e in events],
    )
