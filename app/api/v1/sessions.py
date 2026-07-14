"""Session endpoints: create, join, end, get, upload (physical)."""
import logging
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel

from app.core.dispatch import dispatch
from app.core.exceptions import ValidationError
from app.db.models.enums import Platform, SessionStatus
from app.deps import CurrentUser, DBSession
from app.modules.voxa.service import VoxaService
from app.modules.voxa.state_machine import MeetingStateMachine
from app.schemas.session import (
    CreateSessionRequest,
    EndSessionRequest,
    JoinSessionRequest,
    SessionDetailOut,
    SessionEventOut,
    SessionOut,
)


class DeleteSessionRequest(BaseModel):
    reason: str = ""

_UPLOAD_DIR = Path("/app/tmp")

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


@router.post("/{session_id}/upload", response_model=SessionOut, status_code=202)
async def upload_recording(
    session_id: uuid.UUID,
    db: DBSession,
    user: CurrentUser,
    file: UploadFile = File(...),
) -> SessionOut:
    """Accept a physical meeting recording and kick off the transcription pipeline."""
    service = VoxaService(db)
    session = await service._require_owned(session_id, user.id)

    if session.platform != Platform.PHYSICAL:
        raise ValidationError("File upload is only available for physical sessions.")
    if session.status != SessionStatus.WAITING:
        raise ValidationError("This session has already been processed.")

    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "recording").suffix or ".wav"
    dest = _UPLOAD_DIR / f"{session_id}{suffix}"
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    sm = MeetingStateMachine(db)
    session = await sm.transition(session_id, SessionStatus.PROCESSING)

    from app.workers.upload_worker import transcribe_upload
    await dispatch(
        transcribe_upload,
        args=[str(session_id), str(dest)],
        queue="meeting.audio",
    )

    logger.info("Upload accepted session=%s file=%s", session_id, dest.name)
    return SessionOut.model_validate(session)


@router.post("/{session_id}/start-live", response_model=SessionOut)
async def start_live_transcription(
    session_id: uuid.UUID,
    db: DBSession,
    user: CurrentUser,
) -> SessionOut:
    """Transition a physical session to LISTENING so the browser can stream mic audio."""
    service = VoxaService(db)
    session = await service.start_live_transcription(session_id, user.id)
    return SessionOut.model_validate(session)


@router.get("/{session_id}/transcript")
async def get_session_transcript(
    session_id: uuid.UUID,
    db: DBSession,
    user: CurrentUser,
) -> dict:
    """Return all transcript segments for a session as a single text block."""
    service = VoxaService(db)
    await service._require_owned(session_id, user.id)

    segment_docs = (
        await db.collection("sessions")
        .document(str(session_id))
        .collection("events")
        .where("event_type", "==", "transcript.segment")
        .get()
    )
    segments = sorted(
        [d.to_dict() for d in segment_docs],
        key=lambda s: s.get("timestamp", ""),
    )
    text = " ".join(
        s.get("payload", {}).get("text", "") for s in segments
    ).strip()
    return {"session_id": str(session_id), "transcript": text, "segments": len(segments)}


@router.delete("/{session_id}", response_model=dict)
async def delete_session(
    session_id: uuid.UUID,
    body: DeleteSessionRequest,
    db: DBSession,
    user: CurrentUser,
) -> dict:
    """Permanently delete a session and all its sub-collections."""
    service = VoxaService(db)
    await service._require_owned(session_id, user.id)

    sid = str(session_id)

    # Delete the events sub-collection first
    events_ref = db.collection("sessions").document(sid).collection("events")
    event_docs = await events_ref.get()
    for doc in event_docs:
        await events_ref.document(doc.id).delete()

    # Delete the session document itself
    await db.collection("sessions").document(sid).delete()

    if body.reason:
        logger.info("Session deleted session=%s user=%s reason=%r", sid, user.id, body.reason)
    else:
        logger.info("Session deleted session=%s user=%s", sid, user.id)

    return {"deleted": True}


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
