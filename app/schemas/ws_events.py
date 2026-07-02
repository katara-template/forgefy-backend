"""WebSocket event schemas.

Client→Server messages arrive as JSON with a top-level `type` discriminator.
Server→Client messages are serialized the same way.
"""
from __future__ import annotations

import uuid
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

# ── Client → Server ───────────────────────────────────────────────────────────


class JoinSessionEvent(BaseModel):
    type: Literal["joinSession"]
    session_id: uuid.UUID


class StreamAudioEvent(BaseModel):
    type: Literal["streamAudio"]
    session_id: uuid.UUID
    chunk: str  # base-64 encoded PCM


class EndMeetingEvent(BaseModel):
    type: Literal["endMeeting"]
    session_id: uuid.UUID


class PingEvent(BaseModel):
    type: Literal["ping"]


ClientEvent = Annotated[
    JoinSessionEvent | StreamAudioEvent | EndMeetingEvent | PingEvent,
    Field(discriminator="type"),
]


# ── Server → Client ───────────────────────────────────────────────────────────


class TranscriptEvent(BaseModel):
    type: Literal["transcript"] = "transcript"
    session_id: uuid.UUID
    text: str
    is_final: bool = False


class FeatureDetectedEvent(BaseModel):
    type: Literal["featureDetected"] = "featureDetected"
    session_id: uuid.UUID
    sub_state: str
    payload: dict[str, Any] = {}


class BlueprintReadyEvent(BaseModel):
    type: Literal["blueprintReady"] = "blueprintReady"
    session_id: uuid.UUID
    blueprint_id: uuid.UUID


class MeetingStatusEvent(BaseModel):
    type: Literal["meetingStatus"] = "meetingStatus"
    session_id: uuid.UUID
    status: str


class PongEvent(BaseModel):
    type: Literal["pong"] = "pong"


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    code: str
    detail: str
