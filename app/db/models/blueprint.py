"""Blueprint ORM model — final build-ready specification for a session."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import Boolean, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.db.models.meeting_session import MeetingSession


class Blueprint(Base):
    __tablename__ = "blueprints"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("meeting_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    json_output: Mapped[dict[str, Any] | None] = mapped_column(JSONB(), nullable=True)
    approved: Mapped[bool] = mapped_column(
        Boolean(), nullable=False, default=False, server_default="false"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    session: Mapped[MeetingSession] = relationship(
        "MeetingSession", back_populates="blueprints"
    )
