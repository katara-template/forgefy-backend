"""Requirement ORM model — a single extracted product feature."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum, ForeignKey, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.db.models.enums import Priority

if TYPE_CHECKING:
    from app.db.models.meeting_session import MeetingSession


class Requirement(Base):
    __tablename__ = "requirements"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("meeting_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    feature: Mapped[str] = mapped_column(Text(), nullable=False)
    priority: Mapped[Priority] = mapped_column(
        Enum(Priority, name="priority_level"),
        nullable=False,
        default=Priority.MED,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    session: Mapped[MeetingSession] = relationship(
        "MeetingSession", back_populates="requirements"
    )
