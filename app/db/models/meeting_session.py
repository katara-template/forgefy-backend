"""MeetingSession ORM model."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum, ForeignKey, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.db.models.enums import Platform, SessionStatus

if TYPE_CHECKING:
    from app.db.models.blueprint import Blueprint
    from app.db.models.meeting_event import MeetingEvent
    from app.db.models.memory import Memory
    from app.db.models.requirement import Requirement
    from app.db.models.user import User


class MeetingSession(Base):
    __tablename__ = "meeting_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[SessionStatus] = mapped_column(
        Enum(SessionStatus, name="session_status"),
        nullable=False,
        default=SessionStatus.WAITING,
    )
    platform: Mapped[Platform] = mapped_column(
        Enum(Platform, name="platform_type"),
        nullable=False,
    )
    meeting_url: Mapped[str | None] = mapped_column(Text(), nullable=True)
    start_time: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    end_time: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    user: Mapped[User] = relationship("User", back_populates="sessions")
    events: Mapped[list[MeetingEvent]] = relationship(
        "MeetingEvent", back_populates="session", cascade="all, delete-orphan"
    )
    requirements: Mapped[list[Requirement]] = relationship(
        "Requirement", back_populates="session", cascade="all, delete-orphan"
    )
    blueprints: Mapped[list[Blueprint]] = relationship(
        "Blueprint", back_populates="session", cascade="all, delete-orphan"
    )
    memories: Mapped[list[Memory]] = relationship(
        "Memory", back_populates="session", cascade="all, delete-orphan"
    )
