"""Import all models so SQLAlchemy metadata is fully populated.

Alembic's env.py imports this module to ensure every table is registered
before autogenerate or migration runs.
"""
from app.db.models.blueprint import Blueprint
from app.db.models.meeting_event import MeetingEvent
from app.db.models.meeting_session import MeetingSession
from app.db.models.memory import Memory
from app.db.models.requirement import Requirement
from app.db.models.user import User

__all__ = [
    "Blueprint",
    "MeetingEvent",
    "MeetingSession",
    "Memory",
    "Requirement",
    "User",
]
