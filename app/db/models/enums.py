"""Shared SQLAlchemy-compatible Python enums."""
from enum import Enum


class SessionStatus(str, Enum):
    WAITING = "WAITING"
    JOINING = "JOINING"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    BLUEPRINT_READY = "BLUEPRINT_READY"
    APPROVED = "APPROVED"
    BUILDING = "BUILDING"
    FAILED = "FAILED"


class Platform(str, Enum):
    MEET = "meet"
    ZOOM = "zoom"
    TEAMS = "teams"
    PHYSICAL = "physical"


class Priority(str, Enum):
    LOW = "low"
    MED = "med"
    HIGH = "high"
