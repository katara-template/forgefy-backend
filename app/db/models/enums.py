"""Shared string-valued enums (JSON/Firestore friendly)."""
from enum import StrEnum


class SessionStatus(StrEnum):
    WAITING = "WAITING"
    JOINING = "JOINING"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    BLUEPRINT_READY = "BLUEPRINT_READY"
    APPROVED = "APPROVED"
    BUILDING = "BUILDING"
    FAILED = "FAILED"


class Platform(StrEnum):
    MEET = "meet"
    ZOOM = "zoom"
    TEAMS = "teams"
    PHYSICAL = "physical"


class Priority(StrEnum):
    LOW = "low"
    MED = "med"
    HIGH = "high"
