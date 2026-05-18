"""Shared FastAPI dependencies.

DB session and current_user are added in Steps 2 (Database) and 3 (Auth).
"""
from app.config import Settings, get_settings
from fastapi import Depends
from typing import Annotated

SettingsDep = Annotated[Settings, Depends(get_settings)]
