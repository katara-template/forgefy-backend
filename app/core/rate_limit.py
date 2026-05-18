"""Shared slowapi Limiter singleton.

Both app/main.py (app.state.limiter registration) and route files
(@limiter.limit decorators) must reference the same instance.
"""
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address, default_limits=["300/minute"])
