"""Pydantic schemas for the CLI's browser device-code login.

Same shape as GitHub/Google/Vercel's device authorization flow: the CLI gets
a `device_code` (its own secret, long and opaque) plus a short `user_code` to
show the person; the browser page looks the pending device up by `user_code`
and, once the already-logged-in user approves it, the CLI's next poll of
`device_code` receives a freshly minted API key. See app/api/v1/device_auth.py.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

DeviceStatus = Literal["pending", "approved", "denied", "expired"]


class DeviceStartResponse(BaseModel):
    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str
    expires_in: int  # seconds
    interval: int  # seconds between polls


class DevicePollRequest(BaseModel):
    device_code: str


class DevicePollResponse(BaseModel):
    status: DeviceStatus
    api_key: str | None = None  # present only once, the poll that first sees "approved"


class DeviceApproveRequest(BaseModel):
    user_code: str
