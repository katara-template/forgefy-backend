"""Pydantic schemas for auth endpoints."""
import uuid
from datetime import datetime

from pydantic import BaseModel, EmailStr, field_validator


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    # Explicit marketing-email opt-in captured at signup (an unchecked
    # checkbox on the register form) — never defaulted to true.
    marketing_opt_in: bool = False

    @field_validator("password")
    @classmethod
    def password_min_length(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class ResendVerificationRequest(BaseModel):
    email: EmailStr
    password: str


class MessageResponse(BaseModel):
    message: str


class RefreshRequest(BaseModel):
    refresh_token: str


class GoogleAuthRequest(BaseModel):
    id_token: str
    # Same opt-in semantics as RegisterRequest — only applied when this call
    # creates a brand-new account, never when it signs an existing one in.
    marketing_opt_in: bool = False


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: uuid.UUID
    email: str
    created_at: datetime

    model_config = {"from_attributes": True}
