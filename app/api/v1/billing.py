"""Billing endpoints — Notchpay checkout and subscription status."""
from __future__ import annotations

import logging
from datetime import UTC, datetime

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import get_settings
from app.core.tiers import TIERS, get_tier
from app.core.usage import get_monthly_tokens, get_user_tier
from app.deps import CurrentUser, DBSession

logger = logging.getLogger(__name__)
router = APIRouter()

NOTCHPAY_API = "https://api.notchpay.co"


class CheckoutRequest(BaseModel):
    tier: str   # "starter" | "pro" | "team"


@router.post("/checkout")
async def create_checkout(
    body: CheckoutRequest,
    user: CurrentUser,
    db: DBSession,
) -> dict:
    """Create a Notchpay payment and return the authorization URL."""
    settings = get_settings()

    if body.tier not in TIERS or body.tier == "free":
        raise HTTPException(status_code=400, detail="Invalid tier")

    tier = get_tier(body.tier)

    # Unique reference encodes user + plan so the webhook can parse it
    reference = f"forgefy-{user.id}-{body.tier}-{int(datetime.now(UTC).timestamp())}"

    try:
        async with httpx.AsyncClient() as http_client:
            resp = await http_client.post(
                f"{NOTCHPAY_API}/payments/initialize",
                headers={
                    "Authorization": settings.NOTCHPAY_PUBLIC_KEY,
                    "Content-Type": "application/json",
                },
                json={
                    "amount": tier.price_usd,
                    "currency": "USD",
                    "email": user.email,
                    "reference": reference,
                    "callback": f"{settings.FRONTEND_URL}/billing/success",
                    "description": f"Forgefy {tier.name} Plan — monthly",
                    "meta": {
                        "user_id": str(user.id),
                        "tier": body.tier,
                    },
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        logger.error("Notchpay checkout error: %s", e.response.text)
        raise HTTPException(status_code=502, detail="Payment provider error")
    except Exception as e:
        logger.error("Notchpay request failed: %s", e)
        raise HTTPException(status_code=502, detail="Could not reach payment provider")

    auth_url = (
        data.get("transaction", {}).get("authorization_url")
        or data.get("authorization_url")
    )
    if not auth_url:
        raise HTTPException(status_code=502, detail="No payment URL returned")

    return {"url": auth_url, "reference": reference}


@router.get("/status")
async def billing_status(user: CurrentUser, db: DBSession) -> dict:
    """Return the user's current tier, expiry, and monthly token usage."""
    user_doc = await db.collection("users").document(str(user.id)).get()
    data = user_doc.to_dict() if user_doc.exists else {}

    tier_key = await get_user_tier(db, str(user.id))
    tier = get_tier(tier_key)

    expires_at = data.get("tier_expires_at")
    tokens_used = await get_monthly_tokens(db, str(user.id))

    return {
        "tier": tier_key,
        "tier_name": tier.name,
        "price_usd": tier.price_usd,
        "monthly_tokens": tier.monthly_tokens,
        "tokens_used": tokens_used,
        "tokens_remaining": max(0, tier.monthly_tokens - tokens_used),
        "expires_at": expires_at.isoformat() if expires_at else None,
    }
