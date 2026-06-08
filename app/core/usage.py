"""Token usage tracking — reads and writes monthly usage per user in Firestore.

Storage layout:
  users/{user_id}/usage/{YYYY-MM}
    tokens : int   (total input + output tokens consumed this month)
    builds : int
    updates: int

User doc fields managed here:
  tier            : str      ("free" | "starter" | "pro" | "team")
  tier_expires_at : datetime (None for free, set on paid subscription)
"""
from __future__ import annotations

from datetime import datetime, timezone

from google.cloud.firestore import AsyncClient, Increment


def _period() -> str:
    """Current billing period key, e.g. '2026-06'."""
    return datetime.now(timezone.utc).strftime("%Y-%m")


async def get_monthly_tokens(db: AsyncClient, user_id: str) -> int:
    """Return tokens consumed by this user in the current billing month."""
    doc = await (
        db.collection("users")
        .document(user_id)
        .collection("usage")
        .document(_period())
        .get()
    )
    if not doc.exists:
        return 0
    return doc.to_dict().get("tokens", 0)


async def record_usage(
    db: AsyncClient,
    user_id: str,
    tokens: int,
    *,
    is_build: bool = False,
    is_update: bool = False,
) -> None:
    """Atomically add tokens (and optionally a build/update counter) to the monthly bucket."""
    ref = (
        db.collection("users")
        .document(user_id)
        .collection("usage")
        .document(_period())
    )
    update: dict = {"tokens": Increment(tokens)}
    if is_build:
        update["builds"] = Increment(1)
    if is_update:
        update["updates"] = Increment(1)

    # set with merge so the document is created if it doesn't exist yet
    await ref.set(update, merge=True)


async def is_over_limit(db: AsyncClient, user_id: str, tier_key: str) -> bool:
    """Return True if the user has exhausted their monthly token budget."""
    from app.core.tiers import get_tier
    tier = get_tier(tier_key)
    used = await get_monthly_tokens(db, user_id)
    return used >= tier.monthly_tokens


async def get_user_tier(db: AsyncClient, user_id: str) -> str:
    """Return the user's active tier, downgrading to free if the subscription has expired."""
    from app.core.tiers import DEFAULT_TIER
    doc = await db.collection("users").document(user_id).get()
    if not doc.exists:
        return DEFAULT_TIER

    data = doc.to_dict()
    tier = data.get("tier", DEFAULT_TIER)

    if tier == DEFAULT_TIER:
        return DEFAULT_TIER

    # Check subscription expiry — paid tiers have a tier_expires_at timestamp
    expires_at = data.get("tier_expires_at")
    if expires_at and expires_at < datetime.now(timezone.utc):
        # Subscription lapsed — silently downgrade
        await db.collection("users").document(user_id).update({
            "tier": DEFAULT_TIER,
            "tier_expires_at": None,
        })
        return DEFAULT_TIER

    return tier
