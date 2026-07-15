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

from datetime import UTC, datetime
from typing import NamedTuple

from google.cloud.firestore import AsyncClient, Increment

# The free model a paid user is downgraded to when they exhaust their monthly
# budget, so they keep building instead of being hard-blocked.
_DOWNGRADE_MODEL = "Qwen3"


def _period() -> str:
    """Current billing period key, e.g. '2026-06'."""
    return datetime.now(UTC).strftime("%Y-%m")


def quota_reset_date() -> datetime:
    """UTC datetime when the monthly token bucket resets — first of next month."""
    now = datetime.now(UTC)
    year, month = (now.year + 1, 1) if now.month == 12 else (now.year, now.month + 1)
    return datetime(year, month, 1, tzinfo=UTC)


def quota_reset_label() -> str:
    """Human-readable reset date, e.g. 'August 1, 2026'."""
    d = quota_reset_date()
    return f"{d:%B} {d.day}, {d.year}"


class QuotaOutcome(NamedTuple):
    # "ok"        — under budget, proceed normally
    # "downgrade" — paid user over budget: use `forced_model` and show `message`
    # "block"     — must stop: show `message` (free tier, or free model unavailable)
    action: str
    message: str | None
    forced_model: str | None
    tier_key: str


async def evaluate_quota(db: AsyncClient, settings, user_id: str) -> QuotaOutcome:
    """Decide what to do when a user is (or isn't) over their monthly token budget.

    Policy:
      • Under budget                         → proceed.
      • Free tier over budget                → block, with reset date + upgrade path.
      • Paid tier over budget, free model up  → downgrade to the free model and continue.
      • Paid tier over budget, no free model  → block (can't downgrade), with reset date.

    Non-raising so both the API guards and the workers can share one policy.
    """
    from app.core.tiers import DEFAULT_TIER, get_tier

    tier_key = await get_user_tier(db, user_id)
    if not await is_over_limit(db, user_id, tier_key):
        return QuotaOutcome("ok", None, None, tier_key)

    tier = get_tier(tier_key)
    reset = quota_reset_label()
    used_line = (
        f"You've used all {tier.monthly_tokens:,} tokens in your {tier.name} plan this month. "
        f"Your quota resets on {reset}."
    )

    if tier_key == DEFAULT_TIER:
        return QuotaOutcome("block", f"{used_line} Upgrade your plan to keep building now.", None, tier_key)

    # Paid user — downgrade to the free model instead of hard-blocking, if it's usable.
    if not (getattr(settings, "OPENROUTER_API_KEY", "") or "").strip():
        return QuotaOutcome("block", used_line, None, tier_key)

    return QuotaOutcome(
        "downgrade",
        f"You've used all your {tier.name} tokens for this month (they reset on {reset}). "
        "I've switched you to the free model so you can keep building — responses may be a little slower.",
        _DOWNGRADE_MODEL,
        tier_key,
    )


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


async def check_not_over_limit(db: AsyncClient, user_id: str) -> None:
    """Reject an API request only when the user must be hard-blocked.

    Shares the policy in evaluate_quota: a free user over budget is rejected with
    a clear message + reset date, but a PAID user over budget is let through — the
    worker then downgrades them to the free model and keeps building (so paying
    customers are never hard-stopped mid-month). Call this at the top of any
    endpoint about to dispatch a build/update.
    """
    from app.config import get_settings
    from app.core.exceptions import QuotaExceededError

    outcome = await evaluate_quota(db, get_settings(), user_id)
    if outcome.action == "block":
        raise QuotaExceededError(outcome.message)


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
    if expires_at and expires_at < datetime.now(UTC):
        # Subscription lapsed — silently downgrade
        await db.collection("users").document(user_id).update({
            "tier": DEFAULT_TIER,
            "tier_expires_at": None,
        })
        return DEFAULT_TIER

    return tier
