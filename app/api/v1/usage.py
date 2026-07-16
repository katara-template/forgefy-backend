"""Developer usage endpoint — programmatic quota visibility for API keys.

The machine-authed counterpart of GET /billing/status: integrators poll this
to decide whether to send more work (or when the budget resets) instead of
discovering a 402 mid-batch. Deliberately reuses the platform tiers — API
usage and builds draw from the same monthly token bucket.
"""
import logging

from fastapi import APIRouter, Request

from app.core import usage
from app.core.rate_limit import api_key_ident, limiter
from app.core.tiers import get_tier
from app.deps import ApiKeyDep, DBSession

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("")
@limiter.limit("60/minute", key_func=api_key_ident)
async def get_usage(
    request: Request,
    db: DBSession,
    api_key: ApiKeyDep,
) -> dict:
    """Return the key owner's tier, monthly token budget, and consumption."""
    owner_id = str(api_key.owner_user_id)
    tier_key = await usage.get_user_tier(db, owner_id)
    tier = get_tier(tier_key)
    tokens_used = await usage.get_monthly_tokens(db, owner_id)

    return {
        "tier": tier_key,
        "tier_name": tier.name,
        "monthly_tokens": tier.monthly_tokens,
        "tokens_used": tokens_used,
        "tokens_remaining": max(0, tier.monthly_tokens - tokens_used),
        "resets_at": usage.quota_reset_date().isoformat(),
    }
