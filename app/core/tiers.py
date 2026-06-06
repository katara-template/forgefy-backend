"""Subscription tier definitions and token budgets."""
from __future__ import annotations

from typing import NamedTuple


class Tier(NamedTuple):
    name: str
    price_usd: int          # monthly price in USD (0 = free)
    monthly_tokens: int     # combined input+output tokens per billing month


# Token estimates per operation (averages, actual usage varies by blueprint complexity)
#   Build (full app generation) : ~300 000 tokens
#   Update (single prompt)      : ~50 000 tokens
#
# Tier    | Price | Tokens       | ~Builds | ~Updates
# --------|-------|--------------|---------|----------
# free    |  $0   |    500 000   |    1    |    10
# starter | $19   |  5 000 000   |   16    |   100
# pro     | $49   | 20 000 000   |   66    |   400
# team    | $149  | 75 000 000   |  250    |  1 500

TIERS: dict[str, Tier] = {
    "free":    Tier("Free",    0,    500_000),
    "starter": Tier("Starter", 19,  5_000_000),
    "pro":     Tier("Pro",     49, 20_000_000),
    "team":    Tier("Team",   149, 75_000_000),
}

DEFAULT_TIER = "free"


def get_tier(tier_key: str) -> Tier:
    return TIERS.get(tier_key, TIERS[DEFAULT_TIER])
