"""Development data seeding script.

Usage:
    python scripts/seed.py
"""
import asyncio
import logging

logger = logging.getLogger(__name__)


async def seed() -> None:
    """Insert dev fixtures — users, sample session, etc."""
    # TODO: implement once DB models are in place (Step 2)
    logger.info("Seeding not yet implemented.")


if __name__ == "__main__":
    asyncio.run(seed())
