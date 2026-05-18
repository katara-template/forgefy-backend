"""Alembic async migration environment."""
import asyncio
import logging
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from app.config import get_settings

# Register all ORM models so their tables appear in metadata
import app.db.models  # noqa: F401
from app.db.base import Base

config = context.config
settings = get_settings()

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

logger = logging.getLogger("alembic.env")
target_metadata = Base.metadata


def do_run_migrations(connection):  # type: ignore[no-untyped-def]
    """Configure context and run migrations on a synchronous connection (via run_sync)."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        # Render item-level DROP for enum changes
        include_schemas=False,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_offline() -> None:
    """Generate SQL script without a live DB connection."""
    context.configure(
        url=settings.DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Apply migrations against a live async DB connection."""
    engine = create_async_engine(
        settings.DATABASE_URL,
        poolclass=NullPool,  # fresh connection per Alembic run
    )
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
