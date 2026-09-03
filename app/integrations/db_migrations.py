"""Async executor that applies derived schema migrations to a connected database.

Wraps :mod:`app.integrations.db_schema` with the real storage touchpoint: opens a
Postgres connection (``asyncpg``, lazily imported) and runs the idempotent DDL +
RLS statements inside one transaction. Used by the connect/wire and rebuild paths
as a best-effort backstop — a migration failure is recorded on the project doc,
never allowed to fail the build or the wire (matching the philosophy of
``app/build/workspace.py``).

The connection function is injectable (``connect_fn``) so the executor is fully
testable with a fake connection and never requires a live database in tests.
"""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

from app.integrations.db_schema import (
    render_migration,
    tables_from_entities,
)

logger = logging.getLogger(__name__)

# Callables may be async and may raise; the executor converts exceptions into a
# recorded migration error rather than propagating them.
ConnectFn = Callable[[str, int], Awaitable[Any]]


@dataclass
class MigrationResult:
    """Outcome of one provisioning attempt, safe to persist to the project doc.

    ``applied`` holds the statements actually sent before any failure. On error
    inside a transaction nothing is committed, so treat the list as diagnostics,
    not a commit log.
    """

    provider: str = ""
    tables: list[str] = field(default_factory=list)
    statements: int = 0
    applied: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def status(self) -> str:
        """``ready`` | ``empty`` | ``error`` — the durable, human-meaningful state."""
        if self.error:
            return "error"
        return "ready" if self.tables else "empty"


def supabase_dsn(project_ref: str, db_pass: str) -> str:
    """A direct Postgres DSN for a Supabase project.

    Uses the standard ``db.<ref>.supabase.co`` host. The password is URL-encoded
    because it is a random token that can contain ``@`` or ``:``. This is a real
    secret — it is only ever produced on the server from the encrypted
    ``supabase_db_pass`` and is never written to a generated app.
    """
    return f"postgresql://postgres:{quote(db_pass, safe='')}@db.{project_ref}.supabase.co:5432/postgres"


async def _asyncpg_connect(dsn: str, timeout: int) -> Any:
    """Open an asyncpg connection. Imported lazily so tests never need it."""
    import asyncpg

    return await asyncpg.connect(dsn=dsn, timeout=timeout)


async def run_migrations(
    dsn: str,
    *,
    entities: list[dict[str, Any]],
    provider: str = "supabase",
    connect_fn: ConnectFn | None = None,
    timeout: int = 60,
) -> MigrationResult:
    """Render and apply the schema for ``entities`` against ``dsn``.

    ``provider == "supabase"`` emits RLS policies keyed on ``auth.uid()``; Neon
    (and any other provider) gets schema only. Deterministic, idempotent, and
    transaction-wrapped so a partial application never half-lands.
    """
    specs = tables_from_entities(entities)
    rls = provider == "supabase"
    statements = render_migration(entities, rls=rls)
    tables = [s.table for s in specs]
    applied: list[str] = []

    connect = connect_fn or _asyncpg_connect
    try:
        conn = await connect(dsn, timeout)
        try:
            async with conn.transaction():
                for stmt in statements:
                    await conn.execute(stmt)
                    applied.append(stmt)
        finally:
            try:  # noqa: SIM105 — suppress() is sync-only; the close here is async
                await conn.close()
            except Exception:  # noqa: BLE001 — best-effort close on a failed run
                pass
    except Exception as exc:  # noqa: BLE001 — converted to a recorded error
        logger.warning("db migration failed provider=%s tables=%s: %s", provider, tables, exc)
        return MigrationResult(
            provider=provider, tables=tables, statements=len(statements),
            applied=applied, error=str(exc),
        )
    logger.info("db migration applied provider=%s tables=%s statements=%d", provider, tables, len(applied))
    return MigrationResult(
        provider=provider, tables=tables, statements=len(statements), applied=applied,
    )


def schema_status_fields(result: MigrationResult) -> dict[str, Any]:
    """The project-doc fields describing a provisioning attempt."""
    return {
        "db_schema_version": len(result.applied),
        "db_schema_tables": result.tables,
        "db_status": result.status,
        "db_schema_error": result.error,
    }


async def provision_schema_for_project(
    db: Any, project_id: str, *, raw: dict[str, Any] | None = None
) -> MigrationResult | None:
    """Best-effort: provision schema for a connected project. Never raises.

    Reads the raw project document (the secrets — encrypted Supabase db_pass or
    Neon connection URI — are deliberately absent from the public ProjectOut
    schema), derives DDL from the blueprint's entities, and applies it. The
    resulting ``db_status`` / ``db_schema_*`` fields are persisted to the doc.

    Pass ``raw`` to reuse an already-fetched project document (the build/update
    workers load one anyway) and skip the extra Firestore read.

    Returns ``None`` when there is nothing to do (no backend credential, e.g.
    a Firebase-only project, or no entities), and a ``MigrationResult``
    otherwise. Exceptions inside are swallowed and recorded - this must never
    block a build or a wire.
    """
    if raw is None:
        try:
            doc = await db.collection("projects").document(project_id).get()
            raw = (doc.to_dict() or {}) if doc.exists else {}
        except Exception as exc:  # noqa: BLE001 — provisioning is best-effort
            logger.warning("db provisioning could not read project %s: %s", project_id, exc)
            return None

    entities = (raw.get("blueprint_context") or {}).get("entities") or []
    if not entities:
        return None

    try:
        if raw.get("supabase_project_ref") and raw.get("supabase_db_pass"):
            from app.core.crypto import decrypt

            provider = "supabase"
            dsn = supabase_dsn(raw["supabase_project_ref"], decrypt(raw["supabase_db_pass"]))
        elif raw.get("neon_project_id") and raw.get("neon_connection_uri"):
            from app.core.crypto import decrypt

            provider = "neon"
            dsn = decrypt(raw["neon_connection_uri"])
        else:
            # Firebase has no Postgres schema and keeps its own Security Rules
            # story — nothing to run here.
            return None

        result = await run_migrations(dsn, entities=entities, provider=provider)
    except Exception as exc:  # noqa: BLE001 — never let a failure escape a wire/build
        logger.warning("db provisioning failed for %s: %s", project_id, exc)
        result = MigrationResult(tables=[s.table for s in tables_from_entities(entities)],
                                 error=str(exc))

    try:
        await db.collection("projects").document(project_id).update(
            schema_status_fields(result)
        )
    except Exception as exc:  # noqa: BLE001 — a metadata write must never mask
        logger.warning("db provisioning could not persist status for %s: %s", project_id, exc)
    return result