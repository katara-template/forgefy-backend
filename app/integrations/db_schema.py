"""Derive idempotent Postgres schema + RLS DDL from a Forgefy blueprint.

The build/update agent writes data-access code (ForgefyClient on Supabase, or the
Neon Data API) but nothing ever created the tables those queries hit — a
"connected" app pointed at an empty database. This module turns a blueprint's
entities into safe, re-runnable DDL so a connect/wire step can provision real
storage.

It is pure: no network, no ORM, no provider SDKs. Every function is
unit-testable in isolation, and the emitted SQL uses `IF NOT EXISTS` /
`DROP POLICY IF EXISTS` so re-running is a no-op, not an error.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Postgres reserved words that must never be used as a bare column name. We
# always double-quote identifiers anyway, but skipping these up front keeps the
# generated schema readable and avoids quoting surprises in the client SDKs.
_RESERVED_COLUMNS = frozenset({"id", "owner_id", "created_at", "updated_at"})

# Order matters: the first matching branch wins, so "amount" (numeric) must be
# checked before "int"-bearing words like "count".
_TYPE_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(money|price|amount|salary|balance|cost|cash)\b"), "numeric"),
    (re.compile(r"\b(bool|flag|active|enabled|is_)\b"), "boolean"),
    (re.compile(r"\b(date|time|when|timestamp)\b"), "timestamptz"),
    (re.compile(r"\b(int|count|quantity|stock|qty)\b"), "integer"),
    (re.compile(r"\b(float|double|decimal|real)\b"), "numeric"),
]


@dataclass
class TableSpec:
    """One derived table — table name plus its ordered column definitions.

    ``columns`` entries are dicts shaped for both JSON serialisation (persisting
    ``db_schema_tables`` on the project doc is trivial) and direct DDL use:
    ``{name, type, not_null, primary_key}``.
    """

    table: str
    columns: list[dict[str, Any]] = field(default_factory=list)


def identifier(name: str) -> str:
    """Return `name` as a safe, double-quoted Postgres identifier."""
    return '"' + str(name).replace('"', '""') + '"'


def slugify(part: str) -> str:
    """Snake_case a free-text name into a safe identifier fragment."""
    s = re.sub(r"[^a-z0-9]+", "_", str(part).lower()).strip("_")
    return s or "item"


def pg_type(field_type: str) -> str:
    """Map a blueprint field type string to a conservative Postgres type."""
    t = (field_type or "").lower().strip()
    words = [w for w in re.split(r"[^a-z0-9_]+", t) if w]
    for pattern, pg in _TYPE_RULES:
        if any(pattern.search(w) for w in words):
            return pg
    # We never guess a numeric type from an unknown token — default to text.
    return "text"
def _build_columns(entity_name: str, fields: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the column list for an entity's fields, deduped by slug."""
    columns: list[dict[str, Any]] = []
    seen: set[str] = {*_RESERVED_COLUMNS}
    for f in fields or []:
        name = (str(f.get("name") or "")).strip()
        if not name:
            continue
        slug = slugify(name)
        if slug in seen:
            continue
        seen.add(slug)
        columns.append({
            "name": slug,
            "type": pg_type(str(f.get("type") or "text")),
            "not_null": bool(f.get("required")),
            "primary_key": False,
        })
    return columns


def table_for_entity(entity: dict[str, Any]) -> TableSpec:
    """Turn one blueprint entity dict into a TableSpec.

    Every table carries the row-level-security scaffolding the Supabase anon-key
    model relies on: an ``id`` primary key, an ``owner_id`` reference (the RLS
    column), and created/updated timestamps. ``owner_id`` is NOT NULL so a row
    can never be written ownerless — the anon key is public and committed, so
    ownership scoping is the security boundary.
    """
    name = slugify(str(entity.get("name") or ""))
    columns = [
        {"name": "id", "type": "uuid", "not_null": True, "primary_key": True},
        {"name": "owner_id", "type": "uuid", "not_null": True, "primary_key": False},
        *_build_columns(name, entity.get("fields") or []),
        {"name": "created_at", "type": "timestamptz", "not_null": True, "primary_key": False},
        {"name": "updated_at", "type": "timestamptz", "not_null": True, "primary_key": False},
    ]
    return TableSpec(table=name or "item", columns=columns)


def tables_from_entities(entities: list[dict[str, Any]]) -> list[TableSpec]:
    """Derive a TableSpec per entity, deduped by table name (first wins)."""
    seen: set[str] = set()
    specs: list[TableSpec] = []
    for entity in entities or []:
        if not isinstance(entity, dict):
            continue
        spec = table_for_entity(entity)
        if spec.table in seen:
            continue
        seen.add(spec.table)
        specs.append(spec)
    return specs


def render_table_ddl(spec: TableSpec) -> str:
    """One idempotent ``CREATE TABLE IF NOT EXISTS`` statement."""
    t = identifier(spec.table)
    parts: list[str] = []
    for col in spec.columns:
        cid = identifier(col["name"])
        if col.get("primary_key"):
            parts.append(f"    {cid} UUID PRIMARY KEY DEFAULT gen_random_uuid()")
            continue
        if col["name"] == "created_at":
            parts.append(f"    {cid} TIMESTAMPTZ NOT NULL DEFAULT now()")
            continue
        if col["name"] == "updated_at":
            parts.append(f"    {cid} TIMESTAMPTZ NOT NULL DEFAULT now()")
            continue
        not_null = " NOT NULL" if col.get("not_null") else ""
        parts.append(f"    {cid} {col['type']}{not_null}")
    return (
        f"CREATE TABLE IF NOT EXISTS public.{t} (\n"
        + ",\n".join(parts)
        + "\n);"
    )


def render_rls_policies(spec: TableSpec) -> list[str]:
    """RLS statements that scope reads/writes to the authenticated owner.

    Supabase-only: ``auth.uid()`` does not exist on the Neon Data API / a plain
    Postgres host, so callers pass ``rls=False`` for Neon. The anon key the app
    ships is public and committed, so RLS is the only thing standing between a
    user's data and the world — these policies are the security boundary, not a
    nicety.
    """
    t = identifier(spec.table)
    owner_id = identifier("owner_id")
    policy = f"{spec.table}_owner_isolation"
    return [
        f"ALTER TABLE public.{t} ENABLE ROW LEVEL SECURITY;",
        f"DROP POLICY IF EXISTS {identifier(policy)} ON public.{t};",
        (
            f"CREATE POLICY {identifier(policy)} ON public.{t} "
            f"USING ({owner_id} = (select auth.uid())) "
            f"WITH CHECK ({owner_id} = (select auth.uid()));"
        ),
    ]


def render_migration(
    entities: list[dict[str, Any]], *, rls: bool = True
) -> list[str]:
    """Ordered, idempotent migration statements for a blueprint's entities.

    Each table's DDL comes first (dependencies at schema level are nil since we
    emit no foreign keys), then its RLS policies when the backend supports
    ``auth.uid()``. Safe to re-run at every connect/wire/rebuild.
    """
    statements: list[str] = []
    for spec in tables_from_entities(entities):
        statements.append(render_table_ddl(spec))
        if rls:
            statements.extend(render_rls_policies(spec))
    return statements