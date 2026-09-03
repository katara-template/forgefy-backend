"""Tests for the schema-provisioning backstop (app/integrations/db_schema.py and
app/integrations/db_migrations.py).

No live database is required: the SQL renderer is pure, and the executor's
connection is injected with a fake that records the statements it would run.
"""
from __future__ import annotations

import pytest

from app.integrations import db_migrations
from app.integrations.db_migrations import (
    MigrationResult,
    provision_schema_for_project,
    run_migrations,
    schema_status_fields,
    supabase_dsn,
)
from app.integrations.db_schema import (
    identifier,
    pg_type,
    render_migration,
    render_rls_policies,
    render_table_ddl,
    slugify,
    table_for_entity,
    tables_from_entities,
)

INVOICE = {
    "name": "Invoice",
    "fields": [
        {"name": "Invoice Number", "type": "int", "required": True},
        {"name": "Total Amount", "type": "money", "required": False},
        {"name": "Issued On", "type": "date", "required": True},
        {"name": "Paid", "type": "bool", "required": False},
        {"name": "Bill To", "type": "email", "required": False},
    ],
}


class TestTypeMapping:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("money", "numeric"),
            ("price", "numeric"),
            ("amount", "numeric"),
            ("count", "integer"),
            ("int", "integer"),
            ("bool", "boolean"),
            ("date", "timestamptz"),
            ("float", "numeric"),
            ("email", "text"),
            ("weird thing", "text"),
            ("", "text"),
        ],
    )
    def test_pg_type(self, raw: str, expected: str) -> None:
        assert pg_type(raw) == expected

    def test_substring_boundary(self) -> None:
        # "account"/"amountless" must not be caught by the "count"/"amount" rule.
        assert pg_type("Account Name") == "text"


class TestIdentifiers:
    def test_identifier_escapes_quotes(self) -> None:
        assert identifier('a"b') == '"a""b"'

    def test_slugify_normalises(self) -> None:
        assert slugify("Invoice Number") == "invoice_number"
        assert slugify("   Items  List  ") == "items_list"
        assert slugify("!!!") == "item"


class TestTableDerivation:
    def test_columns_include_ownership_scaffolding(self) -> None:
        spec = table_for_entity(INVOICE)
        names = [c["name"] for c in spec.columns]
        assert names[0] == "id"
        assert names[1] == "owner_id"
        assert names[-2:] == ["created_at", "updated_at"]
        id_col = spec.columns[0]
        assert id_col["primary_key"] is True and id_col["type"] == "uuid"
        owner = spec.columns[1]
        assert owner["not_null"] is True and owner["type"] == "uuid"

    def test_fields_deduped_and_reserved_skipped(self) -> None:
        spec = table_for_entity({
            "name": "Invoice",
            "fields": [
                {"name": "id", "type": "int"},
                {"name": "Total Amount", "type": "money"},
                {"name": "total_amount", "type": "even worse"},
                {"name": "Name", "type": "text", "required": True},
            ],
        })
        names = [c["name"] for c in spec.columns]
        assert names.count("total_amount") == 1
        assert names.count("id") == 1  # reserved — never duplicated from a field
        assert "name" in names

    def test_tables_from_entities_dedupes_by_table_name(self) -> None:
        specs = tables_from_entities([
            {"name": "Invoice"},
            {"name": "Invoice"},
            {"name": "Customer"},
        ])
        assert [s.table for s in specs] == ["invoice", "customer"]
class TestDDLRendering:
    def test_ddl_is_idempotent(self) -> None:
        ddl = render_table_ddl(table_for_entity(INVOICE))
        assert "CREATE TABLE IF NOT EXISTS public.\"invoice\"" in ddl

    def test_ddl_carries_not_null_and_types(self) -> None:
        ddl = render_table_ddl(table_for_entity(INVOICE)).lower()
        assert "invoice_number" in ddl
        assert "total_amount" in ddl
        assert "numeric" in ddl
        assert "timestamptz not null" in ddl
        assert "owner_id" in ddl
        assert "uuid not null" in ddl

    def test_rls_policies_include_auth_uid(self) -> None:
        policies = render_rls_policies(table_for_entity(INVOICE))
        joined = "\n".join(policies)
        assert "ENABLE ROW LEVEL SECURITY" in joined
        assert "auth.uid()" in joined
        assert "DROP POLICY IF EXISTS" in joined

    def test_render_migration_rls_flag(self) -> None:
        with_rls = render_migration([INVOICE], rls=True)
        assert any("ENABLE ROW LEVEL SECURITY" in s for s in with_rls)
        without_rls = render_migration([INVOICE], rls=False)
        assert not any("ENABLE ROW LEVEL SECURITY" in s for s in without_rls)
        # Both re-run safely.
        assert render_migration([INVOICE], rls=True) == with_rls


class _Txn:
    def __init__(self, conn: _FakeConn) -> None:
        self.conn = conn

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> bool:
        return False


class _FakeConn:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.closed = False

    async def execute(self, stmt: str) -> None:
        self.sent.append(stmt)

    async def close(self) -> None:
        self.closed = True

    def transaction(self) -> _Txn:
        return _Txn(self)


class _FailConn(_FakeConn):
    async def execute(self, stmt: str) -> None:
        raise RuntimeError("permission denied")


def _make_connect(conn: _FakeConn):
    async def connect(dsn: str, timeout: int) -> _FakeConn:
        return conn
    return connect


class TestExecutor:
    @pytest.mark.asyncio
    async def test_applies_all_statements_with_rls_for_supabase(self) -> None:
        conn = _FakeConn()
        result = await run_migrations(
            "postgresql://x",
            entities=[INVOICE],
            provider="supabase",
            connect_fn=_make_connect(conn),
        )
        assert result.error is None
        assert result.status == "ready"
        assert result.tables == ["invoice"]
        assert conn.closed is True
        assert any("CREATE TABLE IF NOT EXISTS" in s for s in conn.sent)
        assert any("ENABLE ROW LEVEL SECURITY" in s for s in conn.sent)
        assert result.statements == len(conn.sent)

    @pytest.mark.asyncio
    async def test_neon_skips_rls(self) -> None:
        conn = _FakeConn()
        await run_migrations(
            "postgresql://x",
            entities=[INVOICE],
            provider="neon",
            connect_fn=_make_connect(conn),
        )
        assert not any("ENABLE ROW LEVEL SECURITY" in s for s in conn.sent)

    @pytest.mark.asyncio
    async def test_failure_recorded_not_raised(self) -> None:
        conn = _FailConn()
        result = await run_migrations(
            "postgresql://x",
            entities=[INVOICE],
            provider="supabase",
            connect_fn=_make_connect(conn),
        )
        assert result.status == "error"
        assert result.error == "permission denied"
        assert result.tables == ["invoice"]

    @pytest.mark.asyncio
    async def test_empty_entities_is_empty_status(self) -> None:
        conn = _FakeConn()
        result = await run_migrations(
            "postgresql://x",
            entities=[],
            provider="supabase",
            connect_fn=_make_connect(conn),
        )
        assert result.status == "empty"
        assert result.tables == []
        assert conn.sent == []
class TestStatusFields:
    def test_schema_status_fields(self) -> None:
        result = MigrationResult(provider="supabase", tables=["a"], statements=4, applied=["1", "2"])
        fields = schema_status_fields(result)
        assert fields["db_schema_version"] == 2
        assert fields["db_schema_tables"] == ["a"]
        assert fields["db_status"] == "ready"
        assert fields["db_schema_error"] is None


class _FakeDoc:
    def __init__(self, data: dict, exists: bool = True) -> None:
        self._data = data
        self.exists = exists

    def to_dict(self) -> dict:
        return self._data


class _FakeFirestore:
    def __init__(self, raw: dict) -> None:
        self._raw = raw
        self.updates: list[dict] = []

    def collection(self, _name):
        return self

    def document(self, _id):
        return self

    def get(self) -> _FakeDoc:
        return _FakeDoc(self._raw)

    def update(self, payload: dict) -> None:
        self.updates.append(payload)


def _raw_with_supabase(entities=None) -> dict:
    return {
        "blueprint_context": {"entities": entities} if entities is not None else {},
        "supabase_project_ref": "ref123",
        "supabase_db_pass": "s3cret",
    }


class TestProvisionWiring:
    @pytest.mark.asyncio
    async def test_returns_none_when_no_entities(self, monkeypatch) -> None:
        db = _FakeFirestore(_raw_with_supabase(entities=None))
        result = await provision_schema_for_project(db, "p1", raw=_raw_with_supabase(None))
        assert result is None
        assert db.updates == []

    @pytest.mark.asyncio
    async def test_firebase_only_returns_none(self, monkeypatch) -> None:
        raw = {"blueprint_context": {"entities": [INVOICE]}, "firebase_project_id": "fg-1"}
        db = _FakeFirestore(raw)
        result = await provision_schema_for_project(db, "p1", raw=raw)
        assert result is None
        assert db.updates == []

    @pytest.mark.asyncio
    async def test_runs_and_persists_status(self, monkeypatch) -> None:
        raw = _raw_with_supabase([INVOICE])
        db = _FakeFirestore(raw)

        called: dict = {}

        async def fake_run(dsn, *, entities, provider, connect_fn=None, timeout=60):
            called["dsn"] = dsn
            called["provider"] = provider
            return MigrationResult(provider=provider, tables=["invoice"], statements=4, applied=["a", "b", "c"])

        monkeypatch.setattr(db_migrations, "run_migrations", fake_run)
        monkeypatch.setattr("app.core.crypto.decrypt", lambda c: "plainpass")

        result = await provision_schema_for_project(db, "p1", raw=raw)
        assert called["provider"] == "supabase"
        assert "plainpass" in called["dsn"]
        assert result is not None and result.status == "ready"
        assert db.updates and db.updates[0]["db_status"] == "ready"
        assert db.updates[0]["db_schema_tables"] == ["invoice"]

    @pytest.mark.asyncio
    async def test_never_raises_on_failure(self, monkeypatch) -> None:
        raw = _raw_with_supabase([INVOICE])
        db = _FakeFirestore(raw)

        async def boom(*args, **kwargs):  # noqa: ANN002, ANN003
            raise RuntimeError("unexpected")

        monkeypatch.setattr(db_migrations, "run_migrations", boom)
        monkeypatch.setattr("app.core.crypto.decrypt", lambda c: "plainpass")

        # Must not raise even though the migration fails.
        result = await provision_schema_for_project(db, "p1", raw=raw)
        assert result is None or result.status == "error"

    def test_supabase_dsn_url_encodes_password(self) -> None:
        dsn = supabase_dsn("ref123", "p@ss:w/rd")
        assert "db.ref123.supabase.co" in dsn
        assert "postgres:p%40ss%3Aw%2Frd@" in dsn
        assert "p@ss" not in dsn