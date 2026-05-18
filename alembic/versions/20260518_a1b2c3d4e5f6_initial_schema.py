"""Initial schema: enable pgvector extension and create all tables.

Revision ID: a1b2c3d4e5f6
Revises:
Create Date: 2026-05-18 00:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "a1b2c3d4e5f6"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ── Extensions ────────────────────────────────────────────────────────────
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ── Enum types (DO block to guard against re-runs) ───────────────────────
    op.execute(
        """
        DO $$ BEGIN
            CREATE TYPE session_status AS ENUM (
                'WAITING', 'JOINING', 'LISTENING', 'PROCESSING',
                'BLUEPRINT_READY', 'APPROVED', 'BUILDING'
            );
        EXCEPTION WHEN duplicate_object THEN NULL; END $$
        """
    )
    op.execute(
        """
        DO $$ BEGIN
            CREATE TYPE platform_type AS ENUM ('meet', 'zoom', 'teams', 'physical');
        EXCEPTION WHEN duplicate_object THEN NULL; END $$
        """
    )
    op.execute(
        """
        DO $$ BEGIN
            CREATE TYPE priority_level AS ENUM ('low', 'med', 'high');
        EXCEPTION WHEN duplicate_object THEN NULL; END $$
        """
    )

    # ── users ─────────────────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    # ── meeting_sessions ──────────────────────────────────────────────────────
    op.create_table(
        "meeting_sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "status",
            postgresql.ENUM(name="session_status", create_type=False),
            nullable=False,
            server_default="WAITING",
        ),
        sa.Column(
            "platform",
            postgresql.ENUM(name="platform_type", create_type=False),
            nullable=False,
        ),
        sa.Column("meeting_url", sa.Text(), nullable=True),
        sa.Column("start_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("end_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_meeting_sessions_user_id", "meeting_sessions", ["user_id"]
    )

    # ── meeting_events ────────────────────────────────────────────────────────
    op.create_table(
        "meeting_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("meeting_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("event_type", sa.String(100), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_meeting_events_session_id", "meeting_events", ["session_id"]
    )

    # ── requirements ──────────────────────────────────────────────────────────
    op.create_table(
        "requirements",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("meeting_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("feature", sa.Text(), nullable=False),
        sa.Column(
            "priority",
            postgresql.ENUM(name="priority_level", create_type=False),
            nullable=False,
            server_default="med",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_requirements_session_id", "requirements", ["session_id"]
    )

    # ── blueprints ────────────────────────────────────────────────────────────
    op.create_table(
        "blueprints",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("meeting_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "json_output",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "approved",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_blueprints_session_id", "blueprints", ["session_id"])

    # ── memories (pgvector) ───────────────────────────────────────────────────
    op.create_table(
        "memories",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("meeting_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1536), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_memories_session_id", "memories", ["session_id"])


def downgrade() -> None:
    op.drop_table("memories")
    op.drop_table("blueprints")
    op.drop_table("requirements")
    op.drop_table("meeting_events")
    op.drop_table("meeting_sessions")
    op.drop_table("users")

    op.execute("DROP TYPE IF EXISTS priority_level")
    op.execute("DROP TYPE IF EXISTS platform_type")
    op.execute("DROP TYPE IF EXISTS session_status")
    op.execute("DROP EXTENSION IF EXISTS vector")
