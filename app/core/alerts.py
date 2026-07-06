"""Operator alerts — surfaces "support"-tier build errors to the admin dashboard
instead of showing technical detail to end users. See app/core/build_errors.py.
"""
from __future__ import annotations

from datetime import UTC, datetime

from google.cloud.firestore import AsyncClient

from app.core.build_errors import scrub_sensitive


async def record_operator_alert(
    db: AsyncClient,
    *,
    title: str,
    raw_detail: str,
    source: str,
    session_id: str | None = None,
    project_id: str | None = None,
) -> None:
    """Persist an operator-facing alert. raw_detail is scrubbed of API keys/URLs/tokens."""
    await db.collection("operator_alerts").add({
        "title": title,
        "raw_detail": scrub_sensitive(raw_detail),
        "source": source,
        "session_id": session_id,
        "project_id": project_id,
        "resolved": False,
        "created_at": datetime.now(UTC),
    })
