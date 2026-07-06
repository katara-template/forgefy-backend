"""Blueprint endpoint and aggregator tests."""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import AsyncClient

from app.db.models.blueprint import Blueprint
from app.db.models.enums import Platform, SessionStatus
from app.db.models.meeting_session import MeetingSession
from tests.conftest import make_doc_snapshot

# ── Helpers ───────────────────────────────────────────────────────────────────


def _session(
    user_id: uuid.UUID | None = None,
    status: SessionStatus = SessionStatus.BLUEPRINT_READY,
) -> MeetingSession:
    return MeetingSession(
        id=uuid.uuid4(),
        user_id=user_id or uuid.uuid4(),
        status=status,
        platform=Platform.MEET,
        meeting_url=None,
        start_time=None,
        end_time=None,
        created_at=datetime.now(UTC),
    )


def _blueprint(session: MeetingSession, approved: bool = False) -> Blueprint:
    return Blueprint(
        id=uuid.uuid4(),
        session_id=session.id,
        json_output={
            "version": "1.0",
            "session_id": str(session.id),
            "features": [{"title": "Login", "description": "SSO", "priority": "high"}],
            "open_questions": [],
            "conflicts": [],
            "action_items": [],
            "requirements_count": 1,
        },
        approved=approved,
        created_at=datetime.now(UTC),
    )


# ── GET /voxa/blueprint/{id} ──────────────────────────────────────────────────


class TestGetBlueprint:
    async def test_success(self, auth_client: AsyncClient, test_user) -> None:
        sess = _session(user_id=test_user.id)
        bp = _blueprint(sess)

        with patch("app.api.v1.blueprints._get_owned_blueprint", return_value=bp):
            resp = await auth_client.get(f"/api/v1/voxa/blueprint/{bp.id}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == str(bp.id)
        assert body["approved"] is False
        assert body["json_output"]["features"][0]["title"] == "Login"

    async def test_not_found_returns_404(self, auth_client: AsyncClient) -> None:
        from app.core.exceptions import NotFoundError

        with patch(
            "app.api.v1.blueprints._get_owned_blueprint",
            side_effect=NotFoundError("not found"),
        ):
            resp = await auth_client.get(f"/api/v1/voxa/blueprint/{uuid.uuid4()}")

        assert resp.status_code == 404

    async def test_no_auth_returns_401(self, client: AsyncClient) -> None:
        resp = await client.get(f"/api/v1/voxa/blueprint/{uuid.uuid4()}")
        assert resp.status_code == 401


# ── POST /voxa/blueprint/{id}/approve ────────────────────────────────────────


class TestApproveBlueprint:
    async def test_success_transitions_and_marks_approved(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        sess = _session(user_id=test_user.id, status=SessionStatus.BLUEPRINT_READY)
        bp = _blueprint(sess)

        # approve_blueprint() independently reads sessions/{id} to check the
        # current status before deciding whether to transition.
        mock_db.collection.return_value.document.return_value.get.return_value = (
            make_doc_snapshot({"status": sess.status.value})
        )

        mock_sm = AsyncMock()
        mock_sm.transition = AsyncMock()

        with (
            patch("app.api.v1.blueprints._get_owned_blueprint", return_value=bp),
            patch("app.api.v1.blueprints.MeetingStateMachine", return_value=mock_sm),
            patch(
                "app.build.blueprint_generator.classify_needs_database",
                new=AsyncMock(return_value=(False, "")),
            ),
            patch("app.workers.build_worker.run_build.apply_async") as mock_dispatch,
        ):
            resp = await auth_client.post(f"/api/v1/voxa/blueprint/{bp.id}/approve")

        assert resp.status_code == 200
        body = resp.json()
        assert body["approved"] is True
        mock_sm.transition.assert_called_once_with(
            sess.id,
            SessionStatus.APPROVED,
            extra_payload={"blueprint_id": str(bp.id), "approved_by": str(test_user.id)},
        )
        mock_dispatch.assert_called_once()

        project_payload = mock_db.collection.return_value.document.return_value.set.call_args[0][0]
        assert project_payload["is_updating"] is True
        assert project_payload["db_decision_pending"] is False

    async def test_needs_database_withholds_build(
        self, auth_client: AsyncClient, mock_db: MagicMock, test_user
    ) -> None:
        """If the classifier thinks the app needs persistent storage, the initial
        build must be withheld until the user is asked — never provisioned/wired
        without consent."""
        sess = _session(user_id=test_user.id, status=SessionStatus.BLUEPRINT_READY)
        bp = _blueprint(sess)

        mock_db.collection.return_value.document.return_value.get.return_value = (
            make_doc_snapshot({"status": sess.status.value})
        )

        mock_sm = AsyncMock()
        mock_sm.transition = AsyncMock()

        with (
            patch("app.api.v1.blueprints._get_owned_blueprint", return_value=bp),
            patch("app.api.v1.blueprints.MeetingStateMachine", return_value=mock_sm),
            patch(
                "app.build.blueprint_generator.classify_needs_database",
                new=AsyncMock(return_value=(True, "Tracks inventory items across sessions.")),
            ),
            patch("app.workers.build_worker.run_build.apply_async") as mock_dispatch,
        ):
            resp = await auth_client.post(f"/api/v1/voxa/blueprint/{bp.id}/approve")

        assert resp.status_code == 200
        mock_dispatch.assert_not_called()

        project_payload = mock_db.collection.return_value.document.return_value.set.call_args[0][0]
        assert project_payload["is_updating"] is False
        assert project_payload["db_decision_pending"] is True
        assert project_payload["db_decision_reason"] == "Tracks inventory items across sessions."

    async def test_forbidden_returns_403(self, auth_client: AsyncClient) -> None:
        from app.core.exceptions import ForbiddenError

        with patch(
            "app.api.v1.blueprints._get_owned_blueprint",
            side_effect=ForbiddenError("denied"),
        ):
            resp = await auth_client.post(f"/api/v1/voxa/blueprint/{uuid.uuid4()}/approve")

        assert resp.status_code == 403

    async def test_no_auth_returns_401(self, client: AsyncClient) -> None:
        resp = await client.post(f"/api/v1/voxa/blueprint/{uuid.uuid4()}/approve")
        assert resp.status_code == 401


# BlueprintAggregator and the blueprint Celery worker are covered more
# thoroughly (and against the real Firestore-based implementation) in
# tests/build/test_blueprint_generator.py and tests/workers/test_blueprint_worker.py
# respectively — this file previously had stale, pre-Firestore-migration
# duplicates of both that never actually exercised current code.
