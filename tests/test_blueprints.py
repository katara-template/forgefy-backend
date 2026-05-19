"""Blueprint endpoint and aggregator tests."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient

from app.db.models.blueprint import Blueprint
from app.db.models.enums import Platform, SessionStatus
from app.db.models.meeting_event import MeetingEvent
from app.db.models.meeting_session import MeetingSession


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
        created_at=datetime.now(timezone.utc),
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
        created_at=datetime.now(timezone.utc),
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
        self, auth_client: AsyncClient, test_user
    ) -> None:
        sess = _session(user_id=test_user.id, status=SessionStatus.BLUEPRINT_READY)
        bp = _blueprint(sess)

        mock_sm = AsyncMock()
        mock_sm.transition = AsyncMock()

        with (
            patch("app.api.v1.blueprints._get_owned_blueprint", return_value=bp),
            patch("app.api.v1.blueprints.MeetingStateMachine", return_value=mock_sm),
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


# ── BlueprintAggregator unit tests ────────────────────────────────────────────


class TestBlueprintAggregator:
    def _make_db(self, session: MeetingSession, events: list[MeetingEvent]) -> AsyncMock:
        db = AsyncMock()
        db.add = MagicMock()

        # First execute → session; second execute → events
        sess_result = MagicMock()
        sess_result.scalar_one_or_none.return_value = session

        events_result = MagicMock()
        events_result.scalars.return_value = iter(events)

        db.execute.side_effect = [sess_result, events_result]
        db.flush = AsyncMock()
        return db

    async def test_generate_creates_blueprint_row(self) -> None:
        from app.build.blueprint_generator import BlueprintAggregator

        sess = _session(status=SessionStatus.PROCESSING)
        db = self._make_db(sess, [])

        agg = BlueprintAggregator(db)
        agg._sm = AsyncMock()
        agg._sm.transition = AsyncMock()

        bp = await agg.generate(sess.id)

        db.add.assert_called()
        db.flush.assert_called_once()
        assert bp.session_id == sess.id
        assert bp.approved is False

    async def test_generate_structures_extraction_events(self) -> None:
        from app.build.blueprint_generator import BlueprintAggregator

        sess = _session(status=SessionStatus.PROCESSING)

        feature_event = MeetingEvent(
            id=uuid.uuid4(),
            session_id=sess.id,
            event_type="FEATURE_FOUND",
            payload={"title": "OAuth", "description": "SSO support", "priority": "high"},
            timestamp=datetime.now(timezone.utc),
        )
        question_event = MeetingEvent(
            id=uuid.uuid4(),
            session_id=sess.id,
            event_type="QUESTION_FOUND",
            payload={"text": "Mobile?", "context": "platform"},
            timestamp=datetime.now(timezone.utc),
        )

        db = self._make_db(sess, [feature_event, question_event])

        agg = BlueprintAggregator(db)
        agg._sm = AsyncMock()
        agg._sm.transition = AsyncMock()

        bp = await agg.generate(sess.id)

        assert len(bp.json_output["features"]) == 1
        assert len(bp.json_output["open_questions"]) == 1
        assert bp.json_output["features"][0]["title"] == "OAuth"
        assert bp.json_output["requirements_count"] == 2

    async def test_generate_raises_on_missing_session(self) -> None:
        from app.build.blueprint_generator import BlueprintAggregator

        db = AsyncMock()
        db.add = MagicMock()
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        db.execute.return_value = result

        agg = BlueprintAggregator(db)

        with pytest.raises(ValueError, match="not found"):
            await agg.generate(uuid.uuid4())


# ── Blueprint worker tests ────────────────────────────────────────────────────


class TestBlueprintWorker:
    def test_generate_blueprint_publishes_ready_event(self) -> None:
        import json

        from app.workers.blueprint_worker import generate_blueprint

        session_id = str(uuid.uuid4())
        blueprint_id = str(uuid.uuid4())
        mock_redis = MagicMock()

        # Patch _run_aggregation (the async helper) so asyncio.run receives a
        # coroutine that completes immediately without hitting the real DB.
        async def _fake_run_aggregation(*_args, **_kwargs):
            return blueprint_id

        with (
            patch("app.workers.blueprint_worker._run_aggregation", _fake_run_aggregation),
            patch("app.workers.blueprint_worker.sync_redis.from_url", return_value=mock_redis),
        ):
            generate_blueprint(session_id)

        mock_redis.publish.assert_called_once()
        channel, payload = mock_redis.publish.call_args[0]
        assert channel == f"voxa:session:{session_id}"
        data = json.loads(payload)
        assert data["type"] == "blueprintReady"
        assert data["blueprint_id"] == blueprint_id
