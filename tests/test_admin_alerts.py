"""Admin alerts endpoint tests + record_operator_alert unit tests."""
import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock

from httpx import AsyncClient

from app.core.alerts import record_operator_alert
from app.deps import get_current_user
from app.main import app
from tests.conftest import make_doc_snapshot


def _alert_doc(**overrides) -> MagicMock:
    data = {
        "title": "Your AI service plan has run out of credits. Please contact support.",
        "raw_detail": "429 RESOURCE_EXHAUSTED: credits are depleted",
        "source": "build",
        "session_id": str(uuid.uuid4()),
        "project_id": str(uuid.uuid4()),
        "resolved": False,
        "created_at": datetime.now(UTC),
    }
    data.update(overrides)
    return make_doc_snapshot(data, doc_id=str(uuid.uuid4()))


class TestListAlerts:
    async def test_non_admin_returns_403(self, auth_client: AsyncClient) -> None:
        resp = await auth_client.get("/api/v1/admin/alerts")
        assert resp.status_code == 403

    async def test_no_auth_returns_401(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/admin/alerts")
        assert resp.status_code == 401

    async def test_admin_receives_alert_list(
        self, client: AsyncClient, mock_db: MagicMock, admin_user
    ) -> None:
        app.dependency_overrides[get_current_user] = lambda: admin_user
        doc = _alert_doc()
        mock_db.collection.return_value.where.return_value.get.return_value = [doc]

        resp = await client.get("/api/v1/admin/alerts")

        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        assert body[0]["title"] == doc.to_dict()["title"]
        assert body[0]["resolved"] is False

    async def test_alerts_sorted_newest_first_and_limited_in_python(
        self, client: AsyncClient, mock_db: MagicMock, admin_user
    ) -> None:
        """No Firestore order_by/limit call is needed — sorting/limiting happens
        in Python (see list_alerts), so this must work regardless of doc order."""
        app.dependency_overrides[get_current_user] = lambda: admin_user
        older = _alert_doc(title="older", created_at=datetime(2024, 1, 1, tzinfo=UTC))
        newer = _alert_doc(title="newer", created_at=datetime(2025, 1, 1, tzinfo=UTC))
        # Deliberately returned oldest-first to prove the endpoint re-sorts itself.
        mock_db.collection.return_value.where.return_value.get.return_value = [older, newer]

        resp = await client.get("/api/v1/admin/alerts?limit=1")

        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        assert body[0]["title"] == "newer"


class TestResolveAlert:
    async def test_non_admin_returns_403(self, auth_client: AsyncClient) -> None:
        resp = await auth_client.patch(
            f"/api/v1/admin/alerts/{uuid.uuid4()}", json={"resolved": True}
        )
        assert resp.status_code == 403

    async def test_admin_marks_resolved(
        self, client: AsyncClient, mock_db: MagicMock, admin_user
    ) -> None:
        app.dependency_overrides[get_current_user] = lambda: admin_user
        alert_id = str(uuid.uuid4())

        resp = await client.patch(f"/api/v1/admin/alerts/{alert_id}", json={"resolved": True})

        assert resp.status_code == 200
        mock_db.collection.return_value.document.return_value.update.assert_called_once_with(
            {"resolved": True}
        )


class TestRecordOperatorAlert:
    async def test_writes_scrubbed_alert(self, mock_db: MagicMock) -> None:
        await record_operator_alert(
            mock_db,
            title="Your AI service plan has run out of credits. Please contact support.",
            raw_detail="429 for https://generativelanguage.googleapis.com/v1beta/models/x?key=AIzaSyBQRcnBRW_FhWl_q0LTvnPezwCJze8Q1P8: RESOURCE_EXHAUSTED",
            source="build",
            session_id="sess-1",
            project_id="proj-1",
        )

        mock_db.collection.assert_any_call("operator_alerts")
        payload = mock_db.collection.return_value.add.call_args[0][0]
        assert payload["source"] == "build"
        assert payload["session_id"] == "sess-1"
        assert payload["project_id"] == "proj-1"
        assert payload["resolved"] is False
        assert "AIzaSyBQRcnBRW_FhWl_q0LTvnPezwCJze8Q1P8" not in payload["raw_detail"]
        assert "[redacted]" in payload["raw_detail"]
