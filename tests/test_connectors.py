"""Connector tests — factory routing, stubs, and MeetConnector unit tests."""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.db.models.enums import Platform

# ── Factory tests ─────────────────────────────────────────────────────────────


def _recall_settings(**overrides) -> MagicMock:
    """Mock Settings with valid Recall.ai config, unless overridden."""
    defaults = dict(
        RECALL_API_KEY="fake-recall-key",
        RECALL_REGION="us-east-1",
        REDIS_URL="redis://localhost:6379/0",
        PUBLIC_API_BASE_URL="https://api.example.com",
    )
    defaults.update(overrides)
    return MagicMock(**defaults)


class TestConnectorFactory:
    """MEET/ZOOM/TEAMS all route through Recall.ai's cloud bot service —
    the per-platform MeetConnector/ZoomConnector/TeamsConnector classes are
    legacy, no longer wired into get_connector() (see app/connectors/factory.py).
    """

    def test_meet_returns_recall_connector(self):
        from app.connectors.factory import get_connector
        from app.connectors.recall import RecallConnector

        with patch("app.config.get_settings", return_value=_recall_settings()):
            c = get_connector(Platform.MEET)
        assert isinstance(c, RecallConnector)

    def test_zoom_returns_recall_connector(self):
        from app.connectors.factory import get_connector
        from app.connectors.recall import RecallConnector

        with patch("app.config.get_settings", return_value=_recall_settings()):
            c = get_connector(Platform.ZOOM)
        assert isinstance(c, RecallConnector)

    def test_teams_returns_recall_connector(self):
        from app.connectors.factory import get_connector
        from app.connectors.recall import RecallConnector

        with patch("app.config.get_settings", return_value=_recall_settings()):
            c = get_connector(Platform.TEAMS)
        assert isinstance(c, RecallConnector)

    def test_physical_raises_not_implemented(self):
        from app.connectors.factory import get_connector

        with pytest.raises(NotImplementedError):
            get_connector(Platform.PHYSICAL)

    def test_missing_recall_api_key_raises(self):
        from app.connectors.factory import get_connector

        with patch(
            "app.config.get_settings", return_value=_recall_settings(RECALL_API_KEY="")
        ), pytest.raises(RuntimeError, match="RECALL_API_KEY"):
            get_connector(Platform.MEET)

    def test_missing_public_api_base_url_raises(self):
        from app.connectors.factory import get_connector

        with patch(
            "app.config.get_settings",
            return_value=_recall_settings(PUBLIC_API_BASE_URL=""),
        ), pytest.raises(RuntimeError, match="PUBLIC_API_BASE_URL"):
            get_connector(Platform.MEET)


# ── Stub connector tests ──────────────────────────────────────────────────────


class TestZoomConnector:
    def test_join_raises_not_implemented(self):
        from app.connectors.zoom import ZoomConnector

        with pytest.raises(NotImplementedError):
            ZoomConnector().join("https://zoom.us/j/123", "session-id")

    def test_leave_raises_not_implemented(self):
        from app.connectors.zoom import ZoomConnector

        with pytest.raises(NotImplementedError):
            ZoomConnector().leave()


class TestTeamsConnector:
    def test_join_raises_not_implemented(self):
        from app.connectors.teams import TeamsConnector

        with pytest.raises(NotImplementedError):
            TeamsConnector().join("https://teams.microsoft.com/meet/123", "session-id")

    def test_leave_raises_not_implemented(self):
        from app.connectors.teams import TeamsConnector

        with pytest.raises(NotImplementedError):
            TeamsConnector().leave()


# ── MeetConnector unit tests ──────────────────────────────────────────────────


class TestMeetConnector:
    def _mock_playwright(self):
        """Return a hierarchy of mocks that mimics sync_playwright()."""
        mock_pw = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()

        mock_pw.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.evaluate.return_value = []   # no audio chunks by default
        mock_page.query_selector.return_value = None  # no buttons found

        return mock_pw, mock_page

    def test_join_launches_browser_and_navigates(self):
        from app.connectors.meet import MeetConnector

        mock_pw, mock_page = self._mock_playwright()
        session_id = str(uuid.uuid4())
        url = "https://meet.google.com/abc-def-ghi"

        with (
            patch("app.connectors.meet.sync_playwright") as mock_sp,
            patch("app.connectors.meet.threading.Thread") as mock_thread_cls,
        ):
            mock_sp.return_value.__enter__ = MagicMock(return_value=mock_pw)
            mock_sp.return_value.start = MagicMock(return_value=mock_pw)
            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread

            connector = MeetConnector()
            connector.join(url, session_id)

        mock_page.goto.assert_called_once_with(url, wait_until="domcontentloaded", timeout=30_000)
        mock_thread.start.assert_called_once()
        assert connector._running is True

    def test_leave_stops_capture_and_cleans_up(self):
        from app.connectors.meet import MeetConnector

        mock_pw, mock_page = self._mock_playwright()

        with (
            patch("app.connectors.meet.sync_playwright") as mock_sp,
            patch("app.connectors.meet.threading.Thread"),
        ):
            mock_sp.return_value.start = MagicMock(return_value=mock_pw)
            connector = MeetConnector()
            connector.join("https://meet.google.com/abc", "sid")
            connector.leave()

        assert connector._running is False
        assert connector._page is None
        assert connector._browser is None

    def test_capture_audio_loop_enqueues_chunks(self):
        from app.connectors.meet import MeetConnector

        mock_pw, mock_page = self._mock_playwright()
        session_id = str(uuid.uuid4())
        chunk_b64 = "AQIDBA=="  # base64 for 4 bytes

        # Simulate one batch of chunks then stop
        call_count = 0

        def fake_evaluate(script):
            nonlocal call_count
            if "audioChunks" in script and "recorder" not in script:
                call_count += 1
                if call_count == 1:
                    return [chunk_b64]
            return []

        mock_page.evaluate.side_effect = fake_evaluate

        mock_task = MagicMock()

        with patch(
            "app.connectors.meet.process_audio_chunk",
            mock_task,
        ):
            connector = MeetConnector()
            connector._page = mock_page
            connector._running = True

            # Run one iteration then stop
            import threading

            def _stop_after_one():
                import time
                time.sleep(0.05)
                connector._running = False

            stopper = threading.Thread(target=_stop_after_one, daemon=True)
            stopper.start()
            connector._capture_audio_loop(session_id)

        mock_task.apply_async.assert_called_with(
            args=[session_id, chunk_b64],
            queue="meeting.audio",
        )


# ── Bot avatar tests ──────────────────────────────────────────────────────────


class TestBotAvatar:
    def test_missing_file_returns_none(self, tmp_path):
        from app.connectors.recall import _load_avatar_b64

        assert _load_avatar_b64(str(tmp_path / "nope.jpg")) is None

    def test_empty_path_returns_none(self):
        from app.connectors.recall import _load_avatar_b64

        assert _load_avatar_b64(None) is None
        assert _load_avatar_b64("") is None

    def test_non_jpeg_rejected(self, tmp_path):
        from app.connectors.recall import _load_avatar_b64

        png = tmp_path / "avatar.png"
        png.write_bytes(b"\x89PNG fake")
        assert _load_avatar_b64(str(png)) is None

    def test_oversized_file_rejected(self, tmp_path):
        from app.connectors.recall import _MAX_AVATAR_BYTES, _load_avatar_b64

        big = tmp_path / "avatar.jpg"
        big.write_bytes(b"x" * (_MAX_AVATAR_BYTES + 1))
        assert _load_avatar_b64(str(big)) is None

    def test_valid_jpeg_returns_base64(self, tmp_path):
        import base64

        from app.connectors.recall import _load_avatar_b64

        jpg = tmp_path / "avatar.jpg"
        jpg.write_bytes(b"\xff\xd8\xff fake jpeg")
        result = _load_avatar_b64(str(jpg))
        assert result == base64.b64encode(b"\xff\xd8\xff fake jpeg").decode()

    def _join_payload(self, avatar_path: str | None) -> dict:
        """Run RecallConnector.join with mocked HTTP/Redis; return the bot payload."""
        from app.connectors.recall import RecallConnector

        resp = httpx.Response(
            200,
            request=httpx.Request("POST", "https://x/bot/"),
            json={"id": "bot-1"},
        )
        connector = RecallConnector(
            api_key="fake-key",
            region="us-east-1",
            redis_url="redis://test",
            webhook_base_url="https://api.example.com",
            avatar_path=avatar_path,
        )
        with (
            patch("app.connectors.recall.httpx.Client") as mock_client_cls,
            patch.object(connector, "_store_mapping"),
        ):
            mock_client = mock_client_cls.return_value.__enter__.return_value
            mock_client.post.return_value = resp
            connector.join("https://meet.google.com/abc", "session-1")
            return mock_client.post.call_args.kwargs["json"]

    def test_join_includes_avatar_when_file_exists(self, tmp_path):
        import base64

        jpg = tmp_path / "avatar.jpg"
        jpg.write_bytes(b"\xff\xd8\xff fake jpeg")

        payload = self._join_payload(str(jpg))

        expected_b64 = base64.b64encode(b"\xff\xd8\xff fake jpeg").decode()
        video_out = payload["automatic_video_output"]
        assert video_out["in_call_recording"] == {"kind": "jpeg", "b64_data": expected_b64}
        assert video_out["in_call_not_recording"] == {"kind": "jpeg", "b64_data": expected_b64}

    def test_join_omits_avatar_when_not_configured(self):
        payload = self._join_payload(None)
        assert "automatic_video_output" not in payload


# ── delete_media tests ────────────────────────────────────────────────────────


class TestDeleteMedia:
    _BASE = "https://us-east-1.recall.ai/api/v1"

    def _run(self, status_code: int) -> MagicMock:
        """Call delete_media against a mocked httpx client returning status_code."""
        from app.connectors.recall import delete_media

        resp = httpx.Response(
            status_code,
            request=httpx.Request("POST", f"{self._BASE}/bot/bot-1/delete_media/"),
        )
        with patch("app.connectors.recall.httpx.Client") as mock_client_cls:
            mock_client = mock_client_cls.return_value.__enter__.return_value
            mock_client.post.return_value = resp
            delete_media("bot-1", self._BASE, "fake-key")
            return mock_client

    def test_deletes_media_with_auth(self):
        client = self._run(200)
        client.post.assert_called_once_with(
            f"{self._BASE}/bot/bot-1/delete_media/",
            headers={"Authorization": "Token fake-key"},
        )

    def test_404_means_already_gone(self):
        self._run(404)  # must not raise — bot/media already deleted

    def test_error_raises_so_task_can_retry(self):
        # e.g. media still processing shortly after call end
        with pytest.raises(httpx.HTTPStatusError):
            self._run(409)


# ── Connector Celery worker tests ─────────────────────────────────────────────


class TestConnectorWorker:
    def test_dispatch_meet_connector_calls_join(self):
        from app.workers.connector_worker import dispatch_connector

        session_id = str(uuid.uuid4())
        mock_connector = MagicMock()

        with patch("app.workers.connector_worker.get_connector", return_value=mock_connector):
            dispatch_connector(session_id, "meet", "https://meet.google.com/abc")

        mock_connector.join.assert_called_once_with("https://meet.google.com/abc", session_id)

    def test_dispatch_unknown_platform_logs_and_returns(self):
        from app.workers.connector_worker import dispatch_connector

        # Should not raise
        dispatch_connector(str(uuid.uuid4()), "discord", None)

    def test_dispatch_not_implemented_logs_warning(self):
        from app.workers.connector_worker import dispatch_connector

        mock_connector = MagicMock()
        mock_connector.join.side_effect = NotImplementedError("stub")

        with patch("app.workers.connector_worker.get_connector", return_value=mock_connector):
            # Should not raise
            dispatch_connector(str(uuid.uuid4()), "zoom", "https://zoom.us/j/123")


class TestRecallDeleteMediaTask:
    def test_skips_when_api_key_missing(self):
        from app.workers.connector_worker import recall_delete_media

        settings = MagicMock(RECALL_API_KEY="")
        with (
            patch("app.config.get_settings", return_value=settings),
            patch("app.connectors.recall.delete_media") as mock_delete,
        ):
            recall_delete_media("bot-1")

        mock_delete.assert_not_called()

    def test_deletes_media_via_connector(self):
        from app.workers.connector_worker import recall_delete_media

        settings = MagicMock(RECALL_API_KEY="fake-key", RECALL_REGION="us-east-1")
        with (
            patch("app.config.get_settings", return_value=settings),
            patch("app.connectors.recall.delete_media") as mock_delete,
        ):
            recall_delete_media("bot-1")

        mock_delete.assert_called_once_with(
            bot_id="bot-1",
            base_url="https://us-east-1.recall.ai/api/v1",
            api_key="fake-key",
        )

    def test_remove_bot_enqueues_media_purge(self):
        from app.workers.connector_worker import (
            DELETE_MEDIA_COUNTDOWN_SECONDS,
            recall_remove_bot,
        )

        settings = MagicMock(
            RECALL_API_KEY="fake-key",
            RECALL_REGION="us-east-1",
            REDIS_URL="redis://test",
        )
        redis_mock = MagicMock()
        redis_mock.get.return_value = "bot-1"

        with (
            patch("app.config.get_settings", return_value=settings),
            patch("app.workers.connector_worker.sync_redis.from_url", return_value=redis_mock),
            patch("app.connectors.recall.remove_bot") as mock_remove,
            patch("app.workers.connector_worker.recall_delete_media") as mock_delete,
        ):
            recall_remove_bot("session-1")

        mock_remove.assert_called_once()
        mock_delete.apply_async.assert_called_once_with(
            args=["bot-1"],
            countdown=DELETE_MEDIA_COUNTDOWN_SECONDS,
            queue="meeting.audio",
        )
