"""Connector tests — factory routing, stubs, and MeetConnector unit tests."""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, call, patch

import pytest

from app.db.models.enums import Platform


# ── Factory tests ─────────────────────────────────────────────────────────────


class TestConnectorFactory:
    def test_meet_returns_meet_connector(self):
        from app.connectors.factory import get_connector
        from app.connectors.meet import MeetConnector

        c = get_connector(Platform.MEET)
        assert isinstance(c, MeetConnector)

    def test_zoom_returns_zoom_connector(self):
        from app.connectors.factory import get_connector
        from app.connectors.zoom import ZoomConnector

        c = get_connector(Platform.ZOOM)
        assert isinstance(c, ZoomConnector)

    def test_teams_returns_teams_connector(self):
        from app.connectors.factory import get_connector
        from app.connectors.teams import TeamsConnector

        c = get_connector(Platform.TEAMS)
        assert isinstance(c, TeamsConnector)

    def test_physical_raises_not_implemented(self):
        from app.connectors.factory import get_connector

        with pytest.raises(NotImplementedError):
            get_connector(Platform.PHYSICAL)


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
