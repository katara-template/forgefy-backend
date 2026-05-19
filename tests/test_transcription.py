"""Transcription worker and DeepgramClient unit tests.

All Deepgram and Redis I/O is mocked — no live connections required.
"""
from __future__ import annotations

import base64
import json
import threading
import uuid
from unittest.mock import MagicMock, call, patch


# ── TranscriptionSession unit tests ──────────────────────────────────────────


class TestTranscriptionSession:
    def _make_session(self, mock_connection, mock_redis_cls):
        """Construct a TranscriptionSession with fully mocked Deepgram + Redis."""
        from app.transcription.deepgram_client import TranscriptionSession

        session_id = str(uuid.uuid4())
        # mock_connection is the V1SocketClient returned by __enter__
        mock_ctx = MagicMock()
        mock_ctx.__enter__.return_value = mock_connection
        mock_ctx.__exit__.return_value = False

        # Patch DeepgramClient so connect() returns our mock context
        mock_dg_cls = MagicMock()
        mock_dg_cls.return_value.listen.v1.connect.return_value = mock_ctx

        mock_redis_instance = MagicMock()
        mock_redis_cls.return_value = mock_redis_instance

        with (
            patch("app.transcription.deepgram_client.DeepgramClient", mock_dg_cls),
            patch("app.transcription.deepgram_client.sync_redis.from_url", mock_redis_cls),
        ):
            s = TranscriptionSession(
                session_id=session_id,
                api_key="fake",
                model="nova-3",
                redis_url="redis://localhost:6379/0",
            )

        return s, session_id, mock_redis_instance

    def test_connect_registers_message_callback(self):
        conn = MagicMock()
        conn.start_listening = MagicMock()

        with patch("threading.Thread") as mock_thread_cls:
            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread

            session, sid, _ = self._make_session(conn, MagicMock())

        conn.on.assert_called_once()
        mock_thread.start.assert_called_once()

    def test_send_chunk_forwards_bytes(self):
        conn = MagicMock()
        conn.start_listening = MagicMock()

        with patch("threading.Thread"):
            session, sid, _ = self._make_session(conn, MagicMock())

        chunk = b"\x00\x01\x02\x03"
        session.send_chunk(chunk)
        conn.send_media.assert_called_once_with(chunk)

    def test_send_chunk_noop_when_closed(self):
        conn = MagicMock()
        conn.start_listening = MagicMock()

        with patch("threading.Thread"):
            session, sid, _ = self._make_session(conn, MagicMock())

        session._closed = True
        session.send_chunk(b"\x00\x01")
        conn.send_media.assert_not_called()

    def test_close_sends_close_stream(self):
        conn = MagicMock()
        conn.start_listening = MagicMock()

        with patch("threading.Thread"):
            session, sid, redis_mock = self._make_session(conn, MagicMock())

        session.close()

        conn.send_close_stream.assert_called_once()
        redis_mock.close.assert_called_once()
        assert session._closed is True

    def test_close_idempotent(self):
        conn = MagicMock()
        conn.start_listening = MagicMock()

        with patch("threading.Thread"):
            session, sid, _ = self._make_session(conn, MagicMock())

        session.close()
        session.close()  # second call should be a no-op
        conn.send_close_stream.assert_called_once()

    def test_on_message_publishes_final_transcript(self):
        conn = MagicMock()
        conn.start_listening = MagicMock()
        redis_cls = MagicMock()

        with patch("threading.Thread"):
            session, sid, redis_mock = self._make_session(conn, redis_cls)

        # Build a ListenV1Results-like message
        from deepgram.listen import ListenV1Results

        alt = MagicMock()
        alt.transcript = "hello world"
        channel = MagicMock()
        channel.alternatives = [alt]

        msg = MagicMock(spec=ListenV1Results)
        msg.channel = channel
        msg.is_final = True

        session._on_message(msg)

        redis_mock.publish.assert_called_once()
        channel_arg, payload_arg = redis_mock.publish.call_args[0]
        assert channel_arg == f"voxa:session:{sid}"
        data = json.loads(payload_arg)
        assert data["type"] == "transcript"
        assert data["text"] == "hello world"
        assert data["is_final"] is True

    def test_on_message_ignores_empty_transcript(self):
        conn = MagicMock()
        conn.start_listening = MagicMock()
        redis_cls = MagicMock()

        with patch("threading.Thread"):
            session, sid, redis_mock = self._make_session(conn, redis_cls)

        from deepgram.listen import ListenV1Results

        alt = MagicMock()
        alt.transcript = ""
        channel = MagicMock()
        channel.alternatives = [alt]

        msg = MagicMock(spec=ListenV1Results)
        msg.channel = channel
        msg.is_final = False

        session._on_message(msg)
        redis_mock.publish.assert_not_called()

    def test_on_message_ignores_non_results(self):
        conn = MagicMock()
        conn.start_listening = MagicMock()
        redis_cls = MagicMock()

        with patch("threading.Thread"):
            session, sid, redis_mock = self._make_session(conn, redis_cls)

        # Pass a non-ListenV1Results message
        session._on_message(MagicMock())
        redis_mock.publish.assert_not_called()


# ── Celery task unit tests ────────────────────────────────────────────────────


class TestCeleryTasks:
    def test_process_audio_chunk_creates_session_and_sends(self):
        chunk_bytes = b"\xDE\xAD\xBE\xEF"
        chunk_b64 = base64.b64encode(chunk_bytes).decode()
        session_id = str(uuid.uuid4())

        mock_session = MagicMock()

        with patch(
            "app.workers.transcription_worker._get_or_create_session",
            return_value=mock_session,
        ):
            from app.workers.transcription_worker import process_audio_chunk

            process_audio_chunk(session_id, chunk_b64)

        mock_session.send_chunk.assert_called_once_with(chunk_bytes)

    def test_close_transcription_session_removes_and_closes(self):
        session_id = str(uuid.uuid4())
        mock_session = MagicMock()

        with patch.dict(
            "app.workers.transcription_worker._sessions",
            {session_id: mock_session},
            clear=False,
        ):
            from app.workers.transcription_worker import close_transcription_session

            close_transcription_session(session_id)

        mock_session.close.assert_called_once()

    def test_close_transcription_session_noop_for_unknown(self):
        from app.workers.transcription_worker import close_transcription_session

        # Should not raise
        close_transcription_session("nonexistent-session-id")
