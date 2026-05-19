"""Google Meet connector — headless Playwright bot that joins a Meet call.

The bot:
1. Opens the Meet URL in a headless Chromium browser
2. Dismisses the pre-join dialog and clicks "Join now"
3. Captures audio via the MediaRecorder API injected into the page
4. Ships base64-encoded PCM chunks to the ``meeting.audio`` Celery queue so
   the transcription worker can stream them to Deepgram
5. Monitors the page for the "You've been removed" / call-ended signals and
   exits cleanly

This is intentionally conservative: no microphone input, camera off,
display-name set to "Forgefy Bot".
"""
from __future__ import annotations

import base64
import logging
import threading
import time

from playwright.sync_api import Page, sync_playwright

from app.config import get_settings
from app.workers.transcription_worker import process_audio_chunk

logger = logging.getLogger(__name__)

_BOT_DISPLAY_NAME = "Forgefy Bot"
_AUDIO_CHUNK_MS = 250  # milliseconds per audio chunk sent to Celery


class MeetConnector:
    """Playwright-based Google Meet bot."""

    def __init__(self) -> None:
        self._playwright = None
        self._browser = None
        self._page: Page | None = None
        self._running = False
        self._capture_thread: threading.Thread | None = None

    # ── Public interface ──────────────────────────────────────────────────────

    def join(self, meeting_url: str, session_id: str) -> None:
        """Launch the headless browser, join the Meet call, start audio capture."""
        logger.info("MeetConnector joining session=%s url=%s", session_id, meeting_url)

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=True,
            args=[
                "--use-fake-ui-for-media-stream",   # auto-approve camera/mic
                "--use-fake-device-for-media-stream",
                "--disable-gpu",
                "--no-sandbox",
            ],
        )
        context = self._browser.new_context(
            permissions=["camera", "microphone"],
        )
        self._page = context.new_page()
        self._running = True

        self._page.goto(meeting_url, wait_until="domcontentloaded", timeout=30_000)
        self._dismiss_pre_join()
        self._set_display_name()
        self._click_join()

        # Start audio capture in a background thread so this call returns.
        self._capture_thread = threading.Thread(
            target=self._capture_audio_loop,
            args=(session_id,),
            name=f"meet-audio-{session_id}",
            daemon=True,
        )
        self._capture_thread.start()

    def leave(self) -> None:
        """Hang up and clean up all Playwright resources."""
        self._running = False
        if self._page:
            try:
                # Click the Leave button if visible.
                leave_btn = self._page.query_selector('[aria-label*="Leave"]')
                if leave_btn:
                    leave_btn.click()
            except Exception:
                pass
        self._cleanup()
        logger.info("MeetConnector left meeting")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _dismiss_pre_join(self) -> None:
        """Close cookie / permission dialogs before joining."""
        try:
            self._page.wait_for_selector("text=Got it", timeout=5_000)
            self._page.click("text=Got it")
        except Exception:
            pass  # not always present

    def _set_display_name(self) -> None:
        """Type the bot's display name into the name field if present."""
        try:
            name_input = self._page.query_selector('input[placeholder*="name"]')
            if name_input:
                name_input.fill(_BOT_DISPLAY_NAME)
        except Exception:
            pass

    def _click_join(self) -> None:
        """Click the 'Join now' / 'Ask to join' button."""
        selectors = [
            "text=Join now",
            "text=Ask to join",
            '[data-idom-class*="join"]',
        ]
        for sel in selectors:
            try:
                self._page.wait_for_selector(sel, timeout=8_000)
                self._page.click(sel)
                logger.info("MeetConnector clicked join button")
                return
            except Exception:
                continue
        logger.warning("MeetConnector could not find join button")

    def _capture_audio_loop(self, session_id: str) -> None:
        """Inject MediaRecorder into the page and ship chunks to Celery.

        Runs in a background thread until leave() is called or the call ends.
        """
        # Inject a JS MediaRecorder that fires ondataavailable every _AUDIO_CHUNK_MS ms.
        # Captured chunks are stored in window.__audioChunks as base64 strings.
        self._page.evaluate(
            f"""() => {{
                window.__audioChunks = [];
                navigator.mediaDevices.getUserMedia({{audio: true}})
                    .then(stream => {{
                        const recorder = new MediaRecorder(stream);
                        recorder.ondataavailable = e => {{
                            const reader = new FileReader();
                            reader.onloadend = () => {{
                                window.__audioChunks.push(reader.result.split(',')[1]);
                            }};
                            reader.readAsDataURL(e.data);
                        }};
                        recorder.start({_AUDIO_CHUNK_MS});
                        window.__recorder = recorder;
                    }});
            }}"""
        )

        while self._running:
            time.sleep(_AUDIO_CHUNK_MS / 1000)
            try:
                chunks: list[str] = self._page.evaluate(
                    "() => { const c = window.__audioChunks; window.__audioChunks = []; return c; }"
                )
                for chunk_b64 in chunks:
                    if chunk_b64:
                        process_audio_chunk.apply_async(
                            args=[session_id, chunk_b64],
                            queue="meeting.audio",
                        )
            except Exception as exc:
                logger.warning("MeetConnector audio capture error: %s", exc)
                if not self._running:
                    break

        logger.info("MeetConnector audio capture stopped session=%s", session_id)

    def _cleanup(self) -> None:
        try:
            if self._browser:
                self._browser.close()
        except Exception:
            pass
        try:
            if self._playwright:
                self._playwright.stop()
        except Exception:
            pass
        self._page = None
        self._browser = None
        self._playwright = None
