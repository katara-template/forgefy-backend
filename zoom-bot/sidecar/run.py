"""Sidecar entrypoint — owns the container's process tree.

Responsibilities, in order:

  1. Bind the PCM socket *before* the bot starts, so no audio is ever produced
     with nowhere to go.
  2. Spawn the C++ Meeting SDK client.
  3. Translate its stdout status stream into orchestrator webhooks.
  4. Stream the PCM it produces to Deepgram and forward transcripts.
  5. On SIGTERM, ask the bot to leave the meeting before exiting.

Run as: python3 -m sidecar.run
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import struct
import sys

from sidecar.backend import BackendClient
from sidecar.config import SidecarConfig
from sidecar.deepgram import DeepgramStream

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format='{"level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
    stream=sys.stderr,
)
logger = logging.getLogger("sidecar")

_HEADER_MAGIC = b"FGFYPCM1"
_HEADER_FORMAT = "<8sII"  # magic, sample_rate, channels
_HEADER_SIZE = struct.calcsize(_HEADER_FORMAT)

_PCM_READ_SIZE = 8192

# How long to wait for the bot to leave the meeting after SIGTERM before
# killing it. Zoom's leave handshake is fast; this is a generous ceiling.
_LEAVE_GRACE_SECONDS = 20


class Sidecar:
    def __init__(self, config: SidecarConfig) -> None:
        self._config = config
        self._backend = BackendClient(
            url=config.webhook_url,
            secret=config.webhook_secret,
            session_id=config.session_id,
        )
        self._deepgram: DeepgramStream | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._server: asyncio.Server | None = None
        self._audio_done = asyncio.Event()
        self._stopping = False

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def run(self) -> int:
        await self._listen()
        await self._spawn_bot()

        status_task = asyncio.create_task(self._read_status())

        exit_code = await self._process.wait()
        logger.info("bot exited with code %s", exit_code)

        # The bot half-closes the socket on the way out; give the audio path a
        # moment to drain the tail of the meeting before tearing down.
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._audio_done.wait(), timeout=10)

        status_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await status_task

        await self._shutdown()
        return exit_code

    async def _listen(self) -> None:
        path = self._config.socket_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # A socket left behind by a crashed predecessor would make bind fail.
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)

        self._server = await asyncio.start_unix_server(self._handle_audio, path=path)
        logger.info("listening for audio on %s", path)

    async def _spawn_bot(self) -> None:
        # stderr is inherited so SDK diagnostics land in `docker logs`; stdout
        # is the machine-readable status channel.
        self._process = await asyncio.create_subprocess_exec(
            self._config.bot_binary,
            stdout=asyncio.subprocess.PIPE,
            stderr=None,
        )
        logger.info("spawned bot pid=%s", self._process.pid)

    async def _shutdown(self) -> None:
        if self._deepgram:
            await self._deepgram.finish()
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        await self._backend.aclose()

    async def terminate(self) -> None:
        """Ask the bot to leave the meeting, then wait for it to exit."""
        if self._stopping:
            return
        self._stopping = True

        if not self._process or self._process.returncode is not None:
            return

        logger.info("signalling bot to leave the meeting")
        with contextlib.suppress(ProcessLookupError):
            self._process.send_signal(signal.SIGTERM)

        try:
            await asyncio.wait_for(self._process.wait(), timeout=_LEAVE_GRACE_SECONDS)
        except TimeoutError:
            logger.warning("bot did not exit in time; killing")
            with contextlib.suppress(ProcessLookupError):
                self._process.kill()

    # ── status channel ───────────────────────────────────────────────────────

    async def _read_status(self) -> None:
        """Forward each status line the bot emits to the orchestrator."""
        assert self._process and self._process.stdout

        async for raw in self._process.stdout:
            line = raw.decode(errors="replace").strip()
            if not line:
                continue

            try:
                event = json.loads(line)
            except ValueError:
                # Anything the SDK prints on stdout that is not ours.
                logger.debug("non-JSON stdout: %s", line[:200])
                continue

            kind = event.get("event")
            if kind == "status":
                status = event.get("status", "")
                detail = event.get("detail", "")
                logger.info("bot status: %s %s", status, detail)
                await self._backend.send_status(status, detail)
            elif kind == "audio_format":
                # Informational — the authoritative format comes from the
                # socket header, which is what actually gates the connection.
                logger.info(
                    "bot reports audio %sHz %sch",
                    event.get("sample_rate"), event.get("channels"),
                )

    # ── audio path ───────────────────────────────────────────────────────────

    async def _handle_audio(self, reader: asyncio.StreamReader, _writer) -> None:
        try:
            header = await reader.readexactly(_HEADER_SIZE)
        except asyncio.IncompleteReadError:
            logger.error("bot closed the audio socket before sending a header")
            self._audio_done.set()
            return

        magic, sample_rate, channels = struct.unpack(_HEADER_FORMAT, header)
        if magic != _HEADER_MAGIC:
            logger.error("bad audio header magic %r — version mismatch?", magic)
            self._audio_done.set()
            return

        logger.info("audio stream open: %dHz %dch", sample_rate, channels)

        self._deepgram = DeepgramStream(
            api_key=self._config.deepgram_api_key,
            model=self._config.deepgram_model,
            language=self._config.language,
            on_transcript=self._on_transcript,
        )
        self._deepgram.start(sample_rate, channels)

        try:
            while True:
                chunk = await reader.read(_PCM_READ_SIZE)
                if not chunk:
                    break
                self._deepgram.send(chunk)
        except (ConnectionResetError, asyncio.IncompleteReadError):
            logger.warning("audio socket reset")
        finally:
            logger.info("audio stream closed")
            await self._deepgram.finish()
            self._audio_done.set()

    async def _on_transcript(self, text: str, is_final: bool, speaker: str) -> None:
        await self._backend.send_transcript(text, is_final=is_final, speaker=speaker)


async def main() -> int:
    config = SidecarConfig.from_env()

    problems = config.validate()
    if problems:
        for problem in problems:
            logger.error("%s", problem)
        return 1

    sidecar = Sidecar(config)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(sidecar.terminate()))

    try:
        return await sidecar.run()
    except Exception:
        logger.exception("sidecar failed")
        # Best effort: the orchestrator must not be left waiting on a session
        # whose container has already died.
        with contextlib.suppress(Exception):
            await sidecar._backend.send_status("error", "sidecar crashed")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
