"""Self-hosted Zoom connector — runs our own Meeting SDK bot in a container.

NOT YET WIRED INTO THE BACKEND. This module lives outside `app/` on purpose:
the bot is being built and verified standalone first. Nothing in the running
application imports it, and enabling it is a deliberate, separate step — see
the "Linking into the backend" section of ../README.md.

The alternative to the Recall.ai path: instead of paying per bot-hour for a
cloud bot, we launch one `forgefy-zoom-bot` container per meeting. It joins via
Zoom's Linux Meeting SDK, captures raw audio, and reports transcripts back
through /api/v1/webhooks/zoom-bot — the same shape Recall reports in, so
downstream session handling can be identical once linked.

It satisfies the existing `app.connectors.base.MeetingConnector` protocol
(join/leave), so linking is a routing change rather than a rewrite.

Trade-offs versus Recall, deliberately accepted here:
  • Zoom only. Meet and Teams keep using Recall.
  • Requires a Meeting SDK app with raw-data privileges on a paid Zoom account.
  • Raw audio needs local recording privilege, so the host must approve the bot
    (or pre-authorize it with a join token) before any audio is captured.

See zoom-bot/README.md for the container build and the Zoom account setup.
"""
from __future__ import annotations

import logging
import secrets

import redis as sync_redis

from app.connectors.zoom_meeting import build_sdk_jwt, parse_meeting_url

logger = logging.getLogger(__name__)

_CONTAINER_PREFIX = "forgefy-zoom-bot-"

# Redis keys, mirroring the recall:* namespace so operational tooling that
# inspects one works on the other.
_SESSION_KEY = "zoombot:session:{session_id}"
_CONTAINER_KEY = "zoombot:container:{container_id}"
_SECRET_KEY = "zoombot:secret:{session_id}"

_MAPPING_TTL_SECONDS = 86_400


class ZoomSelfHostedConnector:
    """Launches and stops a per-meeting Zoom bot container."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        image: str,
        redis_url: str,
        webhook_base_url: str,
        deepgram_api_key: str,
        deepgram_model: str,
        display_name: str,
        require_host_consent: bool,
        docker_network: str | None = None,
        leave_after_silence_secs: int = 120,
        user_id: str | None = None,
        settings=None,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._image = image
        self._redis_url = redis_url
        self._webhook_url = webhook_base_url.rstrip("/") + "/api/v1/webhooks/zoom-bot"
        self._deepgram_api_key = deepgram_api_key
        self._deepgram_model = deepgram_model
        self._display_name = display_name
        self._require_host_consent = require_host_consent
        self._docker_network = docker_network
        self._leave_after_silence_secs = leave_after_silence_secs
        # Whose Zoom grant to mint per-meeting tokens from. Without it the bot
        # can only join meetings on our own Zoom account.
        self._user_id = user_id
        self._settings = settings
        self._container_id: str | None = None

    @classmethod
    def from_settings(cls, settings, user_id: str | None = None) -> ZoomSelfHostedConnector:
        """Build from app settings, failing loudly on incomplete configuration.

        Kept here rather than in the connector factory so this path owns its
        own prerequisites and shares nothing with the Recall connector.
        """
        missing = [
            name for name in ("ZOOM_SDK_CLIENT_ID", "ZOOM_SDK_CLIENT_SECRET")
            if not getattr(settings, name, "")
        ]
        if missing:
            raise RuntimeError(
                f"{', '.join(missing)} not configured. Create a Meeting SDK app at "
                "marketplace.zoom.us and set the credentials in the backend .env "
                "to run self-hosted Zoom bots."
            )
        if not settings.DEEPGRAM_API_KEY:
            raise RuntimeError(
                "DEEPGRAM_API_KEY is not configured — the self-hosted bot transcribes "
                "in-container and cannot start without it."
            )

        return cls(
            client_id=settings.ZOOM_SDK_CLIENT_ID,
            client_secret=settings.ZOOM_SDK_CLIENT_SECRET,
            image=settings.ZOOM_BOT_IMAGE,
            redis_url=settings.REDIS_URL,
            # Internal by default: the bot shares a Docker network with the API,
            # so unlike Recall it needs no publicly reachable callback URL.
            webhook_base_url=settings.ZOOM_BOT_CALLBACK_URL,
            deepgram_api_key=settings.DEEPGRAM_API_KEY,
            deepgram_model=settings.DEEPGRAM_MODEL,
            display_name=settings.ZOOM_BOT_DISPLAY_NAME,
            require_host_consent=settings.ZOOM_BOT_REQUIRE_HOST_CONSENT,
            docker_network=settings.ZOOM_BOT_NETWORK or None,
            leave_after_silence_secs=settings.ZOOM_BOT_LEAVE_AFTER_SILENCE_SECS,
            user_id=user_id,
            settings=settings,
        )

    # ── Public interface (MeetingConnector) ──────────────────────────────────

    def join(self, meeting_url: str, session_id: str) -> None:
        """Start a bot container for meeting_url; store the mapping in Redis."""
        meeting_number, passcode = parse_meeting_url(meeting_url)
        if not meeting_number:
            raise ValueError(f"Could not extract a meeting ID from {meeting_url!r}")

        # Per-session secret: a leaked webhook secret compromises one meeting,
        # not every bot we have ever run.
        webhook_secret = secrets.token_urlsafe(32)
        self._store_secret(session_id, webhook_secret)

        env = {
            "ZOOM_SDK_JWT": build_sdk_jwt(self._client_id, self._client_secret),
            "ZOOM_MEETING_NUMBER": meeting_number,
            "ZOOM_MEETING_PASSWORD": passcode or "",
            "ZOOM_DISPLAY_NAME": self._display_name,
            "FORGEFY_REQUIRE_HOST_CONSENT": "true" if self._require_host_consent else "false",
            "FORGEFY_LEAVE_AFTER_SILENCE_SECS": str(self._leave_after_silence_secs),
            "FORGEFY_SESSION_ID": session_id,
            "FORGEFY_WEBHOOK_URL": self._webhook_url,
            "FORGEFY_WEBHOOK_SECRET": webhook_secret,
            "DEEPGRAM_API_KEY": self._deepgram_api_key,
            "DEEPGRAM_MODEL": self._deepgram_model,
        }

        # Minted here, immediately before launch, rather than at dispatch time:
        # the local recording token lives for only ~120 seconds, so a queue
        # backlog between dispatch and launch would deliver an expired one.
        env.update(self._mint_meeting_tokens(meeting_number))

        import docker

        client = docker.from_env()
        container = client.containers.run(
            self._image,
            detach=True,
            environment=env,
            name=f"{_CONTAINER_PREFIX}{session_id}",
            network=self._docker_network,
            # Bots are disposable; never restart one into a meeting that has
            # already moved on.
            restart_policy={"Name": "no"},
            mem_limit="1g",
            labels={
                "forgefy.role": "zoom-bot",
                "forgefy.session_id": session_id,
            },
        )

        self._container_id = container.id
        self._store_mapping(container.id, session_id)
        logger.info(
            "Zoom bot container started session=%s container=%s meeting=%s",
            session_id, container.short_id, meeting_number,
        )

    def leave(self) -> None:
        """Stop the container, giving the bot time to leave the meeting first."""
        if not self._container_id:
            return
        stop_container(self._container_id)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _mint_meeting_tokens(self, meeting_number: str) -> dict[str, str]:
        """Return the per-meeting Zoom tokens the bot should join with.

        Degrades rather than fails. A bot with no OBF token can still join
        meetings hosted on our own Zoom account, and one with no local
        recording token simply asks the host for permission in-meeting — both
        are worth attempting, and both surface clearly in the bot's logs.
        """
        if not (self._user_id and self._settings):
            logger.info("No linked user for this session — joining without OBF token")
            return {}

        import asyncio

        from app.integrations.zoom_oauth import ZoomAuthError, ZoomNotLinked

        async def _resolve() -> dict[str, str]:
            from app.db.firebase import get_firestore_client
            from app.integrations import zoom_oauth

            db = get_firestore_client()
            access_token = await zoom_oauth.get_access_token(
                db, self._user_id, self._settings
            )

            tokens: dict[str, str] = {}

            # Required since 2026-03-02 for meetings on other Zoom accounts.
            obf = await zoom_oauth.get_obf_token(access_token)
            if obf:
                tokens["ZOOM_ON_BEHALF_TOKEN"] = obf

            # Optional: skips the in-meeting consent prompt. Only requested
            # when the operator has chosen not to prompt, so the default
            # remains "ask the host every time".
            if not self._require_host_consent:
                join_token = await zoom_oauth.get_local_recording_token(
                    access_token, meeting_number
                )
                if join_token:
                    tokens["ZOOM_JOIN_TOKEN"] = join_token

            return tokens

        # Matches the sync-Celery-task convention used across app/workers/.
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_resolve())
        except ZoomNotLinked:
            logger.warning(
                "User %s has not linked Zoom — the bot can only join meetings "
                "hosted on our own account", self._user_id,
            )
            return {}
        except ZoomAuthError as exc:
            logger.error("Zoom token minting failed for user %s: %s", self._user_id, exc)
            return {}
        except Exception as exc:
            logger.error("Unexpected error minting Zoom tokens: %s", exc, exc_info=True)
            return {}
        finally:
            loop.close()

    def _store_secret(self, session_id: str, secret: str) -> None:
        _redis_set(self._redis_url, {_SECRET_KEY.format(session_id=session_id): secret})

    def _store_mapping(self, container_id: str, session_id: str) -> None:
        _redis_set(
            self._redis_url,
            {
                _SESSION_KEY.format(session_id=session_id): container_id,
                _CONTAINER_KEY.format(container_id=container_id): session_id,
            },
        )


# ── Module-level helpers (used by workers and the webhook handler) ───────────

def stop_container(container_id: str, timeout: int = 30) -> None:
    """Stop a bot container — safe to call if it has already exited.

    SIGTERM reaches the sidecar, which asks the bot to leave the meeting
    cleanly; the timeout is the ceiling before Docker escalates to SIGKILL.
    """
    try:
        import docker
        from docker.errors import NotFound

        client = docker.from_env()
        try:
            container = client.containers.get(container_id)
        except NotFound:
            logger.info("Zoom bot container already gone id=%s", container_id[:12])
            return

        container.stop(timeout=timeout)
        container.remove(force=True)
        logger.info("Zoom bot container stopped id=%s", container_id[:12])
    except Exception as exc:
        logger.warning("Failed to stop Zoom bot container %s: %s", container_id[:12], exc)


def lookup_secret(session_id: str, redis_url: str) -> str | None:
    """Return the webhook secret issued to this session's bot, if any."""
    r = sync_redis.from_url(redis_url, decode_responses=True)
    try:
        return r.get(_SECRET_KEY.format(session_id=session_id))
    finally:
        r.close()


def lookup_container(session_id: str, redis_url: str) -> str | None:
    r = sync_redis.from_url(redis_url, decode_responses=True)
    try:
        return r.get(_SESSION_KEY.format(session_id=session_id))
    finally:
        r.close()


def clear_mapping(session_id: str, redis_url: str) -> None:
    r = sync_redis.from_url(redis_url, decode_responses=True)
    try:
        container_id = r.get(_SESSION_KEY.format(session_id=session_id))
        keys = [
            _SESSION_KEY.format(session_id=session_id),
            _SECRET_KEY.format(session_id=session_id),
        ]
        if container_id:
            keys.append(_CONTAINER_KEY.format(container_id=container_id))
        r.delete(*keys)
    finally:
        r.close()


def _redis_set(redis_url: str, pairs: dict[str, str]) -> None:
    r = sync_redis.from_url(redis_url, decode_responses=True)
    try:
        for key, value in pairs.items():
            r.set(key, value, ex=_MAPPING_TTL_SECONDS)
    finally:
        r.close()
