"""Self-hosted Zoom bot: URL/JWT helpers, provider routing, webhook auth."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from app.connectors.zoom_meeting import build_sdk_jwt, parse_meeting_url
from app.db.models.enums import Platform


def _b64url_decode(segment: str) -> bytes:
    return base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))


class TestParseMeetingUrl:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("https://zoom.us/j/1234567890?pwd=abc123", ("1234567890", "abc123")),
            ("https://acme.zoom.us/w/9876543210?pwd=xyz", ("9876543210", "xyz")),
            ("https://us02web.zoom.us/j/85512345678", ("85512345678", None)),
            ("1234567890", ("1234567890", None)),
            ("123 456 7890", ("1234567890", None)),
            ("123-456-7890", ("1234567890", None)),
        ],
    )
    def test_extracts_id_and_passcode(self, raw: str, expected: tuple) -> None:
        assert parse_meeting_url(raw) == expected

    @pytest.mark.parametrize("raw", ["", "   ", "https://zoom.us/my/vanityroom"])
    def test_returns_blank_id_when_unresolvable(self, raw: str) -> None:
        """Personal-room links carry no numeric ID — callers must reject, not guess."""
        meeting_id, _ = parse_meeting_url(raw)
        assert meeting_id == ""


class TestBuildSdkJwt:
    def test_carries_the_claims_zoom_requires(self) -> None:
        token = build_sdk_jwt("client-id", "client-secret")
        header_b64, payload_b64, _ = token.split(".")

        assert json.loads(_b64url_decode(header_b64)) == {"alg": "HS256", "typ": "JWT"}

        payload = json.loads(_b64url_decode(payload_b64))
        assert payload["appKey"] == "client-id"
        # Zoom requires tokenExp alongside the standard exp.
        assert payload["tokenExp"] == payload["exp"]
        assert payload["exp"] > payload["iat"]

    def test_signature_verifies_against_the_secret(self) -> None:
        token = build_sdk_jwt("client-id", "s3cret")
        signing_input, signature_b64 = token.rsplit(".", 1)

        expected = hmac.new(
            b"s3cret", signing_input.encode(), hashlib.sha256
        ).digest()
        assert _b64url_decode(signature_b64) == expected

    def test_is_not_padded(self) -> None:
        """JWS forbids base64 padding; a '=' here breaks Zoom's parser."""
        assert "=" not in build_sdk_jwt("client-id", "secret")


class TestProviderRouting:
    """ZOOM_BOT_PROVIDER decides the Zoom bot; Meet and Teams never change."""

    @staticmethod
    def _settings(provider: str) -> MagicMock:
        return MagicMock(
            ZOOM_BOT_PROVIDER=provider,
            ZOOM_SDK_CLIENT_ID="cid",
            ZOOM_SDK_CLIENT_SECRET="csecret",
            ZOOM_BOT_IMAGE="forgefy-zoom-bot:latest",
            ZOOM_BOT_NETWORK="",
            ZOOM_BOT_CALLBACK_URL="http://api:5000",
            ZOOM_BOT_DISPLAY_NAME="Forgefy Notetaker",
            ZOOM_BOT_REQUIRE_HOST_CONSENT=True,
            ZOOM_BOT_LEAVE_AFTER_SILENCE_SECS=120,
            DEEPGRAM_API_KEY="dg-key",
            DEEPGRAM_MODEL="nova-3",
            RECALL_API_KEY="recall-key",
            RECALL_REGION="us-east-1",
            RECALL_BOT_AVATAR_PATH="assets/bot_avatar.jpg",
            REDIS_URL="redis://localhost:6379/0",
            PUBLIC_API_BASE_URL="https://api.example.com",
        )

    def _connector_for(self, provider: str, platform: Platform):
        from app.connectors.factory import get_connector

        with patch("app.config.get_settings", return_value=self._settings(provider)):
            return get_connector(platform)

    def test_self_hosted_routes_zoom_to_the_container_bot(self) -> None:
        from app.connectors.zoom_selfhosted import ZoomSelfHostedConnector

        connector = self._connector_for("self_hosted", Platform.ZOOM)
        assert isinstance(connector, ZoomSelfHostedConnector)

    def test_recall_keeps_zoom_on_the_cloud_bot(self) -> None:
        from app.connectors.recall import RecallConnector

        connector = self._connector_for("recall", Platform.ZOOM)
        assert isinstance(connector, RecallConnector)

    @pytest.mark.parametrize("platform", [Platform.MEET, Platform.TEAMS])
    def test_other_platforms_ignore_the_flag(self, platform: Platform) -> None:
        from app.connectors.recall import RecallConnector

        connector = self._connector_for("self_hosted", platform)
        assert isinstance(connector, RecallConnector)

    def test_missing_sdk_credentials_fail_loudly(self) -> None:
        from app.connectors.zoom_selfhosted import ZoomSelfHostedConnector

        settings = self._settings("self_hosted")
        settings.ZOOM_SDK_CLIENT_SECRET = ""

        with pytest.raises(RuntimeError, match="ZOOM_SDK_CLIENT_SECRET"):
            ZoomSelfHostedConnector.from_settings(settings)

    def test_missing_deepgram_key_fails_loudly(self) -> None:
        from app.connectors.zoom_selfhosted import ZoomSelfHostedConnector

        settings = self._settings("self_hosted")
        settings.DEEPGRAM_API_KEY = ""

        with pytest.raises(RuntimeError, match="DEEPGRAM_API_KEY"):
            ZoomSelfHostedConnector.from_settings(settings)


class TestWebhookVerification:
    """A bot may only post events for the one session it was spawned for."""

    SECRET = "per-session-secret"
    SESSION = "session-123"

    def _verify(self, body: bytes, timestamp: str, signature: str, secret=SECRET):
        from app.api.v1.zoom_bot import _verify

        with patch("app.connectors.zoom_selfhosted.lookup_secret", return_value=secret):
            _verify(self.SESSION, body, timestamp, signature, "redis://localhost")

    def _signed(self, payload: dict, secret: str = SECRET, at: int | None = None):
        body = json.dumps(payload, separators=(",", ":")).encode()
        timestamp = str(at if at is not None else int(time.time()))
        signature = hmac.new(
            secret.encode(), f"{timestamp}.".encode() + body, hashlib.sha256
        ).hexdigest()
        return body, timestamp, signature

    def test_accepts_a_correctly_signed_request(self) -> None:
        body, ts, sig = self._signed({"type": "status", "status": "joining"})
        self._verify(body, ts, sig)  # must not raise

    def test_rejects_a_forged_signature(self) -> None:
        body, ts, _ = self._signed({"type": "status"})
        _, _, wrong_sig = self._signed({"type": "status"}, secret="attacker-secret")

        with pytest.raises(HTTPException) as exc:
            self._verify(body, ts, wrong_sig)
        assert exc.value.status_code == 401

    def test_rejects_a_tampered_body(self) -> None:
        body, ts, sig = self._signed({"type": "status", "status": "joining"})
        tampered = body.replace(b"joining", b"hijackd")  # same length, new content

        with pytest.raises(HTTPException) as exc:
            self._verify(tampered, ts, sig)
        assert exc.value.status_code == 401

    def test_rejects_a_replayed_request(self) -> None:
        stale = int(time.time()) - 3600
        body, ts, sig = self._signed({"type": "status"}, at=stale)

        with pytest.raises(HTTPException, match="Stale"):
            self._verify(body, ts, sig)

    def test_rejects_when_no_bot_is_running_for_the_session(self) -> None:
        """The secret is deleted at teardown, which revokes the container."""
        body, ts, sig = self._signed({"type": "status"})

        with pytest.raises(HTTPException, match="Unknown session"):
            self._verify(body, ts, sig, secret=None)

    def test_rejects_missing_signature_headers(self) -> None:
        body, _, _ = self._signed({"type": "status"})

        with pytest.raises(HTTPException, match="Missing signature"):
            self._verify(body, None, None)
