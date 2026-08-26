"""Zoom OAuth: install flow, refresh-token rotation, and per-meeting tokens."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest

from app.integrations import zoom_oauth
from app.integrations.zoom_oauth import ZoomAuthError, ZoomNotLinked


def _settings(**overrides) -> MagicMock:
    defaults = dict(
        ZOOM_OAUTH_CLIENT_ID="",
        ZOOM_OAUTH_CLIENT_SECRET="",
        ZOOM_SDK_CLIENT_ID="sdk-id",
        ZOOM_SDK_CLIENT_SECRET="sdk-secret",
        PUBLIC_API_BASE_URL="https://api.example.com",
        REDIS_URL="redis://localhost:6379/0",
    )
    defaults.update(overrides)
    return MagicMock(**defaults)


def _user_doc(data: dict) -> MagicMock:
    doc = MagicMock()
    doc.exists = bool(data)
    doc.to_dict.return_value = data
    return doc


def _db_with(data: dict) -> tuple[MagicMock, MagicMock]:
    """Return (db, document_ref) so tests can assert on what was written."""
    ref = MagicMock()
    ref.get = AsyncMock(return_value=_user_doc(data))
    ref.update = AsyncMock()

    db = MagicMock()
    db.collection.return_value.document.return_value = ref
    return db, ref


@pytest.fixture(autouse=True)
def _passthrough_crypto():
    """Encryption is covered by its own tests; keep these readable."""
    with (
        patch("app.core.crypto.encrypt", side_effect=lambda v: f"enc:{v}"),
        patch("app.core.crypto.decrypt", side_effect=lambda v: v.removeprefix("enc:")),
    ):
        yield


@pytest.fixture(autouse=True)
def _no_redis_lock():
    """The refresh lock needs a live Redis; behaviour under contention is
    asserted separately in TestRefreshLocking."""
    lock = MagicMock()
    lock.__enter__ = MagicMock(return_value=lock)
    lock.__exit__ = MagicMock(return_value=False)
    with patch.object(zoom_oauth, "_refresh_lock", return_value=lock):
        yield


class TestAuthorizeUrl:
    def test_requests_the_scopes_obf_needs(self) -> None:
        url = zoom_oauth.authorize_url("state-123", _settings())
        params = parse_qs(urlparse(url).query)

        scopes = params["scope"][0].split()
        # Without user:read:token there is no OBF token, and no external joins.
        assert "user:read:token" in scopes
        assert "meeting:read:local_recording_token" in scopes
        assert params["response_type"] == ["code"]
        assert params["state"] == ["state-123"]

    def test_redirect_uri_matches_the_registered_callback(self) -> None:
        settings = _settings()
        url = zoom_oauth.authorize_url("s", settings)
        params = parse_qs(urlparse(url).query)

        # Zoom rejects any mismatch against the Marketplace registration.
        assert params["redirect_uri"] == ["https://api.example.com/api/v1/zoom/callback"]
        assert zoom_oauth.redirect_uri(settings) == params["redirect_uri"][0]

    def test_falls_back_to_sdk_credentials(self) -> None:
        """One Marketplace app usually serves both features."""
        params = parse_qs(urlparse(zoom_oauth.authorize_url("s", _settings())).query)
        assert params["client_id"] == ["sdk-id"]

    def test_dedicated_oauth_credentials_win(self) -> None:
        settings = _settings(ZOOM_OAUTH_CLIENT_ID="oauth-id")
        params = parse_qs(urlparse(zoom_oauth.authorize_url("s", settings)).query)
        assert params["client_id"] == ["oauth-id"]


class TestAccessToken:
    async def test_unlinked_user_raises(self) -> None:
        db, _ = _db_with({})
        with pytest.raises(ZoomNotLinked):
            await zoom_oauth.get_access_token(db, "user-1", _settings())

    async def test_valid_token_is_reused_without_a_refresh(self) -> None:
        db, ref = _db_with({
            zoom_oauth.FIELD_ACCESS: "enc:live-token",
            zoom_oauth.FIELD_REFRESH: "enc:refresh-1",
            zoom_oauth.FIELD_EXPIRES: datetime.now(UTC) + timedelta(hours=1),
        })

        token = await zoom_oauth.get_access_token(db, "user-1", _settings())

        assert token == "live-token"
        ref.update.assert_not_awaited()

    async def test_token_expiring_within_the_skew_is_refreshed(self) -> None:
        """A token valid for one more minute would expire mid-request."""
        db, ref = _db_with({
            zoom_oauth.FIELD_ACCESS: "enc:nearly-dead",
            zoom_oauth.FIELD_REFRESH: "enc:refresh-1",
            zoom_oauth.FIELD_EXPIRES: datetime.now(UTC) + timedelta(minutes=1),
        })

        with patch.object(
            zoom_oauth, "_token_request",
            AsyncMock(return_value={
                "access_token": "fresh",
                "refresh_token": "refresh-2",
                "expires_in": 3600,
            }),
        ):
            token = await zoom_oauth.get_access_token(db, "user-1", _settings())

        assert token == "fresh"

    async def test_naive_expiry_timestamps_are_treated_as_utc(self) -> None:
        """Firestore can hand back naive datetimes; comparing would explode."""
        db, _ = _db_with({
            zoom_oauth.FIELD_ACCESS: "enc:live",
            zoom_oauth.FIELD_REFRESH: "enc:r",
            zoom_oauth.FIELD_EXPIRES: (datetime.now(UTC) + timedelta(hours=1)).replace(tzinfo=None),
        })

        assert await zoom_oauth.get_access_token(db, "user-1", _settings()) == "live"


class TestRefreshRotation:
    """Zoom invalidates a refresh token as soon as it is used."""

    async def test_the_rotated_refresh_token_is_persisted(self) -> None:
        db, ref = _db_with({
            zoom_oauth.FIELD_REFRESH: "enc:refresh-1",
            zoom_oauth.FIELD_EXPIRES: datetime.now(UTC) - timedelta(minutes=1),
        })

        with patch.object(
            zoom_oauth, "_token_request",
            AsyncMock(return_value={
                "access_token": "fresh",
                "refresh_token": "refresh-2",
                "expires_in": 3600,
            }),
        ):
            await zoom_oauth.get_access_token(db, "user-1", _settings())

        written = ref.update.await_args.args[0]
        # Losing this means the user must reinstall the app.
        assert written[zoom_oauth.FIELD_REFRESH] == "enc:refresh-2"
        assert written[zoom_oauth.FIELD_ACCESS] == "enc:fresh"
        assert written[zoom_oauth.FIELD_EXPIRES] > datetime.now(UTC)

    async def test_a_revoked_grant_surfaces_as_auth_error(self) -> None:
        db, _ = _db_with({
            zoom_oauth.FIELD_REFRESH: "enc:stale",
            zoom_oauth.FIELD_EXPIRES: datetime.now(UTC) - timedelta(minutes=1),
        })

        with patch.object(
            zoom_oauth, "_token_request",
            AsyncMock(side_effect=ZoomAuthError("invalid_grant")),
        ), pytest.raises(ZoomAuthError):
            await zoom_oauth.get_access_token(db, "user-1", _settings())


class TestRefreshLocking:
    async def test_a_concurrent_refresh_is_not_repeated(self) -> None:
        """Whoever loses the lock re-reads and finds the token already fresh,
        instead of spending an already-rotated refresh token."""
        refreshed = {
            zoom_oauth.FIELD_ACCESS: "enc:already-fresh",
            zoom_oauth.FIELD_REFRESH: "enc:refresh-2",
            zoom_oauth.FIELD_EXPIRES: datetime.now(UTC) + timedelta(hours=1),
        }
        stale = {
            zoom_oauth.FIELD_ACCESS: "enc:old",
            zoom_oauth.FIELD_REFRESH: "enc:refresh-1",
            zoom_oauth.FIELD_EXPIRES: datetime.now(UTC) - timedelta(minutes=1),
        }

        ref = MagicMock()
        # First read sees stale state; the read inside the lock sees the
        # winner's write.
        ref.get = AsyncMock(side_effect=[_user_doc(stale), _user_doc(refreshed)])
        ref.update = AsyncMock()
        db = MagicMock()
        db.collection.return_value.document.return_value = ref

        token_request = AsyncMock()
        with patch.object(zoom_oauth, "_token_request", token_request):
            token = await zoom_oauth.get_access_token(db, "user-1", _settings())

        assert token == "already-fresh"
        token_request.assert_not_awaited()
        ref.update.assert_not_awaited()


class TestMeetingTokens:
    async def test_obf_token_uses_the_onbehalf_endpoint(self) -> None:
        with patch.object(
            zoom_oauth, "_get_json", AsyncMock(return_value={"token": "obf-abc"})
        ) as get_json:
            assert await zoom_oauth.get_obf_token("access") == "obf-abc"

        url = get_json.await_args.args[0]
        assert url.endswith("/users/me/token?type=onbehalf")

    async def test_zak_token_uses_the_zak_endpoint(self) -> None:
        with patch.object(
            zoom_oauth, "_get_json", AsyncMock(return_value={"token": "zak-abc"})
        ) as get_json:
            assert await zoom_oauth.get_zak_token("access") == "zak-abc"

        assert get_json.await_args.args[0].endswith("/users/me/token?type=zak")

    async def test_local_recording_token_is_scoped_to_the_meeting(self) -> None:
        with patch.object(
            zoom_oauth, "_get_json", AsyncMock(return_value={"token": "jt-abc"})
        ) as get_json:
            token = await zoom_oauth.get_local_recording_token("access", "8551234567")

        assert token == "jt-abc"
        url = get_json.await_args.args[0]
        assert "/meetings/8551234567/jointoken/local_recording" in url
        assert "bypass_waiting_room=false" in url

    async def test_bypass_waiting_room_is_opt_in(self) -> None:
        with patch.object(
            zoom_oauth, "_get_json", AsyncMock(return_value={"token": "jt"})
        ) as get_json:
            await zoom_oauth.get_local_recording_token(
                "access", "855", bypass_waiting_room=True
            )

        assert "bypass_waiting_room=true" in get_json.await_args.args[0]


class TestTokenRequestConfig:
    async def test_unconfigured_credentials_fail_clearly(self) -> None:
        settings = _settings(ZOOM_SDK_CLIENT_ID="", ZOOM_SDK_CLIENT_SECRET="")

        with pytest.raises(ZoomAuthError, match="not configured"):
            await zoom_oauth._token_request({"grant_type": "refresh_token"}, settings)


class TestConnectorTokenMinting:
    """The bot degrades to a local-account-only join rather than failing."""

    @staticmethod
    def _connector(**kwargs):
        from app.connectors.zoom_selfhosted import ZoomSelfHostedConnector

        defaults = dict(
            client_id="cid", client_secret="csec", image="img",
            redis_url="redis://localhost", webhook_base_url="http://api:5000",
            deepgram_api_key="dg", deepgram_model="nova-3",
            display_name="Forgefy Notetaker", require_host_consent=True,
            user_id="user-1", settings=_settings(),
        )
        defaults.update(kwargs)
        return ZoomSelfHostedConnector(**defaults)

    def test_no_linked_user_yields_no_tokens(self) -> None:
        connector = self._connector(user_id=None, settings=None)
        assert connector._mint_meeting_tokens("855") == {}

    def test_unlinked_zoom_account_is_survivable(self) -> None:
        connector = self._connector()
        with patch(
            "app.integrations.zoom_oauth.get_access_token",
            AsyncMock(side_effect=ZoomNotLinked("nope")),
        ):
            assert connector._mint_meeting_tokens("855") == {}

    def test_obf_token_is_passed_to_the_container(self) -> None:
        connector = self._connector()
        with (
            patch("app.integrations.zoom_oauth.get_access_token", AsyncMock(return_value="at")),
            patch("app.integrations.zoom_oauth.get_obf_token", AsyncMock(return_value="obf-1")),
        ):
            env = connector._mint_meeting_tokens("855")

        assert env == {"ZOOM_ON_BEHALF_TOKEN": "obf-1"}

    def test_join_token_only_requested_when_not_prompting_the_host(self) -> None:
        """With consent required we deliberately do NOT pre-authorize."""
        connector = self._connector(require_host_consent=False)
        with (
            patch("app.integrations.zoom_oauth.get_access_token", AsyncMock(return_value="at")),
            patch("app.integrations.zoom_oauth.get_obf_token", AsyncMock(return_value="obf-1")),
            patch(
                "app.integrations.zoom_oauth.get_local_recording_token",
                AsyncMock(return_value="jt-1"),
            ),
        ):
            env = connector._mint_meeting_tokens("855")

        assert env["ZOOM_JOIN_TOKEN"] == "jt-1"
        assert env["ZOOM_ON_BEHALF_TOKEN"] == "obf-1"

    def test_consent_required_skips_the_join_token(self) -> None:
        connector = self._connector(require_host_consent=True)
        with (
            patch("app.integrations.zoom_oauth.get_access_token", AsyncMock(return_value="at")),
            patch("app.integrations.zoom_oauth.get_obf_token", AsyncMock(return_value="obf")),
            patch(
                "app.integrations.zoom_oauth.get_local_recording_token",
                AsyncMock(return_value="jt"),
            ) as join_token,
        ):
            env = connector._mint_meeting_tokens("855")

        join_token.assert_not_awaited()
        assert "ZOOM_JOIN_TOKEN" not in env
