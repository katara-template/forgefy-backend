"""Unit tests for the Cloudflare Pages custom-domain helpers.

Run:
    venv/Scripts/python -m pytest tests/build/test_cloudflare_pages.py -v

These cover publishing's only REST-API surface: attaching a stable
``<slug>.<base-domain>`` to a Pages project. ``publish_domain`` must be
idempotent (re-publish), collision-tolerant (another app already owns the name),
and must return ``None`` — never raise — when it cannot attach, so a publish can
still fall back to the project's ``*.pages.dev`` URL.
"""
from __future__ import annotations

import httpx
import pytest

from app.build import cloudflare_pages as cf


class TestSlugifyName:
    def test_lowercases_and_replaces_non_alphanumerics(self):
        assert cf.slugify_name("StockFlow Manager!!") == "stockflow-manager"

    def test_empty_name_falls_back(self):
        assert cf.slugify_name("") == "forgefy-app"
        assert cf.slugify_name("!!!") == "forgefy-app"

    def test_capped_so_it_is_a_valid_dns_label(self):
        slug = cf.slugify_name("a" * 40)
        assert len(slug) <= 28
        assert not slug.endswith("-")


class TestCandidates:
    def test_prefers_the_bare_slug(self):
        assert cf._candidates("bun", "abc123def")[0] == "bun"

    def test_adds_an_id_suffixed_fallback(self):
        assert cf._candidates("bun", "abc123def")[1] == "bun-abc123"

    def test_no_fallback_when_the_id_has_no_usable_characters(self):
        assert cf._candidates("bun", "----") == ["bun"]


class TestIsConflict:
    @pytest.mark.parametrize(
        "detail",
        [
            "The domain is already in use by another project",
            "Domain already exists",
            "That name is taken",
        ],
    )
    def test_conflict_phrasings(self, detail: str):
        assert cf.is_conflict(detail) is True

    def test_an_auth_failure_is_not_a_conflict(self):
        assert cf.is_conflict("Invalid API token") is False


class TestAttachDomainHttp:
    """The REST call itself, with httpx stubbed."""

    def test_success(self, monkeypatch):
        monkeypatch.setattr(
            httpx, "post", lambda *a, **k: httpx.Response(200, json={"success": True})
        )
        ok, detail = cf.attach_domain("acct", "bun", "bun.forgefy.dev", "tok")
        assert ok is True
        assert detail == ""

    def test_api_error_message_is_surfaced(self, monkeypatch):
        monkeypatch.setattr(
            httpx,
            "post",
            lambda *a, **k: httpx.Response(
                400, json={"success": False, "errors": [{"code": 1, "message": "bad token"}]}
            ),
        )
        ok, detail = cf.attach_domain("acct", "bun", "bun.forgefy.dev", "tok")
        assert ok is False
        assert "bad token" in detail

    def test_network_error_is_reported_not_raised(self, monkeypatch):
        def boom(*_a, **_k):
            raise httpx.ConnectError("no route to host")

        monkeypatch.setattr(httpx, "post", boom)
        ok, detail = cf.attach_domain("acct", "bun", "bun.forgefy.dev", "tok")
        assert ok is False
        assert "no route" in detail


class TestListDomains:
    def test_returns_attached_names(self, monkeypatch):
        monkeypatch.setattr(
            httpx,
            "get",
            lambda *a, **k: httpx.Response(
                200, json={"success": True, "result": [{"name": "a.dev"}, {"name": "b.dev"}]}
            ),
        )
        assert cf.list_domains("acct", "bun", "tok") == ["a.dev", "b.dev"]

    def test_failure_returns_an_empty_list(self, monkeypatch):
        monkeypatch.setattr(
            httpx, "get", lambda *a, **k: httpx.Response(500, json={"success": False})
        )
        assert cf.list_domains("acct", "bun", "tok") == []

    def test_network_error_returns_an_empty_list(self, monkeypatch):
        def boom(*_a, **_k):
            raise httpx.ConnectError("nope")

        monkeypatch.setattr(httpx, "get", boom)
        assert cf.list_domains("acct", "bun", "tok") == []


class TestPublishDomain:
    def test_returns_none_without_a_base_domain(self):
        assert cf.publish_domain("bun", "pid", "", account_id="a", api_token="t") is None

    def test_returns_none_without_credentials(self):
        assert cf.publish_domain("bun", "pid", "forgefy.dev", account_id="", api_token="") is None

    def test_attaches_the_slug_domain(self, monkeypatch):
        monkeypatch.setattr(cf, "list_domains", lambda *a, **k: [])
        seen: list[tuple] = []

        def fake_attach(*args, **kwargs):
            seen.append(args)
            return True, ""

        monkeypatch.setattr(cf, "attach_domain", fake_attach)

        domain = cf.publish_domain("Bun App", "pid", "forgefy.dev", account_id="a", api_token="t")

        assert domain == "bun-app.forgefy.dev"
        # attach_domain(account_id, project_name, domain, api_token)
        assert seen[0][1] == "bun-app"
        assert seen[0][2] == "bun-app.forgefy.dev"

    def test_base_domain_is_normalised(self, monkeypatch):
        monkeypatch.setattr(cf, "list_domains", lambda *a, **k: [])
        monkeypatch.setattr(cf, "attach_domain", lambda *a, **k: (True, ""))
        domain = cf.publish_domain("bun", "pid", "  .FORGEFY.dev.  ", account_id="a", api_token="t")
        assert domain == "bun.forgefy.dev"

    def test_existing_domain_is_returned_without_reattaching(self, monkeypatch):
        """Re-publishing must be idempotent, not a duplicate-attach error."""
        monkeypatch.setattr(cf, "list_domains", lambda *a, **k: ["bun.forgefy.dev"])
        called = {"attach": False}

        def fake_attach(*_a, **_k):
            called["attach"] = True
            return True, ""

        monkeypatch.setattr(cf, "attach_domain", fake_attach)
        domain = cf.publish_domain("bun", "pid", "forgefy.dev", account_id="a", api_token="t")

        assert domain == "bun.forgefy.dev"
        assert called["attach"] is False

    def test_conflict_retries_with_the_id_suffixed_candidate(self, monkeypatch):
        monkeypatch.setattr(cf, "list_domains", lambda *a, **k: [])
        attempted: list[str] = []

        def fake_attach(_account, _project, domain, _token, **_kw):
            attempted.append(domain)
            if len(attempted) == 1:
                return False, "The domain is already in use by another project"
            return True, ""

        monkeypatch.setattr(cf, "attach_domain", fake_attach)
        domain = cf.publish_domain("bun", "abc123def456", "forgefy.dev", account_id="a", api_token="t")

        assert attempted == ["bun.forgefy.dev", "bun-abc123.forgefy.dev"]
        assert domain == "bun-abc123.forgefy.dev"

    def test_a_non_conflict_failure_gives_up(self, monkeypatch):
        monkeypatch.setattr(cf, "list_domains", lambda *a, **k: [])
        monkeypatch.setattr(cf, "attach_domain", lambda *a, **k: (False, "Invalid API token"))
        assert cf.publish_domain("bun", "pid", "forgefy.dev", account_id="a", api_token="t") is None

    def test_both_candidates_conflicting_gives_up(self, monkeypatch):
        monkeypatch.setattr(cf, "list_domains", lambda *a, **k: [])
        monkeypatch.setattr(cf, "attach_domain", lambda *a, **k: (False, "already exists"))
        assert cf.publish_domain("bun", "abc123def", "forgefy.dev", account_id="a", api_token="t") is None

