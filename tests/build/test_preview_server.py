"""Unit tests for the serve-style local preview helper.

Run:
    venv/Scripts/python -m pytest tests/build/test_preview_server.py -v

The helper is the no-cloud fallback for web previews: when a Cloudflare Pages
deploy is skipped or fails, the compiled artifact is published to a local static
file server instead. Two behaviours matter and are pinned here:

  * a Next.js static export writes ``about.html`` for the ``/about`` route, so
    the stdlib handler's directory-only ``index.html`` fallback is not enough;
  * a preview must never make a build fail, so every failure path returns None.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from app.build import preview_server as pv


def _settings(**overrides) -> SimpleNamespace:
    """Settings stub — port 0 lets the OS pick a free port per test."""
    base = {
        "LOCAL_PREVIEW_ENABLED": True,
        "LOCAL_PREVIEW_HOST": "127.0.0.1",
        "LOCAL_PREVIEW_PORT": 0,
        "LOCAL_PREVIEW_ROOT": "/tmp/forgefy_previews",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _write_site(root: Path) -> Path:
    """A minimal static export: index + an ``about.html`` sibling route."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "index.html").write_text("<h1>home</h1>", encoding="utf-8")
    (root / "about.html").write_text("<h1>about</h1>", encoding="utf-8")
    (root / "_next").mkdir(exist_ok=True)
    (root / "_next" / "app.js").write_text("console.log(1)", encoding="utf-8")
    return root


@pytest.fixture
def clean_server():
    """Isolate the module-level server singleton between tests."""
    pv._server = None
    yield
    server = pv._server
    if server is not None:
        server.shutdown()
        server.server_close()
    pv._server = None


class TestSlugify:
    def test_lowercases_and_collapses_non_alphanumerics(self):
        assert pv.slugify("StockFlow Manager!!") == "stockflow-manager"

    def test_empty_name_falls_back_to_a_default(self):
        assert pv.slugify("") == "forgefy-app"
        assert pv.slugify("!!!") == "forgefy-app"

    def test_truncates_without_leaving_a_trailing_dash(self):
        slug = pv.slugify("a" * 30 + " " + "b" * 30, max_len=32)
        assert len(slug) <= 32
        assert not slug.endswith("-")


class TestPublish:
    def test_copies_the_artifact_under_the_slug(self, tmp_path: Path):
        site = _write_site(tmp_path / "build")
        root = tmp_path / "previews"

        dest = pv.publish_preview(site, "StockFlow", root)

        assert dest == root / "stockflow"
        assert (dest / "index.html").read_text(encoding="utf-8") == "<h1>home</h1>"
        assert (dest / "_next" / "app.js").is_file()

    def test_republish_replaces_stale_files(self, tmp_path: Path):
        site = _write_site(tmp_path / "build")
        root = tmp_path / "previews"
        pv.publish_preview(site, "StockFlow", root)

        (site / "stale.js").write_text("old", encoding="utf-8")
        pv.publish_preview(site, "StockFlow", root)
        assert (root / "stockflow" / "stale.js").is_file()

        (site / "stale.js").unlink()
        pv.publish_preview(site, "StockFlow", root)
        assert not (root / "stockflow" / "stale.js").exists(), "previous build leaked through"


class TestDeployLocalPreview:
    def test_disabled_returns_none_without_touching_disk(self, tmp_path: Path):
        site = _write_site(tmp_path / "build")
        root = tmp_path / "previews"

        url = pv.deploy_local_preview(site, "StockFlow", _settings(LOCAL_PREVIEW_ENABLED=False))

        assert url is None
        assert not root.exists()

    def test_enabled_publishes_and_returns_a_url(self, tmp_path: Path, clean_server):
        site = _write_site(tmp_path / "build")
        root = tmp_path / "previews"

        url = pv.deploy_local_preview(site, "StockFlow", _settings(LOCAL_PREVIEW_ROOT=str(root)))

        assert url is not None and url.startswith("http://127.0.0.1:")
        assert url.endswith("/stockflow/")
        assert (root / "stockflow" / "index.html").is_file()

    def test_missing_artifact_returns_none(self, tmp_path: Path, clean_server):
        url = pv.deploy_local_preview(
            tmp_path / "nope", "StockFlow", _settings(LOCAL_PREVIEW_ROOT=str(tmp_path / "p"))
        )
        assert url is None

    def test_failures_are_swallowed(self, tmp_path: Path, monkeypatch, clean_server):
        """A preview must never fail the build it is attached to."""
        site = _write_site(tmp_path / "build")

        def boom(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(pv, "publish_preview", boom)
        url = pv.deploy_local_preview(site, "StockFlow", _settings(LOCAL_PREVIEW_ROOT=str(tmp_path / "p")))
        assert url is None



class TestServing:
    """End-to-end: publish, then actually fetch over HTTP."""

    def test_serves_index_and_nested_assets(self, tmp_path: Path, clean_server):
        site = _write_site(tmp_path / "build")
        root = tmp_path / "previews"
        url = pv.deploy_local_preview(site, "StockFlow", _settings(LOCAL_PREVIEW_ROOT=str(root)))
        assert url is not None

        with urlopen(url, timeout=5) as resp:  # noqa: S310 - loopback URL built above
            assert resp.status == 200
            assert b"home" in resp.read()

        with urlopen(url + "_next/app.js", timeout=5) as resp:  # noqa: S310
            assert b"console.log" in resp.read()

    def test_extensionless_route_falls_back_to_html(self, tmp_path: Path, clean_server):
        """The reason this handler exists: Next exports ``/about`` as about.html."""
        site = _write_site(tmp_path / "build")
        root = tmp_path / "previews"
        url = pv.deploy_local_preview(site, "StockFlow", _settings(LOCAL_PREVIEW_ROOT=str(root)))
        assert url is not None

        with urlopen(url + "about", timeout=5) as resp:  # noqa: S310
            assert resp.status == 200
            assert b"about" in resp.read()

    def test_unknown_route_is_a_404(self, tmp_path: Path, clean_server):
        site = _write_site(tmp_path / "build")
        root = tmp_path / "previews"
        url = pv.deploy_local_preview(site, "StockFlow", _settings(LOCAL_PREVIEW_ROOT=str(root)))
        assert url is not None

        with pytest.raises(HTTPError) as exc:
            urlopen(url + "does-not-exist", timeout=5)  # noqa: S310
        assert exc.value.code == 404


class TestDeployWebPreview:
    """build_worker's Cloudflare-first, local-second dispatch."""

    def test_cloudflare_url_wins_when_available(self, tmp_path: Path, monkeypatch):
        from app.workers import build_worker

        monkeypatch.setattr(build_worker, "_deploy_cloudflare_pages", lambda *a: "https://x.pages.dev")
        called = {"local": False}
        monkeypatch.setattr(
            "app.build.preview_server.deploy_local_preview",
            lambda *a: called.update(local=True),
        )

        url = build_worker._deploy_web_preview(tmp_path, "StockFlow", _settings())
        assert url == "https://x.pages.dev"
        assert called["local"] is False, "local fallback must not run when Cloudflare succeeds"

    def test_falls_back_to_local_when_cloudflare_is_absent(self, tmp_path: Path, monkeypatch):
        from app.workers import build_worker

        site = _write_site(tmp_path / "build")
        monkeypatch.setattr(build_worker, "_deploy_cloudflare_pages", lambda *a: None)

        url = build_worker._deploy_web_preview(
            site, "StockFlow", _settings(LOCAL_PREVIEW_ROOT=str(tmp_path / "previews"))
        )
        assert url is not None and "/stockflow/" in url

