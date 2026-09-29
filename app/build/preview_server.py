"""Serve-style local preview — a no-cloud alternative to Cloudflare Pages.

The build pipeline normally ships a compiled web artifact (a Next.js static
export ``out/``, the ``.vercel/output/static`` tree from
``@cloudflare/next-on-pages``, or an Expo ``dist/``) to Cloudflare Pages via
``wrangler``. This module serves that same artifact from a local static file
server instead, so a web app still gets a preview with no cloud account at all.

How it fits together:

* :func:`deploy_local_preview` copies the artifact out of the (soon-deleted)
  build workspace into a persistent preview root, then returns
  ``http://<host>:<port>/<slug>/``.
* One :class:`ThreadingHTTPServer` is started per process and reused across
  builds — every published app is a subdirectory of the shared root, so a single
  server fronts all of them and the root doubles as an index page.

Scope and limits (called out on purpose — this is a local/dev convenience, not a
hosting tier):

* The URL is only reachable from wherever the worker runs: ``127.0.0.1`` is the
  *worker's* loopback, not the browser's. Point the browser at the worker's
  host/LAN address (``LOCAL_PREVIEW_HOST=0.0.0.0``) when the two differ.
* Server-side rendering and API routes are NOT reproduced. For a
  ``next-on-pages`` build those live in a Cloudflare Worker, so only the
  pre-rendered HTML/JS/CSS is served here and dynamic routes can come back
  blank.
* Nothing evicts old previews — ``LOCAL_PREVIEW_ROOT`` grows until an operator
  clears it.

Standalone use (serve any built directory without the build pipeline)::

    python -m app.build.preview_server ./out --port 8099
"""
from __future__ import annotations

import argparse
import functools
import logging
import os
import re
import shutil
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Mirrors Workspace's convention (app/build/workspace.py): a predictable root
# outside any build workspace, which is deleted the moment a build finishes.
DEFAULT_PREVIEW_ROOT = Path("/tmp/forgefy_previews")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8099

_lock = threading.Lock()
_server: ThreadingHTTPServer | None = None


def slugify(name: str, *, max_len: int = 40) -> str:
    """Return a filesystem- and URL-safe slug for a preview directory name."""
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    return slug[:max_len].rstrip("-") or "forgefy-app"


class _PreviewHandler(SimpleHTTPRequestHandler):
    """Static file handler with a Next.js-style extensionless route fallback."""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 — stdlib signature
        """Send access logs to logging instead of stderr (one line per asset)."""
        logger.debug("preview %s - %s", self.address_string(), format % args)

    def translate_path(self, path: str) -> str:
        """Map ``/app/about`` onto ``app/about.html`` before the base handler.

        A Next.js static export writes ``about.html`` for the ``/about`` route
        (``about/index.html`` when ``trailingSlash`` is enabled). The stdlib
        handler only falls back to ``index.html`` inside a directory, so the
        extensionless form would otherwise 404.
        """
        translated = super().translate_path(path)
        if os.path.splitext(translated)[1] or os.path.isdir(translated):
            return translated
        html_variant = translated + ".html"
        return html_variant if os.path.isfile(html_variant) else translated


def _display_host(host: str) -> str:
    """Turn a bind-only address into something a browser can actually open."""
    return DEFAULT_HOST if host in ("", "0.0.0.0", "::") else host


def _make_server(host: str, port: int, root: Path) -> ThreadingHTTPServer:
    """Build (but do not start) a static server for ``root``."""
    handler = functools.partial(_PreviewHandler, directory=str(root))
    return ThreadingHTTPServer((host, port), handler)


def ensure_server(host: str, port: int, root: Path) -> tuple[str, int]:
    """Start the shared preview server on first call; return its ``(host, port)``.

    One server per process fronts every published app, since each app is just a
    subdirectory of ``root``. Prefork workers (``CELERY_CONCURRENCY > 1``) share
    the same ``root`` on disk, so a child that cannot bind the port reuses the
    sibling that already owns it instead of failing the build.
    """
    global _server

    root.mkdir(parents=True, exist_ok=True)

    with _lock:
        if _server is not None:
            return _display_host(host), int(_server.server_address[1])

        try:
            server = _make_server(host, port, root)
        except OSError as exc:
            # EADDRINUSE — another worker child owns the port and the same root.
            logger.info(
                "Local preview server already bound at %s:%s (%s) — reusing it",
                host,
                port,
                exc,
            )
            return _display_host(host), port

        threading.Thread(
            target=server.serve_forever,
            name="forgefy-preview-server",
            daemon=True,
        ).start()
        _server = server
        bound_host, bound_port = _display_host(host), int(server.server_address[1])
        logger.info(
            "Local preview server listening on http://%s:%s (root=%s)",
            bound_host,
            bound_port,
            root,
        )
        return bound_host, bound_port


def publish_preview(build_dir: Path, project_name: str, root: Path) -> Path:
    """Copy ``build_dir`` to ``root/<slug>``, replacing any previous build."""
    dest = root / slugify(project_name)
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(build_dir, dest)
    return dest




def deploy_local_preview(
    build_dir: Path,
    project_name: str,
    settings: Any | None = None,
) -> str | None:
    """Publish ``build_dir`` and return its local preview URL, or ``None``.

    Returns ``None`` (never raises) when the feature is disabled, the artifact
    is missing, or publishing fails — a preview is always non-fatal to a build.
    """
    try:
        if settings is None:
            from app.config import get_settings

            settings = get_settings()
        if not getattr(settings, "LOCAL_PREVIEW_ENABLED", False):
            return None
        if not build_dir or not build_dir.is_dir():
            logger.warning("Local preview skipped — %s is not a directory", build_dir)
            return None

        host = getattr(settings, "LOCAL_PREVIEW_HOST", "") or DEFAULT_HOST
        port = int(getattr(settings, "LOCAL_PREVIEW_PORT", 0) or DEFAULT_PORT)
        root = Path(getattr(settings, "LOCAL_PREVIEW_ROOT", "") or DEFAULT_PREVIEW_ROOT)

        slug = slugify(project_name)
        publish_preview(build_dir, project_name, root)
        bound_host, bound_port = ensure_server(host, port, root)
        url = f"http://{bound_host}:{bound_port}/{slug}/"
        logger.info("Local preview deployed → %s", url)
        return url
    except Exception as exc:  # noqa: BLE001 — a preview must never fail a build
        logger.warning("Local preview deploy failed (non-fatal): %s", exc)
        return None


def serve_directory(directory: Path, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    """Serve ``directory`` in the foreground until interrupted (CLI helper)."""
    server = _make_server(host, port, directory)
    bound_host, bound_port = _display_host(host), int(server.server_address[1])
    print(f"Serving {directory} at http://{bound_host}:{bound_port}/  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point — ``python -m app.build.preview_server <directory>``."""
    parser = argparse.ArgumentParser(
        prog="python -m app.build.preview_server",
        description="Serve a compiled web artifact locally — no cloud account required.",
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Built artifact dir (Next.js 'out/', '.vercel/output/static', Expo 'dist/')",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Bind host (default: {DEFAULT_HOST})")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"Bind port (default: {DEFAULT_PORT})"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    directory: Path = args.directory
    if not directory.is_dir():
        parser.error(f"{directory} is not a directory")

    serve_directory(directory, args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

