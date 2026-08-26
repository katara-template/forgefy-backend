"""A standalone stand-in for the Forgefy backend.

Receives the bot's webhooks, verifies their signatures exactly as the real
handler will, and prints transcripts to the terminal. Lets the whole bot be
proven end-to-end — join, consent, audio, Deepgram — before a single line of
the live app changes.

    python scripts/mock_backend.py --secret devsecret

Then point a bot at it with FORGEFY_WEBHOOK_URL=http://host.docker.internal:8080/webhook
(scripts/run_local.py does this for you).

Standard library only — no install required.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Windows defaults stdout to cp1252, which cannot encode the status glyph and
# would otherwise kill the request handler rather than just garble a line.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ANSI colours make the interim/final distinction readable in a live meeting.
# Disabled when output is piped, so logs stay greppable.
_COLOR = sys.stdout.isatty()
_DIM = "\033[2m" if _COLOR else ""
_GREEN = "\033[32m" if _COLOR else ""
_YELLOW = "\033[33m" if _COLOR else ""
_RED = "\033[31m" if _COLOR else ""
_RESET = "\033[0m" if _COLOR else ""

_MAX_SKEW_SECONDS = 300

SECRET = ""
STRICT = True


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 — http.server's required casing
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)

        if STRICT and not self._verify(raw):
            self.send_response(401)
            self.end_headers()
            return

        try:
            body = json.loads(raw)
        except ValueError:
            self.send_response(400)
            self.end_headers()
            return

        # Displaying an event must never fail the request — this stands in for
        # a backend, and a console encoding problem is not a delivery failure.
        try:
            self._render(body)
        except Exception as exc:  # noqa: BLE001
            print(f"[render failed: {exc!r}] {body}")

        self.send_response(204)
        self.end_headers()

    def _verify(self, raw: bytes) -> bool:
        timestamp = self.headers.get("X-Forgefy-Timestamp", "")
        signature = self.headers.get("X-Forgefy-Signature", "")
        if not (timestamp and signature):
            print(f"{_RED}rejected: missing signature headers{_RESET}")
            return False

        try:
            if abs(int(time.time()) - int(timestamp)) > _MAX_SKEW_SECONDS:
                print(f"{_RED}rejected: stale timestamp{_RESET}")
                return False
        except ValueError:
            print(f"{_RED}rejected: bad timestamp{_RESET}")
            return False

        expected = hmac.new(
            SECRET.encode(), f"{timestamp}.".encode() + raw, hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(expected, signature):
            print(f"{_RED}rejected: bad signature{_RESET}")
            return False
        return True

    def _render(self, body: dict) -> None:
        kind = body.get("type")

        if kind == "status":
            status = body.get("status", "")
            detail = body.get("detail", "")
            colour = _RED if status in {"error", "consent_denied"} else _YELLOW
            print(f"{colour}● {status}{_RESET}" + (f" {_DIM}{detail}{_RESET}" if detail else ""))

        elif kind == "transcript":
            text = body.get("text", "")
            speaker = body.get("speaker") or ""
            prefix = f"{speaker}: " if speaker else ""
            if body.get("is_final"):
                print(f"{_GREEN}{prefix}{text}{_RESET}")
            else:
                # Interim results churn; dim them so finals stand out.
                print(f"{_DIM}{prefix}{text}{_RESET}")

        else:
            print(f"{_DIM}unknown event: {body}{_RESET}")

        sys.stdout.flush()

    def log_message(self, *_args) -> None:
        """Silence the default per-request access log."""


def main() -> int:
    global SECRET, STRICT

    parser = argparse.ArgumentParser(description="Mock Forgefy backend for zoom-bot testing")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--secret", default="devsecret",
                        help="must match FORGEFY_WEBHOOK_SECRET given to the bot")
    parser.add_argument("--insecure", action="store_true",
                        help="skip signature verification (debugging only)")
    args = parser.parse_args()

    SECRET = args.secret
    STRICT = not args.insecure

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"mock backend listening on :{args.port}"
          f"{'' if STRICT else '  (signature checks DISABLED)'}")
    print(f"{_DIM}dim = interim   {_GREEN}green = final   {_YELLOW}yellow = status{_RESET}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    return 0


if __name__ == "__main__":
    sys.exit(main())
