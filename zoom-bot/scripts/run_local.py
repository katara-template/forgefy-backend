"""Launch one bot container against a real meeting, with no backend involved.

This is the primary way to prove the bot works before linking anything. It
mints the SDK JWT, resolves the meeting ID, and runs the container in the
foreground so its logs stream to your terminal.

    # terminal 1
    python scripts/mock_backend.py --secret devsecret

    # terminal 2
    export ZOOM_SDK_CLIENT_ID=... ZOOM_SDK_CLIENT_SECRET=... DEEPGRAM_API_KEY=...
    python scripts/run_local.py "https://zoom.us/j/1234567890?pwd=abc"

Ctrl-C stops the container, which makes the bot leave the meeting politely.

Standard library only — run it with any Python 3.11+, no virtualenv needed.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import uuid
from pathlib import Path

# The backend root, so this shares the connector's URL parsing and JWT minting
# rather than keeping a second copy that can drift. Both app/__init__.py and
# app/connectors/__init__.py are empty, so this pulls in no backend deps and
# the script stays runnable with a bare Python.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.connectors.zoom_meeting import build_sdk_jwt, parse_meeting_url  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a Forgefy Zoom bot against a meeting, logging to the terminal",
    )
    parser.add_argument("meeting", help="Zoom join URL or bare meeting ID")
    parser.add_argument("--passcode", default=None,
                        help="overrides any pwd= in the URL")
    parser.add_argument("--image", default="forgefy-zoom-bot:latest")
    parser.add_argument("--name", default="Forgefy Notetaker",
                        help="display name shown to participants")
    parser.add_argument("--session-id", default=None,
                        help="defaults to a random UUID")
    parser.add_argument("--webhook-secret", default="devsecret",
                        help="must match what mock_backend.py was started with")
    parser.add_argument("--webhook-url", default=None,
                        help="defaults to the host's mock backend on :8080")
    parser.add_argument("--join-token", default=None,
                        help="local recording token, to skip the consent prompt")
    parser.add_argument("--no-consent-prompt", action="store_true",
                        help="do not ask the host to record (requires --join-token)")
    parser.add_argument("--leave-after-silence", type=int, default=0,
                        help="seconds alone in the meeting before leaving; 0 disables")
    args = parser.parse_args()

    client_id = os.environ.get("ZOOM_SDK_CLIENT_ID", "")
    client_secret = os.environ.get("ZOOM_SDK_CLIENT_SECRET", "")
    deepgram_key = os.environ.get("DEEPGRAM_API_KEY", "")

    missing = [
        name for name, value in (
            ("ZOOM_SDK_CLIENT_ID", client_id),
            ("ZOOM_SDK_CLIENT_SECRET", client_secret),
            ("DEEPGRAM_API_KEY", deepgram_key),
        ) if not value
    ]
    if missing:
        print(f"error: set {', '.join(missing)} in your environment", file=sys.stderr)
        return 1

    meeting_number, url_passcode = parse_meeting_url(args.meeting)
    if not meeting_number:
        print(f"error: no numeric meeting ID in {args.meeting!r}.\n"
              "Personal-room links (/my/name) are not supported — use the "
              "numeric invite link.", file=sys.stderr)
        return 1

    passcode = args.passcode if args.passcode is not None else (url_passcode or "")
    session_id = args.session_id or str(uuid.uuid4())

    # host.docker.internal resolves to the host from inside the container on
    # Docker Desktop; on plain Linux the --add-host below provides it.
    webhook_url = args.webhook_url or "http://host.docker.internal:8080/webhook"

    env = {
        "ZOOM_SDK_JWT": build_sdk_jwt(client_id, client_secret),
        "ZOOM_MEETING_NUMBER": meeting_number,
        "ZOOM_MEETING_PASSWORD": passcode,
        "ZOOM_DISPLAY_NAME": args.name,
        "ZOOM_JOIN_TOKEN": args.join_token or "",
        "FORGEFY_REQUIRE_HOST_CONSENT": "false" if args.no_consent_prompt else "true",
        "FORGEFY_LEAVE_AFTER_SILENCE_SECS": str(args.leave_after_silence),
        "FORGEFY_SESSION_ID": session_id,
        "FORGEFY_WEBHOOK_URL": webhook_url,
        "FORGEFY_WEBHOOK_SECRET": args.webhook_secret,
        "DEEPGRAM_API_KEY": deepgram_key,
        "DEEPGRAM_MODEL": os.environ.get("DEEPGRAM_MODEL", "nova-3"),
    }

    command = [
        "docker", "run", "--rm", "-i",
        "--name", f"forgefy-zoom-bot-{session_id[:8]}",
        # Makes host.docker.internal work on native Linux too.
        "--add-host", "host.docker.internal:host-gateway",
        "--memory", "1g",
    ]
    for key, value in env.items():
        command += ["-e", f"{key}={value}"]
    command.append(args.image)

    print(f"session   {session_id}")
    print(f"meeting   {meeting_number}")
    print(f"passcode  {'(set)' if passcode else '(none)'}")
    print(f"reporting {webhook_url}")
    print("\nCtrl-C to make the bot leave.\n")

    try:
        # Inherit stdio so SDK logs stream live; docker forwards our SIGINT to
        # the container, which triggers the graceful leave path.
        return subprocess.call(command)
    except KeyboardInterrupt:
        print("\nstopping…")
        return 0
    except FileNotFoundError:
        print("error: docker not found on PATH", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
