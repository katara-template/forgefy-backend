# Forgefy Zoom Bot (self-hosted)

A meeting bot that joins a Zoom call using Zoom's Linux Meeting SDK, captures
raw audio, and streams it to Deepgram for live transcription.

**Wired into the backend and off by default.** Zoom meetings follow
`ZOOM_BOT_PROVIDER` in `.env`:

```bash
ZOOM_BOT_PROVIDER=recall        # Recall.ai cloud bot (default, unchanged)
ZOOM_BOT_PROVIDER=self_hosted   # this bot
```

Meet and Teams always use Recall — this covers Zoom only. See
[Enabling it](#enabling-it) for the full switch-on checklist.

---

## Architecture

One container per meeting, two processes inside it:

```
                    ┌──────────────── bot container ────────────────┐
  Zoom meeting ────►│  forgefy-zoom-bot (C++)                       │
                    │    Meeting SDK: join, consent, raw audio      │
                    │      │ PCM over unix socket   │ status JSON   │
                    │      ▼                        ▼ on stdout     │
                    │  sidecar (Python)                             │
                    │    Deepgram live WS ──► transcripts           │
                    └───────────────────│───────────────────────────┘
                                        ▼  signed HTTP webhooks
                                  Forgefy backend
```

Why split this way: the Meeting SDK is C++ only, but everything downstream
(Deepgram, retries, the backend contract) is far easier in Python. The seam is
a Unix socket carrying raw PCM plus a JSON status stream on stdout.

Design points worth knowing:

- **The sidecar owns the process tree.** It binds the audio socket *before*
  spawning the bot, so audio is never produced with nowhere to go.
- **The bot never blocks on the network.** Audio is queued and drained by a
  writer thread; if the sidecar stalls, the oldest audio is dropped rather than
  stalling the SDK callback or growing memory without bound.
- **The SDK client secret never enters the container.** The orchestrator mints
  the SDK JWT and passes only that in.
- **Audio format is never hardcoded.** The bot reads the sample rate and
  channel count from the SDK and sends them in a header; the sidecar configures
  Deepgram from that.

| Path | What it is |
|---|---|
| `src/` | C++ Meeting SDK client |
| `sidecar/` | Python: PCM → Deepgram → webhooks |
| `scripts/` | Standalone test harness (no backend needed) |
| `lib/zoomsdk/` | Where you unpack the SDK (git-ignored) |

The backend-side pieces live in the app proper:

| Path | What it is |
|---|---|
| [app/connectors/zoom_selfhosted.py](../app/connectors/zoom_selfhosted.py) | Spawns and stops bot containers |
| [app/connectors/zoom_meeting.py](../app/connectors/zoom_meeting.py) | URL parsing + SDK JWT minting |
| [app/api/v1/zoom_bot.py](../app/api/v1/zoom_bot.py) | `/api/v1/webhooks/zoom-bot` receiver |
| [app/workers/zoom_bot_worker.py](../app/workers/zoom_bot_worker.py) | Container teardown tasks |
| [app/integrations/zoom_oauth.py](../app/integrations/zoom_oauth.py) | Host OAuth, OBF + join-token minting |
| [app/api/v1/zoom_oauth.py](../app/api/v1/zoom_oauth.py) | `/api/v1/zoom/*` account linking |

---

## Prerequisites

### 1. Zoom account and app — read this before estimating timeline

This is the part that gates the schedule, not the code.

- You need a **Meeting SDK app** created at [marketplace.zoom.us](https://marketplace.zoom.us).
  It yields a Client ID and Client Secret.
- **Joining meetings on other people's Zoom accounts requires OAuth.** Since
  **2 March 2026**, a Meeting SDK app may not join externally-hosted meetings
  on its JWT alone — it must also present an OBF token, a ZAK token, or use
  RTMS. See [Host account linking](#host-account-linking). Meetings on your own
  account are unaffected.
- **Raw audio requires local recording privilege.** Zoom will not release audio
  frames otherwise. In practice this means either the host approves the bot
  live, or you pre-authorize it with a local recording join token minted through
  the Zoom REST API (which itself requires OAuth scopes granted by the host's
  account).
- **Local recording must be enabled at the account level**, otherwise the
  approval prompt never appears for the host to accept.
- Raw data access has historically required a **paid/ISV account tier**, and
  publishing an app for use outside your own account requires Zoom's app
  review. Budget real calendar time for this — it is an approvals process, not
  a configuration step.

Verify all of the above against Zoom's current docs before committing to a
date; Zoom changes these requirements more often than they change the SDK.

### 2. The SDK bundle

The Meeting SDK is licensed and cannot be committed or fetched at build time.
Download the **Linux Meeting SDK** from the Zoom Marketplace and unpack it so
the layout is:

```
lib/zoomsdk/
├── h/                  # headers, including zoom_sdk.h
├── libmeetingsdk.so
└── qt_libs/
```

`cmake` fails immediately with this instruction if the headers are missing.

### 3. Platform

**x86_64 Linux only.** Zoom ships no ARM build of the Linux SDK, so this cannot
run on Graviton, Ampere, or Apple Silicon (including under emulation in
practice). Docker Desktop on Windows/macOS with an x86_64 host is fine.

Budget roughly **1 GB RAM and a meaningful CPU share per concurrent meeting** —
the SDK runs a full media stack.

---

## Build

```bash
cd forgefy-backend/zoom-bot
docker compose build          # produces forgefy-zoom-bot:latest
```

---

## Testing, milestone by milestone

Each step is independently verifiable. Do them in order — a failure at step 3
is much easier to diagnose if 1 and 2 already passed.

### Milestone 1 — the webhook contract (no Zoom, no Docker)

Already verified, but re-runnable at any time:

```bash
python scripts/mock_backend.py --secret testsecret --port 8098
```

In another terminal, post signed events through the real sidecar client:

```bash
python -c "
import sys, asyncio; sys.path.insert(0,'.')
from sidecar.backend import BackendClient
async def m():
    c = BackendClient('http://127.0.0.1:8098/webhook','testsecret','sess-1')
    await c.send_status('joining')
    await c.send_transcript('A final sentence.', is_final=True, speaker='Speaker 0')
    await c.aclose()
asyncio.run(m())"
```

**Expect:** the mock prints a yellow `● joining` and a green `Speaker 0: A final
sentence.` Repeat with a different `--secret` and expect a `401` — that proves
signature verification is live, not decorative.

### Milestone 2 — the bot joins a meeting

Start a Zoom meeting from a normal client, then:

```bash
python scripts/mock_backend.py --secret devsecret      # terminal 1

export ZOOM_SDK_CLIENT_ID=... ZOOM_SDK_CLIENT_SECRET=... DEEPGRAM_API_KEY=...
python scripts/run_local.py "https://zoom.us/j/1234567890?pwd=abc"   # terminal 2
```

**Expect, in order:** `starting` → `authenticated` → `joining` → `in_meeting`,
and "Forgefy Notetaker" visible in the participant list. If the meeting has a
waiting room you will see `in_waiting_room` and the bot will wait there until
admitted.

**If it stops at `authenticated`:** the JWT was accepted but the join failed —
usually a wrong meeting ID or passcode. The `[error]` line on stderr carries
the SDK error code.

### Milestone 3 — consent and audio

Once the bot is in the meeting it asks the host for recording permission. As
host you will see a permission prompt.

**Expect:** `awaiting_consent`, then on approval `consent_granted` → `recording`,
and on stderr `raw audio flowing at 32000Hz, 1ch`.

Decline it instead and expect `consent_denied` with no audio ever captured.
That is the consent gate working — see [Consent](#consent).

### Milestone 4 — live transcription

Talk in the meeting.

**Expect:** dim interim lines appearing within about a second and firming up
into green final lines. Interim results are what the backend will forward to
the live UI; only finals trigger requirement extraction.

**If statuses arrive but no transcripts:** the Deepgram key or model is wrong.
The sidecar logs the connection failure as JSON on stderr.

### Milestone 5 — clean shutdown

Press Ctrl-C.

**Expect:** `leaving` → `ended` → `stopped`, the bot disappears from the
participant list *before* the container exits, and the final utterance still
arrives — the shutdown path flushes Deepgram rather than dropping the tail.

---

## Host account linking

Whose meeting the bot can join depends entirely on this:

| Meeting hosted on | Needs |
|---|---|
| **Your own Zoom account** | Nothing extra — SDK JWT is enough |
| **Anyone else's account** | That host must link their Zoom account to Forgefy |

Since 2 March 2026 Zoom rejects Meeting SDK apps joining external meetings
without an OBF ("on behalf of") token. OBF tokens are minted from the host's
OAuth grant, so each host installs the Forgefy app once:

```
GET  /api/v1/zoom/authorize    → returns the Zoom consent URL
GET  /api/v1/zoom/callback     → Zoom redirects here; tokens stored encrypted
GET  /api/v1/zoom/status       → { linked: bool, email: str }
POST /api/v1/zoom/disconnect   → forget the grant
```

Register `<PUBLIC_API_BASE_URL>/api/v1/zoom/callback` as a redirect URL on the
Marketplace app, with scopes `user:read:token`, `user:read`, and
`meeting:read:local_recording_token`.

From then on, [zoom_oauth.py](../app/integrations/zoom_oauth.py) mints a fresh
OBF token per meeting, immediately before the container launches. Access tokens
refresh transparently. One detail that matters: **Zoom rotates refresh tokens**
— every refresh invalidates the one used, so the replacement is persisted
before it is handed back, under a Redis lock that stops two simultaneous
meetings racing and spending the same token twice.

If a host has not linked their account the bot still launches; it just cannot
join their meeting, and the reason appears in the container logs rather than
failing silently.

## Consent

The prompt this was built from asked for "a config flag requiring explicit host
confirmation before streaming audio (default: on)." Worth being precise about
what that flag can and cannot do:

**Zoom enforces this itself.** Raw audio is gated behind local recording
privilege, so there is no "just start streaming" mode available to disable.
`FORGEFY_REQUIRE_HOST_CONSENT` therefore chooses between:

- `true` (default) — ask the host live via `RequestLocalRecordingPrivilege()`
  and wait. No audio until they approve.
- `false` — do not prompt. Only works if a local recording join token
  pre-authorized the bot; otherwise it sits in the meeting capturing nothing.

Either way, audio cannot be captured without the host's agreement. Disclosure
is additionally provided by the display name, which is visible to every
participant for the whole meeting.

A chat announcement on join is implemented but **off by default**: Zoom's chat
builder API has changed shape across SDK releases and I could not verify it
against your bundle's headers. Check `meeting_chat_interface.h`, then build
with `-DFORGEFY_ENABLE_CHAT_ANNOUNCE=ON` to enable it.

---

## Configuration

Read by the bot container:

| Variable | Default | Purpose |
|---|---|---|
| `ZOOM_SDK_JWT` | *(required)* | SDK JWT, minted by the orchestrator |
| `ZOOM_MEETING_NUMBER` | *(required)* | Numeric meeting ID |
| `ZOOM_MEETING_PASSWORD` | — | Passcode, if any |
| `ZOOM_DISPLAY_NAME` | `Forgefy Notetaker` | Shown to participants |
| `ZOOM_JOIN_TOKEN` | — | Local recording token, skips the prompt |
| `ZOOM_ON_BEHALF_TOKEN` | — | Required for meetings on other Zoom accounts |
| `ZOOM_ZAK` | — | Join carrying the host's identity, instead of OBF |
| `FORGEFY_SESSION_ID` | *(required)* | Correlates events to a session |
| `FORGEFY_WEBHOOK_URL` | *(required)* | Where events are posted |
| `FORGEFY_WEBHOOK_SECRET` | *(required)* | Per-session HMAC secret |
| `FORGEFY_REQUIRE_HOST_CONSENT` | `true` | See [Consent](#consent) |
| `FORGEFY_LEAVE_AFTER_SILENCE_SECS` | `0` | Leave when alone; 0 disables |
| `DEEPGRAM_API_KEY` | *(required)* | |
| `DEEPGRAM_MODEL` | `nova-3` | |

---

## Enabling it

Setting `ZOOM_BOT_PROVIDER=self_hosted` alone is not enough — four things must
be true or the first Zoom meeting will fail.

1. **The SDK is unpacked** into `lib/zoomsdk/` (see [Prerequisites](#prerequisites)).

2. **The image is built:**

   ```bash
   docker compose --profile build-only build zoom-bot
   ```

3. **Credentials are set** in `.env`:

   ```bash
   ZOOM_BOT_PROVIDER=self_hosted
   ZOOM_SDK_CLIENT_ID=...
   ZOOM_SDK_CLIENT_SECRET=...
   DEEPGRAM_API_KEY=...          # already set if Recall transcription worked
   ZOOM_BOT_NETWORK=forgefy-backend_default
   ```

   `ZOOM_BOT_NETWORK` must match your compose project's network, or the bot
   cannot reach `api` to report events. Check with `docker network ls`.

4. **The worker has the Docker socket.** Already added to
   [docker-compose.yml](../docker-compose.yml). Be deliberate about it: socket
   access is effectively root on the host, so the Celery worker becomes a
   privileged process. Remove that volume line and set `ZOOM_BOT_PROVIDER=recall`
   if that trade is not acceptable; if you want self-hosted bots *without* it,
   put the socket behind a small launcher service and have the connector call
   that instead.

Flipping back is just `ZOOM_BOT_PROVIDER=recall` and a restart. Meetings
already running are unaffected — teardown dispatches on which bot actually
served the session, not on current config.

Because the bot reaches the API over the internal Docker network, the Zoom path
needs no publicly reachable `PUBLIC_API_BASE_URL` and no tunnel — unlike Recall.

### How it hooks in

Nothing about the request flow changes. `POST /api/v1/voxa/session/create`
still dispatches `dispatch_connector`, which still calls
`get_connector(platform).join(...)`. Only what that returns differs:

```
Platform.ZOOM + provider=recall       → RecallConnector        (HTTP to recall.ai)
Platform.ZOOM + provider=self_hosted  → ZoomSelfHostedConnector (spawns container)
Platform.MEET / TEAMS                 → RecallConnector        (always)
```

The self-hosted connector satisfies the same `MeetingConnector` protocol, so
`dispatch_connector` itself was not modified.

---

## Not done

Deliberate omissions, so they are not mistaken for oversights:

- **Per-participant audio.** `FORGEFY_SEPARATE_PARTICIPANT_AUDIO` proves the
  per-speaker callbacks arrive but routes them into one socket, which
  interleaves speakers unusably. Doing it properly needs one sub-stream and one
  Deepgram connection per participant. Mixed audio with Deepgram diarization is
  what MVP uses.
- **Video.** Audio only. The renderer is not wired up.
- **Meet and Teams.** Out of scope by design; they stay on Recall. The
  `MeetingConnector` protocol is the seam if that ever changes.
- **Nothing here has been compiled or run against a real meeting.** The C++ is
  written against interfaces verified from Zoom's current published sample, but
  there is no SDK bundle and no x86_64 Linux toolchain on this machine. The
  Python has been executed and the webhook contract verified end to end;
  the C++ has not.
