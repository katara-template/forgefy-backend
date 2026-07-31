# Forgefy Backend — Meeting Mode

AI-powered meeting assistant that joins planning sessions, transcribes them in real time, extracts product requirements via a LangGraph agent pipeline, and generates build-ready blueprints for multi-platform app generation.

## Architecture Overview

```
Browser / Client
    │  REST (FastAPI)          │  WebSocket /ws/voxa
    ▼                          ▼
┌─────────────────────────────────────┐
│           FastAPI App               │
│  /api/v1/auth   /api/v1/voxa/…      │
│  /api/v1/voxa/blueprint/…           │
│  ConnectionManager (Redis pub/sub)  │
└────────────┬────────────────────────┘
             │ Celery tasks
    ┌────────┼─────────────────────────┐
    │        │                         │
    ▼        ▼                         ▼
meeting.audio   meeting.transcribe   meeting.extract
ConnectorWorker  ExtractionWorker    BlueprintWorker
TranscriptionWkr
    │               │                    │
    ▼               ▼                    ▼
Deepgram WS     LangGraph pipeline    BlueprintAggregator
(nova-3)        4× Claude agents      → Blueprint row
    │               │
    └──── Redis pub/sub ────────────────►  WS clients
           voxa:session:{id}
```

## Requirements

- Python 3.12+
- A Firebase project with Firestore enabled, plus a service-account key
- Redis 7
- Deepgram API key (live transcription)
- Anthropic API key (Claude claude-sonnet-4-5 for agent pipeline)
- OpenAI API key (embeddings, optional)
- Docker (required if using Ollama/Qwen3 locally)

## Environment Variables

Copy `.env.example` to `.env` and fill in your values. Below are the variables most relevant to getting a local instance running — see `.env.example` for the full list (Cloudinary, Cloudflare, GitHub OAuth, Supabase OAuth, Neon, Recall.ai, NotchPay, Sentry, etc. are optional integrations, not required to boot the app).

| Variable | Default | Description |
|---|---|---|
| `APP_ENV` | `development` | `development` / `production` |
| `SECRET_KEY` | `changeme` | JWT signing secret — **change in production** |
| `FIREBASE_CREDENTIALS_JSON` | *(empty)* | Base64-encoded Firebase service-account JSON (preferred for prod/Docker) |
| `FIREBASE_CREDENTIALS_PATH` | `firebase-credentials.json` | Path to the service-account JSON file (used if the JSON var above is unset — convenient for local dev) |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis for pub/sub and app state |
| `CELERY_BROKER_URL` | `redis://localhost:6379/1` | Celery broker |
| `CELERY_RESULT_BACKEND` | `redis://localhost:6379/2` | Celery result store |
| `DEEPGRAM_API_KEY` | *(empty)* | Deepgram API key |
| `DEEPGRAM_MODEL` | `nova-3` | Deepgram STT model |
| `ANTHROPIC_API_KEY` | *(empty)* | Anthropic API key |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-5` | Claude model for agent pipeline |
| `GEMINI_API_KEY` | *(empty)* | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model |
| `OLLAMA_URL` | `http://ollama:11434` | Ollama endpoint — use `http://localhost:11434` outside Docker |
| `OLLAMA_MODEL` | `qwen3:8b` | Ollama model tag to use for blueprint generation |
| `OLLAMA_TIMEOUT` | `300` | Request timeout in seconds — increase for slow hardware or long transcripts |
| `BP_MODEL` | `claude` | Blueprint generation backend: `claude` / `gemini` / `Qwen3` |
| `OPENAI_API_KEY` | *(empty)* | OpenAI API key (embeddings) |
| `CORS_ORIGINS` | `["http://localhost:3000"]` | JSON array or comma-separated origins |

## Local Development

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` is the source of truth for dependencies (it's what the Docker image installs too) and already includes the dev/test tooling (pytest, ruff, mypy, black).

### 2. Start infrastructure (Redis)

```bash
docker compose up redis -d
```

### 3. Set up Firestore credentials

Download a service-account key from Firebase Console → Project settings → Service accounts, save it as `firebase-credentials.json` in the repo root (already gitignored), and leave `FIREBASE_CREDENTIALS_JSON` unset in `.env` — the app will fall back to `FIREBASE_CREDENTIALS_PATH`.

### 4. Start the API server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs at [http://localhost:8000/docs](http://localhost:8000/docs) in development (disabled in production).

### 5. Start Celery workers (separate terminals)

```bash
# Transcription + connector worker (meeting.audio queue)
celery -A app.workers.celery_app worker -Q meeting.audio -c 4 --loglevel=info

# Extraction worker (meeting.transcribe queue)
celery -A app.workers.celery_app worker -Q meeting.transcribe -c 2 --loglevel=info

# Blueprint worker (meeting.extract queue)
celery -A app.workers.celery_app worker -Q meeting.extract -c 2 --loglevel=info
```

### Firestore indexes

Firestore auto-indexes every field individually, which covers almost every query in this codebase (`.where("x", "==", y)` on a single field). Two queries need a manually-declared composite index:
- `blueprints` filtered by `session_id` + `approved` (`app/workers/build_worker.py`) — two equality filters on different fields.
- `operator_alerts` filtered by `resolved` + ordered by `created_at` (`app/api/v1/admin.py`) — a filter and an `order_by` on different fields.

That index is declared in `firestore.indexes.json` at the repo root. Deploy it with the Firebase CLI:

```bash
firebase deploy --only firestore:indexes
```

On a fresh Firestore project without it, that one query fails with a `FailedPrecondition` error (Firestore's error message includes a console link to create the index on the fly — but declaring it here means `firebase deploy` handles it instead of a runtime surprise).

## Blueprint Generation Backends

The `BP_MODEL` env var controls which LLM is used to process Deepgram transcripts and produce structured blueprints.

| `BP_MODEL` value | Provider | Requires |
|---|---|---|
| `claude` (default) | Anthropic Claude — 4-agent LangGraph pipeline | `ANTHROPIC_API_KEY` |
| `gemini` | Google Gemini — single REST call | `GEMINI_API_KEY` |
| `Qwen3` | OpenRouter (hosted, per-action routing) **or** local Qwen3 via Ollama | `OPENROUTER_API_KEY`, else Ollama running |

Switch backends by changing `BP_MODEL` in `.env`. No rebuild is needed — the value is read at runtime.

---

## OpenRouter (hosted `Qwen3`, no local GPU)

Setting `OPENROUTER_API_KEY` makes `BP_MODEL=Qwen3` (and `BUILD_MODEL=Qwen3`) run against
OpenRouter's hosted models instead of a local Ollama. Leave the key blank and the Qwen3
setting behaves exactly as before, calling Ollama — nothing else changes.

```bash
OPENROUTER_API_KEY=sk-or-v1-...
BP_MODEL=Qwen3
BUILD_MODEL=Qwen3
```

### Per-action model routing

There is no single "best" model, so `Qwen3` does not mean one model. Each action is routed
to the model that actually suits it — which is frequently *not* a Qwen model. Chains are
defined in [`app/ai/openrouter.py`](app/ai/openrouter.py):

| Action | Primary model | Why |
|---|---|---|
| `synthesis` / `blueprint` | `nvidia/nemotron-3-super-120b` | 1M context + native JSON — swallows a whole transcript |
| `features` | `nvidia/nemotron-3-super-120b` | Structured JSON, fast |
| `design` | `google/gemma-4-31b-it` | Strongest aesthetic priors for palette/typography |
| `naming` / `classify` | `openai/gpt-oss-20b` | Answers one-liners in ~5s instead of reasoning for 30s |
| `plan` | `nvidia/nemotron-3-ultra-550b` | Deepest reasoning, 1M context |
| `code` | `qwen/qwen3-coder` | 1M context + tool calling — the best free coding model |

Every task defines an **ordered chain**, not one model. Free OpenRouter endpoints
rate-limit upstream without warning (HTTP 429 is routine, not exceptional), so a failed,
truncated, or non-JSON reply drops to the next candidate automatically. A blueprint only
fails if every model in the chain fails.

### Free tier and paid escalation

All primaries above are `:free`. The free tier allows 20 requests/minute and 50/day
(1000/day once you have ever purchased $10 of credits). If the free tier is saturated:

```bash
OPENROUTER_ALLOW_PAID=true   # appends DeepSeek/Gemini/Qwen3-Coder as fallbacks
```

Paid fallbacks are cheap (DeepSeek is ~$0.20 per million input tokens) and are only
reached after every free candidate has failed.

`OPENROUTER_MODEL=<id>` pins one model for every action, bypassing routing — a debugging
aid, not a normal setting.

### A note on token budgets

Most strong free models are *reasoning* models, and their reasoning tokens are billed
against `max_tokens` before any answer is emitted. A 64-token budget for a 6-character app
name returns `content: None` — the budget is gone before the model starts answering, and
disabling reasoning via the API is advisory and widely ignored. The client therefore floors
every request at 2048 output tokens and treats `finish_reason: "length"` as a failure worth
retrying on the next model. Do not lower that floor to "save" tokens; free-tier output is
free, and a starved request fails outright.

---

## Ollama / Qwen3 (local blueprint generation)

Ollama lets you run Qwen3 locally inside Docker at no API cost. All blueprint generation requests are routed to `http://ollama:11434` on the internal Docker network.

### Starting Ollama

Ollama is included as a service in `docker-compose.yml`. Start the full stack (including Ollama) with:

```bash
docker compose up --build
```

Or start only Ollama alongside the infrastructure:

```bash
docker compose up redis ollama -d
```

### Pulling the model (first run only)

The model is pulled **automatically** during `docker compose up`. The `api` and `worker` services wait for Ollama's healthcheck to pass before starting, and the healthcheck only passes once `qwen3:8b` is fully downloaded and listed. You do not need to pull it manually.

Watch the download progress in a separate terminal:

```bash
docker compose logs -f ollama
```

The model (~5 GB) is stored in the `ollama_data` Docker volume. Subsequent starts skip the download entirely — the healthcheck passes in a few seconds.

If the pull fails mid-way and you want to retry manually:

```bash
docker compose exec ollama ollama pull qwen3:8b
```

### Switching to Qwen3

In `.env`, set:

```env
BP_MODEL=Qwen3
OLLAMA_URL=http://ollama:11434   # internal Docker network address
OLLAMA_MODEL=qwen3:8b
```

Restart the worker for the change to take effect:

```bash
docker compose restart worker
```

### Using a different model

Any model available on [ollama.com/library](https://ollama.com/library) can be used. Pull it first, then update `OLLAMA_MODEL`:

```bash
docker compose exec ollama ollama pull llama3.2
# then set OLLAMA_MODEL=llama3.2 in .env and restart worker
```

### Local development (outside Docker)

If you're running the API and workers directly on your machine (not in Docker), Ollama still needs to be accessible. Either:

**Option A — run Ollama via Docker with a port binding** (already configured):

```bash
docker compose up ollama -d
# Ollama is now reachable on your host at http://localhost:11434
```

Then in `.env` change the URL to the localhost address:

```env
OLLAMA_URL=http://localhost:11434
```

**Option B — run Ollama natively**: Download from [ollama.com](https://ollama.com), start with `ollama serve`, then pull the model with `ollama pull qwen3:8b`.

### Checking Ollama status

```bash
# List downloaded models
docker compose exec ollama ollama list

# Check the server is responding
curl http://localhost:11434/api/tags

# View recent Ollama logs
docker compose logs --tail=50 ollama
```

### Deploying to a server

**`qwen3:8b` requires meaningful hardware.** Before deploying, understand what you have:

| Server hardware | Inference speed | Usable in production? |
|---|---|---|
| NVIDIA GPU with 8 GB+ VRAM | 5–30 s per request | Yes |
| CPU only, 16 GB+ RAM | 3–15 min per request | No — will timeout |
| CPU only, < 8 GB RAM | OOM crash | No |

#### GPU server (NVIDIA)

Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host, then uncomment the `deploy` block in the `ollama` service in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

Then bring the stack up normally — Ollama will detect and use the GPU automatically.

#### CPU-only server

Use a much smaller model that can realistically run on CPU:

```bash
# In .env
OLLAMA_MODEL=qwen3:1.7b
OLLAMA_TIMEOUT=120
```

Then on the server:

```bash
docker compose exec ollama ollama pull qwen3:1.7b
```

`qwen3:1.7b` (~1 GB) runs in 20–60 s on a CPU-only VPS. Output quality is lower but it will complete within the timeout.

#### No Ollama server (use a hosted API instead)

If your server has no GPU and quality matters, set `BP_MODEL=gemini` or `BP_MODEL=claude` in your production `.env`. Qwen3/Ollama is best suited for local development or GPU-backed deployments.

### Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `OllamaError: service unavailable` | Ollama container not running | `docker compose up ollama -d` |
| `OllamaError: model not found` | Model not pulled yet | `docker compose exec ollama ollama pull qwen3:8b` |
| `OllamaError: timed out after Xs` | CPU inference is too slow | Use a smaller model (`qwen3:1.7b`) or enable GPU — see deployment section above |
| `OllamaError: not valid JSON` | Model returned malformed output | Retry — more common with smaller models; `qwen3:8b` on GPU is the most reliable |

---

## Docker

Run everything — API, workers, beat scheduler, Redis, and Ollama:

```bash
docker compose up --build
```

Data is stored in Firestore (a managed Google Cloud service, not a container in this stack) — see [Environment Variables](#environment-variables) for credential setup.

Services:

| Service | Port | Description |
|---|---|---|
| `api` | 8000 | FastAPI + Uvicorn |
| `worker` | — | Celery worker (all 3 queues) |
| `beat` | — | Celery beat scheduler |
| `redis` | 6379 | Redis 7 |
| `ollama` | 11434 | Ollama local model server (Qwen3) |

## API Reference

### Authentication

All endpoints except `/health` and `/api/v1/auth/*` require a Bearer token.

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/auth/register` | Create account |
| `POST` | `/api/v1/auth/login` | Get access + refresh tokens |
| `POST` | `/api/v1/auth/refresh` | Exchange refresh token for new access token |

### Sessions

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/voxa/session/create` | Create session (WAITING state) |
| `POST` | `/api/v1/voxa/session/join` | Dispatch bot, transition → JOINING |
| `POST` | `/api/v1/voxa/session/end` | End meeting, transition → PROCESSING |
| `GET` | `/api/v1/voxa/session/{id}` | Get session + 20 recent events |

### Blueprints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/voxa/blueprint/{id}` | Get generated blueprint |
| `POST` | `/api/v1/voxa/blueprint/{id}/approve` | Approve blueprint, transition → APPROVED |

### Developer API — API keys

Key management is JWT-authed (dashboard users). The raw key is returned once
at creation and never again — only its SHA-256 hash is stored.

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/keys` | Create an API key (max 10 active per user) |
| `GET` | `/api/v1/keys` | List your keys (prefixes only, newest first) |
| `DELETE` | `/api/v1/keys/{id}` | Revoke a key (idempotent; propagates within ~30s) |

### Developer API — Extract

Machine-authed with `Authorization: Bearer fgy_live_…`. Runs the same agent
pipeline the meeting workers use, synchronously.

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/extract` | Synchronous — transcript (≤50k chars) → features / questions / conflicts / action items |
| `POST` | `/api/v1/extract/jobs` | Async — transcript up to 200k chars; returns `202 {job_id}` |
| `GET` | `/api/v1/extract/jobs/{id}` | Poll job status; `result` present once `status == "done"` |
| `GET` | `/api/v1/usage` | Key owner's tier, monthly token budget, consumption, reset date |

```json
{
  "transcript": "…up to 50k chars…",
  "extractors": ["features", "questions"],   // optional — default all four
  "model_tier": "standard"                    // "standard" (Claude) | "economy" (Qwen3)
}
```

Response groups extracted items by type and reports `usage`
(`input_tokens`/`output_tokens`, metered against the key owner's monthly
budget). Quota policy matches builds: free-tier owners over budget get a 402;
paid owners are transparently served by the economy tier instead.

Async jobs additionally accept:

- `webhook_url` (https) — the finished job is POSTed there, signed with
  `X-Forgefy-Signature: sha256=<HMAC-SHA256 hex>` using the `webhook_secret`
  returned at creation. Delivery is retried 3× with backoff, independently of
  extraction.
- An `Idempotency-Key` header — replaying the same key returns the existing
  job instead of running a second one.

Jobs run on the `meeting.transcribe` Celery queue and are deleted 30 days
after creation by a daily beat task.

### WebSocket

```
ws://localhost:5000/ws/voxa?token=<access_token>
```

**Client → Server events:**

```json
{"type": "joinSession", "session_id": "<uuid>"}
{"type": "streamAudio", "session_id": "<uuid>", "chunk": "<base64-pcm>"}
{"type": "endMeeting", "session_id": "<uuid>"}
{"type": "ping"}
```

**Server → Client events:**

```json
{"type": "transcript", "session_id": "<uuid>", "text": "...", "is_final": true}
{"type": "featureDetected", "session_id": "<uuid>", "sub_state": "FEATURE_FOUND", "payload": {...}}
{"type": "blueprintReady", "session_id": "<uuid>", "blueprint_id": "<uuid>"}
{"type": "meetingStatus", "session_id": "<uuid>", "status": "joined"}
{"type": "pong"}
{"type": "error", "code": "not_joined", "detail": "..."}
```

## Session State Machine

```
WAITING → JOINING → LISTENING → PROCESSING → BLUEPRINT_READY → APPROVED → BUILDING
                ↘ (cancel) ↗
```

LISTENING sub-states (emitted as `featureDetected` WS events):
- `FEATURE_FOUND` — product feature extracted
- `QUESTION_FOUND` — open question detected
- `CONFLICT_FOUND` — conflicting requirements found
- `ACTION_ITEM_FOUND` — action item extracted

## Running Tests

```bash
# All tests
pytest tests/ -q

# Specific suite
pytest tests/test_auth.py tests/test_sessions.py -v

# With coverage
pytest tests/ --cov=app --cov-report=term-missing
```

Test files are named after what they cover (`test_auth.py`, `test_sessions.py`,
`test_pipeline.py`, `workers/test_blueprint_worker.py`, …). List them with:

```bash
pytest tests/ --collect-only -q
```

Note the suite takes ~17 minutes — several tests exercise real retry/backoff
paths rather than mocking the clock, so budget for it in CI.

## Project Structure

```
app/
├── ai/
│   ├── agents/          # 4 Claude extraction agents
│   └── prompts/         # .txt system prompt files
├── api/
│   ├── v1/              # REST endpoints (auth, sessions, blueprints)
│   └── ws/              # WebSocket gateway + ConnectionManager
├── build/               # BlueprintAggregator
├── connectors/          # Recall.ai cloud bot (Meet/Zoom/Teams) + legacy Playwright Meet bot
├── core/                # Exceptions, rate limiting, security, logging
├── db/
│   ├── firebase.py      # Firebase Admin SDK / Firestore client init
│   └── models/          # Firestore document dataclasses (users, sessions, blueprints, etc.)
├── modules/voxa/        # VoxaService + MeetingStateMachine
├── schemas/             # Pydantic request/response models
├── transcription/       # DeepgramClient streaming wrapper
└── workers/             # Celery tasks (connector, transcription, extraction, blueprint)
tests/                   # 419 pytest tests across 44 files
```
