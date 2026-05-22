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

- Python 3.11+
- PostgreSQL 15 with pgvector extension
- Redis 7
- Deepgram API key (live transcription)
- Anthropic API key (Claude claude-sonnet-4-5 for agent pipeline)
- OpenAI API key (embeddings, optional)
- Docker (required if using Ollama/Qwen3 locally)

## Environment Variables

Copy `.env.example` to `.env` and fill in your values:

| Variable | Default | Description |
|---|---|---|
| `APP_ENV` | `development` | `development` / `production` |
| `SECRET_KEY` | `changeme` | JWT signing secret — **change in production** |
| `DATABASE_URL` | `postgresql+asyncpg://voxa:voxa@localhost:5432/voxa` | Async PostgreSQL URL |
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
pip install -e ".[dev]"
```

### 2. Start infrastructure (PostgreSQL + Redis)

```bash
docker compose up postgres redis -d
```

### 3. Run database migrations

```bash
alembic upgrade head
```

### 4. Start the API server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs at [http://localhost:5000/docs](http://localhost:5000/docs)

### 5. Start Celery workers (separate terminals)

```bash
# Transcription + connector worker (meeting.audio queue)
celery -A app.workers.celery_app worker -Q meeting.audio -c 4 --loglevel=info

# Extraction worker (meeting.transcribe queue)
celery -A app.workers.celery_app worker -Q meeting.transcribe -c 2 --loglevel=info

# Blueprint worker (meeting.extract queue)
celery -A app.workers.celery_app worker -Q meeting.extract -c 2 --loglevel=info
```

## Blueprint Generation Backends

The `BP_MODEL` env var controls which LLM is used to process Deepgram transcripts and produce structured blueprints.

| `BP_MODEL` value | Provider | Requires |
|---|---|---|
| `claude` (default) | Anthropic Claude — 4-agent LangGraph pipeline | `ANTHROPIC_API_KEY` |
| `gemini` | Google Gemini — single REST call | `GEMINI_API_KEY` |
| `Qwen3` | Local Qwen3 via Ollama — no external API | Ollama Docker service running |

Switch backends by changing `BP_MODEL` in `.env`. No rebuild is needed — the value is read at runtime.

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
docker compose up postgres redis ollama -d
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

Run everything — API, workers, beat scheduler, PostgreSQL, Redis, and Ollama:

```bash
docker compose up --build
```

Services:

| Service | Port | Description |
|---|---|---|
| `api` | 8000 | FastAPI + Uvicorn |
| `worker` | — | Celery worker (all 3 queues) |
| `beat` | — | Celery beat scheduler |
| `postgres` | 5432 | PostgreSQL 15 + pgvector |
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
# All 91 tests
pytest tests/ -q

# Specific suite
pytest tests/test_auth.py tests/test_sessions.py -v

# With coverage
pytest tests/ --cov=app --cov-report=term-missing
```

Test suites:

| File | Count | Covers |
|---|---|---|
| `test_auth.py` | 10 | Register, login, refresh endpoints |
| `test_sessions.py` | 10 | Session CRUD endpoints |
| `test_state_machine.py` | 10 | State transition logic |
| `test_ws.py` | 14 | WebSocket gateway, ConnectionManager, event schemas |
| `test_transcription.py` | 11 | DeepgramClient, transcription Celery tasks |
| `test_pipeline.py` | 12 | LangGraph pipeline, 4 agents, extraction worker |
| `test_blueprints.py` | 10 | Blueprint endpoints, aggregator, worker |
| `test_connectors.py` | 14 | Meet/Zoom/Teams connectors, factory, worker |

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
├── connectors/          # Playwright Meet bot + Zoom/Teams stubs
├── core/                # Exceptions, rate limiting, security, logging
├── db/
│   └── models/          # SQLAlchemy ORM models (6 tables)
├── modules/voxa/        # VoxaService + MeetingStateMachine
├── schemas/             # Pydantic request/response models
├── transcription/       # DeepgramClient streaming wrapper
└── workers/             # Celery tasks (connector, transcription, extraction, blueprint)
alembic/                 # Database migrations
tests/                   # 91 pytest tests
```
