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

API docs at [http://localhost:8000/docs](http://localhost:8000/docs)

### 5. Start Celery workers (separate terminals)

```bash
# Transcription + connector worker (meeting.audio queue)
celery -A app.workers.celery_app worker -Q meeting.audio -c 4 --loglevel=info

# Extraction worker (meeting.transcribe queue)
celery -A app.workers.celery_app worker -Q meeting.transcribe -c 2 --loglevel=info

# Blueprint worker (meeting.extract queue)
celery -A app.workers.celery_app worker -Q meeting.extract -c 2 --loglevel=info
```

## Docker

Run everything — API, workers, beat scheduler, PostgreSQL, Redis:

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
ws://localhost:8000/ws/voxa?token=<access_token>
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
