# Forgefy Backend — Quick Usage Guide

This page provides a concise introduction to running the Forgefy backend for development and basic usage notes.

## Purpose
- Explain local setup and common development commands.

## Prerequisites
- Python 3.10+
- Redis (or use the provided `REDIS_URL` in `.env`)
- Optional: Docker for containerized development

## Setup (local)
1. Create a virtual environment and install deps:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Copy `.env` from the repository root and update secrets as needed.

3. Run database migrations (if applicable):

```bash
alembic upgrade head
```

## Running the app
- Development server (example):

```bash
uvicorn main:app --reload
```

- If using the app package entry:

```bash
uvicorn app.main:app --reload
```

## Celery (background tasks)
- Start worker:

```bash
celery -A app.workers worker --loglevel=info
```

- Start beat scheduler:

```bash
celery -A app.workers beat --loglevel=info
```

## Testing
- Run tests with pytest:

```bash
pytest -q
```

## Contributing notes
- Follow existing code style and tests in `tests/`.
- Add documentation or examples to this `docs/` folder when adding features.

---

If you want a different topic (API reference, architecture, or endpoints), tell me which and I will add it or update this page accordingly.
