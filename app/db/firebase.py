"""Firebase / Firestore client initialization.

Credentials are resolved in this order:
  1. FIREBASE_CREDENTIALS_JSON env var — base64-encoded service-account JSON
     (preferred for production / Docker / cloud deployments)
  2. FIREBASE_CREDENTIALS_PATH — path to the credentials file on disk
     (convenient for local development)
"""
from __future__ import annotations

import base64
import json
import os

import firebase_admin
from firebase_admin import credentials, firestore_async
from google.cloud.firestore import AsyncClient


def _load_credentials() -> credentials.Certificate:
    """Build a Certificate from env var JSON or fall back to the file path."""
    raw = os.environ.get("FIREBASE_CREDENTIALS_JSON", "").strip()
    if raw:
        cred_dict = json.loads(base64.b64decode(raw))
        return credentials.Certificate(cred_dict)

    path = os.environ.get("FIREBASE_CREDENTIALS_PATH", "firebase-credentials.json")
    return credentials.Certificate(path)


def init_firebase() -> AsyncClient:
    """Initialize Firebase Admin SDK and return an async Firestore client."""
    cred = _load_credentials()
    firebase_admin.initialize_app(cred)
    return firestore_async.client()


def get_firestore_client() -> AsyncClient:
    """Return an async Firestore client, initializing Firebase if needed.

    Safe to call multiple times (e.g. from Celery workers that each run
    initialize_app once per process).
    """
    try:
        app = firebase_admin.get_app()
    except ValueError:
        cred = _load_credentials()
        app = firebase_admin.initialize_app(cred)
    return firestore_async.client(app=app)
