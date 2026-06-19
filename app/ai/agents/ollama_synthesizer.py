"""Whole-transcript synthesizer using a local Ollama instance.

Single call that extracts all requirement types at once, identical output
schema to the Claude/Gemini synthesizers so all can feed the same pipeline.
Communicates with Ollama via its Docker-internal endpoint (http://ollama:11434).
"""
from __future__ import annotations

import json
import logging

import requests

from app.ai.agents.base import _load_prompt

logger = logging.getLogger(__name__)

_SYSTEM = _load_prompt("synthesizer")
_CHAT_PATH = "/api/chat"


class OllamaError(RuntimeError):
    """Raised when Ollama cannot fulfill the request."""


def call_ollama(
    system_prompt: str,
    user_content: str,
    base_url: str,
    model: str,
    *,
    timeout: int = 300,
) -> dict:
    """Call Ollama chat API and return the parsed JSON response."""
    url = f"{base_url.rstrip('/')}{_CHAT_PATH}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "stream": True,
        "format": "json",
        "options": {
            "num_ctx": 8192,   # cap context window to prevent OOM on small hardware
            "num_predict": 2048,
        },
    }

    # stream=True: timeout applies per-chunk, not for the full response,
    # so long generations on slow hardware don't hit the read timeout.
    try:
        with requests.post(url, json=payload, timeout=(30, None), stream=True) as resp:
            if resp.status_code == 404:
                raise OllamaError(
                    f"Model '{model}' not found in Ollama. "
                    f"Run 'docker compose exec ollama ollama pull {model}' to load it."
                )
            try:
                resp.raise_for_status()
            except requests.exceptions.HTTPError as exc:
                raise OllamaError(
                    f"Ollama returned HTTP {resp.status_code}: {resp.text[:300]}"
                ) from exc

            content_parts: list[str] = []
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                chunk = json.loads(raw_line)
                msg = chunk.get("message", {})
                if msg.get("content"):
                    content_parts.append(msg["content"])
            raw = "".join(content_parts).strip()
    except OllamaError:
        raise
    except requests.exceptions.ConnectionError as exc:
        raise OllamaError(
            f"Ollama service unavailable at {base_url}. "
            "Ensure the ollama Docker service is running."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise OllamaError(
            "Ollama connection timed out. "
            "The model may have crashed or the host is under heavy load."
        ) from exc

    if not raw:
        raise OllamaError("Ollama returned an empty response body.")

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Ollama returned non-JSON content: %s", raw[:200])
        raise OllamaError(f"Ollama response was not valid JSON: {exc}") from exc


def run(transcript: str, base_url: str, model: str, *, timeout: int = 300) -> list[dict]:
    """Return a flat list of extraction events from a single Ollama call."""
    result = call_ollama(_SYSTEM, transcript, base_url, model, timeout=timeout)

    events: list[dict] = []
    if app_desc := result.get("app_description", ""):
        events.append({"sub_state": "APP_DESCRIPTION", "payload": {"text": app_desc}})
    for feature in result.get("features", []):
        events.append({"sub_state": "FEATURE_FOUND", "payload": feature})
    for question in result.get("questions", []):
        events.append({"sub_state": "QUESTION_FOUND", "payload": question})
    for conflict in result.get("conflicts", []):
        events.append({"sub_state": "CONFLICT_FOUND", "payload": conflict})
    for action_item in result.get("action_items", []):
        events.append({"sub_state": "ACTION_ITEM_FOUND", "payload": action_item})
    return events
