# """Whole-transcript synthesizer using Gemini via REST API.

# Single call that extracts all requirement types at once, identical output
# schema to the Claude synthesizer so both can feed the same pipeline.
# """
# from __future__ import annotations

# import json
# import logging
# import time

# import requests

# from app.ai.agents.base import _load_prompt

# logger = logging.getLogger(__name__)

# _SYSTEM = _load_prompt("synthesizer")
# _GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


# def call_gemini(
#     system_prompt: str,
#     user_content: str,
#     api_key: str,
#     model: str,
#     *,
#     max_tokens: int = 2048,
#     max_retries: int = 3,
# ) -> dict:
#     """Call Gemini and return the parsed JSON response.
    
#     Implements exponential backoff for rate limiting (429) errors.
#     """
#     url = _GEMINI_URL.format(model=model)
#     payload = {
#         "system_instruction": {"parts": [{"text": system_prompt}]},
#         "contents": [{"parts": [{"text": user_content}]}],
#         "generationConfig": {
#             "maxOutputTokens": max_tokens,
#             "responseMimeType": "application/json",
#         },
#     }
    
#     for attempt in range(max_retries):
#         try:
#             resp = requests.post(url, params={"key": api_key}, json=payload, timeout=60)
#             resp.raise_for_status()
#             data = resp.json()
#             raw = data["candidates"][0]["content"]["parts"][0]["text"].strip()
#             try:
#                 return json.loads(raw)
#             except json.JSONDecodeError as exc:
#                 logger.warning("Gemini returned non-JSON: %s", raw[:200])
#                 raise ValueError(f"Gemini response was not valid JSON: {exc}") from exc
#         except requests.exceptions.HTTPError as exc:
#             # Handle rate limiting with exponential backoff
#             if exc.response.status_code == 429:
#                 if attempt < max_retries - 1:
#                     wait_time = (2 ** attempt) * 2  # 2s, 4s, 8s
#                     logger.warning(
#                         "Gemini rate limited. Retrying in %ds (attempt %d/%d)",
#                         wait_time,
#                         attempt + 1,
#                         max_retries,
#                     )
#                     time.sleep(wait_time)
#                     continue
#                 else:
#                     logger.error("Gemini rate limited. Max retries exceeded.")
#                     raise
#             # For other HTTP errors, raise immediately
#             raise


# def run(transcript: str, api_key: str, model: str) -> list[dict]:
#     """Return a flat list of extraction events from a single Gemini call."""
#     result = call_gemini(_SYSTEM, transcript, api_key, model, max_tokens=2048)

#     events: list[dict] = []
#     if app_desc := result.get("app_description", ""):
#         events.append({"sub_state": "APP_DESCRIPTION", "payload": {"text": app_desc}})
#     for feature in result.get("features", []):
#         events.append({"sub_state": "FEATURE_FOUND", "payload": feature})
#     for question in result.get("questions", []):
#         events.append({"sub_state": "QUESTION_FOUND", "payload": question})
#     for conflict in result.get("conflicts", []):
#         events.append({"sub_state": "CONFLICT_FOUND", "payload": conflict})
#     for action_item in result.get("action_items", []):
#         events.append({"sub_state": "ACTION_ITEM_FOUND", "payload": action_item})
#     return events
"""Whole-transcript synthesizer using Gemini via REST API.

Single call that extracts all requirement types at once, identical output
schema to the Claude synthesizer so both can feed the same pipeline.
"""
from __future__ import annotations

import json
import logging
import random
import time

import requests

from app.ai.agents.base import _load_prompt

logger = logging.getLogger(__name__)

_SYSTEM = _load_prompt("synthesizer")
_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

_KNOWN_KEYS = {"app_name", "app_description", "features", "questions", "conflicts", "action_items"}

# HTTP status codes that represent transient server-side failures and are safe
# to retry.  503 (Service Unavailable) is the one that triggered this addition.
_RETRYABLE_HTTP = frozenset({429, 500, 502, 503, 529})


def call_gemini(
    system_prompt: str,
    user_content: str,
    api_key: str,
    model: str,
    *,
    max_tokens: int = 2048,
    max_retries: int = 3,
) -> dict:
    """Call Gemini and return the parsed JSON response.

    Implements exponential backoff with jitter for rate limiting (429) errors
    and transient network failures (timeouts, connection errors).
    """
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1")

    url = _GEMINI_URL.format(model=model)
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": user_content}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
        },
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, params={"key": api_key}, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            candidates = data.get("candidates")
            if not candidates:
                block_reason = data.get("promptFeedback", {}).get("blockReason", "unknown")
                raise ValueError(f"Gemini returned no candidates. Block reason: {block_reason}")

            raw = candidates[0]["content"]["parts"][0]["text"].strip()
            try:
                return json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.warning("Gemini returned non-JSON: %s", raw[:200])
                raise ValueError(f"Gemini response was not valid JSON: {exc}") from exc

        except (requests.exceptions.HTTPError, requests.exceptions.Timeout,
                requests.exceptions.ConnectionError) as exc:
            is_retryable = (
                isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError))
                or (
                    isinstance(exc, requests.exceptions.HTTPError)
                    and exc.response is not None
                    and exc.response.status_code in _RETRYABLE_HTTP
                )
            )

            if is_retryable and attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 2 + random.uniform(0, 1)  # 2–3s, 4–5s, 8–9s
                logger.warning(
                    "Gemini request failed (%s). Retrying in %.1fs (attempt %d/%d)",
                    type(exc).__name__,
                    wait_time,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(wait_time)
                continue

            if is_retryable:
                logger.error("Gemini request failed. Max retries (%d) exceeded.", max_retries)
            raise


def run(transcript: str, api_key: str, model: str) -> list[dict]:
    """Return a flat list of extraction events from a single Gemini call."""
    result = call_gemini(_SYSTEM, transcript, api_key, model, max_tokens=2048)

    if unknown := set(result) - _KNOWN_KEYS:
        logger.warning("Gemini returned unexpected keys: %s", unknown)

    events: list[dict] = []
    if app_name := result.get("app_name", ""):
        events.append({"sub_state": "APP_NAME", "payload": {"text": app_name}})
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