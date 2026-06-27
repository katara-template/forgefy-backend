"""Sanitize build/update errors before they are stored in Firestore or shown to users.

Third-party API errors (billing limits, rate limits, authentication failures, internal URLs,
API keys) must NEVER reach end users. Only human-readable, actionable messages are shown.
"""
from __future__ import annotations

import re
from typing import NamedTuple


class BuildError(NamedTuple):
    message: str
    # "retry"    — transient; show a "Try again" button
    # "user_fix" — user can ask the agent to fix; show "Ask agent to fix" + "Cancel"
    # "support"  — needs human intervention; show "Contact support" button
    action: str


# ── Strip sensitive details before any pattern matching ──────────────────────
# Remove API keys, auth tokens, and full URLs from the raw error string.
_SCRUB_PATTERNS: list[re.Pattern] = [
    re.compile(r"https?://\S+", re.I),                      # full URLs
    re.compile(r"key=[A-Za-z0-9_\-]{10,}", re.I),           # ?key=... query params
    re.compile(r"AIza[A-Za-z0-9_\-]{30,}", re.I),           # Google API keys
    re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]{10,}", re.I),    # Bearer tokens
    re.compile(r"ghp_[A-Za-z0-9]{30,}"),                    # GitHub PATs
    re.compile(r"sk-[A-Za-z0-9]{30,}"),                     # OpenAI keys
    re.compile(r"cfat_[A-Za-z0-9_\-]{10,}", re.I),         # Cloudflare tokens
]


def _scrub(raw: str) -> str:
    """Remove URLs, API keys, and tokens from an error string."""
    for pattern in _SCRUB_PATTERNS:
        raw = pattern.sub("[redacted]", raw)
    return raw


# ── User-facing patterns (matched against the scrubbed string) ───────────────
_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    # AI provider billing / credits
    (re.compile(r"credit balance is too low|insufficient.{0,20}credits|upgrade or purchase credits", re.I),
     "Your AI service plan has run out of credits. Please contact support.",
     "support"),

    # Rate limits / overload (transient — retry is sensible)
    (re.compile(r"rate.?limit|too.?many.?requests|overloaded|capacity|quota.?exceeded", re.I),
     "The AI service is busy right now. Please try again in a few minutes.",
     "retry"),

    # Generic HTTP 5xx / transient AI provider errors
    (re.compile(r"server error|502|503|504|bad gateway|service unavailable|internal server", re.I),
     "The AI service returned an error. Please try again in a moment.",
     "retry"),

    # AI provider request failures (Gemini, OpenAI, Anthropic, Ollama)
    (re.compile(r"(gemini|openai|anthropic|ollama|gpt).{0,30}(request failed|error|failed)", re.I),
     "The AI service encountered a problem. Please try again.",
     "retry"),

    # Agent iteration limit — not really an error, more of a status
    (re.compile(r"agent reached iteration limit", re.I),
     "The agent reached its step limit. Send the same request again to continue.",
     "retry"),

    # Authentication / API key misconfiguration
    (re.compile(r"invalid.{0,10}api.?key|authentication.?fail|unauthorized|permission denied", re.I),
     "The AI service is misconfigured. Please contact support.",
     "support"),

    # GitHub authentication
    (re.compile(r"github.*401|github.*403|bad credentials|git.*authentication|remote.*rejected", re.I),
     "GitHub access failed. Please reconnect your GitHub account or contact support.",
     "support"),

    # GitHub repo name conflict
    (re.compile(r"name already exists|repository.*already exists", re.I),
     "A GitHub repository with this name already exists. Try renaming your app and building again.",
     "user_fix"),

    # Deepgram / transcription
    (re.compile(r"deepgram|transcri\w+.*fail", re.I),
     "The transcription service is temporarily unavailable. Please try again.",
     "retry"),

    # Cloudflare / deployment (non-fatal — code is still on GitHub)
    (re.compile(r"cloudflare|pages\.dev.*error|wrangler.*fail", re.I),
     "The preview deployment failed. Your code is still saved on GitHub.",
     "retry"),

    # Compilation errors — generic fallback (auto-fix exhaustion is handled separately)
    (re.compile(r"flutter build|gradle.*failed|npm run build|exit code [0-9]", re.I),
     "The preview build failed. Send a chat message like 'Fix the build errors' to try again.",
     "user_fix"),

    # File system / workspace errors
    (re.compile(r"no such file|permission denied.*workspace|disk quota|no space left", re.I),
     "A file system error occurred. Please try again or contact support.",
     "support"),
]


def sanitize_build_error(exc: Exception) -> BuildError:
    """Return a user-safe BuildError, scrubbing URLs/keys and mapping to friendly messages."""
    raw = str(exc)
    scrubbed = _scrub(raw)

    for pattern, message, action in _PATTERNS:
        if pattern.search(scrubbed):
            return BuildError(message=message, action=action)

    # Fallback: never show raw technical output.
    # Log the real error server-side (callers do this via logger.error).
    return BuildError(
        message="Something went wrong while processing your request. Please try again.",
        action="retry",
    )
