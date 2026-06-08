"""Sanitize build/update errors before they are stored in Firestore or shown to users.

Third-party API errors (billing limits, rate limits, authentication failures) must not
leak provider names, pricing details, or internal credentials to end users.
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


_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    # Anthropic billing / credits
    (re.compile(r"credit balance is too low|insufficient.{0,20}credits|upgrade or purchase credits", re.I),
     "The AI service is temporarily unavailable. Please contact support.",
     "support"),

    # Rate limits / overload (transient — retry is sensible)
    (re.compile(r"rate.?limit|too.?many.?requests|overloaded|capacity|quota.?exceeded", re.I),
     "The AI service is busy right now. Please try again in a few minutes.",
     "retry"),

    # Anthropic / OpenAI authentication misconfiguration
    (re.compile(r"invalid.{0,10}api.?key|authentication.?fail|unauthorized.*anthropic|openai", re.I),
     "The AI service is misconfigured. Please contact support.",
     "support"),

    # GitHub authentication
    (re.compile(r"github.*401|github.*403|bad credentials|git.*authentication|git.*permission denied", re.I),
     "GitHub access failed. Please reconnect your GitHub account or contact support.",
     "support"),

    # GitHub repo name conflict — user can rename and retry
    (re.compile(r"name already exists|repository.*already exists", re.I),
     "A GitHub repository with this name already exists. Try renaming your app and building again.",
     "user_fix"),

    # Deepgram / transcription
    (re.compile(r"deepgram|transcri\w+.*fail|invalid.*api.*key.*deepgram", re.I),
     "The transcription service is temporarily unavailable. Please try again.",
     "retry"),

    # Cloudflare / deployment (non-fatal — code is still on GitHub)
    (re.compile(r"cloudflare|pages\.dev.*error|wrangler.*fail", re.I),
     "The preview deployment failed. Your code is still on GitHub.",
     "retry"),

    # Generic HTTP 5xx
    (re.compile(r"server error|502|503|504|bad gateway|service unavailable", re.I),
     "A service returned an error. Please try again in a moment.",
     "retry"),
]


def sanitize_build_error(exc: Exception) -> BuildError:
    """Return a user-safe BuildError, hiding provider internals."""
    raw = str(exc)
    for pattern, message, action in _PATTERNS:
        if pattern.search(raw):
            return BuildError(message=message, action=action)
    # Not a known provider leak — pass through but cap length
    return BuildError(message=raw[:300], action="retry")
