"""Structured logging configuration.

Dev: human-readable one-liner.
Prod: JSON (one object per line, machine-parseable).
"""
import json
import logging
import sys
from typing import Any

from app.config import get_settings


class _JsonFormatter(logging.Formatter):
    """Minimal RFC-5424-style JSON formatter — no external deps."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        return json.dumps(payload)


def configure_logging() -> None:
    """Set up root logger based on APP_ENV."""
    settings = get_settings()
    handler = logging.StreamHandler(sys.stdout)

    if settings.APP_ENV == "production":
        handler.setFormatter(_JsonFormatter())
        level = logging.INFO
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d — %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        level = logging.DEBUG

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Quiet noisy third-party loggers
    for noisy in ("uvicorn.access", "httpx", "httpcore", "grpc", "grpc._cython", "grpc._cython.cygrpc"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
