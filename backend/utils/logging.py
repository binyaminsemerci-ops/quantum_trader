
"""Application logging helpers."""

from __future__ import annotations

import json
import logging
from logging.config import dictConfig
from typing import Any, Dict


DEFAULT_LOGGING_CONFIG: Dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "structured": {
            "()": "backend.utils.logging._JsonFormatter",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "structured",
            "level": "INFO",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"],
    },
}


class _JsonFormatter(logging.Formatter):
    """Small JSON formatter for structured logs."""

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - trivial
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            payload.update(record.extra)
        return json.dumps(payload, ensure_ascii=False)


_configured = False


def configure_logging(config: Dict[str, Any] | None = None) -> None:
    global _configured
    if _configured:
        return
    dictConfig(config or DEFAULT_LOGGING_CONFIG)
    _configured = True


def get_logger(name: str | None = None) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


__all__ = ["configure_logging", "get_logger"]
