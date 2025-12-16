"""Logging configuration utilities for the backend service."""

from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

_REQUEST_ID_CTX_VAR: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)


class RequestIdFilter(logging.Filter):
    """Inject the current request id (if any) into log records."""

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - trivial
        record.request_id = _REQUEST_ID_CTX_VAR.get()
        return True


class JsonFormatter(logging.Formatter):
    """Render log records as JSON for easier ingestion."""

    _RESERVED_ATTRS = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "request_id",
    }

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        payload: Dict[str, Any] = {
            "timestamp": timestamp,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
        }

        request_id = getattr(record, "request_id", None)
        if request_id:
            payload["request_id"] = request_id

        if record.exc_info:
            exc_type, exc_value, _ = record.exc_info
            payload["exception"] = {
                "type": getattr(exc_type, "__name__", str(exc_type)),
                "message": str(exc_value),
                "stack": self.formatException(record.exc_info),
            }
        elif record.exc_text:
            payload["exception"] = {"stack": record.exc_text}

        extra: Dict[str, Any] = {}
        for key, value in record.__dict__.items():
            if key.startswith("_"):
                continue
            if key in self._RESERVED_ATTRS:
                continue
            extra[key] = value
        if extra:
            payload["extra"] = extra

        return json.dumps(payload, default=str, ensure_ascii=False)


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Populate request identifiers for downstream logging and responses."""

    header_name = "X-Request-ID"

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
        request_id = request.headers.get(self.header_name)
        if not request_id:
            request_id = uuid.uuid4().hex

        token = _REQUEST_ID_CTX_VAR.set(request_id)
        try:
            request.state.request_id = request_id
            response = await call_next(request)
        finally:
            _REQUEST_ID_CTX_VAR.reset(token)

        response.headers[self.header_name] = request_id
        return response


_configured = False


def configure_logging(force: bool = False) -> None:
    """Configure application logging as JSON with optional overrides."""

    global _configured
    if _configured and not force:
        return

    log_level = os.getenv("QT_LOG_LEVEL", "INFO").upper()
    try:
        numeric_level = logging.getLevelName(log_level)
        if isinstance(numeric_level, str):  # getLevelName returns name when unknown
            raise ValueError
    except ValueError:
        numeric_level = logging.INFO

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RequestIdFilter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(numeric_level)
    root_logger.addHandler(handler)

    logging.captureWarnings(True)

    for noisy_logger in ("uvicorn", "uvicorn.access"):
        logging.getLogger(noisy_logger).handlers.clear()
        logging.getLogger(noisy_logger).propagate = True

    _configured = True


__all__ = ["configure_logging", "RequestIdMiddleware", "RequestIdFilter"]
