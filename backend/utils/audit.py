"""Audit logging utilities for admin operations."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from fastapi import Request


_LOGGER = logging.getLogger("quantum_trader.audit.admin")
_LOGGER.setLevel(logging.INFO)

_CURRENT_LOG_PATH: Optional[Path] = None
_LOGGER_INITIALISED = False


def _resolve_log_path() -> Path:
    env_path = os.getenv("QT_ADMIN_AUDIT_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return Path(__file__).resolve().parent.parent / "data" / "admin_audit.log"


def _ensure_logger() -> logging.Logger:
    global _LOGGER_INITIALISED, _CURRENT_LOG_PATH

    log_path = _resolve_log_path()

    if _CURRENT_LOG_PATH != log_path:
        _teardown_logger()
        _CURRENT_LOG_PATH = log_path

    if not _LOGGER_INITIALISED:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        _LOGGER.addHandler(handler)
        _LOGGER.propagate = False
        _LOGGER_INITIALISED = True

    return _LOGGER


def _teardown_logger() -> None:
    global _LOGGER_INITIALISED
    for handler in list(_LOGGER.handlers):
        _LOGGER.removeHandler(handler)
        try:
            handler.close()
        except Exception:  # pragma: no cover - close failures are non-fatal
            pass
    _LOGGER_INITIALISED = False


def reset_admin_audit_logger() -> None:
    """Reset the audit logger so a new handler is created on next use.

    This is primarily intended for tests that need to isolate log files.
    """

    global _CURRENT_LOG_PATH
    _CURRENT_LOG_PATH = None
    _teardown_logger()


def _serialise_details(details: Dict[str, Any]) -> Dict[str, Any]:
    serialised: Dict[str, Any] = {}
    for key, value in details.items():
        try:
            json.dumps(value)
            serialised[key] = value
        except (TypeError, ValueError):
            serialised[key] = str(value)
    return serialised


def record_admin_action(
    event: str,
    *,
    request: Optional["Request"] = None,
    success: bool = True,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Write an admin action entry to the audit log.

    Parameters
    ----------
    event:
        Short identifier for the action (e.g. ``"risk.kill_switch"``).
    request:
        Optional FastAPI request object used to capture request metadata.
    success:
        Indicates whether the action succeeded.
    details:
        Additional structured information to include with the entry.
    """

    entry: Dict[str, Union[str, bool, Dict[str, Any]]] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "success": success,
    }

    if request is not None:
        entry["path"] = request.url.path
        entry["method"] = request.method
        client = getattr(request, "client", None)
        if client is not None and getattr(client, "host", None):
            entry["client_ip"] = client.host
        user_agent = request.headers.get("user-agent")
        if user_agent:
            entry["user_agent"] = user_agent[:200]

    if details:
        entry["details"] = _serialise_details(details)

    try:
        logger = _ensure_logger()
        logger.info(json.dumps(entry, separators=(",", ":")))
    except Exception:  # pragma: no cover - audit logging should never break requests
        logging.getLogger(__name__).exception("Failed to record admin audit action")


__all__ = ["record_admin_action", "reset_admin_audit_logger"]
