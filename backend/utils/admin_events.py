"""Shared admin audit event definitions and helpers."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, Mapping, Optional

from fastapi import Request

from .audit import record_admin_action

try:  # pragma: no cover - import path differs in package usage
    from .telemetry import record_admin_event_metric
except ImportError:  # pragma: no cover - fallback for alternative module layout
    try:
        from utils.telemetry import record_admin_event_metric  # type: ignore
    except ImportError:  # pragma: no cover - telemetry optional in some contexts
        record_admin_event_metric = None  # type: ignore


class AdminEvent(str, Enum):
    """String-valued enumeration of admin audit events."""

    ADMIN_AUTH_MISSING = "admin.auth.missing_token"
    ADMIN_AUTH_INVALID = "admin.auth.invalid_token"
    RISK_AUTH_MISSING = "risk.auth.missing_token"
    RISK_AUTH_INVALID = "risk.auth.invalid_token"
    RISK_SNAPSHOT = "risk.snapshot"
    RISK_KILL_SWITCH = "risk.kill_switch"
    RISK_RESET = "risk.reset"
    SETTINGS_UPDATE = "settings.update"
    SETTINGS_READ = "settings.read"
    TRADES_CREATE = "trades.create"
    TRADES_READ = "trades.read"
    TRADES_RECENT = "trades.recent"
    TRADE_LOGS_READ = "trade_logs.read"
    AI_SCAN = "ai.scan"
    AI_RELOAD = "ai.reload"
    AI_TRAIN = "ai.train"
    AI_TASK_LIST = "ai.tasks.list"
    AI_TASK_DETAIL = "ai.tasks.detail"
    AI_STATUS = "ai.status"
    DASHBOARD_STREAM = "dashboard.stream"
    LIQUIDITY_REFRESH = "liquidity.refresh"
    SCHEDULER_STATUS = "scheduler.status"
    SCHEDULER_WARM = "scheduler.warm"
    SCHEDULER_LIQUIDITY = "scheduler.liquidity"
    SCHEDULER_EXECUTION = "scheduler.execution"


_EVENT_METADATA: Mapping[AdminEvent, Dict[str, str]] = {
    AdminEvent.ADMIN_AUTH_MISSING: {
        "category": "auth",
        "scope": "global",
        "reason": "missing_token",
        "severity": "warning",
    },
    AdminEvent.ADMIN_AUTH_INVALID: {
        "category": "auth",
        "scope": "global",
        "reason": "invalid_token",
        "severity": "warning",
    },
    AdminEvent.RISK_AUTH_MISSING: {
        "category": "auth",
        "scope": "risk",
        "reason": "missing_token",
        "severity": "warning",
    },
    AdminEvent.RISK_AUTH_INVALID: {
        "category": "auth",
        "scope": "risk",
        "reason": "invalid_token",
        "severity": "warning",
    },
    AdminEvent.RISK_SNAPSHOT: {
        "category": "risk",
        "action": "snapshot",
        "severity": "info",
    },
    AdminEvent.RISK_KILL_SWITCH: {
        "category": "risk",
        "action": "kill_switch",
        "severity": "high",
    },
    AdminEvent.RISK_RESET: {
        "category": "risk",
        "action": "reset",
        "severity": "high",
    },
    AdminEvent.SETTINGS_UPDATE: {
        "category": "settings",
        "action": "update",
        "severity": "medium",
    },
    AdminEvent.SETTINGS_READ: {
        "category": "settings",
        "action": "read",
        "severity": "info",
    },
    AdminEvent.TRADES_CREATE: {
        "category": "trade",
        "action": "create",
        "severity": "high",
    },
    AdminEvent.TRADES_READ: {
        "category": "trade",
        "action": "read",
        "severity": "info",
    },
    AdminEvent.TRADES_RECENT: {
        "category": "trade",
        "action": "recent",
        "severity": "info",
    },
    AdminEvent.TRADE_LOGS_READ: {
        "category": "trade",
        "action": "logs",
        "severity": "info",
    },
    AdminEvent.AI_SCAN: {
        "category": "ai",
        "action": "scan",
        "severity": "medium",
    },
    AdminEvent.AI_RELOAD: {
        "category": "ai",
        "action": "reload",
        "severity": "medium",
    },
    AdminEvent.AI_TRAIN: {
        "category": "ai",
        "action": "train",
        "severity": "high",
    },
    AdminEvent.AI_TASK_LIST: {
        "category": "ai",
        "action": "tasks.list",
        "severity": "info",
    },
    AdminEvent.AI_TASK_DETAIL: {
        "category": "ai",
        "action": "tasks.detail",
        "severity": "info",
    },
    AdminEvent.AI_STATUS: {
        "category": "ai",
        "action": "status",
        "severity": "info",
    },
    AdminEvent.DASHBOARD_STREAM: {
        "category": "dashboard",
        "action": "stream",
        "severity": "info",
    },
    AdminEvent.LIQUIDITY_REFRESH: {
        "category": "liquidity",
        "action": "refresh",
        "severity": "medium",
    },
    AdminEvent.SCHEDULER_STATUS: {
        "category": "scheduler",
        "action": "status",
        "severity": "info",
    },
    AdminEvent.SCHEDULER_WARM: {
        "category": "scheduler",
        "action": "warm",
        "severity": "medium",
    },
    AdminEvent.SCHEDULER_LIQUIDITY: {
        "category": "scheduler",
        "action": "liquidity",
        "severity": "high",
    },
    AdminEvent.SCHEDULER_EXECUTION: {
        "category": "scheduler",
        "action": "execution",
        "severity": "high",
    },
}


def get_admin_event_metadata(event: AdminEvent) -> Dict[str, str]:
    """Return metadata associated with an admin audit event."""

    return dict(_EVENT_METADATA.get(event, {}))


def record_admin_event(
    event: AdminEvent,
    *,
    request: Optional[Request] = None,
    success: bool = True,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Record an admin audit event with shared metadata."""

    combined_details: Dict[str, Any] = get_admin_event_metadata(event)
    if details:
        combined_details.update(details)

    if record_admin_event_metric is not None:
        try:
            record_admin_event_metric(
                event=event.value,
                category=combined_details.get("category"),
                severity=combined_details.get("severity"),
                success=success,
            )
        except Exception:  # pragma: no cover - telemetry must not break request flow
            logging.getLogger(__name__).exception("Failed to record admin telemetry")

    record_admin_action(
        event.value,
        request=request,
        success=success,
        details=combined_details or None,
    )


__all__ = [
    "AdminEvent",
    "get_admin_event_metadata",
    "record_admin_event",
]
