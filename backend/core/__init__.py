"""Backend core module - Infrastructure layer.

This module provides the fundamental infrastructure for the quantum_trader system:
- EventBus: Redis Streams-based event distribution
- PolicyStore: Redis-backed global configuration
- Logger: Structured logging with trace_id
- HealthChecker: Dependency and system health monitoring
- TraceContext: Distributed tracing support
"""

from backend.core.event_bus import (
    EventBus,
    get_event_bus,
    initialize_event_bus,
    shutdown_event_bus,
)
from backend.core.health import (
    HealthChecker,
    HealthReport,
    HealthStatus,
    get_health_checker,
    initialize_health_checker,
)
from backend.core.logger import (
    LoggerMixin,
    TradingLogger,
    configure_logging,
    get_logger,
    log_event,
)
from backend.core.policy_store import (
    PolicyStore,
    get_policy_store,
    initialize_policy_store,
    shutdown_policy_store,
)
from backend.core.trace_context import trace_context

__all__ = [
    # EventBus
    "EventBus",
    "get_event_bus",
    "initialize_event_bus",
    "shutdown_event_bus",
    # PolicyStore
    "PolicyStore",
    "get_policy_store",
    "initialize_policy_store",
    "shutdown_policy_store",
    # Logger
    "configure_logging",
    "get_logger",
    "log_event",
    "LoggerMixin",
    "TradingLogger",
    # Health
    "HealthChecker",
    "HealthReport",
    "HealthStatus",
    "get_health_checker",
    "initialize_health_checker",
    # Trace
    "trace_context",
]
