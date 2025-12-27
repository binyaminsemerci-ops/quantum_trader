"""
Observability Module for Quantum Trader v2.0
EPIC-OBS-001 - Phase 2: Common Observability Module

Provides unified interface for:
- Structured logging (JSON, correlation IDs)
- Distributed tracing (OpenTelemetry)
- Metrics (Prometheus)

Usage:
    from backend.infra.observability import init_observability, get_logger, get_tracer, metrics
    
    # At service startup
    init_observability(service_name="ai-engine", log_level="INFO")
    
    # In code
    logger = get_logger(__name__)
    tracer = get_tracer()
    metrics.http_requests_total.labels(method="GET", endpoint="/health", status="200").inc()
"""

from .logging import init_logging, get_logger
from .tracing import init_tracing, get_tracer, instrument_fastapi
from .metrics import init_metrics, metrics
from .config import ObservabilityConfig
from .middleware import add_metrics_middleware
from .contract import (
    ObservableService,
    REQUIRED_ENDPOINTS,
    REQUIRED_LABELS,
    REQUIRED_STARTUP_CALL,
    REQUIRED_TRACE_ATTRIBUTES,
)

__all__ = [
    "init_observability",
    "get_logger",
    "get_tracer",
    "instrument_fastapi",
    "add_metrics_middleware",
    "metrics",
    "ObservabilityConfig",
    "ObservableService",
    "REQUIRED_ENDPOINTS",
    "REQUIRED_LABELS",
    "REQUIRED_STARTUP_CALL",
    "REQUIRED_TRACE_ATTRIBUTES",
]


def init_observability(
    service_name: str,
    log_level: str = "INFO",
    enable_tracing: bool = True,
    enable_metrics: bool = True,
) -> None:
    """
    Initialize all observability components for a service.
    
    Args:
        service_name: Name of the service (e.g., "ai-engine", "execution")
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        enable_tracing: Enable OpenTelemetry tracing
        enable_metrics: Enable Prometheus metrics
    
    Example:
        # In microservices/ai_engine/main.py
        from backend.infra.observability import init_observability
        
        init_observability(
            service_name="ai-engine",
            log_level=settings.LOG_LEVEL,
        )
    """
    # Initialize logging first (always enabled)
    init_logging(service_name=service_name, log_level=log_level)
    
    # Initialize tracing if enabled
    if enable_tracing:
        init_tracing(service_name=service_name)
    
    # Initialize metrics if enabled
    if enable_metrics:
        init_metrics(service_name=service_name)
    
    logger = get_logger(__name__)
    logger.info(
        f"Observability initialized for {service_name}",
        extra={
            "tracing_enabled": enable_tracing,
            "metrics_enabled": enable_metrics,
        }
    )
