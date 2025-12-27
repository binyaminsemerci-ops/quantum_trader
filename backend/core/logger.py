"""Structured logging with automatic metadata injection.

This module provides production-ready logging with:
- Structured logging via structlog
- Automatic trace_id propagation
- Contextual metadata (service, module, event_type, symbol)
- JSON output for log aggregation
- Integration with trace_context
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Optional

import structlog
from structlog.contextvars import merge_contextvars
from structlog.processors import JSONRenderer, TimeStamper, add_log_level

from backend.core.trace_context import trace_context

# Module-level cache for loggers
_loggers: dict[str, structlog.BoundLogger] = {}


def configure_logging(
    service_name: str = "quantum_trader",
    log_level: str = "INFO",
    json_output: bool = True,
) -> None:
    """
    Configure global logging for the application.
    
    Args:
        service_name: Name of the service (for log metadata)
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: If True, output JSON logs (for production)
    
    Should be called once at application startup.
    """
    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )
    
    # Structlog processors
    processors = [
        # Add contextvars (trace_id, etc.)
        merge_contextvars,
        
        # Add log level
        add_log_level,
        
        # Add timestamp
        TimeStamper(fmt="iso", utc=True),
        
        # Inject service_name
        structlog.processors.CallsiteParameterAdder(
            [
                structlog.processors.CallsiteParameter.MODULE,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            ],
        ),
        
        # Add trace_id from context
        _add_trace_id_processor,
        
        # Add service_name
        _add_service_name_processor(service_name),
        
        # Render as JSON (production) or console (dev)
        JSONRenderer() if json_output else structlog.dev.ConsoleRenderer(),
    ]
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def _add_trace_id_processor(
    logger: Any,
    method_name: str,
    event_dict: dict,
) -> dict:
    """
    Processor that adds trace_id from context to every log entry.
    
    Args:
        logger: Logger instance
        method_name: Log method name (info, error, etc.)
        event_dict: Log event dictionary
    
    Returns:
        Modified event_dict with trace_id
    """
    trace_id = trace_context.get()
    if trace_id:
        event_dict["trace_id"] = trace_id
    return event_dict


def _add_service_name_processor(service_name: str):
    """
    Create processor that adds service_name to every log entry.
    
    Args:
        service_name: Name of the service
    
    Returns:
        Processor function
    """
    def processor(logger: Any, method_name: str, event_dict: dict) -> dict:
        event_dict["service"] = service_name
        return event_dict
    
    return processor


def get_logger(
    name: str,
    **default_context: Any,
) -> structlog.BoundLogger:
    """
    Get logger with optional default context.
    
    Args:
        name: Logger name (typically __name__)
        **default_context: Default key-value pairs to include in all logs
    
    Returns:
        Configured structlog logger
    
    Example:
        logger = get_logger(__name__, domain="ai_engine")
        logger.info("signal_generated", symbol="BTCUSDT", confidence=0.85)
        
        # Output:
        # {
        #   "event": "signal_generated",
        #   "symbol": "BTCUSDT",
        #   "confidence": 0.85,
        #   "trace_id": "abc123...",
        #   "service": "quantum_trader",
        #   "module": "ai_engine.orchestrator",
        #   "timestamp": "2025-12-01T12:00:00.000Z",
        #   "level": "info",
        #   "domain": "ai_engine"
        # }
    """
    # Check cache
    cache_key = f"{name}:{str(default_context)}"
    if cache_key in _loggers:
        return _loggers[cache_key]
    
    # Create new logger
    logger = structlog.get_logger(name)
    
    # Bind default context
    if default_context:
        logger = logger.bind(**default_context)
    
    # Cache and return
    _loggers[cache_key] = logger
    return logger


def log_event(
    event_type: str,
    level: str = "info",
    **context: Any,
) -> None:
    """
    Log event with automatic trace_id and metadata.
    
    Convenience function for logging events without getting logger instance.
    
    Args:
        event_type: Type of event (e.g., "signal_generated")
        level: Log level (debug, info, warning, error, critical)
        **context: Additional key-value pairs
    
    Example:
        log_event(
            "trade_executed",
            level="info",
            symbol="BTCUSDT",
            action="LONG",
            price=50000.0,
        )
    """
    logger = get_logger("quantum_trader")
    log_method = getattr(logger, level.lower())
    log_method(event_type, **context)


class LoggerMixin:
    """
    Mixin class that provides logging capability to any class.
    
    Usage:
        class MyService(LoggerMixin):
            def __init__(self):
                self._setup_logger("my_service", domain="execution")
            
            def do_something(self):
                self.logger.info("something_done", status="success")
    """
    
    logger: structlog.BoundLogger
    
    def _setup_logger(self, name: str, **default_context: Any) -> None:
        """
        Setup logger for this instance.
        
        Args:
            name: Logger name
            **default_context: Default context for all logs
        """
        self.logger = get_logger(name, **default_context)


# Structured logging helpers for common patterns
class TradingLogger:
    """Helper for logging trading-specific events."""
    
    @staticmethod
    def signal_generated(
        symbol: str,
        action: str,
        confidence: float,
        trace_id: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """Log AI signal generation."""
        if trace_id:
            trace_context.set(trace_id)
        
        log_event(
            "signal_generated",
            level="info",
            symbol=symbol,
            action=action,
            confidence=confidence,
            **extra,
        )
    
    @staticmethod
    def signal_approved(
        symbol: str,
        action: str,
        trace_id: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """Log signal approval by risk management."""
        if trace_id:
            trace_context.set(trace_id)
        
        log_event(
            "signal_approved",
            level="info",
            symbol=symbol,
            action=action,
            **extra,
        )
    
    @staticmethod
    def signal_rejected(
        symbol: str,
        reason: str,
        trace_id: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """Log signal rejection by risk management."""
        if trace_id:
            trace_context.set(trace_id)
        
        log_event(
            "signal_rejected",
            level="warning",
            symbol=symbol,
            reason=reason,
            **extra,
        )
    
    @staticmethod
    def order_submitted(
        symbol: str,
        order_id: str,
        action: str,
        quantity: float,
        trace_id: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """Log order submission."""
        if trace_id:
            trace_context.set(trace_id)
        
        log_event(
            "order_submitted",
            level="info",
            symbol=symbol,
            order_id=order_id,
            action=action,
            quantity=quantity,
            **extra,
        )
    
    @staticmethod
    def order_filled(
        symbol: str,
        order_id: str,
        fill_price: float,
        trace_id: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """Log order fill."""
        if trace_id:
            trace_context.set(trace_id)
        
        log_event(
            "order_filled",
            level="info",
            symbol=symbol,
            order_id=order_id,
            fill_price=fill_price,
            **extra,
        )
    
    @staticmethod
    def position_opened(
        symbol: str,
        position_id: str,
        entry_price: float,
        quantity: float,
        leverage: float,
        trace_id: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """Log position opening."""
        if trace_id:
            trace_context.set(trace_id)
        
        log_event(
            "position_opened",
            level="info",
            symbol=symbol,
            position_id=position_id,
            entry_price=entry_price,
            quantity=quantity,
            leverage=leverage,
            **extra,
        )
    
    @staticmethod
    def position_closed(
        symbol: str,
        position_id: str,
        exit_price: float,
        pnl_usd: float,
        pnl_pct: float,
        trace_id: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """Log position closing."""
        if trace_id:
            trace_context.set(trace_id)
        
        log_event(
            "position_closed",
            level="info",
            symbol=symbol,
            position_id=position_id,
            exit_price=exit_price,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            **extra,
        )
    
    @staticmethod
    def sl_adjusted(
        symbol: str,
        position_id: str,
        old_sl: float,
        new_sl: float,
        reason: str,
        trace_id: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """Log stop loss adjustment."""
        if trace_id:
            trace_context.set(trace_id)
        
        log_event(
            "sl_adjusted",
            level="info",
            symbol=symbol,
            position_id=position_id,
            old_sl=old_sl,
            new_sl=new_sl,
            reason=reason,
            **extra,
        )
    
    @staticmethod
    def error(
        event: str,
        error_message: str,
        symbol: Optional[str] = None,
        trace_id: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """Log error."""
        if trace_id:
            trace_context.set(trace_id)
        
        log_event(
            event,
            level="error",
            error=error_message,
            symbol=symbol,
            **extra,
        )
