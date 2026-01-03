"""
Standardized JSON logging with correlation_id tracking
P1-B: Ops Hardening - Production-ready logging
"""

import logging
import json
import sys
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from contextvars import ContextVar

# Context variable for correlation_id (thread-safe)
correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


class JSONFormatter(logging.Formatter):
    """
    JSON log formatter with required fields:
    - ts: ISO timestamp
    - level: log level
    - service: service name
    - event: event type
    - correlation_id: tracking ID across services
    - msg: message
    - extra: additional context
    """
    
    def __init__(self, service_name: str):
        super().__init__()
        self.service_name = service_name
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "service": self.service_name,
            "event": getattr(record, 'event', 'LOG'),
            "correlation_id": getattr(record, 'correlation_id', None) or correlation_id_var.get(),
            "msg": record.getMessage(),
        }
        
        # Add optional fields if present
        optional_fields = [
            'symbol', 'order_id', 'intent_id', 'strategy_id', 
            'latency_ms', 'confidence', 'pnl', 'side', 'qty', 'price'
        ]
        
        for field in optional_fields:
            if hasattr(record, field):
                log_data[field] = getattr(record, field)
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add any extra fields from record.__dict__
        extra = {}
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName', 
                          'levelname', 'lineno', 'module', 'msecs', 'message', 'pathname',
                          'process', 'processName', 'relativeCreated', 'thread', 'threadName',
                          'exc_info', 'exc_text', 'stack_info'] and not key.startswith('_'):
                if key not in log_data and key not in optional_fields:
                    extra[key] = value
        
        if extra:
            log_data['extra'] = extra
        
        return json.dumps(log_data, default=str)


class CorrelationAdapter(logging.LoggerAdapter):
    """
    Logger adapter that automatically includes correlation_id in all logs
    """
    
    def process(self, msg, kwargs):
        # Add correlation_id from context if not already present
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        
        if 'correlation_id' not in kwargs['extra']:
            corr_id = correlation_id_var.get()
            if corr_id:
                kwargs['extra']['correlation_id'] = corr_id
        
        return msg, kwargs


def setup_json_logging(service_name: str, level: str = "INFO") -> logging.Logger:
    """
    Setup JSON logging for a service
    
    Args:
        service_name: Name of the service (e.g., 'auto_executor', 'ai_engine')
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add JSON handler to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter(service_name))
    logger.addHandler(handler)
    
    # Return adapter for correlation_id support
    return CorrelationAdapter(logger, {})


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """
    Set correlation_id in context
    
    Args:
        correlation_id: Correlation ID to set, or None to generate new
    
    Returns:
        The correlation_id that was set
    """
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    
    correlation_id_var.set(correlation_id)
    return correlation_id


def get_correlation_id() -> Optional[str]:
    """
    Get current correlation_id from context
    
    Returns:
        Current correlation_id or None
    """
    return correlation_id_var.get()


def clear_correlation_id():
    """
    Clear correlation_id from context
    """
    correlation_id_var.set(None)


def log_event(logger: logging.Logger, level: str, event: str, msg: str, **kwargs):
    """
    Helper to log structured event with correlation_id
    
    Args:
        logger: Logger instance
        level: Log level (debug, info, warning, error, critical)
        event: Event type (INTENT_RECEIVED, ORDER_SUBMIT, etc.)
        msg: Log message
        **kwargs: Additional fields (symbol, order_id, latency_ms, etc.)
    """
    log_func = getattr(logger, level.lower())
    
    # Add correlation_id if in context
    if 'correlation_id' not in kwargs:
        corr_id = get_correlation_id()
        if corr_id:
            kwargs['correlation_id'] = corr_id
    
    log_func(msg, extra={'event': event, **kwargs})


# Convenience functions
def log_intent_received(logger: logging.Logger, intent_id: str, symbol: str, 
                        confidence: float, correlation_id: str, **kwargs):
    """Log INTENT_RECEIVED event"""
    log_event(
        logger, 'info', 'INTENT_RECEIVED',
        f"Intent received: {symbol}",
        intent_id=intent_id,
        symbol=symbol,
        confidence=confidence,
        correlation_id=correlation_id,
        **kwargs
    )


def log_order_submit(logger: logging.Logger, symbol: str, side: str, qty: float,
                    order_type: str, correlation_id: str, **kwargs):
    """Log ORDER_SUBMIT event"""
    log_event(
        logger, 'info', 'ORDER_SUBMIT',
        f"Submitting order: {side} {qty} {symbol}",
        symbol=symbol,
        side=side,
        qty=qty,
        order_type=order_type,
        correlation_id=correlation_id,
        **kwargs
    )


def log_order_response(logger: logging.Logger, order_id: str, symbol: str,
                      status: str, correlation_id: str, **kwargs):
    """Log ORDER_RESPONSE event"""
    log_event(
        logger, 'info', 'ORDER_RESPONSE',
        f"Order response: {order_id} - {status}",
        order_id=order_id,
        symbol=symbol,
        status=status,
        correlation_id=correlation_id,
        **kwargs
    )


def log_order_error(logger: logging.Logger, symbol: str, error: str,
                   correlation_id: str, **kwargs):
    """Log ORDER_ERROR event"""
    log_event(
        logger, 'error', 'ORDER_ERROR',
        f"Order error: {error}",
        symbol=symbol,
        error=error,
        correlation_id=correlation_id,
        **kwargs
    )


def log_corr_assigned(logger: logging.Logger, old_id: Optional[str], new_id: str, **kwargs):
    """Log CORR_ASSIGNED event when correlation_id is generated"""
    log_event(
        logger, 'info', 'CORR_ASSIGNED',
        f"Correlation ID assigned: {new_id}" + (f" (was: {old_id})" if old_id else ""),
        correlation_id=new_id,
        old_correlation_id=old_id,
        **kwargs
    )
