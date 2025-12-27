"""
Structured Logging Module
EPIC-OBS-001 - Phase 2

Provides JSON-formatted logging with:
- Service name tagging
- Correlation ID tracking
- Structured fields for observability
"""

import logging
import sys
from typing import Optional
from pythonjsonlogger import jsonlogger

from .config import config

# Global flag to prevent re-initialization
_logging_initialized = False


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """
    Custom JSON formatter that adds service context to every log.
    
    Adds:
    - service_name
    - service_version
    - environment
    - correlation_id (if available from context)
    """
    
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        
        # Add service context
        log_record["service_name"] = config.service_name
        log_record["service_version"] = config.service_version
        log_record["environment"] = config.environment
        
        # Add correlation ID from record extra fields (set by middleware)
        if hasattr(record, "correlation_id"):
            log_record["correlation_id"] = record.correlation_id


def init_logging(service_name: str, log_level: str = "INFO") -> None:
    """
    Initialize structured logging for the service.
    
    Args:
        service_name: Name of the service
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    
    Note:
        This replaces any existing logging configuration.
        Call once at service startup.
    """
    global _logging_initialized
    
    if _logging_initialized:
        logging.getLogger(__name__).warning(
            f"Logging already initialized for {config.service_name}, skipping"
        )
        return
    
    # Update config
    config.service_name = service_name
    config.log_level = log_level
    
    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    
    # Choose formatter based on config
    if config.log_format == "json":
        formatter = CustomJsonFormatter(
            fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    else:
        # Text format for local development
        formatter = logging.Formatter(
            fmt=f"[%(asctime)s] [{service_name}] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    
    handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    
    _logging_initialized = True
    
    # Log initialization
    logger = logging.getLogger(__name__)
    logger.info(
        f"Structured logging initialized",
        extra={
            "service_name": service_name,
            "log_level": log_level,
            "log_format": config.log_format,
        }
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given name.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance with service context
    
    Example:
        logger = get_logger(__name__)
        logger.info("Processing signal", extra={"symbol": "BTCUSDT"})
    """
    return logging.getLogger(name)


def set_correlation_id(correlation_id: str) -> None:
    """
    Set correlation ID for the current request context.
    
    Note: This should be called by middleware or request handlers.
    Correlation IDs are automatically added to all logs within the request.
    
    Args:
        correlation_id: Unique request identifier
    """
    # Import here to avoid circular dependency with infra/logging
    try:
        from infra.logging.filters import set_correlation_id as _set_correlation_id
        _set_correlation_id(correlation_id)
    except ImportError:
        # Fallback: store in thread-local or context var
        pass
