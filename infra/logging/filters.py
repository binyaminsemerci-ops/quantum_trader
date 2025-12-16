"""
Custom Logging Filters
SPRINT 3 - Module E: Unified Logging

Provides custom filters for adding context to log records.
"""

import logging
import os
import re
import contextvars
from typing import Any, Dict

# Thread-local storage for correlation ID
correlation_id_var = contextvars.ContextVar("correlation_id", default=None)


class CorrelationIdFilter(logging.Filter):
    """
    Adds correlation_id to every log record.
    
    Correlation IDs enable tracing of requests across microservices.
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Get correlation ID from context var (set by middleware)
        correlation_id = correlation_id_var.get()
        
        # If not set, try to get from existing record
        if not correlation_id:
            correlation_id = getattr(record, "correlation_id", None)
        
        # Default to "no-correlation-id" if still not set
        record.correlation_id = correlation_id or "no-correlation-id"
        
        return True


class ServiceNameFilter(logging.Filter):
    """
    Adds service name to every log record.
    
    Service name is determined from environment variable.
    """
    
    def __init__(self):
        super().__init__()
        self.service_name = os.getenv("SERVICE_NAME", "unknown-service")
    
    def filter(self, record: logging.LogRecord) -> bool:
        record.service = self.service_name
        return True


class SensitiveDataFilter(logging.Filter):
    """
    Masks sensitive data in log messages.
    
    Replaces API keys, passwords, tokens, etc. with ***REDACTED***
    """
    
    def __init__(self, fields: list = None):
        super().__init__()
        self.fields = fields or [
            "api_key", "apiKey", "API_KEY",
            "password", "Password", "PASSWORD",
            "secret", "Secret", "SECRET",
            "token", "Token", "TOKEN",
            "authorization", "Authorization",
            "credentials", "Credentials"
        ]
        
        # Compile regex patterns for field names
        self.patterns = []
        for field in self.fields:
            # Match: "field": "value" or field=value
            pattern = rf'("{field}"\s*:\s*"[^"]+"|{field}=[^\s,\)]+)'
            self.patterns.append(re.compile(pattern, re.IGNORECASE))
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Mask message
        if hasattr(record, "msg"):
            record.msg = self._mask_sensitive(str(record.msg))
        
        # Mask args
        if hasattr(record, "args") and record.args:
            record.args = tuple(
                self._mask_sensitive(str(arg)) if isinstance(arg, str) else arg
                for arg in record.args
            )
        
        return True
    
    def _mask_sensitive(self, text: str) -> str:
        """Replace sensitive values with ***REDACTED***"""
        for pattern in self.patterns:
            # Replace value part only
            text = pattern.sub(
                lambda m: m.group(0).split(":", 1)[0] + ': "***REDACTED***"'
                if ":" in m.group(0)
                else m.group(0).split("=", 1)[0] + "=***REDACTED***",
                text
            )
        return text


class EnvironmentFilter(logging.Filter):
    """
    Adds environment info (dev/staging/prod) to log records.
    """
    
    def __init__(self):
        super().__init__()
        self.environment = os.getenv("ENVIRONMENT", "development")
    
    def filter(self, record: logging.LogRecord) -> bool:
        record.environment = self.environment
        return True


class RequestContextFilter(logging.Filter):
    """
    Adds HTTP request context to log records.
    
    Used by FastAPI middleware to attach request_id, method, path, etc.
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Set defaults if not present
        if not hasattr(record, "request_id"):
            record.request_id = "no-request-id"
        
        if not hasattr(record, "request_method"):
            record.request_method = None
        
        if not hasattr(record, "request_path"):
            record.request_path = None
        
        return True


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_correlation_id(correlation_id: str):
    """
    Set correlation ID for current context.
    
    Usage:
        from infra.logging.filters import set_correlation_id
        
        set_correlation_id("abc-123")
        logger.info("This log will have correlation_id=abc-123")
    """
    correlation_id_var.set(correlation_id)


def get_correlation_id() -> str:
    """Get current correlation ID."""
    return correlation_id_var.get() or "no-correlation-id"


def generate_correlation_id() -> str:
    """Generate a new correlation ID (UUID)."""
    import uuid
    return str(uuid.uuid4())


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    import logging
    
    # Configure logger with filters
    logger = logging.getLogger("test")
    logger.setLevel(logging.INFO)
    
    handler = logging.StreamHandler()
    handler.addFilter(CorrelationIdFilter())
    handler.addFilter(ServiceNameFilter())
    handler.addFilter(SensitiveDataFilter())
    
    formatter = logging.Formatter(
        "[%(correlation_id)s] %(service)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Test correlation ID
    set_correlation_id("test-123")
    logger.info("Test message 1")
    
    set_correlation_id("test-456")
    logger.info("Test message 2")
    
    # Test sensitive data masking
    logger.info('User logged in with password="secretpass123"')
    logger.info("API key: api_key=sk-1234567890abcdef")
    logger.info('Auth header: Authorization="Bearer token123"')
    
    # Output:
    # [test-123] ai-engine-service - Test message 1
    # [test-456] ai-engine-service - Test message 2
    # [test-456] ai-engine-service - User logged in with password=***REDACTED***
    # [test-456] ai-engine-service - API key: api_key=***REDACTED***
    # [test-456] ai-engine-service - Auth header: Authorization=***REDACTED***
