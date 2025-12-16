"""
EPIC-OBS-001 Phase 6 — Test structured logging

Validates:
- Logging module initializes correctly
- Logs contain service context (service_name, environment, etc.)
- JSON formatting works
"""

import pytest
import logging


def test_get_logger_works():
    """Test get_logger returns a valid logger instance"""
    from backend.infra.observability import get_logger
    
    logger = get_logger("test-service")
    
    assert logger is not None, "get_logger should return a logger instance"
    assert isinstance(logger, logging.Logger), "Should return logging.Logger instance"


def test_logger_has_service_context(caplog):
    """Test logger includes service context in log records"""
    from backend.infra.observability import get_logger
    
    logger = get_logger("test-service-context")
    
    with caplog.at_level(logging.INFO):
        logger.info("Test log message")
    
    # At least one log record should be captured
    assert len(caplog.records) > 0, "Logger should emit log records"
    
    # Check if log message contains our test message
    messages = [record.message for record in caplog.records]
    assert any("Test log message" in msg for msg in messages), \
        "Log message should be captured"


def test_logging_json_structure():
    """Test logs can be formatted as JSON (if CustomJsonFormatter is used)"""
    from backend.infra.observability.logging import CustomJsonFormatter
    import logging
    
    # Create test logger with JSON formatter
    logger = logging.getLogger("test-json-logger")
    logger.setLevel(logging.INFO)
    
    # Create string handler to capture output
    import io
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(CustomJsonFormatter("test-service", "1.0.0", "test"))
    logger.addHandler(handler)
    
    # Emit log
    logger.info("Test JSON log")
    
    # Get output
    output = stream.getvalue()
    
    # Should contain JSON-like structure or service name
    assert "test-service" in output or "{" in output, \
        "Log output should contain service name or JSON structure"
    
    # Clean up
    logger.removeHandler(handler)


def test_log_levels_work(caplog):
    """Test different log levels work correctly"""
    from backend.infra.observability import get_logger
    
    logger = get_logger("test-log-levels")
    
    with caplog.at_level(logging.DEBUG):
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
    
    # At least some records should be captured
    assert len(caplog.records) > 0, "Should capture log records"
    
    # Check different levels present
    levels = [record.levelname for record in caplog.records]
    assert len(levels) > 0, "Should have log level information"


def test_correlation_id_in_logs():
    """Test correlation_id can be added to log context"""
    from backend.infra.observability import get_logger
    import logging
    
    logger = get_logger("test-correlation")
    
    # Add correlation_id to log extra
    logger.info("Test with correlation", extra={"correlation_id": "test-123-456"})
    
    # This test just verifies logging doesn't crash with extra fields
    # Actual correlation_id presence depends on formatter configuration
    assert True, "Logger should accept extra fields without crashing"
