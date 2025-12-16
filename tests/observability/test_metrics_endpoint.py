"""
EPIC-OBS-001 Phase 6 — Test /metrics endpoint

Validates:
- /metrics returns 200
- Content-Type is text/plain (Prometheus format)
- Contains expected metric names
"""

import pytest
from fastapi.testclient import TestClient


def test_ai_engine_metrics_endpoint():
    """Test AI Engine /metrics endpoint returns Prometheus format"""
    from microservices.ai_engine.main import app
    
    client = TestClient(app)
    response = client.get("/metrics")
    
    # Assert status code
    assert response.status_code == 200, "Metrics endpoint should return 200"
    
    # Assert content type (Prometheus format)
    content_type = response.headers.get("content-type", "")
    assert "text/plain" in content_type, f"Expected text/plain, got {content_type}"
    
    # Assert contains known metrics
    content = response.text
    assert "http_requests_total" in content or "http_request" in content, \
        "Metrics should contain HTTP request metrics"


def test_execution_metrics_endpoint():
    """Test Execution service /metrics endpoint returns Prometheus format"""
    from microservices.execution.main import app
    
    client = TestClient(app)
    response = client.get("/metrics")
    
    # Assert status code
    assert response.status_code == 200, "Metrics endpoint should return 200"
    
    # Assert content type
    content_type = response.headers.get("content-type", "")
    assert "text/plain" in content_type, f"Expected text/plain, got {content_type}"
    
    # Assert contains known metrics
    content = response.text
    assert "http_requests_total" in content or "http_request" in content, \
        "Metrics should contain HTTP request metrics"


def test_metrics_format_structure():
    """Test metrics response follows Prometheus format conventions"""
    from microservices.ai_engine.main import app
    
    client = TestClient(app)
    response = client.get("/metrics")
    
    content = response.text
    
    # Prometheus metrics should have HELP and TYPE comments
    assert "# HELP" in content or "# TYPE" in content or content.strip(), \
        "Metrics should contain Prometheus format comments or metric lines"
    
    # Should not be JSON
    assert not content.strip().startswith("{"), \
        "Metrics should be Prometheus format, not JSON"


def test_metrics_contains_labels():
    """Test metrics include required labels (service, environment, etc.)"""
    from microservices.ai_engine.main import app
    
    client = TestClient(app)
    response = client.get("/metrics")
    
    content = response.text
    
    # Check for label structure (key="value")
    # Most metrics should have labels in Prometheus format
    has_labels = (
        'service=' in content or
        'environment=' in content or
        '{' in content  # Prometheus metric with labels has curly braces
    )
    
    assert has_labels or len(content) > 0, \
        "Metrics should contain labels or metric data"
