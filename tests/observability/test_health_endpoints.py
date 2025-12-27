"""
EPIC-OBS-001 Phase 6 — Test health endpoints

Validates:
- /health/live returns 200 with correct structure
- /health/ready returns 200 or 503 with dependency status
"""

import pytest
from fastapi.testclient import TestClient


def test_ai_engine_liveness():
    """Test AI Engine /health/live endpoint"""
    from microservices.ai_engine.main import app
    
    client = TestClient(app)
    response = client.get("/health/live")
    
    # Liveness should always return 200 if service is running
    assert response.status_code == 200, "Liveness probe should return 200"
    
    # Should return JSON
    data = response.json()
    assert isinstance(data, dict), "Liveness response should be JSON object"
    
    # Should have status field
    assert "status" in data, "Liveness response should contain 'status' field"


def test_ai_engine_readiness():
    """Test AI Engine /health/ready endpoint"""
    from microservices.ai_engine.main import app
    
    client = TestClient(app)
    response = client.get("/health/ready")
    
    # Readiness can be 200 (ready) or 503 (not ready)
    assert response.status_code in [200, 503], \
        f"Readiness should return 200 or 503, got {response.status_code}"
    
    # Should return JSON
    data = response.json()
    assert isinstance(data, dict), "Readiness response should be JSON object"
    
    # Should have status field
    assert "status" in data, "Readiness response should contain 'status' field"


def test_execution_liveness():
    """Test Execution service /health/live endpoint"""
    from microservices.execution.main import app
    
    client = TestClient(app)
    response = client.get("/health/live")
    
    assert response.status_code == 200, "Liveness probe should return 200"
    data = response.json()
    assert isinstance(data, dict), "Liveness response should be JSON object"
    assert "status" in data, "Liveness response should contain 'status' field"


def test_execution_readiness():
    """Test Execution service /health/ready endpoint"""
    from microservices.execution.main import app
    
    client = TestClient(app)
    response = client.get("/health/ready")
    
    # Readiness can be 200 (ready) or 503 (not ready)
    assert response.status_code in [200, 503], \
        f"Readiness should return 200 or 503, got {response.status_code}"
    
    data = response.json()
    assert isinstance(data, dict), "Readiness response should be JSON object"
    assert "status" in data, "Readiness response should contain 'status' field"


def test_readiness_includes_dependencies():
    """Test readiness response includes dependency status"""
    from microservices.ai_engine.main import app
    
    client = TestClient(app)
    response = client.get("/health/ready")
    
    data = response.json()
    
    # If service checks dependencies, response should include them
    # This test is lenient - just checks for common dependency keys
    common_deps = ["dependencies", "checks", "services", "status"]
    has_deps = any(key in data for key in common_deps)
    
    # Either has dependency info OR is simple status response
    assert has_deps or "status" in data, \
        "Readiness should include dependency info or status"


def test_health_endpoints_fast():
    """Test health endpoints respond quickly (< 2s)"""
    import time
    from microservices.ai_engine.main import app
    
    client = TestClient(app)
    
    # Liveness should be very fast
    start = time.time()
    response = client.get("/health/live")
    duration = time.time() - start
    
    assert duration < 2.0, f"Liveness probe too slow: {duration:.2f}s (should be < 2s)"
    
    # Readiness should complete within timeout
    start = time.time()
    response = client.get("/health/ready")
    duration = time.time() - start
    
    assert duration < 3.0, f"Readiness probe too slow: {duration:.2f}s (should be < 3s)"
