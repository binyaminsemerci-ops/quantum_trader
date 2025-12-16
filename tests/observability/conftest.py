"""
Pytest configuration for observability tests

Provides fixtures and setup for testing observability layer.
"""

import pytest
import sys
from pathlib import Path

# Add backend to Python path for imports
backend_path = Path(__file__).parent.parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))


@pytest.fixture(scope="session")
def test_service_name():
    """Fixture providing test service name"""
    return "test-observability-service"


@pytest.fixture(scope="session")
def test_environment():
    """Fixture providing test environment"""
    return "test"


@pytest.fixture(autouse=True)
def reset_prometheus_registry():
    """Reset Prometheus registry between tests to avoid metric collisions"""
    # This is optional - only if tests interfere with each other
    yield
    # Cleanup after test if needed
