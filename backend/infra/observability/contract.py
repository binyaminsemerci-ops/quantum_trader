"""
Observability Contract for Quantum Trader v2.0
EPIC-OBS-001 - Phase 4: Observability Contract

Defines the mandatory observability requirements for all microservices.
Every service MUST comply with this contract to ensure consistent monitoring.
"""

from typing import List, Optional
from fastapi import FastAPI

# ============================================================================
# REQUIRED ENDPOINTS
# ============================================================================

REQUIRED_ENDPOINTS = [
    "/health/live",    # Liveness probe: returns 200 if process is alive
    "/health/ready",   # Readiness probe: returns 200 if deps healthy, 503 if not
    "/metrics",        # Prometheus metrics endpoint (text format)
]

# ============================================================================
# REQUIRED INITIALIZATION
# ============================================================================

REQUIRED_STARTUP_CALL = "init_observability(service_name, log_level)"

# ============================================================================
# REQUIRED METRIC LABELS
# ============================================================================

REQUIRED_LABELS = [
    "service",         # Service name (e.g., "ai-engine", "execution")
    "environment",     # Environment (e.g., "production", "staging", "development")
    "version",         # Service version
    "method",          # HTTP method (GET, POST, etc.)
    "endpoint",        # HTTP endpoint path
    "status",          # HTTP status code
]

# ============================================================================
# REQUIRED TRACE ATTRIBUTES (if tracing enabled)
# ============================================================================

REQUIRED_TRACE_ATTRIBUTES = [
    "service.name",               # Service name
    "service.version",            # Service version
    "deployment.environment",     # Environment (prod/staging/dev)
]

# ============================================================================
# OBSERVABILITY COMPLIANCE HELPER
# ============================================================================

class ObservableService:
    """
    Helper class to ensure a FastAPI service complies with observability contract.
    
    Usage:
        from backend.infra.observability import ObservableService, init_observability
        
        app = FastAPI()
        
        # Initialize observability
        init_observability(service_name="my-service", log_level="INFO")
        
        # Apply full instrumentation
        ObservableService.instrument(app, service_name="my-service")
        
        # Service now has /metrics, /health/live, /health/ready, 
        # plus automatic request tracking
    """
    
    @staticmethod
    def instrument(app: FastAPI, service_name: str) -> None:
        """
        Apply full observability instrumentation to a FastAPI app.
        
        This adds:
        - OpenTelemetry tracing (if available)
        - Metrics middleware (automatic request tracking)
        
        Note: You still need to manually add /health/live, /health/ready, /metrics
        endpoints to maintain explicit control over health checks.
        
        Args:
            app: FastAPI application instance
            service_name: Name of the service (must match init_observability call)
        """
        from . import instrument_fastapi, add_metrics_middleware
        
        # Apply OpenTelemetry instrumentation
        instrument_fastapi(app)
        
        # Apply metrics middleware
        add_metrics_middleware(app)
    
    @staticmethod
    def validate_compliance(app: FastAPI) -> List[str]:
        """
        Validate that a service complies with the observability contract.
        
        Args:
            app: FastAPI application instance
        
        Returns:
            List of missing requirements (empty if fully compliant)
        
        Example:
            missing = ObservableService.validate_compliance(app)
            if missing:
                raise ValueError(f"Service not compliant: {missing}")
        """
        missing = []
        
        # Check required endpoints exist
        routes = {route.path for route in app.routes}
        for endpoint in REQUIRED_ENDPOINTS:
            if endpoint not in routes:
                missing.append(f"Missing endpoint: {endpoint}")
        
        return missing
