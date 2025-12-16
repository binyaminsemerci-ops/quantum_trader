"""
Observability Middleware
EPIC-OBS-001 - Phase 3

FastAPI middleware for automatic HTTP request metrics tracking.
"""

import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Import metrics from existing infra
try:
    from infra.metrics.metrics import (
        http_requests_total,
        http_request_duration_seconds,
        http_requests_in_progress,
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    logging.warning("Metrics not available, HTTP tracking disabled")


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware that automatically tracks HTTP request metrics.
    
    Tracks:
    - Request count (by method, endpoint, status)
    - Request duration (by method, endpoint)
    - Requests in progress (by method, endpoint)
    
    Usage:
        from backend.infra.observability.middleware import MetricsMiddleware
        
        app = FastAPI()
        app.add_middleware(MetricsMiddleware)
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not METRICS_AVAILABLE:
            return await call_next(request)
        
        method = request.method
        # Use path template instead of actual path to avoid high cardinality
        endpoint = request.url.path
        
        # Increment in-progress counter
        http_requests_in_progress.labels(method=method, endpoint=endpoint).inc()
        
        start_time = time.time()
        status_code = 500  # Default to error if exception occurs
        
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        
        except Exception as e:
            # Log exception but let it propagate
            logging.error(f"Request failed: {method} {endpoint}", exc_info=e)
            raise
        
        finally:
            # Record metrics
            duration = time.time() - start_time
            
            http_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status=str(status_code)
            ).inc()
            
            http_request_duration_seconds.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
            http_requests_in_progress.labels(method=method, endpoint=endpoint).dec()


def add_metrics_middleware(app) -> None:
    """
    Add metrics middleware to FastAPI app.
    
    Args:
        app: FastAPI application instance
    
    Example:
        from backend.infra.observability.middleware import add_metrics_middleware
        
        app = FastAPI()
        add_metrics_middleware(app)
    """
    if not METRICS_AVAILABLE:
        logging.warning("Metrics not available, skipping middleware")
        return
    
    app.add_middleware(MetricsMiddleware)
    logging.info("Metrics middleware added to FastAPI app")
