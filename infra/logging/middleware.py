"""
FastAPI Logging Middleware
SPRINT 3 - Module E: Unified Logging

Adds correlation ID and request context to all FastAPI requests.
"""

import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from infra.logging.filters import set_correlation_id, generate_correlation_id

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds correlation IDs and logs all HTTP requests.
    
    Features:
    - Generates or propagates correlation IDs
    - Logs request/response with timing
    - Adds correlation ID to response headers
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate or extract correlation ID
        correlation_id = request.headers.get("X-Correlation-ID") or generate_correlation_id()
        
        # Set correlation ID for this request context
        set_correlation_id(correlation_id)
        
        # Log request
        start_time = time.time()
        logger.info(
            f"{request.method} {request.url.path}",
            extra={
                "correlation_id": correlation_id,
                "request_method": request.method,
                "request_path": str(request.url.path),
                "request_query": str(request.url.query),
                "client_host": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
            }
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Log response
            logger.info(
                f"{request.method} {request.url.path} - {response.status_code}",
                extra={
                    "correlation_id": correlation_id,
                    "response_status": response.status_code,
                    "duration_ms": round(duration_ms, 2),
                }
            )
            
            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id
            
            return response
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            # Log error
            logger.error(
                f"{request.method} {request.url.path} - ERROR",
                extra={
                    "correlation_id": correlation_id,
                    "error": str(e),
                    "duration_ms": round(duration_ms, 2),
                },
                exc_info=True
            )
            
            raise


# ============================================================================
# USAGE IN FASTAPI APP
# ============================================================================

"""
# backend/services/ai_engine/main.py

import logging
import logging.config
import yaml
from fastapi import FastAPI
from infra.logging.middleware import LoggingMiddleware

# Load logging config
with open("infra/logging/logging_config.yml", "r") as f:
    config = yaml.safe_load(f)
    logging.config.dictConfig(config)

# Create app
app = FastAPI()

# Add logging middleware
app.add_middleware(LoggingMiddleware)

@app.get("/health")
async def health():
    return {"status": "healthy"}

# All requests will now have correlation IDs and structured logging
"""

# ============================================================================
# EXAMPLE LOG OUTPUT
# ============================================================================

"""
{
  "asctime": "2025-12-04T12:34:56.789Z",
  "name": "ai_engine",
  "levelname": "INFO",
  "message": "POST /api/signals",
  "correlation_id": "abc-123-def-456",
  "service": "ai-engine-service",
  "request_method": "POST",
  "request_path": "/api/signals",
  "request_query": "",
  "client_host": "172.17.0.1",
  "user_agent": "httpx/0.24.1"
}

{
  "asctime": "2025-12-04T12:34:56.892Z",
  "name": "ai_engine",
  "levelname": "INFO",
  "message": "POST /api/signals - 200",
  "correlation_id": "abc-123-def-456",
  "service": "ai-engine-service",
  "response_status": 200,
  "duration_ms": 103.45
}
"""
