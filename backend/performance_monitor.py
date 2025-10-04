"""
Performance monitoring middleware and utilities for Quantum Trader.

This module provides comprehensive performance monitoring including:
- HTTP request/response timing
- Database query performance tracking
- Memory and CPU usage monitoring
- Custom metric collection

Usage:
    from performance_monitor import add_monitoring_middleware
    from fastapi import FastAPI

    app = FastAPI()
    add_monitoring_middleware(app)
"""

import time
import psutil
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from sqlalchemy.event import listen
from sqlalchemy.engine import Engine

# Set up performance logger
perf_logger = logging.getLogger("quantum_trader.performance")


@dataclass
class RequestMetrics:
    """Metrics for a single HTTP request."""
    method: str
    path: str
    status_code: int
    duration_ms: float
    timestamp: str
    db_queries: int = 0
    db_time_ms: float = 0.0
    memory_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatabaseMetrics:
    """Metrics for database operations."""
    query: str
    duration_ms: float
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PerformanceCollector:
    """Centralized performance metrics collection."""

    def __init__(self, max_metrics: int = 1000):
        self.max_metrics = max_metrics
        self.request_metrics: deque = deque(maxlen=max_metrics)
        self.db_metrics: deque = deque(maxlen=max_metrics)
        self.current_request_db_queries = 0
        self.current_request_db_time = 0.0

    def add_request_metric(self, metric: RequestMetrics):
        """Add a request metric to the collection."""
        self.request_metrics.append(metric)
        perf_logger.info(
            f"Request {metric.method} {metric.path} - "
            f"{metric.status_code} - {metric.duration_ms:.2f}ms - "
            f"DB: {metric.db_queries} queries ({metric.db_time_ms:.2f}ms)"
        )

    def add_db_metric(self, metric: DatabaseMetrics):
        """Add a database metric to the collection."""
        self.db_metrics.append(metric)

    def get_request_summary(self) -> Dict[str, Any]:
        """Get summary statistics for requests."""
        if not self.request_metrics:
            return {"message": "No request metrics available"}

        metrics = list(self.request_metrics)
        durations = [m.duration_ms for m in metrics]
        db_queries = [m.db_queries for m in metrics]

        # Group by endpoint
        endpoint_stats = defaultdict(list)
        for m in metrics:
            endpoint_stats[f"{m.method} {m.path}"].append(m.duration_ms)

        return {
            "total_requests": len(metrics),
            "avg_duration_ms": sum(durations) / len(durations),
            "max_duration_ms": max(durations),
            "min_duration_ms": min(durations),
            "avg_db_queries": sum(db_queries) / len(db_queries),
            "slow_requests": len([d for d in durations if d > 1000]),  # > 1 second
            "endpoint_performance": {
                endpoint: {
                    "count": len(times),
                    "avg_ms": sum(times) / len(times),
                    "max_ms": max(times)
                }
                for endpoint, times in endpoint_stats.items()
            }
        }

    def get_db_summary(self) -> Dict[str, Any]:
        """Get summary statistics for database queries."""
        if not self.db_metrics:
            return {"message": "No database metrics available"}

        metrics = list(self.db_metrics)
        durations = [m.duration_ms for m in metrics]

        return {
            "total_queries": len(metrics),
            "avg_duration_ms": sum(durations) / len(durations),
            "max_duration_ms": max(durations),
            "min_duration_ms": min(durations),
            "slow_queries": len([d for d in durations if d > 100]),  # > 100ms
            "recent_slow_queries": [
                {"query": m.query[:100] + "...", "duration_ms": m.duration_ms}
                for m in metrics[-10:] if m.duration_ms > 100
            ]
        }

    def reset_request_counters(self):
        """Reset per-request database counters."""
        self.current_request_db_queries = 0
        self.current_request_db_time = 0.0


# Global collector instance
collector = PerformanceCollector()


class PerformanceMiddleware(BaseHTTPMiddleware):
    """Middleware to collect performance metrics for HTTP requests."""

    async def dispatch(self, request: Request, call_next):
        # Reset per-request counters
        collector.reset_request_counters()

        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Time the request
        start_time = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000

        # Create metrics
        metric = RequestMetrics(
            method=request.method,
            path=str(request.url.path),
            status_code=response.status_code,
            duration_ms=duration_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
            db_queries=collector.current_request_db_queries,
            db_time_ms=collector.current_request_db_time,
            memory_mb=initial_memory
        )

        collector.add_request_metric(metric)

        # Add performance headers
        response.headers["X-Response-Time-MS"] = str(round(duration_ms, 2))
        response.headers["X-DB-Queries"] = str(collector.current_request_db_queries)

        return response


def setup_db_monitoring():
    """Set up database query monitoring."""

    def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        context._query_start_time = time.perf_counter()

    def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        if hasattr(context, '_query_start_time'):
            duration_ms = (time.perf_counter() - context._query_start_time) * 1000

            # Update per-request counters
            collector.current_request_db_queries += 1
            collector.current_request_db_time += duration_ms

            # Store individual query metrics
            metric = DatabaseMetrics(
                query=statement[:200],  # Truncate long queries
                duration_ms=duration_ms,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            collector.add_db_metric(metric)

    # Register the event listeners
    listen(Engine, "before_cursor_execute", receive_before_cursor_execute)
    listen(Engine, "after_cursor_execute", receive_after_cursor_execute)


@asynccontextmanager
async def performance_timer(operation_name: str):
    """Context manager for timing custom operations."""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start_time) * 1000
        perf_logger.info(f"Operation '{operation_name}' took {duration_ms:.2f}ms")


def add_monitoring_middleware(app):
    """Add performance monitoring to a FastAPI app."""
    app.add_middleware(PerformanceMiddleware)
    setup_db_monitoring()

    # Add metrics endpoints
    @app.get("/api/metrics/requests")
    async def get_request_metrics():
        """Get request performance metrics summary."""
        return collector.get_request_summary()

    @app.get("/api/metrics/database")
    async def get_database_metrics():
        """Get database performance metrics summary."""
        return collector.get_db_summary()

    @app.get("/api/metrics/system")
    async def get_system_metrics():
        """Get current system metrics."""
        process = psutil.Process()
        cpu_percent = process.cpu_percent()
        memory_info = process.memory_info()

        return {
            "cpu_percent": cpu_percent,
            "memory_mb": memory_info.rss / 1024 / 1024,
            "memory_percent": process.memory_percent(),
            "open_files": len(process.open_files()),
            "connections": len(process.connections()),
            "threads": process.num_threads()
        }

    perf_logger.info("Performance monitoring enabled with metrics endpoints")


def get_performance_summary() -> Dict[str, Any]:
    """Get overall performance summary for logging/debugging."""
    return {
        "requests": collector.get_request_summary(),
        "database": collector.get_db_summary(),
        "system": {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3)
        }
    }
