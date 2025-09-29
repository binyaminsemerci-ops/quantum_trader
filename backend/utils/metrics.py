
"""Prometheus metrics helpers."""

from __future__ import annotations

import time
from typing import Dict

from fastapi import APIRouter, Request, Response
from prometheus_client import Counter, Gauge, Histogram, CONTENT_TYPE_LATEST, generate_latest

REQUEST_COUNT = Counter(
    "quantum_requests_total",
    "Total HTTP requests",
    labelnames=("method", "path", "status"),
)
REQUEST_LATENCY = Histogram(
    "quantum_request_latency_seconds",
    "Latency per HTTP request",
    labelnames=("method", "path"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1.0, 5.0),
)
INFLIGHT_REQUESTS = Gauge(
    "quantum_requests_in_progress",
    "Number of in-flight HTTP requests",
)
HEALTH_STATUS = Gauge(
    "quantum_health_status",
    "Latest health check status (1=ok, 0=failed)",
)

router = APIRouter()


@router.get("/", include_in_schema=False)
def metrics_endpoint() -> Response:
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


def add_metrics_middleware(app) -> None:
    @app.middleware("http")
    async def _metrics_middleware(request: Request, call_next):  # type: ignore
        path = request.url.path
        if path.startswith("/metrics"):
            return await call_next(request)
        start = time.perf_counter()
        INFLIGHT_REQUESTS.inc()
        status_code = '500'
        try:
            response = await call_next(request)
            status_code = str(response.status_code)
            return response
        finally:
            duration = time.perf_counter() - start
            REQUEST_COUNT.labels(request.method, path, status_code).inc()
            REQUEST_LATENCY.labels(request.method, path).observe(duration)
            INFLIGHT_REQUESTS.dec()


def update_health_metric(ok: bool) -> None:
    HEALTH_STATUS.set(1.0 if ok else 0.0)


__all__ = [
    "router",
    "add_metrics_middleware",
    "update_health_metric",
]
