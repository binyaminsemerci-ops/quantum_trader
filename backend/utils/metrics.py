"""Prometheus metrics helpers."""

from __future__ import annotations

import time

from fastapi import APIRouter, Request, Response
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    CONTENT_TYPE_LATEST,
    generate_latest,
    Info,
)

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

# Active model info (version/tag). Uses Info (labels converted to metrics)
MODEL_INFO = Info(
    "quantum_model",
    "Active model metadata (version, tag)",
)

MODEL_PERF_SHARPE = Gauge(
    "quantum_model_sharpe",
    "Active model backtest Sharpe ratio (rolling training insert)",
)

MODEL_PERF_MAX_DRAWDOWN = Gauge(
    "quantum_model_max_drawdown",
    "Active model max drawdown from last backtest (fraction)",
)

MODEL_PERF_SORTINO = Gauge(
    "quantum_model_sortino",
    "Active model backtest Sortino ratio (downside risk adjusted)",
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
        status_code = "500"
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


def update_model_info(version: str | None, tag: str | None = None) -> None:
    """Publish active model metadata.

    Falls back to placeholders if None provided.
    """
    MODEL_INFO.info(
        {
            "version": version or "unknown",
            "tag": tag or "none",
        }
    )


def update_model_perf(sharpe: float | None, max_drawdown: float | None) -> None:
    if sharpe is not None:
        try:
            MODEL_PERF_SHARPE.set(float(sharpe))
        except Exception:
            pass
    if max_drawdown is not None:
        try:
            MODEL_PERF_MAX_DRAWDOWN.set(float(max_drawdown))
        except Exception:
            pass


def update_model_sortino(sortino: float | None) -> None:
    if sortino is not None:
        try:
            MODEL_PERF_SORTINO.set(float(sortino))
        except Exception:
            pass


__all__ = [
    "router",
    "add_metrics_middleware",
    "update_health_metric",
    "update_model_info",
    "MODEL_INFO",
    "MODEL_PERF_SHARPE",
    "MODEL_PERF_MAX_DRAWDOWN",
    "MODEL_PERF_SORTINO",
    "update_model_perf",
    "update_model_sortino",
]
