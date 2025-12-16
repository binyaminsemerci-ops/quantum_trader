"""Prometheus telemetry helpers for the Quantum Trader backend."""

from __future__ import annotations

import functools
from contextlib import contextmanager
from time import perf_counter
from typing import Awaitable, Callable, Dict, Iterator

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest, REGISTRY, CollectorRegistry

# Use a custom registry to avoid conflicts with reload
_metrics_registry = CollectorRegistry()

# HTTP metrics
API_REQUEST_DURATION = Histogram(
    "qt_api_request_duration_seconds",
    "Latency for HTTP requests",
    labelnames=("route", "method", "status_code"),
    registry=_metrics_registry,
)
API_REQUEST_TOTAL = Counter(
    "qt_api_request_total",
    "Total HTTP requests",
    labelnames=("route", "method", "status_code"),
    registry=_metrics_registry,
)

# Scheduler metrics
SCHEDULER_RUN_DURATION = Histogram(
    "qt_scheduler_run_duration_seconds",
    "Duration of scheduler runs",
    labelnames=("job_id", "status"),
    registry=_metrics_registry,
)
SCHEDULER_RUN_TOTAL = Counter(
    "qt_scheduler_runs_total",
    "Total scheduler runs by status",
    labelnames=("job_id", "status"),
    registry=_metrics_registry,
)
PROVIDER_FAILURES = Counter(
    "qt_provider_failures_total",
    "Provider failure counter",
    labelnames=("provider",),
    registry=_metrics_registry,
)
PROVIDER_SUCCESSES = Counter(
    "qt_provider_success_total",
    "Provider success counter",
    labelnames=("provider",),
    registry=_metrics_registry,
)
PROVIDER_CIRCUIT_OPEN = Gauge(
    "qt_provider_circuit_open",
    "Circuit breaker status (1=open)",
    labelnames=("provider",),
    registry=_metrics_registry,
)

# Cache metrics
CACHE_LOOKUPS = Counter(
    "qt_cache_hits_total",
    "Cache lookup results by outcome",
    labelnames=("cache_name", "result"),
    registry=_metrics_registry,
)

# Model inference metrics
MODEL_INFERENCE_DURATION = Histogram(
    "qt_model_inference_duration_seconds",
    "Duration of model inference calls",
    labelnames=("model_name", "outcome"),
    registry=_metrics_registry,
)

# Admin audit metrics
ADMIN_EVENTS_TOTAL = Counter(
    "qt_admin_events_total",
    "Administrative actions grouped by event, category, severity and outcome",
    labelnames=("event", "category", "severity", "success"),
    registry=_metrics_registry,
)

# Risk guard metrics
RISK_DENIALS = Counter(
    "qt_risk_denials_total",
    "Risk guard denials by reason",
    labelnames=("reason",),
    registry=_metrics_registry,
)
RISK_DAILY_LOSS = Gauge(
    "qt_risk_daily_loss",
    "Rolling 24h loss tracked by risk guard",
    registry=_metrics_registry,
)


def instrument_app(app: FastAPI) -> None:
    """Attach Prometheus middleware and /metrics endpoint."""

    @app.middleware("http")
    async def _prometheus_http_middleware(request: Request, call_next: Callable[[Request], Awaitable[Response]]):
        start = perf_counter()
        status_code = "500"
        response: Response | None = None
        try:
            response = await call_next(request)
            status_code = str(response.status_code)
            return response
        finally:
            duration = perf_counter() - start
            route = request.url.path
            labels = {
                "route": route,
                "method": request.method,
                "status_code": status_code,
            }
            API_REQUEST_DURATION.labels(**labels).observe(duration)
            API_REQUEST_TOTAL.labels(**labels).inc()

    @app.get("/metrics")
    async def metrics_endpoint() -> Response:  # pragma: no cover - exercised in integration tests
        payload = generate_latest(_metrics_registry)
        return Response(payload, media_type=CONTENT_TYPE_LATEST)


def record_scheduler_run(job_id: str, status: str, duration_seconds: float) -> None:
    SCHEDULER_RUN_DURATION.labels(job_id=job_id, status=status).observe(duration_seconds)
    SCHEDULER_RUN_TOTAL.labels(job_id=job_id, status=status).inc()


def record_provider_success(provider: str) -> None:
    PROVIDER_SUCCESSES.labels(provider=provider).inc()
    PROVIDER_CIRCUIT_OPEN.labels(provider=provider).set(0)


def record_provider_failure(provider: str, circuit_open: bool) -> None:
    PROVIDER_FAILURES.labels(provider=provider).inc()
    PROVIDER_CIRCUIT_OPEN.labels(provider=provider).set(1 if circuit_open else 0)


def set_provider_circuit(provider: str, circuit_open: bool) -> None:
    PROVIDER_CIRCUIT_OPEN.labels(provider=provider).set(1 if circuit_open else 0)


def record_cache_lookup(cache_name: str, result: str) -> None:
    """Increment cache lookup counter for a given result."""

    CACHE_LOOKUPS.labels(cache_name=cache_name, result=result).inc()


def record_cache_hit(cache_name: str) -> None:
    record_cache_lookup(cache_name, "hit")


def record_cache_miss(cache_name: str) -> None:
    record_cache_lookup(cache_name, "miss")


def record_cache_write(cache_name: str) -> None:
    record_cache_lookup(cache_name, "write")


def record_risk_denial(reason: str) -> None:
    RISK_DENIALS.labels(reason=reason).inc()


def set_risk_daily_loss(value: float) -> None:
    RISK_DAILY_LOSS.set(value)


def record_admin_event_metric(
    *, event: str, category: str | None, severity: str | None, success: bool
) -> None:
    labels = {
        "event": event,
        "category": category or "unknown",
        "severity": severity or "info",
        "success": "true" if success else "false",
    }
    ADMIN_EVENTS_TOTAL.labels(**labels).inc()


def reset_admin_event_metrics() -> None:
    ADMIN_EVENTS_TOTAL._metrics.clear()


@contextmanager
def track_model_inference(model_name: str) -> Iterator[None]:
    """Record duration and outcome for a model inference block."""

    start = perf_counter()
    outcome = "success"
    try:
        yield
    except Exception:
        outcome = "error"
        raise
    finally:
        duration = perf_counter() - start
        MODEL_INFERENCE_DURATION.labels(model_name=model_name, outcome=outcome).observe(duration)