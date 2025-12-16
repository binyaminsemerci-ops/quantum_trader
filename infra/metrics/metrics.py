"""
Basic Metrics Instrumentation
SPRINT 3 - Module F: Prometheus Metrics

Provides Prometheus-compatible metrics for microservices.
"""

import time
import os
from typing import Optional, Dict, Any
from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, REGISTRY
from prometheus_client.exposition import choose_encoder
from fastapi import Response

# ============================================================================
# SERVICE INFO
# ============================================================================

service_info = Info(
    'service',
    'Service information',
    registry=REGISTRY
)
service_info.info({
    'name': os.getenv('SERVICE_NAME', 'unknown'),
    'version': os.getenv('SERVICE_VERSION', '1.0.0'),
    'environment': os.getenv('ENVIRONMENT', 'development')
})


# ============================================================================
# HTTP METRICS
# ============================================================================

http_requests_total = Counter(
    'http_requests_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status'],
    registry=REGISTRY
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=REGISTRY
)

http_requests_in_progress = Gauge(
    'http_requests_in_progress',
    'Number of HTTP requests in progress',
    ['method', 'endpoint'],
    registry=REGISTRY
)


# ============================================================================
# EVENTBUS METRICS
# ============================================================================

eventbus_messages_published_total = Counter(
    'eventbus_messages_published_total',
    'Total number of messages published to EventBus',
    ['event_type'],
    registry=REGISTRY
)

eventbus_messages_consumed_total = Counter(
    'eventbus_messages_consumed_total',
    'Total number of messages consumed from EventBus',
    ['event_type'],
    registry=REGISTRY
)

eventbus_message_processing_duration_seconds = Histogram(
    'eventbus_message_processing_duration_seconds',
    'EventBus message processing duration in seconds',
    ['event_type'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0],
    registry=REGISTRY
)

eventbus_queue_lag = Gauge(
    'eventbus_queue_lag',
    'Number of messages waiting in EventBus queue',
    ['queue_name'],
    registry=REGISTRY
)


# ============================================================================
# REDIS METRICS
# ============================================================================

redis_operations_total = Counter(
    'redis_operations_total',
    'Total number of Redis operations',
    ['operation', 'status'],
    registry=REGISTRY
)

redis_operation_duration_seconds = Histogram(
    'redis_operation_duration_seconds',
    'Redis operation duration in seconds',
    ['operation'],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
    registry=REGISTRY
)

redis_connection_pool_size = Gauge(
    'redis_connection_pool_size',
    'Current size of Redis connection pool',
    registry=REGISTRY
)


# ============================================================================
# DATABASE METRICS
# ============================================================================

database_operations_total = Counter(
    'database_operations_total',
    'Total number of database operations',
    ['operation', 'status'],
    registry=REGISTRY
)

database_operation_duration_seconds = Histogram(
    'database_operation_duration_seconds',
    'Database operation duration in seconds',
    ['operation'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0],
    registry=REGISTRY
)

database_connection_pool_size = Gauge(
    'database_connection_pool_size',
    'Current size of database connection pool',
    registry=REGISTRY
)


# ============================================================================
# TRADING METRICS (AI Engine, Execution)
# ============================================================================

trades_executed_total = Counter(
    'trades_executed_total',
    'Total number of trades executed',
    ['symbol', 'side', 'status'],
    registry=REGISTRY
)

trade_execution_duration_seconds = Histogram(
    'trade_execution_duration_seconds',
    'Trade execution duration in seconds',
    ['symbol'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    registry=REGISTRY
)

positions_open = Gauge(
    'positions_open',
    'Number of currently open positions',
    ['symbol'],
    registry=REGISTRY
)

portfolio_value_usd = Gauge(
    'portfolio_value_usd',
    'Total portfolio value in USD',
    registry=REGISTRY
)

signals_generated_total = Counter(
    'signals_generated_total',
    'Total number of trading signals generated',
    ['signal_type', 'model'],
    registry=REGISTRY
)


# ============================================================================
# RISK METRICS
# ============================================================================

risk_checks_total = Counter(
    'risk_checks_total',
    'Total number of risk checks performed',
    ['check_type', 'result'],
    registry=REGISTRY
)

emergency_stop_triggered_total = Counter(
    'emergency_stop_triggered_total',
    'Total number of times Emergency Stop was triggered',
    ['reason'],
    registry=REGISTRY
)

policy_violations_total = Counter(
    'policy_violations_total',
    'Total number of policy violations',
    ['policy_name', 'severity'],
    registry=REGISTRY
)

# EPIC-STRESS-DASH-001: Risk & Resilience Dashboard Metrics
risk_gate_decisions_total = Counter(
    'risk_gate_decisions_total',
    'Count of RiskGate v3 decisions',
    ['decision', 'reason', 'account', 'exchange', 'strategy'],
    registry=REGISTRY
)

ess_triggers_total = Counter(
    'ess_triggers_total',
    'Total ESS (Emergency Stop System) triggers',
    ['source'],
    registry=REGISTRY
)

exchange_failover_events_total = Counter(
    'exchange_failover_events_total',
    'Count of exchange failover events',
    ['primary', 'selected'],
    registry=REGISTRY
)

stress_scenario_runs_total = Counter(
    'stress_scenario_runs_total',
    'Stress scenarios executed',
    ['scenario', 'success'],
    registry=REGISTRY
)


# ============================================================================
# ERROR METRICS
# ============================================================================

errors_total = Counter(
    'errors_total',
    'Total number of errors',
    ['error_type', 'severity'],
    registry=REGISTRY
)

exceptions_total = Counter(
    'exceptions_total',
    'Total number of exceptions',
    ['exception_class'],
    registry=REGISTRY
)


# ============================================================================
# HEALTH METRICS
# ============================================================================

service_health_status = Gauge(
    'service_health_status',
    'Service health status (1=healthy, 0=unhealthy)',
    ['component'],
    registry=REGISTRY
)

dependency_health_status = Gauge(
    'dependency_health_status',
    'Dependency health status (1=healthy, 0=unhealthy)',
    ['dependency'],
    registry=REGISTRY
)


# ============================================================================
# METRICS ENDPOINT (FastAPI)
# ============================================================================

def metrics_endpoint() -> Response:
    """
    Prometheus metrics endpoint.
    
    Usage in FastAPI:
        from infra.metrics.metrics import metrics_endpoint
        
        @app.get("/metrics")
        async def metrics():
            return metrics_endpoint()
    """
    encoder, content_type = choose_encoder(None)
    return Response(content=encoder(REGISTRY), media_type=content_type)


# ============================================================================
# HELPER DECORATORS
# ============================================================================

def track_http_request(method: str, endpoint: str):
    """
    Decorator to track HTTP request metrics.
    
    Usage:
        @track_http_request("GET", "/health")
        async def health():
            return {"status": "healthy"}
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            http_requests_in_progress.labels(method=method, endpoint=endpoint).inc()
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                status = "200"
                return result
            except Exception as e:
                status = "500"
                raise
            finally:
                duration = time.time() - start_time
                http_requests_total.labels(method=method, endpoint=endpoint, status=status).inc()
                http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)
                http_requests_in_progress.labels(method=method, endpoint=endpoint).dec()
        
        return wrapper
    return decorator


def track_eventbus_message(event_type: str):
    """
    Decorator to track EventBus message processing.
    
    Usage:
        @track_eventbus_message("trade.executed")
        async def handle_trade_executed(event):
            pass
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            eventbus_messages_consumed_total.labels(event_type=event_type).inc()
            
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                eventbus_message_processing_duration_seconds.labels(event_type=event_type).observe(duration)
        
        return wrapper
    return decorator


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
# Example 1: FastAPI integration

from fastapi import FastAPI
from infra.metrics.metrics import metrics_endpoint, http_requests_total

app = FastAPI()

@app.get("/metrics")
async def metrics():
    return metrics_endpoint()

@app.get("/health")
async def health():
    http_requests_total.labels(method="GET", endpoint="/health", status="200").inc()
    return {"status": "healthy"}


# Example 2: Manual metric recording

from infra.metrics.metrics import trades_executed_total, trade_execution_duration_seconds
import time

start = time.time()
# ... execute trade ...
duration = time.time() - start

trades_executed_total.labels(symbol="BTCUSDT", side="long", status="filled").inc()
trade_execution_duration_seconds.labels(symbol="BTCUSDT").observe(duration)


# Example 3: EventBus integration

from infra.metrics.metrics import eventbus_messages_published_total

async def publish_event(event_type: str, data: dict):
    await event_bus.publish(event_type, data)
    eventbus_messages_published_total.labels(event_type=event_type).inc()
"""
