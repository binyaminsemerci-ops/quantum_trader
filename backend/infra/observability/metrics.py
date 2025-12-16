"""
Metrics Module
EPIC-OBS-001 - Phase 2

Wrapper around existing infra/metrics/metrics.py with service-aware helpers.
"""

import logging
from typing import Optional

# Import existing Prometheus metrics
# These are already defined in infra/metrics/metrics.py
try:
    from infra.metrics import metrics as _base_metrics
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    logging.warning("infra.metrics not found, metrics disabled")

from .config import config

# Re-export all metrics from base module
if METRICS_AVAILABLE:
    # HTTP metrics
    http_requests_total = _base_metrics.http_requests_total
    http_request_duration_seconds = _base_metrics.http_request_duration_seconds
    http_requests_in_progress = _base_metrics.http_requests_in_progress
    
    # EventBus metrics
    eventbus_messages_published_total = _base_metrics.eventbus_messages_published_total
    eventbus_messages_consumed_total = _base_metrics.eventbus_messages_consumed_total
    eventbus_message_processing_duration_seconds = _base_metrics.eventbus_message_processing_duration_seconds
    
    # Trading metrics
    trades_executed_total = _base_metrics.trades_executed_total
    trade_execution_duration_seconds = _base_metrics.trade_execution_duration_seconds
    positions_open = _base_metrics.positions_open
    signals_generated_total = _base_metrics.signals_generated_total
    
    # Risk metrics
    risk_checks_total = _base_metrics.risk_checks_total
    emergency_stop_triggered_total = _base_metrics.emergency_stop_triggered_total
    
    # EPIC-STRESS-DASH-001: Risk & Resilience Dashboard Metrics
    risk_gate_decisions_total = _base_metrics.risk_gate_decisions_total
    ess_triggers_total = _base_metrics.ess_triggers_total
    exchange_failover_events_total = _base_metrics.exchange_failover_events_total
    stress_scenario_runs_total = _base_metrics.stress_scenario_runs_total
    
    # Error metrics
    errors_total = _base_metrics.errors_total
    exceptions_total = _base_metrics.exceptions_total
    
    # Health metrics
    service_health_status = _base_metrics.service_health_status
    dependency_health_status = _base_metrics.dependency_health_status
    
    # Metrics endpoint
    metrics_endpoint = _base_metrics.metrics_endpoint
else:
    # Dummy placeholders if metrics not available
    http_requests_total = None
    http_request_duration_seconds = None
    http_requests_in_progress = None
    eventbus_messages_published_total = None
    eventbus_messages_consumed_total = None
    eventbus_message_processing_duration_seconds = None
    trades_executed_total = None
    trade_execution_duration_seconds = None
    positions_open = None
    signals_generated_total = None
    risk_checks_total = None
    emergency_stop_triggered_total = None
    risk_gate_decisions_total = None
    ess_triggers_total = None
    exchange_failover_events_total = None
    stress_scenario_runs_total = None
    errors_total = None
    exceptions_total = None
    service_health_status = None
    dependency_health_status = None
    metrics_endpoint = None


class MetricsRegistry:
    """
    Service-aware metrics registry.
    
    Provides convenient access to all Prometheus metrics with service context.
    """
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.enabled = METRICS_AVAILABLE and config.enable_metrics
    
    # HTTP metrics
    @property
    def http_requests_total(self):
        return http_requests_total if self.enabled else None
    
    @property
    def http_request_duration_seconds(self):
        return http_request_duration_seconds if self.enabled else None
    
    @property
    def http_requests_in_progress(self):
        return http_requests_in_progress if self.enabled else None
    
    # EventBus metrics
    @property
    def eventbus_messages_published_total(self):
        return eventbus_messages_published_total if self.enabled else None
    
    @property
    def eventbus_messages_consumed_total(self):
        return eventbus_messages_consumed_total if self.enabled else None
    
    @property
    def eventbus_message_processing_duration_seconds(self):
        return eventbus_message_processing_duration_seconds if self.enabled else None
    
    # Trading metrics
    @property
    def trades_executed_total(self):
        return trades_executed_total if self.enabled else None
    
    @property
    def trade_execution_duration_seconds(self):
        return trade_execution_duration_seconds if self.enabled else None
    
    @property
    def positions_open(self):
        return positions_open if self.enabled else None
    
    @property
    def signals_generated_total(self):
        return signals_generated_total if self.enabled else None
    
    # Risk metrics
    @property
    def risk_checks_total(self):
        return risk_checks_total if self.enabled else None
    
    @property
    def emergency_stop_triggered_total(self):
        return emergency_stop_triggered_total if self.enabled else None
    
    # Error metrics
    @property
    def errors_total(self):
        return errors_total if self.enabled else None
    
    @property
    def exceptions_total(self):
        return exceptions_total if self.enabled else None
    
    # Health metrics
    @property
    def service_health_status(self):
        return service_health_status if self.enabled else None
    
    @property
    def dependency_health_status(self):
        return dependency_health_status if self.enabled else None
    
    def get_metrics_endpoint(self):
        """Get FastAPI response for /metrics endpoint."""
        if not self.enabled or not metrics_endpoint:
            from fastapi import Response
            return Response(content="Metrics disabled", status_code=503)
        return metrics_endpoint()


# Global metrics instance
metrics: Optional[MetricsRegistry] = None


def init_metrics(service_name: str) -> MetricsRegistry:
    """
    Initialize metrics for the service.
    
    Args:
        service_name: Name of the service
    
    Returns:
        MetricsRegistry instance
    
    Example:
        metrics = init_metrics(service_name="ai-engine")
        
        # Later in code
        metrics.http_requests_total.labels(method="GET", endpoint="/health", status="200").inc()
    """
    global metrics
    
    if not METRICS_AVAILABLE:
        logging.warning(f"Metrics not available for {service_name}")
        return MetricsRegistry(service_name)
    
    if metrics is not None:
        logging.warning(f"Metrics already initialized for {config.service_name}, skipping")
        return metrics
    
    config.service_name = service_name
    metrics = MetricsRegistry(service_name)
    
    logging.info(f"Metrics initialized for {service_name}")
    return metrics
