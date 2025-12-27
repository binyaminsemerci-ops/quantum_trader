"""
FastAPI Application for Monitoring Health Service

Exposes REST API endpoints for health monitoring and system status.

Author: Quantum Trader AI Team
Date: December 4, 2025
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .collectors import HealthCollector, ServiceTarget, InfraTarget
from .aggregators import HealthAggregator, SystemStatus
from .alerting import AlertManager, AlertLevel

logger = logging.getLogger(__name__)


# Global state (will be populated by main.py)
_health_collector: Optional[HealthCollector] = None
_health_aggregator: Optional[HealthAggregator] = None
_alert_manager: Optional[AlertManager] = None
_latest_health: Optional[Any] = None


def create_health_app(
    collector: HealthCollector,
    aggregator: HealthAggregator,
    alert_manager: AlertManager,
) -> FastAPI:
    """
    Create FastAPI application for monitoring health service.
    
    Args:
        collector: HealthCollector instance
        aggregator: HealthAggregator instance
        alert_manager: AlertManager instance
    
    Returns:
        FastAPI application
    """
    global _health_collector, _health_aggregator, _alert_manager
    
    _health_collector = collector
    _health_aggregator = aggregator
    _alert_manager = alert_manager
    
    app = FastAPI(
        title="Monitoring Health Service",
        description="Health monitoring and telemetry aggregator for Quantum Trader",
        version="1.0.0",
    )
    
    # Register routes
    app.add_api_route("/health", health_check, methods=["GET"], tags=["Health"])
    app.add_api_route("/health/system", get_system_health, methods=["GET"], tags=["System Health"])
    app.add_api_route("/health/metrics", get_health_metrics, methods=["GET"], tags=["Metrics"])
    app.add_api_route("/alerts/active", get_active_alerts, methods=["GET"], tags=["Alerts"])
    app.add_api_route("/alerts/history", get_alert_history, methods=["GET"], tags=["Alerts"])
    app.add_api_route("/alerts/{alert_id}/clear", clear_alert, methods=["POST"], tags=["Alerts"])
    
    return app


# ============================================================================
# API ENDPOINTS
# ============================================================================

async def health_check() -> Dict[str, Any]:
    """
    Basic health check for monitoring-health-service itself.
    
    Returns:
        200: Service is operational
    """
    return {
        "status": "healthy",
        "service": "monitoring-health-service",
        "timestamp": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }


async def get_system_health() -> Dict[str, Any]:
    """
    Get aggregated system health status.
    
    This endpoint collects fresh health data from all services and infrastructure,
    aggregates it, and returns the overall system status.
    
    Returns:
        200: Aggregated system health
        {
            "status": "OK" | "DEGRADED" | "CRITICAL",
            "timestamp": "2025-12-04T10:30:00Z",
            "services": {
                "ok": ["service1", "service2"],
                "degraded": [],
                "down": []
            },
            "infrastructure": {
                "ok": ["redis", "postgres", "binance_api"],
                "degraded": [],
                "down": []
            },
            "critical_failures": [],
            "performance": {
                "avg_service_latency_ms": 45.3,
                "max_service_latency_ms": 120.5
            }
        }
    """
    global _health_collector, _health_aggregator, _alert_manager, _latest_health
    
    if not _health_collector or not _health_aggregator:
        raise HTTPException(status_code=503, detail="Health monitoring not initialized")
    
    try:
        # Collect fresh snapshot
        snapshot = await _health_collector.collect_snapshot()
        
        # Aggregate
        aggregated = _health_aggregator.aggregate(snapshot)
        
        # Store latest
        _latest_health = aggregated
        
        # Process alerts
        if _alert_manager:
            await _alert_manager.process_health(aggregated)
        
        return aggregated.to_dict()
    
    except Exception as e:
        logger.error(f"[Monitoring Health API] Failed to get system health: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


async def get_health_metrics() -> Dict[str, Any]:
    """
    Get key health metrics (for Dashboard / Grafana).
    
    Returns:
        200: Key metrics
        {
            "timestamp": "2025-12-04T10:30:00Z",
            "counters": {
                "total_services": 8,
                "services_ok": 7,
                "services_degraded": 1,
                "services_down": 0,
                "infra_ok": 3,
                "infra_down": 0
            },
            "gauges": {
                "system_status_code": 0,  // 0=OK, 1=DEGRADED, 2=CRITICAL
                "avg_service_latency_ms": 45.3,
                "max_service_latency_ms": 120.5
            },
            "alerts": {
                "active_count": 1,
                "critical_count": 0,
                "warning_count": 1
            }
        }
    """
    global _latest_health, _alert_manager
    
    if not _latest_health:
        return {
            "timestamp": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            "status": "no_data",
            "message": "No health data collected yet",
        }
    
    # Map status to numeric code
    status_code_map = {
        SystemStatus.OK: 0,
        SystemStatus.DEGRADED: 1,
        SystemStatus.CRITICAL: 2,
        SystemStatus.UNKNOWN: 3,
    }
    
    # Count active alerts by level
    active_alerts = _alert_manager.get_active_alerts() if _alert_manager else []
    critical_alerts = sum(1 for a in active_alerts if a.level == AlertLevel.CRITICAL)
    warning_alerts = sum(1 for a in active_alerts if a.level == AlertLevel.WARNING)
    
    return {
        "timestamp": _latest_health.timestamp,
        "counters": {
            "total_services": (
                len(_latest_health.services_ok) +
                len(_latest_health.services_degraded) +
                len(_latest_health.services_down)
            ),
            "services_ok": len(_latest_health.services_ok),
            "services_degraded": len(_latest_health.services_degraded),
            "services_down": len(_latest_health.services_down),
            "infra_ok": len(_latest_health.infra_ok),
            "infra_degraded": len(_latest_health.infra_degraded),
            "infra_down": len(_latest_health.infra_down),
        },
        "gauges": {
            "system_status_code": status_code_map.get(_latest_health.status, 3),
            "avg_service_latency_ms": _latest_health.avg_service_latency_ms,
            "max_service_latency_ms": _latest_health.max_service_latency_ms,
        },
        "alerts": {
            "active_count": len(active_alerts),
            "critical_count": critical_alerts,
            "warning_count": warning_alerts,
        },
    }


async def get_active_alerts() -> Dict[str, Any]:
    """
    Get all active alerts.
    
    Returns:
        200: List of active alerts
    """
    global _alert_manager
    
    if not _alert_manager:
        return {"alerts": [], "count": 0}
    
    active = _alert_manager.get_active_alerts()
    
    return {
        "alerts": [alert.to_dict() for alert in active],
        "count": len(active),
    }


async def get_alert_history(limit: int = 100) -> Dict[str, Any]:
    """
    Get alert history.
    
    Args:
        limit: Maximum number of alerts to return (default: 100, max: 500)
    
    Returns:
        200: Alert history
    """
    global _alert_manager
    
    if not _alert_manager:
        return {"alerts": [], "count": 0}
    
    # Clamp limit
    limit = min(max(1, limit), 500)
    
    history = _alert_manager.get_alert_history(limit=limit)
    
    return {
        "alerts": [alert.to_dict() for alert in history],
        "count": len(history),
        "limit": limit,
    }


async def clear_alert(alert_id: str) -> Dict[str, Any]:
    """
    Clear/acknowledge an active alert.
    
    Args:
        alert_id: Alert ID to clear
    
    Returns:
        200: Alert cleared successfully
        404: Alert not found
    """
    global _alert_manager
    
    if not _alert_manager:
        raise HTTPException(status_code=503, detail="Alert manager not available")
    
    success = _alert_manager.clear_alert(alert_id)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Alert not found: {alert_id}")
    
    return {
        "status": "ok",
        "message": f"Alert {alert_id} cleared",
    }
