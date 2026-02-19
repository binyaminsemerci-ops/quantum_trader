"""
Unit Tests for Monitoring Health Service (Sprint 2 - Service #6)

Tests collectors, aggregators, and alerting logic with mocked dependencies.

Author: Quantum Trader AI Team
Date: December 4, 2025
"""

import pytest
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

# Import components to test
from backend.services.monitoring_health_service.collectors import (
    HealthCollector,
    ServiceTarget,
    InfraTarget,
)
from backend.services.monitoring_health_service.aggregators import (
    HealthAggregator,
    SystemStatus,
    AggregatedHealth,
)
from backend.services.monitoring_health_service.alerting import (
    AlertManager,
    AlertLevel,
    HealthAlert,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_http_client():
    """Mock httpx.AsyncClient."""
    client = AsyncMock()
    return client


@pytest.fixture
def mock_redis_client():
    """Mock Redis client."""
    client = AsyncMock()
    client.ping = AsyncMock(return_value=True)
    return client


@pytest.fixture
def mock_event_bus():
    """Mock EventBus."""
    bus = AsyncMock()
    bus.publish = AsyncMock()
    return bus


@pytest.fixture
def sample_service_targets():
    """Sample service targets for testing."""
    return [
        ServiceTarget("service_a", "http://localhost:8000/health", critical=True),
        ServiceTarget("service_b", "http://localhost:8001/health", critical=True),
        ServiceTarget("service_c", "http://localhost:8002/health", critical=False),
    ]


@pytest.fixture
def sample_infra_targets():
    """Sample infrastructure targets for testing."""
    return [
        InfraTarget("redis", "redis", {"url": "redis://localhost:6379"}),
        InfraTarget("postgres", "postgres", {"health_url": "http://localhost:8000/health"}),
        InfraTarget("binance_api", "binance_api", {"url": "https://api.binance.com/api/v3/ping"}),
    ]


@pytest.fixture
def health_collector(mock_http_client, mock_redis_client, sample_service_targets, sample_infra_targets):
    """Create HealthCollector instance."""
    return HealthCollector(
        service_targets=sample_service_targets,
        infra_targets=sample_infra_targets,
        http_client=mock_http_client,
        redis_client=mock_redis_client,
    )


@pytest.fixture
def health_aggregator():
    """Create HealthAggregator instance."""
    return HealthAggregator()


@pytest.fixture
def alert_manager(mock_event_bus):
    """Create AlertManager instance."""
    return AlertManager(event_bus=mock_event_bus)


# ============================================================================
# TEST: HEALTH COLLECTOR
# ============================================================================

@pytest.mark.asyncio
async def test_collect_snapshot_all_services_ok(health_collector, mock_http_client):
    """Test collect_snapshot when all services are OK."""
    # Mock all HTTP responses as OK
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "healthy"}
    mock_http_client.get = AsyncMock(return_value=mock_response)
    
    # Collect snapshot
    snapshot = await health_collector.collect_snapshot()
    
    # Verify structure
    assert "timestamp" in snapshot
    assert "services" in snapshot
    assert "infra" in snapshot
    
    # Verify all services OK
    for service_name, health in snapshot["services"].items():
        assert health["status"] == "OK"
        assert health["latency_ms"] is not None
        assert "error" not in health


@pytest.mark.asyncio
async def test_collect_snapshot_service_down(health_collector, mock_http_client):
    """Test collect_snapshot when a service is DOWN."""
    # Mock one service timeout, others OK
    call_count = 0
    
    async def mock_get(url):
        nonlocal call_count
        call_count += 1
        
        if call_count == 1:
            # First service times out
            raise asyncio.TimeoutError("Connection timeout")
        else:
            # Others OK
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "healthy"}
            return mock_response
    
    mock_http_client.get = mock_get
    
    # Collect snapshot
    snapshot = await health_collector.collect_snapshot()
    
    # Verify at least one service is DOWN
    services = snapshot["services"]
    down_services = [name for name, health in services.items() if health["status"] == "DOWN"]
    assert len(down_services) >= 1
    
    # Verify DOWN service has error
    for service_name in down_services:
        assert "error" in services[service_name]


@pytest.mark.asyncio
async def test_collect_snapshot_service_degraded(health_collector, mock_http_client):
    """Test collect_snapshot when a service is DEGRADED."""
    # Mock one service returning 503, others OK
    call_count = 0
    
    async def mock_get(url):
        nonlocal call_count
        call_count += 1
        
        if call_count == 1:
            # First service degraded
            mock_response = MagicMock()
            mock_response.status_code = 503
            return mock_response
        else:
            # Others OK
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "healthy"}
            return mock_response
    
    mock_http_client.get = mock_get
    
    # Collect snapshot
    snapshot = await health_collector.collect_snapshot()
    
    # Verify at least one service is DOWN (503 = server error = DOWN)
    services = snapshot["services"]
    down_services = [name for name, health in services.items() if health["status"] == "DOWN"]
    assert len(down_services) >= 1


@pytest.mark.asyncio
async def test_collect_snapshot_redis_ok(health_collector, mock_redis_client):
    """Test Redis health check when Redis is OK."""
    # Mock services OK
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "healthy"}
    health_collector.http_client.get = AsyncMock(return_value=mock_response)
    
    # Collect snapshot
    snapshot = await health_collector.collect_snapshot()
    
    # Verify Redis is OK
    assert "redis" in snapshot["infra"]
    assert snapshot["infra"]["redis"]["status"] == "OK"
    assert snapshot["infra"]["redis"]["latency_ms"] is not None


@pytest.mark.asyncio
async def test_collect_snapshot_redis_down(health_collector, mock_redis_client):
    """Test Redis health check when Redis is DOWN."""
    # Mock Redis ping failure
    mock_redis_client.ping = AsyncMock(side_effect=Exception("Connection refused"))
    
    # Mock services OK
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "healthy"}
    health_collector.http_client.get = AsyncMock(return_value=mock_response)
    
    # Collect snapshot
    snapshot = await health_collector.collect_snapshot()
    
    # Verify Redis is DOWN
    assert "redis" in snapshot["infra"]
    assert snapshot["infra"]["redis"]["status"] == "DOWN"
    assert "error" in snapshot["infra"]["redis"]


# ============================================================================
# TEST: HEALTH AGGREGATOR
# ============================================================================

def test_aggregator_all_healthy(health_aggregator):
    """Test aggregator when all services are healthy."""
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {
            "service_a": {"status": "OK", "latency_ms": 50, "critical": True},
            "service_b": {"status": "OK", "latency_ms": 80, "critical": True},
        },
        "infra": {
            "redis": {"status": "OK", "latency_ms": 5},
            "postgres": {"status": "OK", "latency_ms": 10},
        },
    }
    
    aggregated = health_aggregator.aggregate(snapshot)
    
    assert aggregated.status == SystemStatus.OK
    assert len(aggregated.services_ok) == 2
    assert len(aggregated.services_down) == 0
    assert len(aggregated.critical_failures) == 0
    assert aggregated.avg_service_latency_ms is not None


def test_aggregator_service_degraded(health_aggregator):
    """Test aggregator when a non-critical service is degraded."""
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {
            "service_a": {"status": "OK", "latency_ms": 50, "critical": True},
            "service_b": {"status": "DEGRADED", "latency_ms": 1500, "critical": False},
        },
        "infra": {
            "redis": {"status": "OK", "latency_ms": 5},
        },
    }
    
    aggregated = health_aggregator.aggregate(snapshot)
    
    assert aggregated.status == SystemStatus.DEGRADED
    assert "service_b" in aggregated.services_degraded
    assert len(aggregated.critical_failures) == 0


def test_aggregator_critical_service_down(health_aggregator):
    """Test aggregator when a critical service is DOWN."""
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {
            "service_a": {"status": "DOWN", "latency_ms": None, "critical": True, "error": "Timeout"},
            "service_b": {"status": "OK", "latency_ms": 80, "critical": True},
        },
        "infra": {
            "redis": {"status": "OK", "latency_ms": 5},
        },
    }
    
    aggregated = health_aggregator.aggregate(snapshot)
    
    assert aggregated.status == SystemStatus.CRITICAL
    assert "service_a" in aggregated.services_down
    assert len(aggregated.critical_failures) > 0
    assert any("service_a" in failure for failure in aggregated.critical_failures)


def test_aggregator_infra_down(health_aggregator):
    """Test aggregator when infrastructure is DOWN."""
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {
            "service_a": {"status": "OK", "latency_ms": 50, "critical": True},
        },
        "infra": {
            "redis": {"status": "DOWN", "latency_ms": None, "error": "Connection refused"},
        },
    }
    
    aggregated = health_aggregator.aggregate(snapshot)
    
    assert aggregated.status == SystemStatus.CRITICAL
    assert "redis" in aggregated.infra_down
    assert len(aggregated.critical_failures) > 0


def test_aggregator_high_latency_degraded(health_aggregator):
    """Test aggregator marks DEGRADED for high latency."""
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {
            "service_a": {"status": "OK", "latency_ms": 1500, "critical": True},  # High latency
        },
        "infra": {
            "redis": {"status": "OK", "latency_ms": 5},
        },
    }
    
    aggregated = health_aggregator.aggregate(snapshot)
    
    # Should be DEGRADED due to high latency
    assert aggregated.status == SystemStatus.DEGRADED
    assert aggregated.max_service_latency_ms > health_aggregator.LATENCY_WARNING_THRESHOLD_MS


# ============================================================================
# TEST: ALERT MANAGER
# ============================================================================

@pytest.mark.asyncio
async def test_alert_manager_critical_system(alert_manager, mock_event_bus):
    """Test alert manager raises CRITICAL alert for system failure."""
    # Create mock aggregated health with CRITICAL status
    aggregated = AggregatedHealth(
        status=SystemStatus.CRITICAL,
        timestamp=datetime.now(timezone.utc).isoformat(),
        services_ok=["service_b"],
        services_down=["service_a"],
        critical_failures=["Service: service_a"],
    )
    
    # Process health
    alerts = await alert_manager.process_health(aggregated)
    
    # Verify alert raised
    assert len(alerts) > 0
    critical_alerts = [a for a in alerts if a.level == AlertLevel.CRITICAL]
    assert len(critical_alerts) > 0
    
    # Verify event published (multiple calls expected for system + component alerts)
    assert mock_event_bus.publish.called
    assert mock_event_bus.publish.call_count >= 1
    
    # Verify at least one call was health.alert_raised
    call_args_list = [call[0] for call in mock_event_bus.publish.call_args_list]
    assert any("health.alert_raised" in str(args) for args in call_args_list)


@pytest.mark.asyncio
async def test_alert_manager_degraded_system(alert_manager, mock_event_bus):
    """Test alert manager raises WARNING alert for degraded system."""
    aggregated = AggregatedHealth(
        status=SystemStatus.DEGRADED,
        timestamp=datetime.now(timezone.utc).isoformat(),
        services_ok=["service_a"],
        services_degraded=["service_b"],
    )
    
    # Process health
    alerts = await alert_manager.process_health(aggregated)
    
    # Verify warning alert raised
    assert len(alerts) > 0
    warning_alerts = [a for a in alerts if a.level == AlertLevel.WARNING]
    assert len(warning_alerts) > 0


@pytest.mark.asyncio
async def test_alert_manager_ess_tripped(alert_manager, mock_event_bus):
    """Test alert manager handles ess.tripped event."""
    event_data = {
        "reason": "Daily loss limit exceeded",
        "severity": "CRITICAL",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    # Process ESS event
    await alert_manager.process_ess_tripped(event_data)
    
    # Verify alert created
    active_alerts = alert_manager.get_active_alerts()
    assert len(active_alerts) > 0
    
    ess_alerts = [a for a in active_alerts if "Emergency Stop" in a.title]
    assert len(ess_alerts) > 0
    assert ess_alerts[0].level == AlertLevel.CRITICAL
    
    # Verify event published
    assert mock_event_bus.publish.called


@pytest.mark.asyncio
async def test_alert_manager_tracks_active_alerts(alert_manager):
    """Test alert manager tracks active alerts."""
    aggregated = AggregatedHealth(
        status=SystemStatus.CRITICAL,
        timestamp=datetime.now(timezone.utc).isoformat(),
        critical_failures=["Service: service_a"],
        services_down=["service_a"],
    )
    
    # Process health
    await alert_manager.process_health(aggregated)
    
    # Verify active alerts tracked
    active = alert_manager.get_active_alerts()
    assert len(active) > 0
    
    # Clear an alert
    alert_id = active[0].alert_id
    success = alert_manager.clear_alert(alert_id)
    assert success
    
    # Verify alert removed
    active_after = alert_manager.get_active_alerts()
    assert len(active_after) == len(active) - 1


def test_alert_manager_history(alert_manager):
    """Test alert manager maintains history."""
    # Create multiple alerts
    for i in range(5):
        alert = alert_manager._create_alert(
            level=AlertLevel.INFO,
            title=f"Test alert {i}",
            message=f"Test message {i}",
            component="test",
        )
    
    # Get history
    history = alert_manager.get_alert_history(limit=10)
    assert len(history) == 5


# ============================================================================
# TEST: INTEGRATION
# ============================================================================

@pytest.mark.asyncio
async def test_end_to_end_health_check(
    health_collector,
    health_aggregator,
    alert_manager,
    mock_http_client,
    mock_event_bus,
):
    """Test end-to-end health check flow."""
    # Setup: Mock one service down, others OK
    call_count = 0
    
    async def mock_get(url):
        nonlocal call_count
        call_count += 1
        
        if "service_a" in url or call_count == 1:
            raise asyncio.TimeoutError("Timeout")
        else:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "healthy"}
            return mock_response
    
    mock_http_client.get = mock_get
    
    # Step 1: Collect snapshot
    snapshot = await health_collector.collect_snapshot()
    assert "services" in snapshot
    
    # Step 2: Aggregate
    aggregated = health_aggregator.aggregate(snapshot)
    assert aggregated.status in [SystemStatus.CRITICAL, SystemStatus.DEGRADED]
    
    # Step 3: Process alerts
    alerts = await alert_manager.process_health(aggregated)
    assert len(alerts) > 0
    
    # Verify event published
    assert mock_event_bus.publish.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
