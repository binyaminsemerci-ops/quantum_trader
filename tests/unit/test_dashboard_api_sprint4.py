"""
Unit Tests for Dashboard API (Sprint 4)

Tests the REST snapshot endpoint and WebSocket functionality.
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime, timezone

# Import dashboard components
from backend.api.dashboard.models import (
    DashboardSnapshot,
    DashboardEvent,
    DashboardPosition,
    DashboardSignal,
    DashboardPortfolio,
    DashboardRisk,
    DashboardSystemHealth,
    ServiceHealthInfo,
    ESSState,
    ServiceStatus,
    SignalDirection,
    PositionSide,
    EventType,
    create_position_updated_event,
    create_pnl_updated_event,
    create_signal_generated_event,
    create_ess_state_changed_event,
    create_health_alert_event,
)
from backend.api.dashboard.routes import (
    aggregate_portfolio_data,
    aggregate_positions,
    aggregate_signals,
    aggregate_risk,
    aggregate_system_health,
)
from backend.api.dashboard.websocket import (
    DashboardConnectionManager,
)


# ========== FIXTURES ==========

@pytest.fixture
def mock_portfolio_snapshot():
    """Mock portfolio service /snapshot response."""
    return {
        "equity": 100000.0,
        "cash": 50000.0,
        "margin_used": 30000.0,
        "margin_available": 20000.0,
        "total_pnl": 5000.0,
        "positions": [
            {"symbol": "BTCUSDT", "quantity": 0.5},
            {"symbol": "ETHUSDT", "quantity": 2.0},
        ]
    }


@pytest.fixture
def mock_portfolio_pnl():
    """Mock portfolio service /pnl response."""
    return {
        "daily_pnl": 150.25,
        "daily_pnl_pct": 0.15,
        "weekly_pnl": 750.50,
        "monthly_pnl": 3500.75,
        "realized_pnl": 2000.0,
        "unrealized_pnl": 3000.0,
    }


@pytest.fixture
def mock_execution_positions():
    """Mock execution service /positions response."""
    return {
        "positions": [
            {
                "symbol": "BTCUSDT",
                "side": "LONG",
                "quantity": 0.5,
                "entry_price": 95000.0,
                "current_price": 96000.0,
                "unrealized_pnl": 500.0,
                "unrealized_pnl_pct": 1.05,
            },
            {
                "symbol": "ETHUSDT",
                "side": "SHORT",
                "quantity": 2.0,
                "entry_price": 3500.0,
                "current_price": 3450.0,
                "unrealized_pnl": 100.0,
                "unrealized_pnl_pct": 1.43,
            }
        ]
    }


@pytest.fixture
def mock_risk_ess():
    """Mock risk service /ess/status response."""
    return {
        "state": "ARMED",
        "reason": None,
        "tripped_at": None,
    }


@pytest.fixture
def mock_portfolio_drawdown():
    """Mock portfolio service /drawdown response."""
    return {
        "daily_dd_pct": 1.5,
        "weekly_dd_pct": 3.2,
        "max_dd_pct": 8.7,
    }


@pytest.fixture
def mock_portfolio_exposure():
    """Mock portfolio service /exposure response."""
    return {
        "total_exposure": 50000.0,
        "long_exposure": 30000.0,
        "short_exposure": 20000.0,
        "net_exposure": 10000.0,
    }


@pytest.fixture
def mock_monitoring_services():
    """Mock monitoring service /services response."""
    return {
        "services": [
            {
                "name": "portfolio-intelligence",
                "status": "OK",
                "latency_ms": 15,
                "last_check": "2025-01-27T10:00:00Z"
            },
            {
                "name": "ai-engine",
                "status": "OK",
                "latency_ms": 25,
                "last_check": "2025-01-27T10:00:00Z"
            },
            {
                "name": "execution",
                "status": "DEGRADED",
                "latency_ms": 150,
                "last_check": "2025-01-27T10:00:00Z"
            }
        ]
    }


@pytest.fixture
def mock_monitoring_alerts():
    """Mock monitoring service /alerts response."""
    return {
        "total": 2,
        "alerts": [
            {
                "timestamp": "2025-01-27T09:55:00Z",
                "service": "execution",
                "severity": "warning",
                "message": "High latency detected"
            }
        ]
    }


# ========== MODEL TESTS ==========

def test_dashboard_position_creation():
    """Test DashboardPosition model creation."""
    position = DashboardPosition(
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        size=0.5,
        entry_price=95000.0,
        current_price=96000.0,
        unrealized_pnl=500.0,
        unrealized_pnl_pct=1.05,
        value=48000.0
    )
    
    assert position.symbol == "BTCUSDT"
    assert position.side == PositionSide.LONG
    assert position.size == 0.5
    assert position.unrealized_pnl == 500.0
    
    # Test to_dict
    data = position.to_dict()
    assert data["symbol"] == "BTCUSDT"
    assert data["side"] == "LONG"
    assert data["unrealized_pnl"] == 500.0


def test_dashboard_signal_creation():
    """Test DashboardSignal model creation."""
    signal = DashboardSignal(
        timestamp="2025-01-27T10:00:00Z",
        symbol="ETHUSDT",
        direction=SignalDirection.BUY,
        confidence=0.85,
        strategy="ensemble",
        target_size=2.0
    )
    
    assert signal.symbol == "ETHUSDT"
    assert signal.direction == SignalDirection.BUY
    assert signal.confidence == 0.85
    
    # Test to_dict
    data = signal.to_dict()
    assert data["direction"] == "BUY"
    assert data["confidence"] == 0.85


def test_dashboard_portfolio_creation():
    """Test DashboardPortfolio model creation."""
    portfolio = DashboardPortfolio(
        equity=100000.0,
        cash=50000.0,
        margin_used=30000.0,
        margin_available=20000.0,
        total_pnl=5000.0,
        daily_pnl=150.25,
        daily_pnl_pct=0.15,
        weekly_pnl=750.50,
        monthly_pnl=3500.75,
        realized_pnl=2000.0,
        unrealized_pnl=3000.0,
        position_count=2
    )
    
    assert portfolio.equity == 100000.0
    assert portfolio.daily_pnl == 150.25
    assert portfolio.position_count == 2
    
    # Test to_dict
    data = portfolio.to_dict()
    assert data["equity"] == 100000.0
    assert data["daily_pnl_pct"] == 0.15


def test_dashboard_risk_creation():
    """Test DashboardRisk model creation."""
    risk = DashboardRisk(
        ess_state=ESSState.ARMED,
        ess_reason=None,
        ess_tripped_at=None,
        daily_drawdown_pct=1.5,
        weekly_drawdown_pct=3.2,
        max_drawdown_pct=8.7,
        exposure_total=50000.0,
        exposure_long=30000.0,
        exposure_short=20000.0,
        exposure_net=10000.0,
        risk_limit_used_pct=15.0
    )
    
    assert risk.ess_state == ESSState.ARMED
    assert risk.daily_drawdown_pct == 1.5
    assert risk.exposure_net == 10000.0
    
    # Test to_dict
    data = risk.to_dict()
    assert data["ess_state"] == "ARMED"
    assert data["daily_drawdown_pct"] == 1.5


def test_dashboard_snapshot_creation():
    """Test DashboardSnapshot model creation."""
    portfolio = DashboardPortfolio(
        equity=100000, cash=50000, margin_used=30000, margin_available=20000,
        total_pnl=5000, daily_pnl=150, daily_pnl_pct=0.15,
        weekly_pnl=750, monthly_pnl=3500, realized_pnl=2000,
        unrealized_pnl=3000, position_count=2
    )
    
    risk = DashboardRisk(
        ess_state=ESSState.ARMED, ess_reason=None, ess_tripped_at=None,
        daily_drawdown_pct=1.5, weekly_drawdown_pct=3.2, max_drawdown_pct=8.7,
        exposure_total=50000, exposure_long=30000, exposure_short=20000,
        exposure_net=10000, risk_limit_used_pct=15.0
    )
    
    system = DashboardSystemHealth(
        overall_status=ServiceStatus.OK,
        services=[],
        alerts_count=0
    )
    
    snapshot = DashboardSnapshot(
        timestamp="2025-01-27T10:00:00Z",
        portfolio=portfolio,
        positions=[],
        signals=[],
        risk=risk,
        system=system
    )
    
    assert snapshot.portfolio.equity == 100000
    assert snapshot.risk.ess_state == ESSState.ARMED
    
    # Test to_dict
    data = snapshot.to_dict()
    assert "portfolio" in data
    assert "risk" in data
    assert "system" in data


def test_dashboard_event_creation():
    """Test DashboardEvent model creation."""
    event = DashboardEvent(
        type="position_updated",
        timestamp="2025-01-27T10:00:00Z",
        payload={"symbol": "BTCUSDT", "pnl": 500.0}
    )
    
    assert event.type == "position_updated"
    assert event.payload["symbol"] == "BTCUSDT"
    
    # Test to_dict
    data = event.to_dict()
    assert data["type"] == "position_updated"
    assert data["payload"]["symbol"] == "BTCUSDT"


def test_event_helper_functions():
    """Test event helper functions."""
    # Position updated event
    position = DashboardPosition(
        symbol="BTCUSDT", side=PositionSide.LONG, size=0.5,
        entry_price=95000, current_price=96000,
        unrealized_pnl=500, unrealized_pnl_pct=1.05, value=48000
    )
    event1 = create_position_updated_event(position)
    assert event1.type == "position_updated"
    assert event1.payload["symbol"] == "BTCUSDT"
    
    # PnL updated event
    portfolio = DashboardPortfolio(
        equity=100000, cash=50000, margin_used=30000, margin_available=20000,
        total_pnl=5000, daily_pnl=150, daily_pnl_pct=0.15,
        weekly_pnl=750, monthly_pnl=3500, realized_pnl=2000,
        unrealized_pnl=3000, position_count=2
    )
    event2 = create_pnl_updated_event(portfolio)
    assert event2.type == "pnl_updated"
    assert event2.payload["daily_pnl"] == 150
    
    # Signal generated event
    signal = DashboardSignal(
        timestamp="2025-01-27T10:00:00Z", symbol="ETHUSDT",
        direction=SignalDirection.BUY, confidence=0.85,
        strategy="ensemble", target_size=2.0
    )
    event3 = create_signal_generated_event(signal)
    assert event3.type == "signal_generated"
    assert event3.payload["symbol"] == "ETHUSDT"
    
    # ESS state changed event
    event4 = create_ess_state_changed_event(ESSState.TRIPPED, "Daily loss limit exceeded")
    assert event4.type == "ess_state_changed"
    assert event4.payload["ess_state"] == "TRIPPED"
    
    # Health alert event
    event5 = create_health_alert_event("execution", ServiceStatus.DEGRADED, "High latency")
    assert event5.type == "health_alert"
    assert event5.payload["service"] == "execution"


# ========== AGGREGATION TESTS ==========

@pytest.mark.asyncio
async def test_aggregate_portfolio_data(mock_portfolio_snapshot, mock_portfolio_pnl):
    """Test portfolio data aggregation."""
    with patch('backend.api.dashboard.routes.fetch_json') as mock_fetch:
        # Mock service responses
        async def fetch_side_effect(url, timeout=5.0):
            if "snapshot" in url:
                return mock_portfolio_snapshot
            elif "pnl" in url:
                return mock_portfolio_pnl
            return None
        
        mock_fetch.side_effect = fetch_side_effect
        
        # Aggregate
        portfolio = await aggregate_portfolio_data()
        
        # Verify
        assert portfolio.equity == 100000.0
        assert portfolio.cash == 50000.0
        assert portfolio.daily_pnl == 150.25
        assert portfolio.daily_pnl_pct == 0.15
        assert portfolio.position_count == 2


@pytest.mark.asyncio
async def test_aggregate_portfolio_data_service_down():
    """Test portfolio aggregation when service is down."""
    with patch('backend.api.dashboard.routes.fetch_json') as mock_fetch:
        # Mock service down
        mock_fetch.return_value = None
        
        # Aggregate
        portfolio = await aggregate_portfolio_data()
        
        # Should return default values
        assert portfolio.equity == 0.0
        assert portfolio.position_count == 0


@pytest.mark.asyncio
async def test_aggregate_positions(mock_execution_positions):
    """Test positions aggregation."""
    with patch('backend.api.dashboard.routes.fetch_json') as mock_fetch:
        mock_fetch.return_value = mock_execution_positions
        
        # Aggregate
        positions = await aggregate_positions()
        
        # Verify
        assert len(positions) == 2
        assert positions[0].symbol == "BTCUSDT"
        assert positions[0].side == PositionSide.LONG
        assert positions[0].unrealized_pnl == 500.0
        assert positions[1].symbol == "ETHUSDT"
        assert positions[1].side == PositionSide.SHORT


@pytest.mark.asyncio
async def test_aggregate_positions_service_down():
    """Test positions aggregation when service is down."""
    with patch('backend.api.dashboard.routes.fetch_json') as mock_fetch:
        mock_fetch.return_value = None
        
        # Aggregate
        positions = await aggregate_positions()
        
        # Should return empty list
        assert positions == []


@pytest.mark.asyncio
async def test_aggregate_signals():
    """Test signals aggregation (currently returns empty list)."""
    with patch('backend.api.dashboard.routes.fetch_json') as mock_fetch:
        mock_fetch.return_value = None
        
        # Aggregate
        signals = await aggregate_signals()
        
        # Should return empty list (endpoint doesn't exist yet)
        assert signals == []


@pytest.mark.asyncio
async def test_aggregate_risk(mock_risk_ess, mock_portfolio_drawdown, mock_portfolio_exposure):
    """Test risk data aggregation."""
    with patch('backend.api.dashboard.routes.fetch_json') as mock_fetch:
        # Mock service responses
        async def fetch_side_effect(url, timeout=5.0):
            if "ess/status" in url:
                return mock_risk_ess
            elif "drawdown" in url:
                return mock_portfolio_drawdown
            elif "exposure" in url:
                return mock_portfolio_exposure
            return None
        
        mock_fetch.side_effect = fetch_side_effect
        
        # Aggregate
        risk = await aggregate_risk()
        
        # Verify
        assert risk.ess_state == ESSState.ARMED
        assert risk.daily_drawdown_pct == 1.5
        assert risk.exposure_total == 50000.0
        assert risk.exposure_net == 10000.0


@pytest.mark.asyncio
async def test_aggregate_system_health(mock_monitoring_services, mock_monitoring_alerts):
    """Test system health aggregation."""
    with patch('backend.api.dashboard.routes.fetch_json') as mock_fetch:
        # Mock service responses
        async def fetch_side_effect(url, timeout=5.0):
            if "services" in url:
                return mock_monitoring_services
            elif "alerts" in url:
                return mock_monitoring_alerts
            return None
        
        mock_fetch.side_effect = fetch_side_effect
        
        # Aggregate
        system = await aggregate_system_health()
        
        # Verify
        assert system.overall_status == ServiceStatus.DEGRADED  # One service degraded
        assert len(system.services) == 3
        assert system.alerts_count == 2
        assert system.services[2].status == ServiceStatus.DEGRADED


# ========== WEBSOCKET TESTS ==========

@pytest.mark.asyncio
async def test_connection_manager_connect():
    """Test WebSocket connection manager connect."""
    manager = DashboardConnectionManager()
    mock_ws = AsyncMock()
    
    await manager.connect(mock_ws)
    
    assert manager.connection_count == 1
    assert mock_ws in manager.active_connections
    mock_ws.accept.assert_called_once()


@pytest.mark.asyncio
async def test_connection_manager_disconnect():
    """Test WebSocket connection manager disconnect."""
    manager = DashboardConnectionManager()
    mock_ws = AsyncMock()
    
    await manager.connect(mock_ws)
    assert manager.connection_count == 1
    
    await manager.disconnect(mock_ws)
    assert manager.connection_count == 0


@pytest.mark.asyncio
async def test_connection_manager_broadcast():
    """Test WebSocket broadcast to multiple clients."""
    manager = DashboardConnectionManager()
    mock_ws1 = AsyncMock()
    mock_ws2 = AsyncMock()
    
    # Connect two clients
    await manager.connect(mock_ws1)
    await manager.connect(mock_ws2)
    
    # Broadcast event
    event = DashboardEvent(
        type="position_updated",
        timestamp="2025-01-27T10:00:00Z",
        payload={"symbol": "BTCUSDT", "pnl": 500.0}
    )
    
    await manager.broadcast(event)
    
    # Verify both clients received message
    assert mock_ws1.send_text.called
    assert mock_ws2.send_text.called


@pytest.mark.asyncio
async def test_connection_manager_broadcast_handles_disconnected():
    """Test broadcast handles disconnected clients."""
    manager = DashboardConnectionManager()
    mock_ws1 = AsyncMock()
    mock_ws2 = AsyncMock()
    
    # Make ws2 fail on send
    mock_ws2.send_text.side_effect = Exception("Connection closed")
    
    await manager.connect(mock_ws1)
    await manager.connect(mock_ws2)
    
    # Broadcast event
    event = DashboardEvent(
        type="position_updated",
        timestamp="2025-01-27T10:00:00Z",
        payload={"symbol": "BTCUSDT"}
    )
    
    await manager.broadcast(event)
    
    # ws2 should be removed
    assert manager.connection_count == 1
    assert mock_ws1 in manager.active_connections
    assert mock_ws2 not in manager.active_connections


# ========== INTEGRATION TESTS ==========

@pytest.mark.asyncio
async def test_full_snapshot_aggregation():
    """Test full snapshot aggregation with all services."""
    from backend.api.dashboard.routes import get_dashboard_snapshot
    
    with patch('backend.api.dashboard.routes.fetch_json') as mock_fetch:
        # Mock all service responses
        async def fetch_side_effect(url, timeout=5.0):
            if "portfolio/snapshot" in url:
                return {"equity": 100000, "cash": 50000, "margin_used": 30000, "margin_available": 20000, "total_pnl": 5000, "positions": []}
            elif "portfolio/pnl" in url:
                return {"daily_pnl": 150, "daily_pnl_pct": 0.15, "weekly_pnl": 750, "monthly_pnl": 3500, "realized_pnl": 2000, "unrealized_pnl": 3000}
            elif "execution/positions" in url:
                return {"positions": []}
            elif "ess/status" in url:
                return {"state": "ARMED", "reason": None, "tripped_at": None}
            elif "drawdown" in url:
                return {"daily_dd_pct": 1.5, "weekly_dd_pct": 3.2, "max_dd_pct": 8.7}
            elif "exposure" in url:
                return {"total_exposure": 50000, "long_exposure": 30000, "short_exposure": 20000, "net_exposure": 10000}
            elif "health/services" in url:
                return {"services": [{"name": "test", "status": "OK", "latency_ms": 10, "last_check": "2025-01-27T10:00:00Z"}]}
            elif "health/alerts" in url:
                return {"total": 0, "alerts": []}
            return None
        
        mock_fetch.side_effect = fetch_side_effect
        
        # Get snapshot
        snapshot_dict = await get_dashboard_snapshot()
        
        # Verify structure
        assert "timestamp" in snapshot_dict
        assert "portfolio" in snapshot_dict
        assert "positions" in snapshot_dict
        assert "signals" in snapshot_dict
        assert "risk" in snapshot_dict
        assert "system" in snapshot_dict
        
        # Verify portfolio data
        assert snapshot_dict["portfolio"]["equity"] == 100000
        assert snapshot_dict["portfolio"]["daily_pnl"] == 150
        
        # Verify risk data
        assert snapshot_dict["risk"]["ess_state"] == "ARMED"
        assert snapshot_dict["risk"]["daily_drawdown_pct"] == 1.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
