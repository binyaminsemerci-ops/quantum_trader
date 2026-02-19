"""
Test: Portfolio Intelligence Service - Sprint 2 Service #4

Tests:
- build_snapshot() with fake TradeStore + Binance data
- Total equity calculation (cash + unrealized PnL)
- Unrealized PnL calculation (LONG/SHORT positions)
- Realized PnL aggregation (today/week/month)
- Exposure breakdown by symbol
- Daily drawdown calculation
- Event handlers (trade.opened, trade.closed, order.executed)
- Event publishing (portfolio.* events)
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone, timedelta

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from microservices.portfolio_intelligence.service import PortfolioIntelligenceService
from microservices.portfolio_intelligence.models import (
    PositionInfo, PortfolioSnapshot, PnLBreakdown, ExposureBreakdown, DrawdownMetrics
)


@pytest.fixture
async def service():
    """Create Portfolio Intelligence service with mocked dependencies."""
    with patch('microservices.portfolio_intelligence.service.EventBus') as mock_event_bus, \
         patch('microservices.portfolio_intelligence.service.EventBuffer') as mock_event_buffer, \
         patch('microservices.portfolio_intelligence.service.httpx.AsyncClient') as mock_http_client:
        
        # Configure mocks
        mock_event_bus.return_value.subscribe = MagicMock()
        mock_event_bus.return_value.publish = AsyncMock()
        
        service = PortfolioIntelligenceService()
        
        # Manually set mocks
        service.event_bus = mock_event_bus.return_value
        service.event_buffer = mock_event_buffer.return_value
        service.http_client = mock_http_client.return_value
        service._running = True
        
        # Mock TradeStore
        service.trade_store = MagicMock()
        service.trade_store.get_open_trades = MagicMock(return_value=[])
        service.trade_store.get_closed_trades_since = MagicMock(return_value=[])
        
        yield service


@pytest.mark.asyncio
async def test_build_snapshot_empty_portfolio(service):
    """Test snapshot generation with empty portfolio."""
    # Mock empty trades
    service.trade_store.get_open_trades = MagicMock(return_value=[])
    service._get_cash_balance = AsyncMock(return_value=10000.0)
    
    snapshot = await service._generate_snapshot()
    
    assert snapshot is not None
    assert snapshot.total_equity == 10000.0
    assert snapshot.cash_balance == 10000.0
    assert snapshot.total_exposure == 0.0
    assert snapshot.num_positions == 0
    assert len(snapshot.positions) == 0
    assert snapshot.unrealized_pnl == 0.0
    assert snapshot.realized_pnl_today == 0.0
    assert snapshot.daily_pnl == 0.0


@pytest.mark.asyncio
async def test_build_snapshot_with_long_position(service):
    """Test snapshot with one LONG position (profit)."""
    # Mock open trades
    service.trade_store.get_open_trades = MagicMock(return_value=[
        {
            "trade_id": "trade_1",
            "symbol": "BTCUSDT",
            "side": "LONG",
            "quantity": 0.2,
            "entry_price": 42000.0,
            "leverage": 5.0,
            "category": "CORE"
        }
    ])
    service._get_cash_balance = AsyncMock(return_value=10000.0)
    service._get_current_price = AsyncMock(return_value=42741.13)  # +1.76% gain
    service._get_realized_pnl_today = AsyncMock(return_value=0.0)
    
    snapshot = await service._generate_snapshot()
    
    # Verify snapshot
    assert snapshot.num_positions == 1
    assert len(snapshot.positions) == 1
    
    # Verify position
    pos = snapshot.positions[0]
    assert pos.symbol == "BTCUSDT"
    assert pos.side == "LONG"
    assert pos.size == 0.2
    assert pos.entry_price == 42000.0
    assert pos.current_price == 42741.13
    
    # Verify unrealized PnL: (42741.13 - 42000.0) * 0.2 = 148.226
    assert abs(pos.unrealized_pnl - 148.226) < 0.01
    assert abs(pos.unrealized_pnl_pct - 0.0176) < 0.0001
    
    # Verify exposure: 0.2 * 42741.13 = 8548.226
    assert abs(pos.exposure - 8548.226) < 0.01
    assert pos.leverage == 5.0
    
    # Verify total equity: 10000 + 148.226 = 10148.226
    assert abs(snapshot.total_equity - 10148.226) < 0.01
    assert abs(snapshot.unrealized_pnl - 148.226) < 0.01


@pytest.mark.asyncio
async def test_build_snapshot_with_short_position(service):
    """Test snapshot with one SHORT position (profit)."""
    # Mock open trades
    service.trade_store.get_open_trades = MagicMock(return_value=[
        {
            "trade_id": "trade_2",
            "symbol": "SOLUSDT",
            "side": "SHORT",
            "quantity": 45.0,
            "entry_price": 98.50,
            "leverage": 10.0,
            "category": "EXPANSION"
        }
    ])
    service._get_cash_balance = AsyncMock(return_value=10000.0)
    service._get_current_price = AsyncMock(return_value=95.09)  # -3.46% (profit for SHORT)
    service._get_realized_pnl_today = AsyncMock(return_value=0.0)
    
    snapshot = await service._generate_snapshot()
    
    # Verify position
    pos = snapshot.positions[0]
    assert pos.symbol == "SOLUSDT"
    assert pos.side == "SHORT"
    assert pos.size == 45.0
    assert pos.entry_price == 98.50
    assert pos.current_price == 95.09
    
    # Verify unrealized PnL: (98.50 - 95.09) * 45.0 = 153.45
    assert abs(pos.unrealized_pnl - 153.45) < 0.01
    assert abs(pos.unrealized_pnl_pct - 0.0346) < 0.001
    
    # Verify total equity: 10000 + 153.45 = 10153.45
    assert abs(snapshot.total_equity - 10153.45) < 0.01


@pytest.mark.asyncio
async def test_build_snapshot_with_multiple_positions(service):
    """Test snapshot with multiple positions (mixed LONG/SHORT)."""
    # Mock open trades
    service.trade_store.get_open_trades = MagicMock(return_value=[
        {
            "symbol": "BTCUSDT",
            "side": "LONG",
            "quantity": 0.2,
            "entry_price": 42000.0,
            "leverage": 5.0
        },
        {
            "symbol": "ETHUSDT",
            "side": "LONG",
            "quantity": 5.0,
            "entry_price": 2680.0,
            "leverage": 5.0
        },
        {
            "symbol": "SOLUSDT",
            "side": "SHORT",
            "quantity": 45.0,
            "entry_price": 98.50,
            "leverage": 10.0
        }
    ])
    service._get_cash_balance = AsyncMock(return_value=10000.0)
    
    # Mock current prices
    async def mock_get_price(symbol):
        prices = {
            "BTCUSDT": 42741.13,  # +1.76%
            "ETHUSDT": 2833.30,   # +5.72%
            "SOLUSDT": 95.09      # -3.46% (profit for SHORT)
        }
        return prices.get(symbol, 100.0)
    
    service._get_current_price = mock_get_price
    service._get_realized_pnl_today = AsyncMock(return_value=0.0)
    
    snapshot = await service._generate_snapshot()
    
    # Verify 3 positions
    assert snapshot.num_positions == 3
    assert len(snapshot.positions) == 3
    
    # Verify symbols
    symbols = [pos.symbol for pos in snapshot.positions]
    assert "BTCUSDT" in symbols
    assert "ETHUSDT" in symbols
    assert "SOLUSDT" in symbols
    
    # Calculate expected unrealized PnL
    # BTC: (42741.13 - 42000.0) * 0.2 = 148.226
    # ETH: (2833.30 - 2680.0) * 5.0 = 766.50
    # SOL: (98.50 - 95.09) * 45.0 = 153.45
    # Total: 148.226 + 766.50 + 153.45 = 1068.176
    expected_unrealized_pnl = 148.226 + 766.50 + 153.45
    assert abs(snapshot.unrealized_pnl - expected_unrealized_pnl) < 0.01
    
    # Verify total equity
    expected_equity = 10000.0 + expected_unrealized_pnl
    assert abs(snapshot.total_equity - expected_equity) < 0.01


@pytest.mark.asyncio
async def test_realized_pnl_calculation(service):
    """Test realized PnL aggregation from closed trades."""
    # Mock closed trades today
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    
    closed_trades = [
        {"trade_id": "trade_1", "realized_pnl": 125.50},
        {"trade_id": "trade_2", "realized_pnl": 89.75},
        {"trade_id": "trade_3", "realized_pnl": -45.20},
        {"trade_id": "trade_4", "realized_pnl": 250.10},
    ]
    
    service.trade_store.get_closed_trades_since = MagicMock(return_value=closed_trades)
    service.trade_store.get_open_trades = MagicMock(return_value=[])
    service._get_cash_balance = AsyncMock(return_value=10000.0)
    
    snapshot = await service._generate_snapshot()
    
    # Expected realized PnL: 125.50 + 89.75 - 45.20 + 250.10 = 420.15
    expected_realized_pnl = 125.50 + 89.75 - 45.20 + 250.10
    assert abs(snapshot.realized_pnl_today - expected_realized_pnl) < 0.01
    
    # Daily PnL = realized_pnl_today + unrealized_pnl (0 in this case)
    assert abs(snapshot.daily_pnl - expected_realized_pnl) < 0.01


@pytest.mark.asyncio
async def test_exposure_calculation(service):
    """Test exposure breakdown by symbol."""
    # Mock open trades
    service.trade_store.get_open_trades = MagicMock(return_value=[
        {"symbol": "BTCUSDT", "side": "LONG", "quantity": 0.2, "entry_price": 42000.0, "leverage": 5.0},
        {"symbol": "ETHUSDT", "side": "LONG", "quantity": 5.0, "entry_price": 2680.0, "leverage": 5.0},
        {"symbol": "SOLUSDT", "side": "SHORT", "quantity": 45.0, "entry_price": 98.50, "leverage": 10.0},
    ])
    
    async def mock_get_price(symbol):
        return {"BTCUSDT": 42741.13, "ETHUSDT": 2833.30, "SOLUSDT": 95.09}.get(symbol, 100.0)
    
    service._get_current_price = mock_get_price
    service._get_cash_balance = AsyncMock(return_value=10000.0)
    service._get_realized_pnl_today = AsyncMock(return_value=0.0)
    
    snapshot = await service._generate_snapshot()
    exposure = await service.get_exposure_breakdown()
    
    # Expected exposures:
    # BTC: 0.2 * 42741.13 = 8548.226
    # ETH: 5.0 * 2833.30 = 14166.50
    # SOL: 45.0 * 95.09 = 4279.05
    # Total: 8548.226 + 14166.50 + 4279.05 = 26993.776
    
    assert abs(exposure.total_exposure - 26993.776) < 0.01
    
    # Long exposure: BTC + ETH = 8548.226 + 14166.50 = 22714.726
    assert abs(exposure.long_exposure - 22714.726) < 0.01
    
    # Short exposure: SOL = 4279.05
    assert abs(exposure.short_exposure - 4279.05) < 0.01
    
    # Net exposure: 22714.726 - 4279.05 = 18435.676
    assert abs(exposure.net_exposure - 18435.676) < 0.01
    
    # Exposure by symbol
    assert "BTCUSDT" in exposure.exposure_by_symbol
    assert "ETHUSDT" in exposure.exposure_by_symbol
    assert "SOLUSDT" in exposure.exposure_by_symbol


@pytest.mark.asyncio
async def test_daily_drawdown_calculation(service):
    """Test daily drawdown calculation."""
    # Start with high equity
    service.trade_store.get_open_trades = MagicMock(return_value=[])
    service._get_cash_balance = AsyncMock(return_value=12000.0)  # Peak equity
    service._get_realized_pnl_today = AsyncMock(return_value=0.0)
    
    # Generate first snapshot (peak)
    await service._generate_snapshot()
    assert service._peak_equity_today == 12000.0
    
    # Simulate loss (equity drops to 10000)
    service._get_cash_balance = AsyncMock(return_value=10000.0)
    
    # Generate second snapshot
    snapshot = await service._generate_snapshot()
    
    # Expected drawdown: ((12000 - 10000) / 12000) * 100 = 16.67%
    expected_drawdown = ((12000.0 - 10000.0) / 12000.0) * 100
    assert abs(snapshot.daily_drawdown_pct - expected_drawdown) < 0.01


@pytest.mark.asyncio
async def test_handle_trade_opened_event(service):
    """Test trade.opened event handler triggers snapshot rebuild."""
    service.trade_store.get_open_trades = MagicMock(return_value=[
        {"symbol": "BTCUSDT", "side": "LONG", "quantity": 0.1, "entry_price": 42000.0, "leverage": 5.0}
    ])
    service._get_cash_balance = AsyncMock(return_value=10000.0)
    service._get_current_price = AsyncMock(return_value=42000.0)
    service._get_realized_pnl_today = AsyncMock(return_value=0.0)
    
    # Track snapshot generation
    initial_count = service._snapshots_generated
    
    # Trigger event
    await service._handle_trade_opened({"symbol": "BTCUSDT", "side": "LONG", "quantity": 0.1})
    
    # Verify snapshot was regenerated
    assert service._snapshots_generated == initial_count + 1
    
    # Verify portfolio.snapshot_updated was published
    service.event_bus.publish.assert_called()
    call_args = service.event_bus.publish.call_args_list[-1]
    assert call_args[0][0] == "portfolio.snapshot_updated"


@pytest.mark.asyncio
async def test_handle_trade_closed_event(service):
    """Test trade.closed event triggers PnL update."""
    service.trade_store.get_open_trades = MagicMock(return_value=[])
    service.trade_store.get_closed_trades_since = MagicMock(return_value=[
        {"realized_pnl": 150.50}
    ])
    service._get_cash_balance = AsyncMock(return_value=10000.0)
    service._get_realized_pnl_today = AsyncMock(return_value=150.50)
    
    # Trigger event
    await service._handle_trade_closed({
        "symbol": "BTCUSDT",
        "realized_pnl": 150.50
    })
    
    # Verify both portfolio.snapshot_updated and portfolio.pnl_updated were published
    assert service.event_bus.publish.call_count >= 2
    
    # Check event types
    event_types = [call[0][0] for call in service.event_bus.publish.call_args_list]
    assert "portfolio.snapshot_updated" in event_types
    assert "portfolio.pnl_updated" in event_types


@pytest.mark.asyncio
async def test_pnl_breakdown_response(service):
    """Test PnL breakdown API response."""
    # Setup mock snapshot
    service._current_snapshot = PortfolioSnapshot(
        total_equity=10500.0,
        cash_balance=10000.0,
        total_exposure=5000.0,
        num_positions=2,
        positions=[],
        unrealized_pnl=500.0,
        realized_pnl_today=250.0,
        daily_pnl=750.0,
        daily_drawdown_pct=0.0,
        timestamp=datetime.now(timezone.utc).isoformat()
    )
    
    pnl = await service.get_pnl_breakdown()
    
    assert pnl.realized_pnl_today == 250.0
    assert pnl.unrealized_pnl == 500.0
    assert pnl.total_pnl == 750.0  # realized_today + unrealized


@pytest.mark.asyncio
async def test_service_health_check(service):
    """Test service health check returns all components."""
    health = await service.get_health()
    
    assert health["service"] == "portfolio-intelligence"
    assert "version" in health
    assert "status" in health
    assert "components" in health
    assert len(health["components"]) >= 3  # EventBus, TradeStore, EventBuffer
    
    # Verify component names
    component_names = [c["name"] for c in health["components"]]
    assert "EventBus" in component_names
    assert "TradeStore" in component_names
    assert "EventBuffer" in component_names


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
