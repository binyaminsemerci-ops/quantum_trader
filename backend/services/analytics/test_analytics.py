"""
Tests for Analytics Service.
"""

import asyncio
import pytest
from datetime import datetime

from backend.services.eventbus import InMemoryEventBus, PolicyUpdatedEvent, TradeExecutedEvent
from backend.services.analytics import (
    AnalyticsService,
    StrategyMetrics,
    ModelMetrics,
    TradeMetrics,
    InMemoryMetricsRepository,
)
from backend.services.policy_store import RiskMode


@pytest.fixture
def repository():
    return InMemoryMetricsRepository()


@pytest.fixture
def eventbus():
    return InMemoryEventBus()


@pytest.fixture
def analytics(eventbus, repository):
    service = AnalyticsService(eventbus, repository)
    service.subscribe_to_events()
    return service


@pytest.fixture
async def running_bus(eventbus):
    """Create and start an EventBus, cleanup after test."""
    task = asyncio.create_task(eventbus.run_forever())
    await asyncio.sleep(0.05)
    yield eventbus
    eventbus.stop()
    await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_strategy_metrics_calculation():
    """Test strategy metrics calculations."""
    metrics = StrategyMetrics(strategy_id="test_strat")
    
    # Add winning trades
    metrics.total_trades = 10
    metrics.winning_trades = 7
    metrics.losing_trades = 3
    metrics.avg_win = 100.0
    metrics.avg_loss = -50.0
    
    win_rate = metrics.calculate_win_rate()
    assert win_rate == 0.7
    
    profit_factor = metrics.calculate_profit_factor()
    expected_pf = (7 * 100.0) / (3 * 50.0)
    assert abs(profit_factor - expected_pf) < 0.01


@pytest.mark.asyncio
async def test_trade_metrics_pnl_calculation():
    """Test trade PnL calculations."""
    # Long trade
    long_trade = TradeMetrics(
        trade_id="trade1",
        symbol="BTCUSDT",
        strategy_id="momentum",
        side="LONG",
        entry_price=50000.0,
        exit_price=51000.0,
        quantity=1.0,
    )
    
    pnl = long_trade.calculate_pnl()
    assert pnl == 1000.0
    
    pnl_pct = long_trade.calculate_pnl_percent()
    assert abs(pnl_pct - 2.0) < 0.01
    
    # Short trade
    short_trade = TradeMetrics(
        trade_id="trade2",
        symbol="ETHUSDT",
        strategy_id="mean_rev",
        side="SHORT",
        entry_price=3000.0,
        exit_price=2900.0,
        quantity=10.0,
    )
    
    pnl = short_trade.calculate_pnl()
    assert pnl == 1000.0


@pytest.mark.asyncio
async def test_repository_save_and_retrieve(repository):
    """Test repository save and retrieve operations."""
    # Strategy metrics
    strategy = StrategyMetrics(strategy_id="test1")
    await repository.save_strategy_metrics(strategy)
    
    retrieved = await repository.get_strategy_metrics("test1")
    assert retrieved is not None
    assert retrieved.strategy_id == "test1"
    
    # Model metrics
    model = ModelMetrics(model_name="lstm", version="1.0")
    await repository.save_model_metrics(model)
    
    retrieved = await repository.get_model_metrics("lstm")
    assert retrieved is not None
    assert retrieved.model_name == "lstm"


@pytest.mark.asyncio
async def test_analytics_on_policy_updated(running_bus, analytics):
    """Test analytics handles policy update events."""
    event = PolicyUpdatedEvent.create(
        risk_mode=RiskMode.AGGRESSIVE,
        allowed_strategies=["strat1"],
        global_min_confidence=0.7,
        max_risk_per_trade=0.02,
        max_positions=10,
    )
    
    await running_bus.publish(event)
    await asyncio.sleep(0.1)
    
    system_metrics = await analytics.get_system_metrics()
    assert system_metrics.current_risk_mode == "AGGRESSIVE"
    assert system_metrics.policy_changes_count == 1


@pytest.mark.asyncio
async def test_analytics_on_trade_executed(running_bus, analytics):
    """Test analytics handles trade execution events."""
    event = TradeExecutedEvent.create(
        order_id="trade123",
        symbol="BTCUSDT",
        strategy_id="momentum_v1",
        side="BUY",
        size=1.0,
        price=51000.0,
        pnl=1000.0,
    )
    
    await running_bus.publish(event)
    await asyncio.sleep(0.1)
    
    # Check strategy metrics updated
    strategy_metrics = await analytics.get_strategy_metrics("momentum_v1")
    assert strategy_metrics is not None
    assert strategy_metrics.total_trades == 1
    assert strategy_metrics.winning_trades == 1
    assert strategy_metrics.total_pnl == 1000.0
    
    # Check system metrics
    system_metrics = await analytics.get_system_metrics()
    assert system_metrics.closed_positions == 1
    assert system_metrics.total_pnl == 1000.0


@pytest.mark.asyncio
async def test_trade_history(analytics):
    """Test trade history retrieval."""
    # Add some trades
    for i in range(5):
        trade = TradeMetrics(
            trade_id=f"trade{i}",
            symbol="BTCUSDT",
            strategy_id="momentum",
            side="LONG",
            entry_price=50000.0 + i * 100,
            quantity=1.0,
        )
        await analytics.repository.save_trade_metrics(trade)
    
    history = await analytics.get_trade_history(limit=3)
    assert len(history) == 3
    
    # Most recent first
    assert history[0].trade_id == "trade4"


@pytest.mark.asyncio
async def test_performance_report(analytics):
    """Test comprehensive performance report generation."""
    # Add some data
    strategy = StrategyMetrics(
        strategy_id="strat1",
        total_trades=10,
        winning_trades=7,
        stage="LIVE",
    )
    await analytics.repository.save_strategy_metrics(strategy)
    
    model = ModelMetrics(
        model_name="lstm",
        version="1.0",
        accuracy=0.85,
        stage="LIVE",
    )
    await analytics.repository.save_model_metrics(model)
    
    report = await analytics.generate_performance_report()
    
    assert "system" in report
    assert "strategies" in report
    assert "models" in report
    assert "summary" in report
    assert report["summary"]["total_strategies"] == 1
    assert report["summary"]["live_strategies"] == 1
    assert report["summary"]["total_models"] == 1


@pytest.mark.asyncio
async def test_multiple_trades_update_strategy(analytics):
    """Test multiple trades correctly update strategy metrics."""
    trades = [
        ("trade1", 50000.0, 51000.0, 1.0),  # Win: +1000
        ("trade2", 51000.0, 51500.0, 1.0),  # Win: +500
        ("trade3", 51500.0, 51000.0, 1.0),  # Loss: -500
    ]
    
    for trade_id, entry, exit_price, qty in trades:
        event = TradeExecutedEvent.create(
            order_id=trade_id,
            symbol="BTCUSDT",
            strategy_id="test_strat",
            side="BUY",
            size=qty,
            price=exit_price,
            pnl=(exit_price - entry) * qty,
        )
        await analytics.on_trade_executed(event)
    
    metrics = await analytics.get_strategy_metrics("test_strat")
    assert metrics.total_trades == 3
    assert metrics.winning_trades == 2
    assert metrics.losing_trades == 1
    assert metrics.total_pnl == 1000.0
    assert abs(metrics.win_rate - 0.6667) < 0.01
