"""
Tests for Emergency Stop System (ESS)

Validates all ESS functionality including:
- Condition evaluators
- Controller activation/reset
- State persistence
- Event publishing
- Full system integration
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from backend.services.risk.emergency_stop_system import (
    # Core classes
    EmergencyStopController,
    EmergencyStopSystem,
    EmergencyState,
    ESSStatus,
    # Events
    EmergencyStopEvent,
    EmergencyResetEvent,
    # Evaluators
    DrawdownEmergencyEvaluator,
    SystemHealthEmergencyEvaluator,
    ExecutionErrorEmergencyEvaluator,
    DataFeedEmergencyEvaluator,
    ManualTriggerEmergencyEvaluator,
    # Fakes
    FakePolicyStore,
    FakeExchangeClient,
    FakeEventBus,
    FakeMetricsRepository,
    FakeSystemHealthMonitor,
    FakeDataFeedMonitor,
)


# ============================================================================
# CONTROLLER TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_controller_activation():
    """Test basic ESS activation."""
    policy_store = FakePolicyStore()
    exchange = FakeExchangeClient()
    event_bus = FakeEventBus()
    
    controller = EmergencyStopController(
        policy_store=policy_store,
        exchange=exchange,
        event_bus=event_bus,
    )
    
    # Initially inactive
    assert not controller.is_active
    assert controller.state.status == ESSStatus.INACTIVE
    
    # Activate
    await controller.activate("Test emergency")
    
    # Should be active
    assert controller.is_active
    assert controller.state.status == ESSStatus.ACTIVE
    assert controller.state.reason == "Test emergency"
    assert controller.state.activation_count == 1
    
    # Should have closed positions and canceled orders
    assert exchange.positions_closed == 3
    assert exchange.orders_canceled == 5
    
    # Should have published event
    assert len(event_bus.events) == 1
    assert isinstance(event_bus.events[0], EmergencyStopEvent)
    assert event_bus.events[0].reason == "Test emergency"


@pytest.mark.asyncio
async def test_controller_reset():
    """Test ESS manual reset."""
    policy_store = FakePolicyStore()
    exchange = FakeExchangeClient()
    event_bus = FakeEventBus()
    
    controller = EmergencyStopController(
        policy_store=policy_store,
        exchange=exchange,
        event_bus=event_bus,
    )
    
    # Activate then reset
    await controller.activate("Test")
    assert controller.is_active
    
    await controller.reset("admin")
    
    # Should be inactive
    assert not controller.is_active
    assert controller.state.status == ESSStatus.INACTIVE
    assert controller.state.reason is None
    
    # Should have published reset event
    assert len(event_bus.events) == 2
    assert isinstance(event_bus.events[1], EmergencyResetEvent)


@pytest.mark.asyncio
async def test_controller_state_persistence():
    """Test ESS state is persisted to PolicyStore."""
    policy_store = FakePolicyStore()
    exchange = FakeExchangeClient()
    event_bus = FakeEventBus()
    
    controller = EmergencyStopController(
        policy_store=policy_store,
        exchange=exchange,
        event_bus=event_bus,
    )
    
    await controller.activate("Persistence test")
    
    # Check PolicyStore
    ess_data = policy_store.get("emergency_stop")
    assert ess_data["active"] is True
    assert ess_data["reason"] == "Persistence test"
    assert ess_data["auto_recover"] is False
    assert "timestamp" in ess_data


@pytest.mark.asyncio
async def test_controller_double_activation():
    """Test that double activation is ignored."""
    policy_store = FakePolicyStore()
    exchange = FakeExchangeClient()
    event_bus = FakeEventBus()
    
    controller = EmergencyStopController(
        policy_store=policy_store,
        exchange=exchange,
        event_bus=event_bus,
    )
    
    await controller.activate("First")
    await controller.activate("Second")  # Should be ignored
    
    # Should only activate once
    assert controller.state.activation_count == 1
    assert controller.state.reason == "First"
    assert len(event_bus.events) == 1


# ============================================================================
# EVALUATOR TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_drawdown_evaluator_daily_loss():
    """Test DrawdownEvaluator triggers on daily loss."""
    metrics = FakeMetricsRepository()
    evaluator = DrawdownEmergencyEvaluator(
        metrics_repo=metrics,
        max_daily_loss_percent=10.0,
    )
    
    # Normal condition
    metrics.daily_pnl_pct = -5.0
    triggered, reason = await evaluator.check()
    assert not triggered
    
    # Catastrophic loss
    metrics.daily_pnl_pct = -12.0
    triggered, reason = await evaluator.check()
    assert triggered
    assert "Daily PnL catastrophic" in reason
    assert "-12.00%" in reason


@pytest.mark.asyncio
async def test_drawdown_evaluator_equity_dd():
    """Test DrawdownEvaluator triggers on equity drawdown."""
    metrics = FakeMetricsRepository()
    evaluator = DrawdownEmergencyEvaluator(
        metrics_repo=metrics,
        max_equity_drawdown_percent=25.0,
    )
    
    # Normal drawdown
    metrics.drawdown_pct = 15.0
    triggered, reason = await evaluator.check()
    assert not triggered
    
    # Catastrophic drawdown
    metrics.drawdown_pct = 30.0
    triggered, reason = await evaluator.check()
    assert triggered
    assert "Equity drawdown catastrophic" in reason
    assert "30.00%" in reason


@pytest.mark.asyncio
async def test_health_evaluator():
    """Test SystemHealthEvaluator triggers on CRITICAL status."""
    health = FakeSystemHealthMonitor()
    evaluator = SystemHealthEmergencyEvaluator(health_monitor=health)
    
    # Healthy
    health.status = "HEALTHY"
    triggered, reason = await evaluator.check()
    assert not triggered
    
    # Degraded (not critical yet)
    health.status = "DEGRADED"
    triggered, reason = await evaluator.check()
    assert not triggered
    
    # Critical
    health.status = "CRITICAL"
    triggered, reason = await evaluator.check()
    assert triggered
    assert "CRITICAL" in reason


@pytest.mark.asyncio
async def test_execution_error_evaluator():
    """Test ExecutionErrorEvaluator triggers on excessive SL hits."""
    metrics = FakeMetricsRepository()
    evaluator = ExecutionErrorEmergencyEvaluator(
        metrics_repo=metrics,
        max_sl_hits_per_period=10,
        period_minutes=15,
    )
    
    # Normal
    metrics.sl_hits = 5
    triggered, reason = await evaluator.check()
    assert not triggered
    
    # Excessive
    metrics.sl_hits = 15
    triggered, reason = await evaluator.check()
    assert triggered
    assert "Excessive SL hits" in reason
    assert "15" in reason


@pytest.mark.asyncio
async def test_data_feed_evaluator_corrupted():
    """Test DataFeedEvaluator triggers on corrupted data."""
    data_feed = FakeDataFeedMonitor()
    evaluator = DataFeedEmergencyEvaluator(
        data_feed_monitor=data_feed,
        max_staleness_minutes=5,
    )
    
    # Normal
    data_feed.corrupted = False
    triggered, reason = await evaluator.check()
    assert not triggered
    
    # Corrupted
    data_feed.corrupted = True
    triggered, reason = await evaluator.check()
    assert triggered
    assert "corrupted" in reason.lower()


@pytest.mark.asyncio
async def test_data_feed_evaluator_stale():
    """Test DataFeedEvaluator triggers on stale data."""
    data_feed = FakeDataFeedMonitor()
    evaluator = DataFeedEmergencyEvaluator(
        data_feed_monitor=data_feed,
        max_staleness_minutes=5,
    )
    
    # Fresh data
    data_feed.last_update = datetime.utcnow()
    triggered, reason = await evaluator.check()
    assert not triggered
    
    # Stale data
    data_feed.last_update = datetime.utcnow() - timedelta(minutes=10)
    triggered, reason = await evaluator.check()
    assert triggered
    assert "stale" in reason.lower()


@pytest.mark.asyncio
async def test_manual_trigger_evaluator():
    """Test ManualTriggerEvaluator."""
    evaluator = ManualTriggerEmergencyEvaluator()
    
    # Not triggered
    triggered, reason = await evaluator.check()
    assert not triggered
    
    # Trigger manually
    evaluator.trigger("Admin emergency stop")
    triggered, reason = await evaluator.check()
    assert triggered
    assert reason == "Admin emergency stop"
    
    # Reset
    evaluator.reset()
    triggered, reason = await evaluator.check()
    assert not triggered


# ============================================================================
# SYSTEM TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_ess_full_system():
    """Test full ESS system with multiple evaluators."""
    policy_store = FakePolicyStore()
    exchange = FakeExchangeClient()
    event_bus = FakeEventBus()
    metrics = FakeMetricsRepository()
    health = FakeSystemHealthMonitor()
    
    evaluators = [
        DrawdownEmergencyEvaluator(metrics, max_daily_loss_percent=10.0),
        SystemHealthEmergencyEvaluator(health),
    ]
    
    controller = EmergencyStopController(
        policy_store=policy_store,
        exchange=exchange,
        event_bus=event_bus,
    )
    
    ess = EmergencyStopSystem(
        evaluators=evaluators,
        controller=controller,
        policy_store=policy_store,
        check_interval_sec=1,
    )
    
    # Start ESS
    task = ess.start()
    
    # Run for a bit (all conditions normal)
    await asyncio.sleep(1.5)
    assert not controller.is_active
    
    # Trigger health failure
    health.status = "CRITICAL"
    
    # Wait for ESS to detect
    await asyncio.sleep(2)
    
    # Should be activated
    assert controller.is_active
    assert "CRITICAL" in controller.state.reason
    
    # Stop ESS
    await ess.stop()
    await task


@pytest.mark.asyncio
async def test_ess_first_trigger_wins():
    """Test that only the first triggered condition activates ESS."""
    policy_store = FakePolicyStore()
    exchange = FakeExchangeClient()
    event_bus = FakeEventBus()
    metrics = FakeMetricsRepository()
    health = FakeSystemHealthMonitor()
    
    evaluators = [
        DrawdownEmergencyEvaluator(metrics, max_daily_loss_percent=10.0),
        SystemHealthEmergencyEvaluator(health),
    ]
    
    controller = EmergencyStopController(
        policy_store=policy_store,
        exchange=exchange,
        event_bus=event_bus,
    )
    
    ess = EmergencyStopSystem(
        evaluators=evaluators,
        controller=controller,
        policy_store=policy_store,
        check_interval_sec=1,
    )
    
    # Trigger BOTH conditions
    metrics.daily_pnl_pct = -15.0  # Drawdown trigger
    health.status = "CRITICAL"      # Health trigger
    
    # Start ESS
    task = ess.start()
    await asyncio.sleep(2)
    
    # Should be activated by first evaluator (Drawdown)
    assert controller.is_active
    assert "Daily PnL" in controller.state.reason
    
    # Should only activate once
    assert controller.state.activation_count == 1
    
    await ess.stop()
    await task


@pytest.mark.asyncio
async def test_ess_skip_checks_when_active():
    """Test that ESS skips condition checks when already active."""
    policy_store = FakePolicyStore()
    exchange = FakeExchangeClient()
    event_bus = FakeEventBus()
    metrics = FakeMetricsRepository()
    
    evaluator = DrawdownEmergencyEvaluator(metrics, max_daily_loss_percent=10.0)
    
    controller = EmergencyStopController(
        policy_store=policy_store,
        exchange=exchange,
        event_bus=event_bus,
    )
    
    ess = EmergencyStopSystem(
        evaluators=[evaluator],
        controller=controller,
        policy_store=policy_store,
        check_interval_sec=1,
    )
    
    # Pre-activate ESS
    await controller.activate("Pre-activated")
    
    # Start ESS
    task = ess.start()
    
    # Trigger condition (should be ignored)
    metrics.daily_pnl_pct = -20.0
    await asyncio.sleep(2)
    
    # Should still show original activation
    assert controller.is_active
    assert controller.state.reason == "Pre-activated"
    assert controller.state.activation_count == 1
    
    await ess.stop()
    await task


@pytest.mark.asyncio
async def test_ess_multiple_activation_cycles():
    """Test ESS can be activated, reset, and activated again."""
    policy_store = FakePolicyStore()
    exchange = FakeExchangeClient()
    event_bus = FakeEventBus()
    
    controller = EmergencyStopController(
        policy_store=policy_store,
        exchange=exchange,
        event_bus=event_bus,
    )
    
    # First activation
    await controller.activate("First emergency")
    assert controller.is_active
    assert controller.state.activation_count == 1
    
    # Reset
    await controller.reset()
    assert not controller.is_active
    
    # Second activation
    await controller.activate("Second emergency")
    assert controller.is_active
    assert controller.state.activation_count == 2
    assert controller.state.reason == "Second emergency"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_integration_drawdown_emergency():
    """Integration test: Drawdown triggers full emergency shutdown."""
    # Setup full system
    policy_store = FakePolicyStore()
    exchange = FakeExchangeClient()
    event_bus = FakeEventBus()
    metrics = FakeMetricsRepository()
    
    evaluator = DrawdownEmergencyEvaluator(
        metrics_repo=metrics,
        max_daily_loss_percent=10.0,
        max_equity_drawdown_percent=25.0,
    )
    
    controller = EmergencyStopController(
        policy_store=policy_store,
        exchange=exchange,
        event_bus=event_bus,
    )
    
    ess = EmergencyStopSystem(
        evaluators=[evaluator],
        controller=controller,
        policy_store=policy_store,
        check_interval_sec=1,
    )
    
    # Start system
    task = ess.start()
    await asyncio.sleep(1)
    
    # Simulate catastrophic loss
    metrics.daily_pnl_pct = -15.0
    
    # Wait for detection and activation
    await asyncio.sleep(2)
    
    # Verify full shutdown
    assert controller.is_active
    assert exchange.positions_closed == 3
    assert exchange.orders_canceled == 5
    assert len(event_bus.events) == 1
    
    # Verify PolicyStore
    ess_data = policy_store.get("emergency_stop")
    assert ess_data["active"] is True
    assert ess_data["auto_recover"] is False
    
    # Cleanup
    await ess.stop()
    await task


@pytest.mark.asyncio
async def test_integration_manual_trigger():
    """Integration test: Manual trigger works end-to-end."""
    policy_store = FakePolicyStore()
    exchange = FakeExchangeClient()
    event_bus = FakeEventBus()
    
    manual_trigger = ManualTriggerEmergencyEvaluator()
    
    controller = EmergencyStopController(
        policy_store=policy_store,
        exchange=exchange,
        event_bus=event_bus,
    )
    
    ess = EmergencyStopSystem(
        evaluators=[manual_trigger],
        controller=controller,
        policy_store=policy_store,
        check_interval_sec=1,
    )
    
    # Start system
    task = ess.start()
    await asyncio.sleep(1)
    
    # Admin triggers emergency
    manual_trigger.trigger("Suspicious trading activity detected")
    
    # Wait for activation
    await asyncio.sleep(2)
    
    # Verify
    assert controller.is_active
    assert "Suspicious trading activity" in controller.state.reason
    assert exchange.positions_closed == 3
    assert exchange.orders_canceled == 5
    
    # Cleanup
    await ess.stop()
    await task


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
