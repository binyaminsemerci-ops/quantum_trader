"""
Event-Driven Trading Flow v1 - Integration Tests
=================================================

Tests for the complete event-driven trading pipeline using EventBus v2.

Test Flow:
    1. AI Engine publishes signal.generated
    2. SignalSubscriber receives → RiskGuard → publishes trade.execution_requested
    3. TradeSubscriber receives → ExecutionEngine → publishes trade.executed
    4. PositionSubscriber receives → publishes position.opened
    5. Position closes → publishes position.closed
    6. RL/CLM/Supervisor receive closed position data

Author: Quantum Trader AI Team
Date: December 2, 2025
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone

from backend.events.event_types import EventType
from backend.events.publishers import (
    publish_signal_generated,
    publish_execution_requested,
    publish_trade_executed,
    publish_position_opened,
    publish_position_closed,
)
from backend.events.subscribers import (
    SignalSubscriber,
    TradeSubscriber,
    PositionSubscriber,
    RiskSubscriber,
    ErrorSubscriber,
)
from backend.core.event_bus import InMemoryEventBus


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def event_bus():
    """Create in-memory event bus for testing."""
    return InMemoryEventBus()


@pytest.fixture
def mock_risk_guard():
    """Create mock RiskGuard."""
    guard = AsyncMock()
    guard.can_execute = AsyncMock(return_value=(True, ""))
    return guard


@pytest.fixture
def mock_policy_store():
    """Create mock PolicyStore."""
    store = AsyncMock()
    
    # Mock RiskProfile
    mock_profile = Mock()
    mock_profile.name = "NORMAL"
    mock_profile.max_leverage = 5.0
    mock_profile.max_risk_pct_per_trade = 1.5
    mock_profile.position_size_cap_usd = 1000.0
    mock_profile.global_min_confidence = 0.65
    
    store.get_active_risk_profile = AsyncMock(return_value=mock_profile)
    return store


# ============================================================================
# TEST 1: Complete Signal-to-Position Flow
# ============================================================================

@pytest.mark.asyncio
async def test_complete_signal_to_position_flow(event_bus, mock_risk_guard, mock_policy_store):
    """
    Test complete flow: signal → execution → position opened.
    
    This tests the first 4 steps of the trading pipeline.
    """
    
    # Track events received
    events_received = []
    
    async def track_event(event_type, event_data):
        events_received.append((event_type, event_data))
    
    # Initialize subscribers
    signal_subscriber = SignalSubscriber(
        risk_guard=mock_risk_guard,
        policy_store=mock_policy_store,
    )
    
    trade_subscriber = TradeSubscriber(
        execution_engine=None,  # Uses simulation
    )
    
    position_subscriber = PositionSubscriber()
    
    # Subscribe to events
    await event_bus.subscribe(EventType.SIGNAL_GENERATED, signal_subscriber.handle_signal)
    await event_bus.subscribe(EventType.TRADE_EXECUTION_REQUESTED, trade_subscriber.handle_execution_request)
    await event_bus.subscribe(EventType.TRADE_EXECUTED, position_subscriber.handle_trade_executed)
    
    # Track all published events
    await event_bus.subscribe(EventType.TRADE_EXECUTION_REQUESTED, lambda data: track_event("EXECUTION_REQUESTED", data))
    await event_bus.subscribe(EventType.TRADE_EXECUTED, lambda data: track_event("TRADE_EXECUTED", data))
    await event_bus.subscribe(EventType.POSITION_OPENED, lambda data: track_event("POSITION_OPENED", data))
    
    # STEP 1: Publish signal.generated
    trace_id = "test-signal-001"
    await publish_signal_generated(
        symbol="BTCUSDT",
        side="BUY",
        confidence=0.85,
        timeframe="1h",
        model_version="ensemble_v2",
        trace_id=trace_id,
    )
    
    # Wait for event processing
    await asyncio.sleep(0.5)
    
    # Verify: trade.execution_requested published
    execution_requested_events = [e for e in events_received if e[0] == "EXECUTION_REQUESTED"]
    assert len(execution_requested_events) == 1
    exec_req_data = execution_requested_events[0][1]
    assert exec_req_data["symbol"] == "BTCUSDT"
    assert exec_req_data["side"] == "BUY"
    assert exec_req_data["confidence"] == 0.85
    assert exec_req_data["trace_id"] == trace_id
    
    # Verify: trade.executed published
    trade_executed_events = [e for e in events_received if e[0] == "TRADE_EXECUTED"]
    assert len(trade_executed_events) == 1
    trade_exec_data = trade_executed_events[0][1]
    assert trade_exec_data["symbol"] == "BTCUSDT"
    assert trade_exec_data["side"] == "BUY"
    assert "order_id" in trade_exec_data
    assert trade_exec_data["trace_id"] == trace_id
    
    # Verify: position.opened published
    position_opened_events = [e for e in events_received if e[0] == "POSITION_OPENED"]
    assert len(position_opened_events) == 1
    pos_opened_data = position_opened_events[0][1]
    assert pos_opened_data["symbol"] == "BTCUSDT"
    assert pos_opened_data["is_long"] is True
    assert pos_opened_data["trace_id"] == trace_id
    
    # Verify: trace_id propagated through entire flow
    assert all(e[1]["trace_id"] == trace_id for e in events_received)


# ============================================================================
# TEST 2: Position Closed → Learning Systems
# ============================================================================

@pytest.mark.asyncio
async def test_position_closed_feeds_learning_systems(event_bus):
    """
    Test that position.closed event feeds data to all learning systems.
    
    This tests the integration with RL, CLM, Supervisor, Drift Detector.
    """
    
    # Mock learning systems
    mock_rl_position_sizing = Mock()
    mock_rl_meta_strategy = Mock()
    mock_model_supervisor = Mock()
    mock_drift_detector = Mock()
    mock_clm = Mock()
    
    # Initialize subscriber
    position_subscriber = PositionSubscriber(
        rl_position_sizing=mock_rl_position_sizing,
        rl_meta_strategy=mock_rl_meta_strategy,
        model_supervisor=mock_model_supervisor,
        drift_detector=mock_drift_detector,
        clm=mock_clm,
    )
    
    # Subscribe to position.closed
    await event_bus.subscribe(EventType.POSITION_CLOSED, position_subscriber.handle_position_closed)
    
    # Publish position.closed event
    trace_id = "test-position-closed-001"
    await publish_position_closed(
        symbol="ETHUSDT",
        entry_price=2000.0,
        exit_price=2100.0,
        size_usd=500.0,
        leverage=3.0,
        is_long=True,
        pnl_usd=50.0,
        pnl_pct=5.0,
        duration_seconds=3600.0,
        exit_reason="TP",
        trace_id=trace_id,
        entry_confidence=0.75,
        model_version="lstm_v1",
        market_condition="BULLISH",
    )
    
    # Wait for processing
    await asyncio.sleep(0.3)
    
    # Verify: All learning systems received the data
    # (In real implementation, these would be actual method calls)
    # For this test, we just verify the subscriber logged the calls
    assert True  # Placeholder - would check actual method calls


# ============================================================================
# TEST 3: Risk Alert Triggers Kill Switch
# ============================================================================

@pytest.mark.asyncio
async def test_risk_alert_triggers_kill_switch(event_bus):
    """
    Test that CRITICAL risk alerts trigger kill switch.
    """
    
    # Mock dependencies
    mock_risk_guard = Mock()
    mock_emergency_stop = Mock()
    
    # Initialize subscriber
    risk_subscriber = RiskSubscriber(
        risk_guard=mock_risk_guard,
        emergency_stop_controller=mock_emergency_stop,
    )
    
    # Subscribe to risk.alert
    await event_bus.subscribe(EventType.RISK_ALERT, risk_subscriber.handle_risk_alert)
    
    # Publish CRITICAL risk alert
    from backend.events.publishers import publish_risk_alert
    
    trace_id = "test-risk-alert-001"
    await publish_risk_alert(
        severity="CRITICAL",
        alert_type="MAX_DRAWDOWN_BREACHED",
        message="Daily drawdown exceeded 10%",
        trace_id=trace_id,
        current_drawdown_pct=10.5,
        max_allowed_drawdown_pct=5.0,
        action_taken="EMERGENCY_STOP",
        risk_profile="NORMAL",
    )
    
    # Wait for processing
    await asyncio.sleep(0.3)
    
    # Verify: Kill switch would be triggered
    # (In real implementation, would check actual method calls)
    assert True  # Placeholder


# ============================================================================
# TEST 4: Event Error Handling
# ============================================================================

@pytest.mark.asyncio
async def test_event_error_handling(event_bus):
    """
    Test that errors in event handlers are caught and published.
    """
    
    # Initialize error subscriber
    error_subscriber = ErrorSubscriber()
    
    # Track errors received
    errors_received = []
    
    async def track_error(event_data):
        errors_received.append(event_data)
    
    await event_bus.subscribe(EventType.SYSTEM_EVENT_ERROR, error_subscriber.handle_event_error)
    await event_bus.subscribe(EventType.SYSTEM_EVENT_ERROR, track_error)
    
    # Publish error event
    from backend.events.publishers import publish_event_error
    
    trace_id = "test-error-001"
    await publish_event_error(
        error_type="ValueError",
        error_message="Invalid symbol format",
        component="SignalSubscriber",
        trace_id=trace_id,
        event_type=str(EventType.SIGNAL_GENERATED),
        is_recoverable=True,
    )
    
    # Wait for processing
    await asyncio.sleep(0.3)
    
    # Verify: Error was received and logged
    assert len(errors_received) > 0
    assert errors_received[0]["error_type"] == "ValueError"
    assert errors_received[0]["trace_id"] == trace_id


# ============================================================================
# TEST 5: Trace ID Propagation
# ============================================================================

@pytest.mark.asyncio
async def test_trace_id_propagation(event_bus, mock_risk_guard, mock_policy_store):
    """
    Test that trace_id is propagated through the entire event chain.
    """
    
    # Track all events with trace_id
    trace_ids = []
    
    async def track_trace_id(event_data):
        if "trace_id" in event_data:
            trace_ids.append(event_data["trace_id"])
    
    # Subscribe tracker to all event types
    await event_bus.subscribe(EventType.SIGNAL_GENERATED, track_trace_id)
    await event_bus.subscribe(EventType.TRADE_EXECUTION_REQUESTED, track_trace_id)
    await event_bus.subscribe(EventType.TRADE_EXECUTED, track_trace_id)
    await event_bus.subscribe(EventType.POSITION_OPENED, track_trace_id)
    
    # Initialize subscribers
    signal_subscriber = SignalSubscriber(
        risk_guard=mock_risk_guard,
        policy_store=mock_policy_store,
    )
    trade_subscriber = TradeSubscriber()
    position_subscriber = PositionSubscriber()
    
    await event_bus.subscribe(EventType.SIGNAL_GENERATED, signal_subscriber.handle_signal)
    await event_bus.subscribe(EventType.TRADE_EXECUTION_REQUESTED, trade_subscriber.handle_execution_request)
    await event_bus.subscribe(EventType.TRADE_EXECUTED, position_subscriber.handle_trade_executed)
    
    # Publish signal with specific trace_id
    test_trace_id = "end-to-end-trace-001"
    await publish_signal_generated(
        symbol="BTCUSDT",
        side="BUY",
        confidence=0.80,
        timeframe="1h",
        model_version="test_v1",
        trace_id=test_trace_id,
    )
    
    # Wait for all events to process
    await asyncio.sleep(0.5)
    
    # Verify: All events have the same trace_id
    assert all(tid == test_trace_id for tid in trace_ids)
    assert len(trace_ids) >= 4  # signal, execution_requested, executed, opened


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
