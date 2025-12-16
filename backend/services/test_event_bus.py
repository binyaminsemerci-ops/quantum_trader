"""
Comprehensive tests for EventBus module.

Tests cover:
- Event model serialization
- Event factory methods
- Publishing and subscribing
- Async and sync handlers
- Error isolation
- Multiple handlers per event
- Unsubscribe functionality
- Statistics tracking
- Graceful shutdown

Run: pytest backend/services/test_event_bus.py -v
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import Mock

from backend.services.event_bus import (
    Event,
    PolicyUpdatedEvent,
    StrategyPromotedEvent,
    ModelPromotedEvent,
    HealthStatusChangedEvent,
    OpportunitiesUpdatedEvent,
    TradeExecutedEvent,
    InMemoryEventBus,
    create_event_bus,
)


# ============================================================================
# Event Model Tests
# ============================================================================

def test_base_event_creation():
    """Test basic Event creation."""
    ts = datetime.utcnow()
    event = Event(
        type="test.event",
        timestamp=ts,
        payload={"key": "value"},
        source="test_module",
        correlation_id="test-123"
    )
    
    assert event.type == "test.event"
    assert event.payload == {"key": "value"}
    assert event.source == "test_module"
    assert event.correlation_id == "test-123"


def test_event_to_dict():
    """Test Event serialization."""
    ts = datetime.utcnow()
    event = Event(
        type="test.event",
        timestamp=ts,
        payload={"foo": "bar"},
        source="test"
    )
    
    d = event.to_dict()
    
    assert d["type"] == "test.event"
    assert d["timestamp"] == ts.isoformat()
    assert d["payload"] == {"foo": "bar"}
    assert d["source"] == "test"


def test_policy_updated_event_factory():
    """Test PolicyUpdatedEvent factory method."""
    event = PolicyUpdatedEvent.create(
        risk_mode="AGGRESSIVE",
        max_risk_per_trade=0.03,
        max_positions=15,
        global_min_confidence=0.7,
        changes={"risk_mode": "NORMAL -> AGGRESSIVE"}
    )
    
    assert event.type == "policy.updated"
    assert event.source == "msc_ai"
    assert event.payload["risk_mode"] == "AGGRESSIVE"
    assert event.payload["max_risk_per_trade"] == 0.03
    assert event.payload["max_positions"] == 15
    assert event.payload["global_min_confidence"] == 0.7
    assert "risk_mode" in event.payload["changes"]


def test_strategy_promoted_event_factory():
    """Test StrategyPromotedEvent factory method."""
    event = StrategyPromotedEvent.create(
        strategy_id="strat-123",
        strategy_name="MeanReversion_v1",
        from_state="SHADOW",
        to_state="LIVE",
        performance_score=0.85,
        reason="Outperformed benchmark by 15%"
    )
    
    assert event.type == "strategy.promoted"
    assert event.source == "strategy_generator"
    assert event.payload["strategy_id"] == "strat-123"
    assert event.payload["from_state"] == "SHADOW"
    assert event.payload["to_state"] == "LIVE"
    assert event.payload["performance_score"] == 0.85


def test_model_promoted_event_factory():
    """Test ModelPromotedEvent factory method."""
    event = ModelPromotedEvent.create(
        model_name="xgboost",
        old_version="v1.2.3",
        new_version="v1.3.0",
        improvement_pct=12.5,
        shadow_test_results={"accuracy": 0.92, "sharpe": 2.1}
    )
    
    assert event.type == "model.promoted"
    assert event.source == "continuous_learning"
    assert event.payload["model_name"] == "xgboost"
    assert event.payload["new_version"] == "v1.3.0"
    assert event.payload["improvement_pct"] == 12.5


def test_health_status_changed_event_factory():
    """Test HealthStatusChangedEvent factory method."""
    event = HealthStatusChangedEvent.create(
        status="CRITICAL",
        previous_status="WARNING",
        failed_modules=["policy_store", "redis"],
        warning_modules=["ai_services"],
        details={"error_count": 5}
    )
    
    assert event.type == "health.status_changed"
    assert event.source == "system_health_monitor"
    assert event.payload["status"] == "CRITICAL"
    assert event.payload["previous_status"] == "WARNING"
    assert len(event.payload["failed_modules"]) == 2
    assert "policy_store" in event.payload["failed_modules"]


def test_opportunities_updated_event_factory():
    """Test OpportunitiesUpdatedEvent factory method."""
    event = OpportunitiesUpdatedEvent.create(
        top_symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        rankings={"BTCUSDT": 0.95, "ETHUSDT": 0.88, "SOLUSDT": 0.82},
        num_symbols_scored=50,
        top_score=0.95
    )
    
    assert event.type == "opportunities.updated"
    assert event.source == "opportunity_ranker"
    assert len(event.payload["top_symbols"]) == 3
    assert event.payload["rankings"]["BTCUSDT"] == 0.95


def test_trade_executed_event_factory():
    """Test TradeExecutedEvent factory method."""
    event = TradeExecutedEvent.create(
        symbol="BTCUSDT",
        side="BUY",
        quantity=0.5,
        price=45000.0,
        order_id="order-123",
        strategy_id="strat-456",
        pnl=250.0
    )
    
    assert event.type == "trade.executed"
    assert event.source == "executor"
    assert event.payload["symbol"] == "BTCUSDT"
    assert event.payload["side"] == "BUY"
    assert event.payload["quantity"] == 0.5
    assert event.payload["pnl"] == 250.0


# ============================================================================
# EventBus Tests
# ============================================================================

@pytest.mark.asyncio
async def test_eventbus_creation():
    """Test EventBus instantiation."""
    bus = InMemoryEventBus(max_queue_size=100)
    
    stats = bus.get_stats()
    assert stats["published"] == 0
    assert stats["processed"] == 0
    assert stats["queue_size"] == 0


@pytest.mark.asyncio
async def test_publish_and_subscribe_async_handler():
    """Test publishing event with async handler."""
    bus = InMemoryEventBus()
    
    received_events = []
    
    async def handler(event: Event):
        received_events.append(event)
    
    bus.subscribe("test.event", handler)
    
    # Start event loop
    loop_task = asyncio.create_task(bus.run_forever())
    
    # Publish event
    test_event = Event(type="test.event", payload={"foo": "bar"})
    await bus.publish(test_event)
    
    # Wait for processing
    await asyncio.sleep(0.1)
    
    # Stop bus
    bus.stop()
    await loop_task
    
    # Verify
    assert len(received_events) == 1
    assert received_events[0].type == "test.event"
    assert received_events[0].payload["foo"] == "bar"


@pytest.mark.asyncio
async def test_publish_and_subscribe_sync_handler():
    """Test publishing event with sync handler."""
    bus = InMemoryEventBus()
    
    received_events = []
    
    def sync_handler(event: Event):
        # Sync function
        received_events.append(event)
    
    bus.subscribe("test.sync", sync_handler)
    
    # Start event loop
    loop_task = asyncio.create_task(bus.run_forever())
    
    # Publish event
    test_event = Event(type="test.sync")
    await bus.publish(test_event)
    
    # Wait for processing
    await asyncio.sleep(0.1)
    
    # Stop bus
    bus.stop()
    await loop_task
    
    # Verify
    assert len(received_events) == 1
    assert received_events[0].type == "test.sync"


@pytest.mark.asyncio
async def test_multiple_handlers_same_event():
    """Test multiple handlers for the same event type."""
    bus = InMemoryEventBus()
    
    handler1_calls = []
    handler2_calls = []
    
    async def handler1(event: Event):
        handler1_calls.append(event)
    
    async def handler2(event: Event):
        handler2_calls.append(event)
    
    bus.subscribe("multi.event", handler1)
    bus.subscribe("multi.event", handler2)
    
    # Start event loop
    loop_task = asyncio.create_task(bus.run_forever())
    
    # Publish event
    test_event = Event(type="multi.event")
    await bus.publish(test_event)
    
    # Wait for processing
    await asyncio.sleep(0.1)
    
    # Stop bus
    bus.stop()
    await loop_task
    
    # Both handlers should have been called
    assert len(handler1_calls) == 1
    assert len(handler2_calls) == 1


@pytest.mark.asyncio
async def test_handler_error_isolation():
    """Test that handler errors don't crash the bus."""
    bus = InMemoryEventBus()
    
    successful_calls = []
    
    async def failing_handler(event: Event):
        raise ValueError("Intentional error")
    
    async def successful_handler(event: Event):
        successful_calls.append(event)
    
    bus.subscribe("error.test", failing_handler)
    bus.subscribe("error.test", successful_handler)
    
    # Start event loop
    loop_task = asyncio.create_task(bus.run_forever())
    
    # Publish event
    test_event = Event(type="error.test")
    await bus.publish(test_event)
    
    # Wait for processing
    await asyncio.sleep(0.1)
    
    # Stop bus
    bus.stop()
    await loop_task
    
    # Successful handler should still have been called
    assert len(successful_calls) == 1
    
    # Stats should show error
    stats = bus.get_stats()
    assert stats["errors"] >= 1


@pytest.mark.asyncio
async def test_unsubscribe():
    """Test unsubscribing a handler."""
    bus = InMemoryEventBus()
    
    calls = []
    
    async def handler(event: Event):
        calls.append(event)
    
    # Subscribe
    bus.subscribe("unsub.test", handler)
    
    # Unsubscribe
    result = bus.unsubscribe("unsub.test", handler)
    assert result is True
    
    # Start event loop
    loop_task = asyncio.create_task(bus.run_forever())
    
    # Publish event
    test_event = Event(type="unsub.test")
    await bus.publish(test_event)
    
    # Wait for processing
    await asyncio.sleep(0.1)
    
    # Stop bus
    bus.stop()
    await loop_task
    
    # Handler should NOT have been called
    assert len(calls) == 0


@pytest.mark.asyncio
async def test_no_handlers_for_event():
    """Test publishing event with no registered handlers."""
    bus = InMemoryEventBus()
    
    # Start event loop
    loop_task = asyncio.create_task(bus.run_forever())
    
    # Publish event with no handlers
    test_event = Event(type="no.handlers")
    await bus.publish(test_event)
    
    # Wait for processing
    await asyncio.sleep(0.1)
    
    # Stop bus
    bus.stop()
    await loop_task
    
    # Should not crash - just no-op
    stats = bus.get_stats()
    assert stats["published"] == 1
    # Note: processed count may be 0 since no handlers were called
    # The event was dequeued but no handlers executed


@pytest.mark.asyncio
async def test_statistics_tracking():
    """Test EventBus statistics."""
    bus = InMemoryEventBus()
    
    call_count = 0
    
    async def handler(event: Event):
        nonlocal call_count
        call_count += 1
    
    bus.subscribe("stats.test", handler)
    
    # Start event loop
    loop_task = asyncio.create_task(bus.run_forever())
    
    # Publish 5 events
    for i in range(5):
        await bus.publish(Event(type="stats.test"))
    
    # Wait for processing
    await asyncio.sleep(0.2)
    
    # Stop bus
    bus.stop()
    await loop_task
    
    # Check stats
    stats = bus.get_stats()
    assert stats["published"] == 5
    assert stats["processed"] == 5
    assert call_count == 5


@pytest.mark.asyncio
async def test_graceful_shutdown():
    """Test graceful shutdown waits for queue to empty."""
    bus = InMemoryEventBus()
    
    processed_events = []
    
    async def slow_handler(event: Event):
        await asyncio.sleep(0.05)  # Simulate slow processing
        processed_events.append(event)
    
    bus.subscribe("shutdown.test", slow_handler)
    
    # Start event loop
    loop_task = asyncio.create_task(bus.run_forever())
    
    # Publish multiple events
    for i in range(3):
        await bus.publish(Event(
            type="shutdown.test",
            payload={"index": i}
        ))
    
    # Shutdown (should wait for all events)
    await bus.shutdown()
    await loop_task
    
    # All events should have been processed
    assert len(processed_events) == 3


@pytest.mark.asyncio
async def test_factory_function():
    """Test create_event_bus factory function."""
    bus = create_event_bus(max_queue_size=500)
    
    assert isinstance(bus, InMemoryEventBus)
    
    # Should work normally
    calls = []
    
    async def handler(event: Event):
        calls.append(event)
    
    bus.subscribe("factory.test", handler)
    
    loop_task = asyncio.create_task(bus.run_forever())
    
    await bus.publish(Event(type="factory.test"))
    
    await asyncio.sleep(0.1)
    
    bus.stop()
    await loop_task
    
    assert len(calls) == 1


# ============================================================================
# Integration Test - Real-World Scenario
# ============================================================================

@pytest.mark.asyncio
async def test_integration_scenario():
    """
    Test a realistic scenario with multiple event types and subscribers.
    
    Simulates:
    - MSC AI publishes PolicyUpdatedEvent
    - Logger subscribes to all events
    - Health Monitor subscribes to policy changes
    - Multiple events published
    """
    bus = InMemoryEventBus()
    
    logged_events = []
    health_events = []
    
    # Logger subscribes to everything
    async def logger_handler(event: Event):
        logged_events.append(event)
    
    # Health monitor only cares about policy and health events
    async def health_handler(event: Event):
        health_events.append(event)
    
    bus.subscribe("policy.updated", logger_handler)
    bus.subscribe("policy.updated", health_handler)
    bus.subscribe("strategy.promoted", logger_handler)
    bus.subscribe("health.status_changed", logger_handler)
    bus.subscribe("health.status_changed", health_handler)
    
    # Start event loop
    loop_task = asyncio.create_task(bus.run_forever())
    
    # Publish various events
    await bus.publish(PolicyUpdatedEvent.create(
        risk_mode="DEFENSIVE",
        max_positions=8
    ))
    
    await bus.publish(StrategyPromotedEvent.create(
        strategy_id="strat-1",
        strategy_name="MomentumStrategy",
        from_state="SHADOW",
        to_state="LIVE",
        performance_score=0.9
    ))
    
    await bus.publish(HealthStatusChangedEvent.create(
        status="WARNING",
        previous_status="HEALTHY",
        warning_modules=["redis"]
    ))
    
    # Wait for processing
    await asyncio.sleep(0.2)
    
    # Stop bus
    bus.stop()
    await loop_task
    
    # Verify
    # Logger should see all 3 events
    assert len(logged_events) == 3
    
    # Health handler should see 2 events (policy + health)
    assert len(health_events) == 2
    assert health_events[0].type == "policy.updated"
    assert health_events[1].type == "health.status_changed"
    
    # Check stats
    stats = bus.get_stats()
    assert stats["published"] == 3
    assert stats["processed"] == 3
    assert stats["errors"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
