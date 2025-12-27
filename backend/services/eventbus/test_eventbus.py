"""
Unit tests for the EventBus subsystem.

Tests cover:
- Event publishing and subscription
- Async and sync handlers
- Error handling and resilience
- Multiple handlers per event type
- Event type routing
- Statistics tracking
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from backend.services.eventbus import (
    InMemoryEventBus,
    Event,
    PolicyUpdatedEvent,
    StrategyPromotedEvent,
    HealthStatusChangedEvent,
    RiskMode,
    StrategyLifecycle,
    HealthStatus,
)


@pytest.fixture
def event_bus():
    """Create a fresh EventBus for each test."""
    return InMemoryEventBus(max_queue_size=100, max_workers=2)


@pytest.fixture
async def running_bus(event_bus):
    """Create and start an EventBus, cleanup after test."""
    task = asyncio.create_task(event_bus.run_forever())
    yield event_bus
    event_bus.stop()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


class TestEventCreation:
    """Test event dataclass creation."""
    
    def test_basic_event_creation(self):
        """Test creating a basic Event."""
        event = Event(
            type="test.event",
            timestamp=datetime.utcnow(),
            payload={"key": "value"}
        )
        
        assert event.type == "test.event"
        assert isinstance(event.timestamp, datetime)
        assert event.payload["key"] == "value"
    
    def test_policy_updated_event_factory(self):
        """Test PolicyUpdatedEvent factory method."""
        event = PolicyUpdatedEvent.create(
            risk_mode=RiskMode.AGGRESSIVE,
            allowed_strategies=["strat1", "strat2"],
            global_min_confidence=0.65,
            max_risk_per_trade=0.03,
            max_positions=10,
        )
        
        assert event.type == "policy.updated"
        assert event.payload["risk_mode"] == "AGGRESSIVE"
        assert len(event.payload["allowed_strategies"]) == 2
        assert event.payload["global_min_confidence"] == 0.65
    
    def test_strategy_promoted_event_factory(self):
        """Test StrategyPromotedEvent factory method."""
        event = StrategyPromotedEvent.create(
            strategy_id="test_strat",
            from_stage=StrategyLifecycle.SHADOW,
            to_stage=StrategyLifecycle.LIVE,
            reason="Passed validation",
            metrics={"sharpe": 2.0},
        )
        
        assert event.type == "strategy.promoted"
        assert event.payload["strategy_id"] == "test_strat"
        assert event.payload["from_stage"] == "SHADOW"
        assert event.payload["to_stage"] == "LIVE"


class TestEventBusBasics:
    """Test basic EventBus functionality."""
    
    @pytest.mark.asyncio
    async def test_publish_and_subscribe(self, running_bus):
        """Test basic publish/subscribe flow."""
        received_events = []
        
        async def handler(event: Event):
            received_events.append(event)
        
        running_bus.subscribe("test.event", handler)
        
        event = Event("test.event", datetime.utcnow(), {"data": "test"})
        await running_bus.publish(event)
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        assert len(received_events) == 1
        assert received_events[0].type == "test.event"
        assert received_events[0].payload["data"] == "test"
    
    @pytest.mark.asyncio
    async def test_multiple_handlers_per_event(self, running_bus):
        """Test multiple handlers can subscribe to same event type."""
        calls_1 = []
        calls_2 = []
        
        async def handler_1(event: Event):
            calls_1.append(event)
        
        async def handler_2(event: Event):
            calls_2.append(event)
        
        running_bus.subscribe("multi.event", handler_1)
        running_bus.subscribe("multi.event", handler_2)
        
        event = Event("multi.event", datetime.utcnow(), {})
        await running_bus.publish(event)
        
        await asyncio.sleep(0.1)
        
        assert len(calls_1) == 1
        assert len(calls_2) == 1
    
    @pytest.mark.asyncio
    async def test_event_type_routing(self, running_bus):
        """Test events only go to handlers for their type."""
        type_a_calls = []
        type_b_calls = []
        
        async def handler_a(event: Event):
            type_a_calls.append(event)
        
        async def handler_b(event: Event):
            type_b_calls.append(event)
        
        running_bus.subscribe("type.a", handler_a)
        running_bus.subscribe("type.b", handler_b)
        
        await running_bus.publish(Event("type.a", datetime.utcnow(), {}))
        await running_bus.publish(Event("type.b", datetime.utcnow(), {}))
        
        await asyncio.sleep(0.1)
        
        assert len(type_a_calls) == 1
        assert len(type_b_calls) == 1
        assert type_a_calls[0].type == "type.a"
        assert type_b_calls[0].type == "type.b"
    
    @pytest.mark.asyncio
    async def test_unsubscribe(self, running_bus):
        """Test unsubscribing handlers."""
        calls = []
        
        async def handler(event: Event):
            calls.append(event)
        
        running_bus.subscribe("test.event", handler)
        await running_bus.publish(Event("test.event", datetime.utcnow(), {}))
        await asyncio.sleep(0.1)
        
        assert len(calls) == 1
        
        # Unsubscribe and publish again
        running_bus.unsubscribe("test.event", handler)
        await running_bus.publish(Event("test.event", datetime.utcnow(), {}))
        await asyncio.sleep(0.1)
        
        # Should still be 1 (not 2)
        assert len(calls) == 1


class TestSyncHandlers:
    """Test synchronous handler support."""
    
    @pytest.mark.asyncio
    async def test_sync_handler(self, running_bus):
        """Test sync handlers run in thread pool."""
        calls = []
        
        def sync_handler(event: Event):
            # This is a sync function
            calls.append(event)
        
        running_bus.subscribe("sync.event", sync_handler)
        
        event = Event("sync.event", datetime.utcnow(), {"msg": "sync"})
        await running_bus.publish(event)
        
        await asyncio.sleep(0.2)  # More time for thread pool
        
        assert len(calls) == 1
        assert calls[0].payload["msg"] == "sync"


class TestErrorHandling:
    """Test error handling and resilience."""
    
    @pytest.mark.asyncio
    async def test_handler_exception_does_not_crash_bus(self, running_bus):
        """Test that handler exceptions don't stop the bus."""
        good_calls = []
        
        async def bad_handler(event: Event):
            raise ValueError("Handler error!")
        
        async def good_handler(event: Event):
            good_calls.append(event)
        
        running_bus.subscribe("test.event", bad_handler)
        running_bus.subscribe("test.event", good_handler)
        
        await running_bus.publish(Event("test.event", datetime.utcnow(), {}))
        await asyncio.sleep(0.1)
        
        # Good handler should still have been called
        assert len(good_calls) == 1
        
        # Bus should still work for next event
        await running_bus.publish(Event("test.event", datetime.utcnow(), {}))
        await asyncio.sleep(0.1)
        
        assert len(good_calls) == 2
    
    @pytest.mark.asyncio
    async def test_multiple_events_after_error(self, running_bus):
        """Test bus continues processing after handler error."""
        calls = []
        error_on_first = True
        
        async def sometimes_fails(event: Event):
            nonlocal error_on_first
            if error_on_first:
                error_on_first = False
                raise RuntimeError("First call fails")
            calls.append(event)
        
        running_bus.subscribe("test.event", sometimes_fails)
        
        # First event causes error
        await running_bus.publish(Event("test.event", datetime.utcnow(), {"n": 1}))
        await asyncio.sleep(0.1)
        
        # Second event should work
        await running_bus.publish(Event("test.event", datetime.utcnow(), {"n": 2}))
        await asyncio.sleep(0.1)
        
        assert len(calls) == 1
        assert calls[0].payload["n"] == 2


class TestStatistics:
    """Test statistics tracking."""
    
    @pytest.mark.asyncio
    async def test_published_count(self, running_bus):
        """Test published event counter."""
        async def dummy_handler(event: Event):
            pass
        
        running_bus.subscribe("test", dummy_handler)
        
        await running_bus.publish(Event("test", datetime.utcnow(), {}))
        await running_bus.publish(Event("test", datetime.utcnow(), {}))
        await running_bus.publish(Event("test", datetime.utcnow(), {}))
        
        await asyncio.sleep(0.1)
        
        stats = running_bus.get_stats()
        assert stats["published"] == 3
        assert stats["dispatched"] == 3
    
    @pytest.mark.asyncio
    async def test_error_count(self, running_bus):
        """Test error counter increments on handler failure."""
        async def failing_handler(event: Event):
            raise Exception("Test error")
        
        running_bus.subscribe("test", failing_handler)
        
        await running_bus.publish(Event("test", datetime.utcnow(), {}))
        await running_bus.publish(Event("test", datetime.utcnow(), {}))
        
        await asyncio.sleep(0.1)
        
        stats = running_bus.get_stats()
        assert stats["errors"] >= 2  # At least 2 errors
    
    @pytest.mark.asyncio
    async def test_handler_counts(self, event_bus):
        """Test handler count statistics."""
        async def h1(e): pass
        async def h2(e): pass
        async def h3(e): pass
        
        event_bus.subscribe("type.a", h1)
        event_bus.subscribe("type.a", h2)
        event_bus.subscribe("type.b", h3)
        
        stats = event_bus.get_stats()
        assert stats["handler_types"] == 2
        assert stats["total_handlers"] == 3


class TestRealWorldScenarios:
    """Test realistic integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_msc_ai_to_orchestrator_flow(self, running_bus):
        """Test MSC AI → Orchestrator policy update flow."""
        orchestrator_state = {}
        
        async def orchestrator_handler(event: Event):
            orchestrator_state.update(event.payload)
        
        running_bus.subscribe("policy.updated", orchestrator_handler)
        
        # MSC AI publishes policy update
        event = PolicyUpdatedEvent.create(
            risk_mode=RiskMode.DEFENSIVE,
            allowed_strategies=["strat1"],
            global_min_confidence=0.75,
            max_risk_per_trade=0.01,
            max_positions=3,
        )
        await running_bus.publish(event)
        
        await asyncio.sleep(0.1)
        
        assert orchestrator_state["risk_mode"] == "DEFENSIVE"
        assert orchestrator_state["max_positions"] == 3
    
    @pytest.mark.asyncio
    async def test_health_monitor_to_discord_flow(self, running_bus):
        """Test Health Monitor → Discord alert flow."""
        alerts_sent = []
        
        async def discord_handler(event: Event):
            if event.payload["new_status"] == "CRITICAL":
                alerts_sent.append(event.payload)
        
        running_bus.subscribe("health.status_changed", discord_handler)
        
        # Health monitor publishes critical status
        event = HealthStatusChangedEvent.create(
            old_status=HealthStatus.HEALTHY,
            new_status=HealthStatus.CRITICAL,
            component="DrawdownGuard",
            reason="DD exceeded",
            metrics={"dd": -6.0},
        )
        await running_bus.publish(event)
        
        await asyncio.sleep(0.1)
        
        assert len(alerts_sent) == 1
        assert alerts_sent[0]["component"] == "DrawdownGuard"
    
    @pytest.mark.asyncio
    async def test_high_volume_event_processing(self, running_bus):
        """Test bus handles high volume of events."""
        processed = []
        
        async def counter_handler(event: Event):
            processed.append(event.payload["n"])
        
        running_bus.subscribe("perf.test", counter_handler)
        
        # Publish 100 events
        for i in range(100):
            await running_bus.publish(
                Event("perf.test", datetime.utcnow(), {"n": i})
            )
        
        # Wait for all to process
        await asyncio.sleep(1.0)
        
        assert len(processed) == 100
        assert sorted(processed) == list(range(100))


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.mark.asyncio
    async def test_publish_without_subscribers(self, running_bus):
        """Test publishing event with no subscribers doesn't error."""
        # Should not raise
        await running_bus.publish(
            Event("no.subscribers", datetime.utcnow(), {})
        )
        await asyncio.sleep(0.1)
        
        stats = running_bus.get_stats()
        assert stats["published"] == 1
    
    @pytest.mark.asyncio
    async def test_subscribe_before_bus_starts(self, event_bus):
        """Test subscribing before run_forever is called."""
        calls = []
        
        async def handler(event: Event):
            calls.append(event)
        
        # Subscribe before starting
        event_bus.subscribe("test", handler)
        
        # Start bus
        task = asyncio.create_task(event_bus.run_forever())
        
        await event_bus.publish(Event("test", datetime.utcnow(), {}))
        await asyncio.sleep(0.1)
        
        assert len(calls) == 1
        
        # Cleanup
        event_bus.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
