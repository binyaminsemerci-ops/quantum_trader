# EventBus Integration Guide

## Quick Start

This guide shows how to integrate the EventBus into existing Quantum Trader components.

## Step 1: Initialize EventBus on Startup

Add to your main application startup (e.g., `backend/main.py`):

```python
from backend.services.eventbus import InMemoryEventBus
import asyncio

# Global event bus instance
event_bus = InMemoryEventBus(max_queue_size=10000)

# Start the event loop in background
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(event_bus.run_forever())
    
@app.on_event("shutdown")
async def shutdown_event():
    event_bus.stop()
```

## Step 2: Inject EventBus into Components

Update component constructors to accept the EventBus:

### Example: MSC AI (Publisher)

```python
from backend.services.eventbus import EventBus, PolicyUpdatedEvent, RiskMode

class MetaStrategyController:
    def __init__(
        self,
        event_bus: EventBus,
        policy_store: PolicyStore,
    ):
        self.event_bus = event_bus
        self.policy_store = policy_store
    
    async def update_risk_mode(self, new_mode: RiskMode):
        """Update risk mode and notify the system."""
        # 1. Update storage
        await self.policy_store.set_risk_mode(new_mode)
        
        # 2. Publish event
        event = PolicyUpdatedEvent.create(
            risk_mode=new_mode,
            allowed_strategies=await self.policy_store.get_allowed_strategies(),
            global_min_confidence=0.7 if new_mode == RiskMode.DEFENSIVE else 0.6,
            max_risk_per_trade=0.01 if new_mode == RiskMode.DEFENSIVE else 0.02,
            max_positions=3 if new_mode == RiskMode.DEFENSIVE else 5,
        )
        await self.event_bus.publish(event)
```

### Example: Orchestrator (Subscriber)

```python
from backend.services.eventbus import EventBus, Event

class Orchestrator:
    def __init__(
        self,
        event_bus: EventBus,
        # ... other dependencies
    ):
        self.event_bus = event_bus
        self.current_policy = {}
        
        # Subscribe to policy updates
        event_bus.subscribe("policy.updated", self.on_policy_updated)
        event_bus.subscribe("opportunities.updated", self.on_opportunities_updated)
    
    async def on_policy_updated(self, event: Event):
        """Handle policy update events."""
        logger.info("Policy updated, reloading configuration")
        self.current_policy = event.payload
        
        # Update internal state
        self.global_min_confidence = event.payload["global_min_confidence"]
        self.max_risk_per_trade = event.payload["max_risk_per_trade"]
        # ... etc
    
    async def on_opportunities_updated(self, event: Event):
        """Handle new symbol rankings."""
        self.top_symbols = event.payload["top_symbols"]
        logger.info(f"Updated top symbols: {self.top_symbols}")
```

## Step 3: Wire Components Together

In your application factory or startup:

```python
from backend.services.eventbus import InMemoryEventBus

def create_app():
    # Create shared event bus
    event_bus = InMemoryEventBus()
    
    # Create components with event bus
    msc_ai = MetaStrategyController(event_bus=event_bus, ...)
    orchestrator = Orchestrator(event_bus=event_bus, ...)
    health_monitor = SystemHealthMonitor(event_bus=event_bus, ...)
    discord_notifier = DiscordNotifier(event_bus=event_bus, ...)
    
    # Start event loop
    asyncio.create_task(event_bus.run_forever())
    
    return {
        "event_bus": event_bus,
        "msc_ai": msc_ai,
        "orchestrator": orchestrator,
        # ... etc
    }
```

## Common Patterns

### Pattern 1: MSC AI → Multiple Consumers

**MSC AI publishes:**
```python
await self.event_bus.publish(PolicyUpdatedEvent.create(...))
```

**Multiple subscribers react:**
- Orchestrator updates its policy
- Risk Guard reloads risk limits
- Portfolio Balancer adjusts constraints
- Analytics logs the change

### Pattern 2: Health Monitor → Alert Channels

**Health Monitor publishes:**
```python
event = HealthStatusChangedEvent.create(
    old_status=HealthStatus.HEALTHY,
    new_status=HealthStatus.CRITICAL,
    component="DrawdownGuard",
    reason="Daily drawdown exceeded",
    metrics={"dd": -5.2},
)
await self.event_bus.publish(event)
```

**Multiple notification channels:**
```python
# Discord
class DiscordNotifier:
    def __init__(self, event_bus: EventBus):
        event_bus.subscribe("health.status_changed", self.on_health_changed)
    
    async def on_health_changed(self, event: Event):
        if event.payload["new_status"] == "CRITICAL":
            await self.send_discord_alert(event.payload)

# Email
class EmailNotifier:
    def __init__(self, event_bus: EventBus):
        event_bus.subscribe("health.status_changed", self.on_health_changed)
    
    async def on_health_changed(self, event: Event):
        if event.payload["new_status"] == "CRITICAL":
            await self.send_email_alert(event.payload)
```

### Pattern 3: Executor → Analytics Pipeline

**Executor publishes every trade:**
```python
# After order fills
event = TradeExecutedEvent.create(
    order_id=order.id,
    symbol=order.symbol,
    side=order.side,
    size=order.size,
    price=fill_price,
    strategy_id=signal.strategy_id,
    model=signal.model,
    pnl=pnl if closing else None,
)
await self.event_bus.publish(event)
```

**Analytics components subscribe:**
```python
# Trade logger
class TradeLogger:
    def __init__(self, event_bus: EventBus):
        event_bus.subscribe("trade.executed", self.log_trade)
    
    async def log_trade(self, event: Event):
        await self.db.insert_trade(event.payload)

# Performance tracker
class PerformanceTracker:
    def __init__(self, event_bus: EventBus):
        event_bus.subscribe("trade.executed", self.update_metrics)
    
    async def update_metrics(self, event: Event):
        if event.payload["pnl"] is not None:
            self.total_pnl += event.payload["pnl"]
            self.trade_count += 1
```

## Testing with EventBus

### Unit Testing Individual Components

```python
import pytest
from backend.services.eventbus import InMemoryEventBus

@pytest.mark.asyncio
async def test_orchestrator_reacts_to_policy():
    # Create test bus
    bus = InMemoryEventBus()
    
    # Create component
    orchestrator = Orchestrator(event_bus=bus)
    
    # Start bus
    task = asyncio.create_task(bus.run_forever())
    
    # Publish event
    event = PolicyUpdatedEvent.create(
        risk_mode=RiskMode.DEFENSIVE,
        allowed_strategies=["test"],
        global_min_confidence=0.8,
        max_risk_per_trade=0.01,
        max_positions=2,
    )
    await bus.publish(event)
    
    # Wait for processing
    await asyncio.sleep(0.1)
    
    # Assert
    assert orchestrator.current_policy["risk_mode"] == "DEFENSIVE"
    
    # Cleanup
    bus.stop()
    task.cancel()
```

### Integration Testing

```python
@pytest.mark.asyncio
async def test_full_policy_update_flow():
    bus = InMemoryEventBus()
    
    # Create full system
    msc_ai = MetaStrategyController(event_bus=bus, ...)
    orchestrator = Orchestrator(event_bus=bus, ...)
    risk_guard = RiskGuard(event_bus=bus, ...)
    
    # Start bus
    task = asyncio.create_task(bus.run_forever())
    
    # Trigger policy change
    await msc_ai.update_risk_mode(RiskMode.AGGRESSIVE)
    await asyncio.sleep(0.1)
    
    # Verify all components updated
    assert orchestrator.current_policy["risk_mode"] == "AGGRESSIVE"
    assert risk_guard.max_risk_per_trade == 0.02
    
    # Cleanup
    bus.stop()
    task.cancel()
```

## Event Types Quick Reference

| Event Type | When to Publish | Who Subscribes |
|------------|----------------|----------------|
| `policy.updated` | MSC AI changes global policy | Orchestrator, Risk Guard, Portfolio Balancer |
| `strategy.promoted` | SG AI promotes/demotes strategy | Strategy Runtime Engine, Analytics |
| `model.promoted` | CLM promotes new model version | Ensemble Manager, Analytics |
| `health.status_changed` | Health Monitor detects issue | Discord, Safety Governor, Logger |
| `opportunities.updated` | OppRank refreshes rankings | Strategy Runtime Engine, Executor |
| `trade.executed` | Executor fills an order | Analytics, Performance Tracker, Logger |

## Best Practices

### DO ✅
- Use typed factory methods: `PolicyUpdatedEvent.create(...)`
- Keep handlers fast (<100ms)
- Handle exceptions in your handlers
- Make handlers idempotent
- Use async handlers when possible
- Log important events in handlers

### DON'T ❌
- Don't block in handlers (use `await` or offload to workers)
- Don't create circular event dependencies
- Don't assume event ordering
- Don't publish events too frequently (>1000/sec sustained)
- Don't put business logic in the EventBus

## Monitoring

```python
# Get EventBus statistics
stats = event_bus.get_stats()
logger.info(f"EventBus stats: {stats}")

# {
#     "published": 1234,
#     "dispatched": 1230,
#     "errors": 4,
#     "queue_size": 3,
#     "handler_types": 6,
#     "total_handlers": 12,
# }
```

## Troubleshooting

### Events Not Being Delivered

1. **Is the bus running?**
   ```python
   # Check if run_forever() task is active
   ```

2. **Are handlers subscribed?**
   ```python
   stats = event_bus.get_stats()
   print(f"Handlers: {stats['total_handlers']}")
   ```

3. **Is the queue full?**
   ```python
   stats = event_bus.get_stats()
   print(f"Queue size: {stats['queue_size']}")
   ```

### Handler Errors

Check logs for handler exceptions:
```python
logger.error(f"Handler errors: {event_bus.get_stats()['errors']}")
```

Handlers should catch and log their own exceptions:
```python
async def safe_handler(event: Event):
    try:
        # Do work
        await risky_operation()
    except Exception as e:
        logger.error(f"Handler failed: {e}", exc_info=True)
        # Don't re-raise - let bus continue
```

## Summary

The EventBus enables:
- 🔌 **Decoupled architecture** - no direct dependencies
- 📡 **Reactive components** - automatic event propagation
- 🔍 **System observability** - all events in one place
- 🛡️ **Error resilience** - failures don't cascade
- ⚡ **High performance** - async, non-blocking design

Follow this guide to integrate the EventBus into your Quantum Trader components!
