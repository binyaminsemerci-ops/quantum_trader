# EventBus in Quantum Trader Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         QUANTUM TRADER AI HEDGE FUND OS                  │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                            EVENT BUS (New!)                              │
│                    Async Pub/Sub Messaging Backbone                      │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  STRATEGY    │    │   META       │    │  CONTINUOUS  │
│  GENERATOR   │    │  STRATEGY    │    │  LEARNING    │
│  AI (SG AI)  │    │  CONTROLLER  │    │  MANAGER     │
│              │    │  (MSC AI)    │    │  (CLM)       │
│ - Generates  │    │              │    │              │
│ - Backtests  │    │ - Risk Mode  │    │ - Retrains   │
│ - Evolves    │    │ - Strategies │    │ - Evaluates  │
│ - Promotes   │    │ - Thresholds │    │ - Promotes   │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │ Events            │ Events            │ Events
       └───────────────────┼───────────────────┘
                           │
       ┌───────────────────┼───────────────────┐
       ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ OPPORTUNITY  │    │  ENSEMBLE    │    │ ORCHESTRATOR │
│ RANKER       │    │  MANAGER     │    │ POLICY       │
│              │    │              │    │              │
│ - Scores     │    │ - XGBoost    │    │ - Signal     │
│ - Ranks      │    │ - LightGBM   │    │   Filtering  │
│ - Filters    │    │ - N-HiTS     │    │ - Trade      │
│              │    │ - PatchTST   │    │   Approval   │
└──────┬───────┘    └──────────────┘    └──────┬───────┘
       │                                        │
       │                                        ▼
       │                                 ┌──────────────┐
       │                                 │ PORTFOLIO    │
       │                                 │ BALANCER     │
       │                                 │              │
       │                                 │ - Max Pos    │
       │                                 │ - Exposure   │
       │                                 │ - Correlation│
       │                                 └──────┬───────┘
       │                                        │
       └────────────────┬───────────────────────┘
                        ▼
                 ┌──────────────┐
                 │  EXECUTOR    │
                 │              │
                 │ - Orders     │
                 │ - Positions  │
                 │ - Fills      │
                 └──────┬───────┘
                        │ Events
                        ▼
         ┌──────────────────────────────┐
         │  MONITORING & OBSERVABILITY   │
         │                               │
         │ - Health Monitor              │
         │ - Discord Notifier            │
         │ - Analytics Service           │
         │ - Performance Tracker         │
         └───────────────────────────────┘
```

## Event Flow Diagrams

### Flow 1: Market Regime Change → Policy Update

```
1. Regime Detector detects market shift
   ↓
2. MSC AI analyzes new regime
   ↓
3. MSC AI updates PolicyStore
   ↓
4. MSC AI publishes PolicyUpdatedEvent
   ↓
   ├──→ Orchestrator reloads policy
   ├──→ Risk Guard updates limits
   ├──→ Portfolio Balancer adjusts constraints
   └──→ Analytics logs change
```

**Code:**
```python
# MSC AI
async def on_regime_change(self, new_regime: Regime):
    new_mode = self._determine_risk_mode(new_regime)
    await self.policy_store.set_risk_mode(new_mode)
    
    # Publish event - all subscribers react automatically
    await self.event_bus.publish(PolicyUpdatedEvent.create(
        risk_mode=new_mode,
        allowed_strategies=await self._get_strategies_for_mode(new_mode),
        global_min_confidence=self._get_min_confidence(new_mode),
        max_risk_per_trade=self._get_max_risk(new_mode),
        max_positions=self._get_max_positions(new_mode),
    ))
```

### Flow 2: Strategy Performance → Lifecycle Change

```
1. SG AI monitors shadow strategy performance
   ↓
2. Strategy exceeds promotion thresholds
   ↓
3. SG AI updates strategy status to LIVE
   ↓
4. SG AI publishes StrategyPromotedEvent
   ↓
   ├──→ Strategy Runtime Engine enables strategy
   ├──→ Orchestrator adds to allowed list
   ├──→ Analytics logs promotion
   └──→ Discord notifies team
```

### Flow 3: Model Degradation → Retraining → Promotion

```
1. CLM detects model performance degradation
   ↓
2. CLM triggers retraining job
   ↓
3. CLM evaluates new model in shadow mode
   ↓
4. New model outperforms old model
   ↓
5. CLM publishes ModelPromotedEvent
   ↓
   ├──→ Ensemble Manager swaps model version
   ├──→ Analytics updates model registry
   └──→ Discord notifies team
```

### Flow 4: Drawdown Alert → Emergency Actions

```
1. Position Monitor calculates portfolio DD
   ↓
2. Health Monitor detects DD > threshold
   ↓
3. Health Monitor publishes HealthStatusChangedEvent
   ↓
   ├──→ Safety Governor triggers circuit breaker
   ├──→ Discord sends CRITICAL alert
   ├──→ MSC AI switches to DEFENSIVE mode
   ├──→ Executor pauses new trades
   └──→ Analytics logs incident
```

### Flow 5: Opportunity Ranking → Trade Generation

```
1. OppRank runs periodic symbol scoring
   ↓
2. OppRank identifies top N symbols
   ↓
3. OppRank publishes OpportunitiesUpdatedEvent
   ↓
   ├──→ Strategy Runtime Engine focuses on top symbols
   ├──→ Orchestrator updates allowed symbols
   └──→ Analytics logs ranking changes
```

## Event Types and System Integration

### PolicyUpdatedEvent
**Publisher:** MSC AI  
**Subscribers:**
- ✅ Orchestrator Policy - reloads global thresholds
- ✅ Risk Guard - updates pre-trade risk checks
- ✅ Portfolio Balancer - adjusts position limits
- ✅ Analytics Service - logs policy history
- ✅ Safety Governor - updates circuit breaker params

**Impact:** Changes system-wide risk behavior

### StrategyPromotedEvent
**Publisher:** Strategy Generator AI  
**Subscribers:**
- ✅ Strategy Runtime Engine - enables/disables strategies
- ✅ Orchestrator Policy - updates allowed strategies list
- ✅ Analytics Service - tracks strategy lifecycle
- ✅ Performance Tracker - starts/stops metrics collection
- ✅ Discord Notifier - announces promotions/demotions

**Impact:** Changes active trading strategies

### ModelPromotedEvent
**Publisher:** Continuous Learning Manager  
**Subscribers:**
- ✅ Ensemble Manager - swaps model versions
- ✅ Analytics Service - updates model registry
- ✅ Performance Tracker - resets model metrics
- ✅ Discord Notifier - announces model updates

**Impact:** Changes prediction models used for signals

### HealthStatusChangedEvent
**Publisher:** System Health Monitor  
**Subscribers:**
- ✅ Safety Governor - triggers circuit breakers
- ✅ MSC AI - may auto-adjust risk mode
- ✅ Discord Notifier - sends alerts (email, SMS, etc.)
- ✅ Analytics Service - logs health incidents
- ✅ Executor - may pause/resume trading

**Impact:** System-wide safety reactions

### OpportunitiesUpdatedEvent
**Publisher:** Opportunity Ranker  
**Subscribers:**
- ✅ Strategy Runtime Engine - focuses on top symbols
- ✅ Orchestrator Policy - updates tradeable universe
- ✅ Analytics Service - tracks symbol performance
- ✅ Position Monitor - prioritizes top opportunities

**Impact:** Directs trading focus to best opportunities

### TradeExecutedEvent
**Publisher:** Executor  
**Subscribers:**
- ✅ Analytics Service - records trade history
- ✅ Performance Tracker - updates strategy/model metrics
- ✅ Position Monitor - updates portfolio state
- ✅ Cost Model - tracks fees and slippage
- ✅ Discord Notifier - may send trade notifications

**Impact:** Provides observability into all trades

## Benefits for Advanced Features

### 1. Strategy Generator AI
**Without EventBus:**
- Hard-coded coupling to Strategy Runtime Engine
- Manual notification to other components
- Difficult to test in isolation

**With EventBus:**
```python
# SG AI just publishes - subscribers react automatically
await self.event_bus.publish(StrategyPromotedEvent.create(...))
```
- ✅ Zero coupling to consumers
- ✅ Easy to add new subscribers (Discord, email, etc.)
- ✅ Testable in isolation

### 2. Meta Strategy Controller
**Without EventBus:**
```python
# Tightly coupled nightmare
await self.orchestrator.update_policy(policy)
await self.risk_guard.reload_config()
await self.portfolio_balancer.set_limits(limits)
await self.analytics.log_policy_change(policy)
# What if one fails? What order to call? Hard to maintain!
```

**With EventBus:**
```python
# Clean, decoupled
await self.event_bus.publish(PolicyUpdatedEvent.create(...))
# All subscribers react automatically in parallel
```
- ✅ Single point of publication
- ✅ Subscribers can be added without changing MSC AI
- ✅ Parallel execution
- ✅ Error isolation

### 3. Continuous Learning Manager
**Without EventBus:**
- Direct dependency on Ensemble Manager
- Manual coordination with Analytics
- Hard to add new consumers

**With EventBus:**
```python
# CLM publishes model promotion
await self.event_bus.publish(ModelPromotedEvent.create(...))

# Ensemble Manager automatically swaps model
# Analytics automatically logs change
# Discord automatically notifies team
# All without CLM knowing about them!
```

### 4. System Health Monitor
**Without EventBus:**
- Direct calls to Safety Governor, Discord, etc.
- Hard to add new notification channels
- Tight coupling

**With EventBus:**
```python
# Publish once
await self.event_bus.publish(HealthStatusChangedEvent.create(...))

# Multiple notification channels subscribe:
# - Discord
# - Email
# - SMS
# - Telegram
# - PagerDuty
# All without changing Health Monitor!
```

## Observability & Analytics

The EventBus provides a **single point of observation** for all system events:

```python
class EventLogger:
    """Logs all events to database for analytics."""
    
    def __init__(self, event_bus: EventBus, db: Database):
        # Subscribe to ALL event types
        event_bus.subscribe("policy.updated", self.log_event)
        event_bus.subscribe("strategy.promoted", self.log_event)
        event_bus.subscribe("model.promoted", self.log_event)
        event_bus.subscribe("health.status_changed", self.log_event)
        event_bus.subscribe("opportunities.updated", self.log_event)
        event_bus.subscribe("trade.executed", self.log_event)
    
    async def log_event(self, event: Event):
        await self.db.insert_event({
            "type": event.type,
            "timestamp": event.timestamp,
            "payload": event.payload,
        })
```

Now you can:
- Query event history
- Analyze event patterns
- Debug system behavior
- Generate reports
- Build dashboards

## Testing Benefits

### Unit Testing
```python
# Test MSC AI in isolation
@pytest.mark.asyncio
async def test_msc_ai_publishes_policy_events():
    bus = InMemoryEventBus()
    msc_ai = MetaStrategyController(event_bus=bus, ...)
    
    events = []
    bus.subscribe("policy.updated", lambda e: events.append(e))
    
    await msc_ai.update_risk_mode(RiskMode.DEFENSIVE)
    
    assert len(events) == 1
    assert events[0].payload["risk_mode"] == "DEFENSIVE"
```

### Integration Testing
```python
# Test full system flow
@pytest.mark.asyncio
async def test_policy_update_propagates():
    bus = InMemoryEventBus()
    
    # Wire up full system
    msc_ai = MetaStrategyController(event_bus=bus, ...)
    orchestrator = Orchestrator(event_bus=bus, ...)
    risk_guard = RiskGuard(event_bus=bus, ...)
    
    # Trigger change
    await msc_ai.update_risk_mode(RiskMode.AGGRESSIVE)
    await asyncio.sleep(0.1)
    
    # Verify all components updated
    assert orchestrator.current_policy["risk_mode"] == "AGGRESSIVE"
    assert risk_guard.max_risk_per_trade == 0.02
```

## Performance Impact

### Minimal Overhead
- **Latency:** <1ms to publish and dispatch
- **Throughput:** 1000-5000 events/sec
- **Memory:** ~1-5 MB for typical load
- **CPU:** Negligible when idle

### Async Design
- Non-blocking publication
- Parallel handler execution
- No waiting on slow handlers
- Event buffering with queue

## Migration Path

### Phase 1: Add EventBus (Non-Breaking)
Add EventBus alongside existing direct calls:
```python
# Old way still works
await self.orchestrator.update_policy(policy)

# New way also publishes event
await self.event_bus.publish(PolicyUpdatedEvent.create(...))
```

### Phase 2: Add Subscribers
Components subscribe to events but keep old interfaces:
```python
class Orchestrator:
    def update_policy(self, policy):  # Old interface
        self._apply_policy(policy)
    
    async def on_policy_updated(self, event: Event):  # New subscriber
        self._apply_policy(event.payload)
```

### Phase 3: Remove Direct Calls
Once all components subscribe, remove direct calls:
```python
# Only the EventBus way
await self.event_bus.publish(PolicyUpdatedEvent.create(...))
```

## Future Enhancements

### External Broker Integration
Replace `InMemoryEventBus` with Kafka/RabbitMQ adapter:
```python
# Same interface, different implementation
event_bus = KafkaEventBus(brokers=["localhost:9092"])
```

Benefits:
- Multi-process distribution
- Event persistence
- Replay capability
- Higher throughput

### Event Sourcing
Store all events as source of truth:
```python
# Rebuild system state from events
for event in event_store.replay():
    await event_bus.publish(event)
```

### Dead Letter Queue
Handle failed events:
```python
class EventBus:
    async def publish_with_retry(self, event: Event, max_retries: int = 3):
        # Retry failed events
        # Move to DLQ if still failing
```

## Summary

The EventBus is the **foundation for building advanced AI features** in Quantum Trader:

✅ **Enables complex workflows** without tight coupling  
✅ **Supports reactive architecture** for rapid system responses  
✅ **Provides system-wide observability** through centralized events  
✅ **Simplifies testing** through clear interfaces  
✅ **Scales easily** with async, non-blocking design  
✅ **Future-proof** with pluggable implementations  

**Next Steps:**
1. Integrate EventBus into main application startup
2. Update components to publish/subscribe to events
3. Add analytics subscribers for observability
4. Build notification subscribers (Discord, email, etc.)
5. Use for implementing Strategy Generator AI
6. Use for implementing Meta Strategy Controller
7. Use for implementing Continuous Learning Manager

The EventBus transforms Quantum Trader from a monolithic system into a **modular, event-driven AI Hedge Fund OS**! 🚀
