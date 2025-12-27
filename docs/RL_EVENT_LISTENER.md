# RL Event Listener - Architecture Documentation

**Event-Driven Trading Flow v1**  
**Module**: `backend/events/subscribers/rl_subscriber.py`  
**Version**: 1.0  
**Date**: January 2024

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Event Flow Diagram](#event-flow-diagram)
3. [Reward Formulas](#reward-formulas)
4. [PolicyStore Integration](#policystore-integration)
5. [trace_id Propagation](#trace_id-propagation)
6. [EventFlow v1 Integration](#eventflow-v1-integration)
7. [Usage Examples](#usage-examples)
8. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

The **RL Event Listener** is a production-ready module that bridges Event-Driven Trading Flow v1 with Reinforcement Learning agents. It listens to trading events and updates RL agents with states, actions, and rewards in real-time.

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    RL Event Listener                        │
│                  (Event Subscriber)                         │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│signal.       │    │trade.        │    │position.     │
│generated     │    │executed      │    │closed        │
└──────────────┘    └──────────────┘    └──────────────┘
        │                   │                   │
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Meta         │    │ Position     │    │ Both Agents: │
│ Strategy     │    │ Sizing       │    │ Reward       │
│ Agent:       │    │ Agent:       │    │ Update       │
│ set_current  │    │ set_executed │    │              │
│ _state()     │    │ _action()    │    │              │
└──────────────┘    └──────────────┘    └──────────────┘
```

### Key Features

- **Event-Driven Architecture**: Listens to 3 event types via EventBus v2
- **RL Integration**: Updates Meta Strategy Agent and Position Sizing Agent
- **PolicyStore Control**: Respects `enable_rl` flag for dynamic control
- **Trace ID Propagation**: Full traceability from signal to reward
- **Error Handling**: Publishes error events on failures
- **Async/Await**: Non-blocking, production-ready implementation

---

## Event Flow Diagram

### Complete Trading Cycle with RL Integration

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRADING CYCLE                                │
└─────────────────────────────────────────────────────────────────────┘

1. SIGNAL GENERATION
   ┌──────────────────────────────────────┐
   │ AI Model generates trading signal    │
   │ • Symbol: BTCUSDT                    │
   │ • Signal: BUY                        │
   │ • Confidence: 0.85                   │
   │ • Timeframe: 1h                      │
   └──────────────────────────────────────┘
                    │
                    │ EventBus.publish("signal.generated")
                    ▼
   ┌──────────────────────────────────────┐
   │ RL Event Listener receives event     │
   │ • Extracts state features            │
   │ • Checks PolicyStore enable_rl flag  │
   └──────────────────────────────────────┘
                    │
                    │ meta_agent.set_current_state(trace_id, state)
                    ▼
   ┌──────────────────────────────────────┐
   │ Meta Strategy Agent stores state     │
   │ • state = {symbol, confidence, tf}   │
   │ • Indexed by trace_id                │
   └──────────────────────────────────────┘

2. TRADE EXECUTION
   ┌──────────────────────────────────────┐
   │ Trade executed on exchange           │
   │ • Leverage: 10x                      │
   │ • Size: 500 USD                      │
   │ • Entry Price: 43,000                │
   └──────────────────────────────────────┘
                    │
                    │ EventBus.publish("trade.executed")
                    ▼
   ┌──────────────────────────────────────┐
   │ RL Event Listener receives event     │
   │ • Extracts action parameters         │
   └──────────────────────────────────────┘
                    │
                    │ size_agent.set_executed_action(trace_id, action)
                    ▼
   ┌──────────────────────────────────────┐
   │ Position Sizing Agent stores action  │
   │ • action = {leverage, size_usd}      │
   │ • Indexed by trace_id                │
   └──────────────────────────────────────┘

3. POSITION CLOSED (REWARD CALCULATION)
   ┌──────────────────────────────────────┐
   │ Position closed with P&L             │
   │ • P&L: +150 USD (+3.5%)              │
   │ • Max Drawdown: 1.2%                 │
   │ • Duration: 25 minutes               │
   └──────────────────────────────────────┘
                    │
                    │ EventBus.publish("position.closed")
                    ▼
   ┌──────────────────────────────────────┐
   │ RL Event Listener calculates rewards │
   │ • meta_reward = pnl% - dd% * 0.5     │
   │ • size_reward = pnl%                 │
   └──────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌────────────────┐      ┌────────────────┐
│ Meta Agent:    │      │ Size Agent:    │
│ update(        │      │ update(        │
│   trace_id,    │      │   trace_id,    │
│   reward=2.9   │      │   reward=3.5   │
│ )              │      │ )              │
└────────────────┘      └────────────────┘
```

---

## Reward Formulas

### Meta Strategy Agent Reward

The Meta Strategy Agent is responsible for selecting the right trading strategy (when to trade, which signal to follow). Its reward penalizes drawdown to encourage risk-aware strategy selection.

**Formula**:
```
meta_reward = pnl_pct - (max_drawdown_pct * 0.5)
```

**Rationale**:
- **pnl_pct**: Positive reward for profitable trades
- **max_drawdown_pct * 0.5**: Penalty for drawdown (reduces reward by half the drawdown)
- **Goal**: Learn to select signals with high profit AND low drawdown

**Examples**:

| P&L % | Max DD % | meta_reward | Interpretation |
|-------|----------|-------------|----------------|
| +5.0  | 1.0      | 4.5         | Excellent (high profit, low drawdown) |
| +3.5  | 1.2      | 2.9         | Good (moderate profit, low drawdown) |
| +8.0  | 5.0      | 5.5         | Risky (high profit, high drawdown) |
| -2.0  | 3.5      | -3.75       | Bad (loss + high drawdown) |

---

### Position Sizing Agent Reward

The Position Sizing Agent determines leverage and position size. Its reward is purely based on P&L percentage to learn optimal sizing.

**Formula**:
```
size_reward = pnl_pct
```

**Rationale**:
- **Simple P&L-based reward**: Encourages sizing that maximizes returns
- **No drawdown penalty**: Size agent focuses on position sizing, not strategy selection
- **Direct feedback**: Clear signal on whether size was too small or too large

**Examples**:

| P&L % | size_reward | Interpretation |
|-------|-------------|----------------|
| +5.0  | 5.0         | Excellent sizing |
| +3.5  | 3.5         | Good sizing |
| +0.5  | 0.5         | Size too small |
| -2.0  | -2.0        | Size or leverage too high |

---

## PolicyStore Integration

### enable_rl Flag

The RL Event Listener respects the `enable_rl` flag in PolicyStore v2, allowing dynamic control of RL behavior without code changes.

**Flow**:
```python
# In RLEventListener._check_rl_enabled()

profile = await policy_store.get_active_profile()
if profile and profile.get("enable_rl", False):
    return True  # RL updates enabled
else:
    return False  # RL updates disabled (skip processing)
```

### Use Cases

1. **Development Mode**: Disable RL during testing
   ```python
   policy_store.update_profile({
       "enable_rl": False,
       "profile_name": "test_mode"
   })
   ```

2. **Production Gradual Rollout**: Enable RL for specific profiles
   ```python
   policy_store.update_profile({
       "enable_rl": True,
       "profile_name": "aggressive_rl"
   })
   ```

3. **Emergency Shutdown**: Disable RL immediately without restart
   ```python
   policy_store.update_profile({"enable_rl": False})
   # RL Event Listener will skip all events
   ```

---

## trace_id Propagation

### Complete Trace Flow

The `trace_id` is the key that links signal → trade → position → reward across the entire system.

```
┌─────────────────────────────────────────────────────────────────┐
│                    trace_id: "abc-123"                          │
└─────────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Signal Event │    │ Trade Event  │    │ Position     │
│ trace_id:    │    │ trace_id:    │    │ Event        │
│ "abc-123"    │    │ "abc-123"    │    │ trace_id:    │
│              │    │              │    │ "abc-123"    │
└──────────────┘    └──────────────┘    └──────────────┘
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ meta_agent.  │    │ size_agent.  │    │ Both agents: │
│ states[      │    │ actions[     │    │ experiences[ │
│ "abc-123"    │    │ "abc-123"    │    │ "abc-123"    │
│ ] = state    │    │ ] = action   │    │ ] += reward  │
└──────────────┘    └──────────────┘    └──────────────┘
```

### trace_id Generation

Generated by the signal generation module:
```python
import uuid
trace_id = str(uuid.uuid4())
```

### trace_id Usage in RL Event Listener

```python
# 1. Signal → State
await meta_agent.set_current_state(trace_id, state)

# 2. Trade → Action
await size_agent.set_executed_action(trace_id, action)

# 3. Position → Reward
await meta_agent.update(trace_id, reward=meta_reward)
await size_agent.update(trace_id, reward=size_reward)
```

---

## EventFlow v1 Integration

### Integration Points in main.py

#### 1. Startup (After EventBus v2 Initialization)

```python
# File: backend/main.py
# Lines: 375-398

from backend.events.subscribers.rl_subscriber import RLEventListener

# Get RL agents from app state
meta_agent = getattr(app_instance.state, 'rl_meta_strategy_agent', None)
size_agent = getattr(app_instance.state, 'rl_position_sizing_agent', None)

# Create RL Event Listener
rl_listener = RLEventListener(
    event_bus=event_bus_v2,
    policy_store=policy_store_v2,
    meta_strategy_agent=meta_agent,
    position_sizing_agent=size_agent
)

# Start listening to events
await rl_listener.start()

# Store in app state for cleanup
app_instance.state.rl_event_listener = rl_listener

logger.info(
    "[v2] RL Event Listener started",
    meta_agent_available=meta_agent is not None,
    size_agent_available=size_agent is not None
)
```

#### 2. Shutdown (Before EventBus v2 Shutdown)

```python
# File: backend/main.py
# Lines: ~1748-1757

if hasattr(app_instance.state, 'rl_event_listener') and app_instance.state.rl_event_listener:
    try:
        logger.info("[v2] Stopping RL Event Listener...")
        await app_instance.state.rl_event_listener.stop()
        logger.info("[v2] RL Event Listener stopped")
    except Exception as e:
        logger.error(f"[ERROR] RL Event Listener shutdown failed: {e}")
```

### Dependencies

```python
# EventBus v2 (Redis Streams)
from backend.events.event_bus_v2 import EventBusV2

# PolicyStore v2 (Risk Profiles)
from backend.core.policy_store_v2 import PolicyStoreV2

# RL Agents
from backend.services.rl_meta_strategy_agent import RLMetaStrategyAgent
from backend.services.rl_position_sizing_agent import RLPositionSizingAgent

# Event Schemas
from backend.events.schemas import (
    SignalGeneratedEvent,
    TradeExecutedEvent,
    PositionClosedEvent
)

# Logger v2 (structlog)
from backend.core.logger_v2 import get_logger
```

---

## Usage Examples

### Example 1: Manual Event Publishing (Testing)

```python
import asyncio
from backend.events.event_bus_v2 import EventBusV2

async def test_signal_event():
    # Initialize EventBus
    event_bus = EventBusV2(redis_url="redis://localhost:6379")
    await event_bus.start()
    
    # Publish signal.generated event
    await event_bus.publish({
        "event_type": "signal.generated",
        "trace_id": "test-signal-001",
        "timestamp": "2024-01-15T10:00:00Z",
        "data": {
            "symbol": "BTCUSDT",
            "signal": "BUY",
            "confidence": 0.85,
            "timeframe": "1h"
        }
    })
    
    # RL Event Listener will automatically process this event
    print("Signal event published!")
    
    await event_bus.stop()

asyncio.run(test_signal_event())
```

### Example 2: Checking RL Agent State

```python
# Access RL agents from app state
from backend.main import app

# Get Meta Strategy Agent
meta_agent = app.state.rl_meta_strategy_agent

# Check if state was stored for a trace_id
trace_id = "test-signal-001"
if hasattr(meta_agent, 'current_states') and trace_id in meta_agent.current_states:
    state = meta_agent.current_states[trace_id]
    print(f"State for {trace_id}: {state}")
else:
    print(f"No state found for {trace_id}")
```

### Example 3: Disabling RL Dynamically

```python
# Access PolicyStore from app state
from backend.main import app

policy_store = app.state.policy_store_v2

# Disable RL for all events
await policy_store.update_profile({
    "enable_rl": False,
    "profile_name": "emergency_disable"
})

# All subsequent events will be ignored by RL Event Listener
```

### Example 4: End-to-End Test via HTTP

```bash
# 1. Publish signal event
curl -X POST http://localhost:8000/testevents/generate_signal \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "ETHUSDT",
    "signal": "BUY",
    "confidence": 0.80,
    "timeframe": "5m"
  }'

# Response: {"trace_id": "abc-123", "message": "Signal published"}

# 2. Check event flow status
curl http://localhost:8000/testevents/event_flow_status

# Response includes:
# {
#   "event_flow_enabled": true,
#   "event_types": [...],
#   "subscribers": {
#     "signal.generated": ["rl_listener", ...]
#   }
# }
```

---

## Troubleshooting

### Problem: RL Event Listener not processing events

**Symptoms**:
- Events published but no RL agent updates
- No logs from RL Event Listener

**Solutions**:

1. **Check enable_rl flag**:
   ```python
   profile = await policy_store.get_active_profile()
   print(f"enable_rl: {profile.get('enable_rl', False)}")
   ```

2. **Verify RL Event Listener is started**:
   ```python
   from backend.main import app
   print(hasattr(app.state, 'rl_event_listener'))
   ```

3. **Check EventBus subscriptions**:
   ```bash
   # Look for logs:
   # [v2] RL Event Listener subscribed to signal.generated
   # [v2] RL Event Listener subscribed to trade.executed
   # [v2] RL Event Listener subscribed to position.closed
   ```

---

### Problem: Agents not receiving updates

**Symptoms**:
- RL Event Listener processes events
- But agent.current_states or agent.executed_actions is empty

**Solutions**:

1. **Check if agents are initialized**:
   ```python
   from backend.main import app
   print(f"Meta agent: {app.state.rl_meta_strategy_agent}")
   print(f"Size agent: {app.state.rl_position_sizing_agent}")
   ```

2. **Verify trace_id consistency**:
   ```python
   # All events in a cycle must have the SAME trace_id
   # Check event data:
   print(f"Signal trace_id: {signal_event['trace_id']}")
   print(f"Trade trace_id: {trade_event['trace_id']}")
   print(f"Position trace_id: {position_event['trace_id']}")
   ```

3. **Enable debug logging**:
   ```python
   import logging
   logging.getLogger("backend.events.subscribers.rl_subscriber").setLevel(logging.DEBUG)
   ```

---

### Problem: Rewards not calculated correctly

**Symptoms**:
- Agents receive updates but reward values seem wrong

**Solutions**:

1. **Verify position.closed event data**:
   ```python
   # Required fields:
   # - pnl_pct (float)
   # - max_drawdown_pct (float)
   
   event_data = {
       "pnl_pct": 3.5,  # Must be present
       "max_drawdown_pct": 1.2  # Must be present
   }
   ```

2. **Check reward formula**:
   ```python
   # Meta Agent:
   meta_reward = pnl_pct - (max_drawdown_pct * 0.5)
   # Example: 3.5 - (1.2 * 0.5) = 2.9
   
   # Size Agent:
   size_reward = pnl_pct
   # Example: 3.5
   ```

3. **Look for error events**:
   ```python
   # Check for system.event_error events:
   # {"event_type": "system.event_error", "data": {"error_type": "RL_POSITION_HANDLER_ERROR"}}
   ```

---

### Problem: Redis connection issues

**Symptoms**:
- EventBus fails to start
- "Redis connection error" in logs

**Solutions**:

1. **Verify Redis is running**:
   ```bash
   docker ps | grep redis
   # Should show quantum_redis container
   ```

2. **Check Redis connectivity**:
   ```bash
   redis-cli -h localhost -p 6379 ping
   # Should return: PONG
   ```

3. **Restart Redis container**:
   ```bash
   docker stop quantum_redis
   docker rm quantum_redis
   docker run -d --name quantum_redis -p 6379:6379 redis:latest
   ```

---

## Performance Considerations

### Event Processing Latency

- **Target**: < 10ms per event
- **Typical**: 2-5ms per event
- **Factors**:
  * PolicyStore query: ~1ms
  * Agent update: ~1-2ms
  * Redis publish (errors): ~1ms

### Memory Usage

- **Per Event**: ~500 bytes (event data)
- **Per Agent**: ~1KB per stored state/action
- **Total**: Scales with number of active trace_ids

### Scalability

- **Events/Second**: 1000+ (Redis Streams capacity)
- **Concurrent Listeners**: Multiple RL Event Listeners can run (consumer groups)
- **Horizontal Scaling**: Deploy multiple backend instances with same consumer group

---

## Future Extensions

### Planned Features

1. **Multi-Agent Support**: Extend to more RL agents (e.g., risk management, entry timing)
2. **Reward Shaping**: Configurable reward formulas via PolicyStore
3. **Event Replay**: Replay historical events for offline training
4. **Performance Metrics**: Track RL agent performance over time
5. **A/B Testing**: Compare RL-enabled vs. manual policies

### Integration Points

- **Monitoring**: Add Prometheus metrics for event processing
- **Alerting**: Send alerts when RL agents fail to update
- **Visualization**: Dashboard showing RL agent states and rewards

---

## References

- **EventBus v2 Documentation**: `docs/EVENTBUS_V2.md`
- **PolicyStore v2 Guide**: `docs/POLICY_STORE_V2.md`
- **RL Meta Strategy Agent**: `backend/services/rl_meta_strategy_agent.py`
- **RL Position Sizing Agent**: `backend/services/rl_position_sizing_agent.py`
- **Event Schemas**: `backend/events/schemas.py`

---

**Document Version**: 1.0  
**Last Updated**: January 2024  
**Maintained By**: Quantum Trader Development Team
