# 🏗️ Architecture v2 - Domain-Driven Design

## 📂 Complete Directory Structure

```
backend/
├── core/                           # Shared infrastructure (allowed everywhere)
│   ├── __init__.py
│   ├── event_bus.py               # EventBus v2 (Redis Streams)
│   ├── policy_store.py            # PolicyStore v2 (Redis + JSON)
│   ├── logger.py                  # Structured logging with trace_id
│   ├── health.py                  # Health check system
│   └── trace_context.py           # trace_id propagation
│
├── domains/                        # Business domains (NO cross-domain imports)
│   │
│   ├── ai_engine/                 # AI/ML Intelligence Domain
│   │   ├── __init__.py
│   │   ├── orchestrator.py        # AI Orchestrator
│   │   ├── model_supervisor.py    # Model health & bias detection
│   │   ├── meta_strategy.py       # Meta-Strategy Selector
│   │   ├── rl_position_sizing.py  # RL Position Sizing
│   │   └── continuous_learning.py # Continuous Learning Manager
│   │
│   ├── execution/                 # Order Execution Domain
│   │   ├── __init__.py
│   │   ├── event_driven_executor.py  # Main executor
│   │   ├── order_manager.py       # Order lifecycle
│   │   └── slippage_monitor.py    # AELM - Adaptive Execution
│   │
│   ├── risk_safety/               # Risk & Safety Domain
│   │   ├── __init__.py
│   │   ├── global_risk_controller.py
│   │   ├── safety_governor.py
│   │   ├── emergency_stop.py      # ESS
│   │   └── trade_lifecycle_manager.py
│   │
│   ├── portfolio/                 # Portfolio Management Domain
│   │   ├── __init__.py
│   │   ├── position_monitor.py    # Dynamic TP/SL
│   │   ├── portfolio_analyzer.py  # PAL - P&L Analysis
│   │   ├── balance_allocator.py   # PBA - Portfolio Balance
│   │   └── liquidity_filter.py    # Universe selection
│   │
│   ├── learning/                  # Learning & Adaptation Domain
│   │   ├── __init__.py
│   │   ├── retraining_system.py
│   │   ├── drift_detector.py
│   │   ├── shadow_tester.py
│   │   └── strategy_generator.py  # SG AI
│   │
│   └── core_os/                   # Operating System Domain
│       ├── __init__.py
│       ├── hedgefund_os.py        # AI-HFOS coordination
│       ├── regime_detector.py     # PIL - Position Inference
│       ├── self_healing.py
│       └── universe_os.py
│
├── services/                       # Legacy (to be migrated to domains)
│   └── ...                        # Existing services
│
├── models/                         # Shared data models (Pydantic)
│   ├── __init__.py
│   ├── events.py                  # Event schemas
│   ├── policy.py                  # Policy schemas
│   └── trade.py                   # Trade/Position schemas
│
├── config/                         # Configuration
│   ├── __init__.py
│   └── settings.py
│
└── api/                           # FastAPI routes
    ├── __init__.py
    └── routes/
        ├── health.py
        ├── trading.py
        └── admin.py
```

---

## 🚫 Import Rules - STRICTLY ENFORCED

### ✅ ALLOWED Imports

```python
# ALL domains can import from core/
from backend.core.event_bus import EventBus
from backend.core.policy_store import PolicyStore
from backend.core.logger import get_logger
from backend.core.health import HealthChecker
from backend.core.trace_context import trace_context

# ALL domains can import from models/
from backend.models.events import SignalEvent, TradeEvent
from backend.models.policy import RiskMode, PolicyConfig

# Standard library & third-party
import asyncio
import redis
from pydantic import BaseModel
```

### ❌ FORBIDDEN Imports

```python
# NEVER import between domains!
from backend.domains.ai_engine.orchestrator import AIOrchestrator  # ❌ NO!
from backend.domains.execution.executor import Executor            # ❌ NO!
from backend.domains.risk_safety.safety_governor import ...        # ❌ NO!

# WHY? Domains must be decoupled for microservices split
```

---

## 🔗 Inter-Domain Communication - EventBus ONLY

### Pattern: Publish-Subscribe

```python
# Domain A: AI Engine publishes signal
from backend.core.event_bus import EventBus
from backend.core.logger import get_logger

logger = get_logger(__name__)
event_bus = EventBus()

async def generate_signal():
    signal = {
        "symbol": "BTCUSDT",
        "action": "LONG",
        "confidence": 0.85,
        "trace_id": trace_context.get()
    }
    
    await event_bus.publish("ai.signal.generated", signal)
    logger.info("signal_published", symbol="BTCUSDT", confidence=0.85)
```

```python
# Domain B: Risk Safety subscribes to signals
from backend.core.event_bus import EventBus
from backend.core.logger import get_logger

logger = get_logger(__name__)
event_bus = EventBus()

async def handle_signal(event_data: dict):
    trace_id = event_data.get("trace_id")
    
    # Validation logic
    approved = validate_risk(event_data)
    
    if approved:
        await event_bus.publish("risk.signal.approved", event_data)
        logger.info("signal_approved", trace_id=trace_id)
    else:
        await event_bus.publish("risk.signal.rejected", event_data)
        logger.warning("signal_rejected", trace_id=trace_id)

# Register subscriber
event_bus.subscribe("ai.signal.generated", handle_signal)
```

---

## 🎯 Domain Responsibilities

### 1. **ai_engine/** - AI/ML Intelligence
- Generate trading signals
- Supervise model health
- Detect model bias
- Meta-strategy selection
- RL-based position sizing
- Continuous learning coordination

**Events Published:**
- `ai.signal.generated`
- `ai.model.degraded`
- `ai.strategy.changed`

**Events Subscribed:**
- `trade.closed` (for learning)
- `portfolio.performance` (for adaptation)

---

### 2. **execution/** - Order Execution
- Execute approved trades
- Manage order lifecycle
- Monitor slippage
- Handle exchange connectivity
- Retry failed orders

**Events Published:**
- `execution.order.submitted`
- `execution.order.filled`
- `execution.order.failed`
- `execution.slippage.high`

**Events Subscribed:**
- `risk.signal.approved`
- `safety.emergency.stop`

---

### 3. **risk_safety/** - Risk & Safety
- Global risk validation
- Safety governor checks
- Circuit breaker
- Emergency stop system
- Trade lifecycle approval

**Events Published:**
- `risk.signal.approved`
- `risk.signal.rejected`
- `risk.circuit_breaker.activated`
- `safety.emergency.triggered`

**Events Subscribed:**
- `ai.signal.generated`
- `portfolio.drawdown.high`
- `execution.order.failed`

---

### 4. **portfolio/** - Portfolio Management
- Monitor open positions
- Adjust TP/SL dynamically
- Calculate P&L
- Portfolio balancing
- Liquidity filtering

**Events Published:**
- `portfolio.position.opened`
- `portfolio.position.closed`
- `portfolio.sl.adjusted`
- `portfolio.performance` (periodic)

**Events Subscribed:**
- `execution.order.filled`
- `market.price.update`

---

### 5. **learning/** - Learning & Adaptation
- Model retraining
- Drift detection
- Shadow testing
- Strategy generation (genetic algo)
- Performance evaluation

**Events Published:**
- `learning.drift.detected`
- `learning.model.retrained`
- `learning.strategy.promoted`

**Events Subscribed:**
- `trade.closed`
- `ai.model.degraded`
- `portfolio.performance`

---

### 6. **core_os/** - Operating System
- AI-HFOS coordination
- Regime detection
- Self-healing
- Universe management
- System orchestration

**Events Published:**
- `os.regime.changed`
- `os.self_healing.triggered`
- `os.universe.updated`

**Events Subscribed:**
- `ai.signal.generated`
- `portfolio.performance`
- `learning.drift.detected`

---

## 🔄 Lifecycle Example: Signal → Trade → Learning

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. AI Engine Domain                                             │
│    - Orchestrator generates signal                              │
│    - publish("ai.signal.generated", {...})                      │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. Risk Safety Domain                                           │
│    - subscribe("ai.signal.generated")                           │
│    - Safety Governor validates                                  │
│    - Global Risk Controller checks limits                       │
│    - publish("risk.signal.approved", {...})                     │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. Execution Domain                                             │
│    - subscribe("risk.signal.approved")                          │
│    - Event Driven Executor submits order                        │
│    - publish("execution.order.filled", {...})                   │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. Portfolio Domain                                             │
│    - subscribe("execution.order.filled")                        │
│    - Position Monitor tracks position                           │
│    - publish("portfolio.position.opened", {...})                │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. Portfolio Domain (later)                                     │
│    - Position closed by TP/SL                                   │
│    - publish("portfolio.position.closed", {...})                │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. Learning Domain                                              │
│    - subscribe("portfolio.position.closed")                     │
│    - Update model performance                                   │
│    - Trigger retraining if needed                               │
│    - publish("learning.model.retrained", {...})                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛡️ Why This Architecture?

### 1. **Microservices Ready**
Each domain can be extracted into a separate service with ZERO code changes:
- Just change EventBus backend from in-process to network (Kafka/RabbitMQ)
- PolicyStore already uses Redis (external state)
- No direct dependencies between domains

### 2. **Reduced Bugs**
- No circular dependencies
- Clear ownership boundaries
- Easier to test in isolation
- Predictable data flow

### 3. **Race Condition Prevention**
- EventBus handles message ordering
- PolicyStore has atomic reads
- Each domain owns its state
- No shared mutable state between domains

### 4. **Technical Debt Reduction**
- Easy to add new domains without touching existing ones
- Easy to replace/upgrade individual domains
- Clear contracts via events
- Self-documenting architecture

### 5. **Scalability**
- Can run multiple instances of same domain (horizontal scaling)
- Can dedicate more resources to bottleneck domains
- Can deploy domains independently
- Can version domains independently

---

## 📋 Migration Strategy (Legacy → Domains)

1. **Keep existing code running** in `backend/services/`
2. **Create new core/** modules (EventBus, PolicyStore, Logger, Health)
3. **Gradually migrate** services into domains:
   - Start with least-coupled modules
   - Replace direct calls with EventBus
   - Move file to appropriate domain
4. **Deprecate old paths** once migration complete
5. **Remove `backend/services/`** when empty

---

## 🎓 Developer Guidelines

### When to Create New Module in Domain?

✅ **YES** - Create in domain if:
- Implements business logic for that domain
- Needs to publish/subscribe to domain-specific events
- Owns specific state or data

❌ **NO** - Put in `core/` if:
- Used by multiple domains
- Infrastructure concern (logging, events, config)
- No business logic (pure utility)

### How to Add New Event Type?

1. Define schema in `backend/models/events.py`:
```python
class NewEvent(BaseModel):
    event_type: str = "domain.action.verb"
    trace_id: str
    timestamp: datetime
    payload: dict
```

2. Publish from domain:
```python
await event_bus.publish("domain.action.verb", event.dict())
```

3. Subscribe in another domain:
```python
event_bus.subscribe("domain.action.verb", handle_new_event)
```

4. Document in ARCHITECTURE_V2_DOMAINS.md

---

## 🔍 Debugging Cross-Domain Issues

Use `trace_id` to follow request through entire system:

```bash
# Find all logs for specific trace
journalctl -u quantum_backend.service | grep "trace_id=abc-123-def"

# Output shows flow through all domains:
[ai_engine.orchestrator] signal_generated trace_id=abc-123-def
[risk_safety.safety_governor] signal_approved trace_id=abc-123-def
[execution.executor] order_submitted trace_id=abc-123-def
[portfolio.monitor] position_opened trace_id=abc-123-def
```

---

## 📈 Performance Considerations

### EventBus Throughput
- Redis Streams: ~50,000 msgs/sec single instance
- Consumer groups: parallel processing per domain
- Maxlen 10,000: prevents memory bloat

### PolicyStore Latency
- Redis GET: <1ms
- In-memory cache: <0.01ms (future optimization)
- JSON snapshot: async background task (non-blocking)

### Health Checks
- Run every 30 seconds
- Async non-blocking
- Cached for 5 seconds (avoid check spam)

---

## 🚀 Future Extensions

### Phase 2: Multi-Tenancy
- Add `tenant_id` to all events
- PolicyStore per tenant
- Isolated Redis keyspaces

### Phase 3: Multi-Region
- EventBus: Kafka + region-aware routing
- PolicyStore: Redis Cluster with geo-replication
- Health: Regional aggregation

### Phase 4: Full Microservices
- Each domain → separate Docker container
- Kubernetes deployment
- Service mesh (Istio) for observability
- Distributed tracing (Jaeger)

---

*This architecture is the foundation for a world-class algorithmic trading platform.*

