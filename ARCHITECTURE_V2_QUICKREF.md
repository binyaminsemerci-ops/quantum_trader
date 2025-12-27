# 🚀 Architecture v2 - Quick Reference

## 📋 File Structure

```
backend/
├── core/                          # ✅ Import from anywhere
│   ├── event_bus.py              # Redis Streams event system
│   ├── policy_store.py           # Risk mode configuration
│   ├── logger.py                 # Structured logging
│   ├── health.py                 # Health checks
│   └── trace_context.py          # Distributed tracing
│
├── models/                        # ✅ Import from anywhere
│   ├── events.py                 # Event schemas
│   └── policy.py                 # Policy schemas
│
└── domains/                       # ❌ NO cross-domain imports!
    ├── ai_engine/                # AI/ML domain
    ├── execution/                # Order execution
    ├── risk_safety/              # Risk management
    ├── portfolio/                # Position monitoring
    ├── learning/                 # Model training
    └── core_os/                  # System orchestration
```

---

## ⚡ Quick Start

### Import Pattern (Use Everywhere)

```python
# ✅ ALLOWED: Import from core
from backend.core import (
    get_event_bus,
    get_policy_store,
    get_logger,
    trace_context,
)

# ✅ ALLOWED: Import from models
from backend.models import SignalEvent, RiskMode

# ❌ FORBIDDEN: Import between domains
from backend.domains.ai_engine.orchestrator import ...  # NO!
```

---

## 📡 EventBus Cheat Sheet

### Publish Event

```python
from backend.core import get_event_bus, trace_context

event_bus = get_event_bus()
trace_id = trace_context.generate()

await event_bus.publish(
    "ai.signal.generated",
    {
        "symbol": "BTCUSDT",
        "confidence": 0.85,
        "action": "LONG",
    },
    trace_id=trace_id,
)
```

### Subscribe to Event

```python
from backend.core import get_event_bus

event_bus = get_event_bus()

async def handle_signal(event_data: dict):
    symbol = event_data["symbol"]
    print(f"Received signal for {symbol}")

event_bus.subscribe("ai.signal.generated", handle_signal)
```

### Common Event Types

| Event Type | Published By | Consumed By |
|------------|--------------|-------------|
| `ai.signal.generated` | ai_engine | risk_safety |
| `risk.signal.approved` | risk_safety | execution |
| `execution.order.filled` | execution | portfolio |
| `portfolio.position.closed` | portfolio | learning |
| `policy.mode.changed` | policy_store | ALL domains |

---

## 🎛️ PolicyStore Cheat Sheet

### Get Current Policy

```python
from backend.core import get_policy_store

policy_store = get_policy_store()
policy = await policy_store.get_policy()
config = policy.get_active_config()

print(f"Max leverage: {config.max_leverage}x")
print(f"Risk per trade: {config.max_risk_pct_per_trade:.2%}")
```

### Switch Risk Mode

```python
from backend.core import get_policy_store
from backend.models import RiskMode

policy_store = get_policy_store()

await policy_store.switch_mode(
    RiskMode.DEFENSIVE,
    updated_by="risk_manager",
)
# Broadcasts "policy.mode.changed" event
```

### Risk Modes

| Mode | Leverage | Risk/Trade | Daily DD | Min Confidence |
|------|----------|------------|----------|----------------|
| `AGGRESSIVE_SMALL_ACCOUNT` | 30x | 2% | 10% | 45% |
| `NORMAL` | 20x | 1% | 5% | 50% |
| `DEFENSIVE` | 10x | 0.5% | 3% | 60% |

---

## 📝 Logging Cheat Sheet

### Get Logger

```python
from backend.core import get_logger

logger = get_logger(__name__, domain="ai_engine")

logger.info(
    "signal_generated",
    symbol="BTCUSDT",
    confidence=0.85,
    action="LONG",
)

# Output (JSON):
# {
#   "event": "signal_generated",
#   "symbol": "BTCUSDT",
#   "confidence": 0.85,
#   "action": "LONG",
#   "trace_id": "abc123...",
#   "timestamp": "2025-12-01T12:00:00.000Z",
#   "module": "ai_engine.orchestrator",
#   "domain": "ai_engine"
# }
```

### Log Levels

```python
logger.debug("low_level_detail", ...)     # Debug info
logger.info("normal_event", ...)          # Normal operation
logger.warning("potential_issue", ...)     # Warning
logger.error("error_occurred", ...)        # Error
logger.critical("system_failure", ...)     # Critical
```

### Trading-Specific Logs

```python
from backend.core import TradingLogger

# Signal events
TradingLogger.signal_generated("BTCUSDT", "LONG", 0.85, trace_id="abc")
TradingLogger.signal_approved("BTCUSDT", "LONG", trace_id="abc")
TradingLogger.signal_rejected("BTCUSDT", "Low confidence", trace_id="abc")

# Order events
TradingLogger.order_submitted("BTCUSDT", "ORD-123", "LONG", 1.0, trace_id="abc")
TradingLogger.order_filled("BTCUSDT", "ORD-123", 50000.0, trace_id="abc")

# Position events
TradingLogger.position_opened("BTCUSDT", "POS-456", 50000.0, 1.0, 20.0, trace_id="abc")
TradingLogger.position_closed("BTCUSDT", "POS-456", 51000.0, 1000.0, 2.0, trace_id="abc")
TradingLogger.sl_adjusted("BTCUSDT", "POS-456", 49000.0, 50000.0, "breakeven", trace_id="abc")
```

---

## 🔍 trace_id Cheat Sheet

### Generate trace_id

```python
from backend.core import trace_context

# Generate new trace_id
trace_id = trace_context.generate()

# Set trace_id
trace_context.set("my-custom-id")

# Get current trace_id
trace_id = trace_context.get()

# Get or generate
trace_id = trace_context.get_or_generate()
```

### Use as Context Manager

```python
from backend.core import trace_context

with trace_context.scope("my-trace-id") as trace_id:
    # All operations here have trace_id="my-trace-id"
    logger.info("event_happened")  # Automatically includes trace_id
    await event_bus.publish("event.type", {...})  # Automatically includes trace_id
```

### Debug with trace_id

```bash
# Filter logs by trace_id
docker logs quantum_backend | grep "trace_id=abc123"

# Shows complete flow:
# [ai_engine] signal_generated trace_id=abc123
# [risk_safety] signal_approved trace_id=abc123
# [execution] order_submitted trace_id=abc123
# [portfolio] position_opened trace_id=abc123
```

---

## 🏥 Health Check Cheat Sheet

### Get Health Report

```python
from backend.core import get_health_checker

checker = get_health_checker()
report = await checker.get_health_report()

print(f"Status: {report.status}")  # HEALTHY, DEGRADED, CRITICAL
print(f"Uptime: {report.uptime_seconds}s")

for name, dep in report.dependencies.items():
    print(f"{name}: {dep.status} ({dep.latency_ms}ms)")
```

### Health Endpoint (FastAPI)

```python
from fastapi import FastAPI
from backend.core import get_health_checker

app = FastAPI()

@app.get("/health")
async def health():
    checker = get_health_checker()
    report = await checker.get_health_report()
    return report.to_dict()
```

### Check Specific Dependency

```python
checker = get_health_checker()

# Check individual dependencies
redis_health = await checker.check_redis()
binance_health = await checker.check_binance_rest()
system_health = await checker.check_system()
```

---

## 🔧 Common Patterns

### Pattern 1: Domain Module Template

```python
"""My Domain - Description."""

from backend.core import get_event_bus, get_policy_store, get_logger, trace_context
from backend.models import SignalEvent

logger = get_logger(__name__, domain="my_domain")


class MyService:
    """My service description."""
    
    def __init__(self):
        self.event_bus = get_event_bus()
        self.policy_store = get_policy_store()
        logger.info("MyService initialized")
    
    async def handle_event(self, event_data: dict):
        """Handle incoming event."""
        # Set trace_id
        trace_id = event_data.get("trace_id")
        if trace_id:
            trace_context.set(trace_id)
        
        # Get policy
        policy = await self.policy_store.get_policy()
        config = policy.get_active_config()
        
        # Process
        logger.info("event_processed", trace_id=trace_id)
        
        # Publish result
        await self.event_bus.publish(
            "my_domain.event.completed",
            {"status": "success"},
            trace_id=trace_id,
        )
```

### Pattern 2: Error Handling

```python
async def handle_event(self, event_data: dict):
    """Handle event with error recovery."""
    try:
        symbol = event_data["symbol"]
        
        # Process
        result = await self.process(symbol)
        
        logger.info("event_processed", symbol=symbol, result=result)
    
    except KeyError as e:
        logger.error("invalid_event_data", error=str(e), event=event_data)
    
    except Exception as e:
        logger.error(
            "handler_error",
            error=str(e),
            event=event_data,
            traceback=traceback.format_exc(),
        )
```

### Pattern 3: Policy-Aware Logic

```python
async def should_trade(self, symbol: str, confidence: float) -> bool:
    """Check if trade meets policy requirements."""
    policy = await self.policy_store.get_policy()
    config = policy.get_active_config()
    
    # Check confidence
    if confidence < config.global_min_confidence:
        logger.debug(
            "trade_filtered",
            symbol=symbol,
            confidence=confidence,
            required=config.global_min_confidence,
        )
        return False
    
    # Check if RL enabled
    if not config.enable_rl:
        logger.debug("rl_disabled", symbol=symbol)
        return False
    
    return True
```

---

## 🚨 Common Mistakes to Avoid

### ❌ DON'T: Import between domains

```python
# BAD!
from backend.domains.ai_engine.orchestrator import AIOrchestrator
orchestrator = AIOrchestrator()
signal = orchestrator.generate_signal()  # Direct coupling!
```

### ✅ DO: Use EventBus

```python
# GOOD!
await event_bus.publish("request.signal", {"symbol": "BTCUSDT"})
# AI Engine subscribes, generates signal, publishes result
```

---

### ❌ DON'T: Hardcode configuration

```python
# BAD!
MAX_LEVERAGE = 30
if leverage > MAX_LEVERAGE:
    reject()
```

### ✅ DO: Use PolicyStore

```python
# GOOD!
policy = await policy_store.get_policy()
config = policy.get_active_config()
if leverage > config.max_leverage:
    reject()
```

---

### ❌ DON'T: String logs

```python
# BAD!
logger.info(f"Trade executed: {symbol}, PnL: ${pnl}")
# Not searchable, not parseable
```

### ✅ DO: Structured logs

```python
# GOOD!
logger.info("trade_executed", symbol=symbol, pnl_usd=pnl)
# Searchable: pnl_usd > 100
```

---

### ❌ DON'T: Forget trace_id

```python
# BAD!
await event_bus.publish("event.type", payload)
logger.info("event_published")
# Can't trace flow!
```

### ✅ DO: Always include trace_id

```python
# GOOD!
trace_id = trace_context.get_or_generate()
await event_bus.publish("event.type", payload, trace_id=trace_id)
logger.info("event_published", trace_id=trace_id)
```

---

## 📞 Quick Commands

### Grep logs by trace_id
```bash
docker logs quantum_backend | grep "trace_id=abc123"
```

### Check health
```bash
curl http://localhost:8000/health
```

### View Redis Streams
```bash
redis-cli XLEN quantum:stream:ai.signal.generated
redis-cli XRANGE quantum:stream:ai.signal.generated - + COUNT 10
```

### View PolicyStore
```bash
redis-cli GET quantum:policy:current
```

---

## 📚 Full Documentation

- **Architecture**: `ARCHITECTURE_V2_DOMAINS.md`
- **Integration Guide**: `ARCHITECTURE_V2_INTEGRATION.md`
- **Why v2?**: `ARCHITECTURE_V2_WHY.md`
- **This Quick Ref**: `ARCHITECTURE_V2_QUICKREF.md`

---

**Keep this file open while coding!** 🚀
