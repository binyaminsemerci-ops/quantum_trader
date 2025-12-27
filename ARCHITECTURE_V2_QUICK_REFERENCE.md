# Architecture v2 Quick Reference

Fast reference for using v2 components in quantum_trader.

---

## Import Patterns

```python
# Structured Logger
from backend.core import get_logger
logger = get_logger(__name__, component="my_service")

# PolicyStore v2
from backend.core import get_policy_store_v2
from backend.core.policy_store import RiskMode
policy_store = get_policy_store_v2()

# EventBus
from backend.core import get_event_bus
event_bus = get_event_bus()

# HealthChecker
from backend.core import get_health_checker
health_checker = get_health_checker()

# Trace Context
from backend.core import trace_context
```

---

## Logger Usage

### Basic Logging
```python
logger.info("Order placed successfully")
logger.warning("High latency detected")
logger.error("Failed to connect to database")
logger.debug("Processing signal: BTCUSDT LONG")
```

### With Context (f-strings only!)
```python
# ✅ CORRECT (f-string)
logger.info(f"Order placed: symbol={symbol}, quantity={quantity}")
logger.error(f"Connection failed: error={str(e)}, retry={retry_count}")

# ❌ WRONG (keyword arguments NOT supported)
logger.info("Order placed", symbol=symbol, quantity=quantity)
```

### With Trace Context
```python
from backend.core import trace_context

with trace_context(trace_id="order_123", user_id="user_456"):
    logger.info("Processing order")  # Automatically includes trace_id and user_id
```

---

## PolicyStore v2

### Get Current Policy
```python
policy_store = get_policy_store_v2()
policy = await policy_store.get_policy()

print(f"Active mode: {policy.active_mode}")
print(f"Max leverage: {policy.modes[policy.active_mode].max_leverage}")
print(f"Min confidence: {policy.modes[policy.active_mode].global_min_confidence}")
```

### Switch Risk Mode
```python
from backend.core.policy_store import RiskMode

policy_store = get_policy_store_v2()

# Switch to defensive mode
await policy_store.switch_mode(RiskMode.DEFENSIVE)

# Switch to aggressive mode
await policy_store.switch_mode(RiskMode.AGGRESSIVE_SMALL_ACCOUNT)

# Back to normal
await policy_store.switch_mode(RiskMode.NORMAL)
```

### Update Policy Parameters
```python
policy = await policy_store.get_policy()
policy.modes[RiskMode.NORMAL].max_risk_pct_per_trade = 0.015  # 1.5% instead of 1%
await policy_store.save_policy(policy)
```

### Check if Module Enabled
```python
policy = await policy_store.get_policy()
mode_config = policy.modes[policy.active_mode]

if mode_config.enable_rl:
    logger.info("RL position sizing enabled")

if mode_config.enable_meta_strategy:
    logger.info("Meta Strategy Controller enabled")
```

---

## EventBus

### Publish Events
```python
event_bus = get_event_bus()

# Publish signal generated event
await event_bus.publish("ai.signal.generated", {
    "symbol": "BTCUSDT",
    "action": "LONG",
    "confidence": 0.85,
    "timestamp": datetime.utcnow().isoformat()
})

# Publish order placed event
await event_bus.publish("execution.order.placed", {
    "order_id": "order_123",
    "symbol": "ETHUSDT",
    "side": "BUY",
    "quantity": 10.0
})

# Publish risk event
await event_bus.publish("risk.limit.exceeded", {
    "type": "daily_drawdown",
    "current": 0.06,
    "limit": 0.05
})
```

### Subscribe to Events
```python
event_bus = get_event_bus()

# Subscribe to specific event
async def handle_signal(event_data: dict):
    symbol = event_data["symbol"]
    action = event_data["action"]
    confidence = event_data["confidence"]
    
    logger.info(f"Received signal: {symbol} {action} (conf={confidence})")
    # Your logic here

event_bus.subscribe("ai.signal.generated", handle_signal)

# Subscribe to wildcard (all AI events)
async def handle_ai_event(event_data: dict):
    logger.info(f"AI event: {event_data}")

event_bus.subscribe("ai.*", handle_ai_event)

# Subscribe to multiple event types
event_bus.subscribe("execution.order.placed", handle_order)
event_bus.subscribe("execution.order.filled", handle_order)
event_bus.subscribe("execution.order.failed", handle_order)
```

### Unsubscribe
```python
event_bus.unsubscribe("ai.signal.generated", handle_signal)
```

---

## HealthChecker

### Get Health Report
```python
health_checker = get_health_checker()
report = await health_checker.get_health_report()

print(f"Service: {report.service}")
print(f"Status: {report.status}")  # HEALTHY, DEGRADED, CRITICAL, UNKNOWN
print(f"Uptime: {report.uptime_seconds}s")

# Check specific dependency
redis_health = report.dependencies["redis"]
print(f"Redis: {redis_health.status} (latency: {redis_health.latency_ms}ms)")

# Check system resources
print(f"CPU: {report.system.cpu_percent}%")
print(f"Memory: {report.system.memory_percent}%")
print(f"Disk: {report.system.disk_percent}%")
```

### Check Specific Dependency
```python
# Manual dependency checks
redis_health = await health_checker.check_redis()
print(f"Redis status: {redis_health.status}")

binance_health = await health_checker.check_binance_rest()
print(f"Binance REST latency: {binance_health.latency_ms}ms")
```

---

## Common Event Types

### AI Signals
- `ai.signal.generated`: New trading signal from AI models
- `ai.signal.validated`: Signal passed validation
- `ai.signal.rejected`: Signal failed validation
- `ai.model.retrained`: AI model retrained
- `ai.model.performance`: Model performance metrics

### Execution
- `execution.order.placed`: Order submitted to exchange
- `execution.order.filled`: Order executed successfully
- `execution.order.cancelled`: Order cancelled
- `execution.order.failed`: Order submission failed
- `execution.position.opened`: New position opened
- `execution.position.closed`: Position closed

### Risk Management
- `risk.limit.exceeded`: Risk limit breached
- `risk.mode.changed`: Risk mode switched (NORMAL → DEFENSIVE)
- `risk.exposure.high`: Portfolio exposure too high
- `risk.drawdown.warning`: Approaching drawdown limit

### System
- `system.startup`: Backend started
- `system.shutdown`: Backend shutting down
- `system.health.degraded`: System health degraded
- `system.error`: Critical error occurred

---

## Environment Variables

```bash
# Redis
REDIS_URL=redis://redis:6379  # Docker: redis:6379, Local: localhost:6379

# Logging
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR

# Binance
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
BINANCE_TESTNET=true  # true for testnet, false for mainnet
```

---

## Health Endpoint

### Check System Health
```bash
curl http://localhost:8000/api/v2/health | jq
```

**Response:**
```json
{
  "service": "quantum_trader",
  "status": "HEALTHY",
  "uptime_seconds": 123.45,
  "dependencies": {
    "redis": {"status": "HEALTHY", "latency_ms": 0.77, ...},
    "binance_rest": {"status": "HEALTHY", "latency_ms": 257.44, ...},
    "binance_ws": {"status": "HEALTHY", ...}
  },
  "system": {
    "cpu_percent": 1.3,
    "memory_percent": 19.5,
    "disk_percent": 10.7,
    "status": "HEALTHY"
  },
  "v2_available": true
}
```

### Health Status Levels
- `HEALTHY`: All systems operational
- `DEGRADED`: Some non-critical issues (e.g., high latency)
- `CRITICAL`: Critical dependency down (e.g., Redis unavailable)
- `UNKNOWN`: Health check failed or not initialized

---

## Redis CLI Commands

### Check Policy
```bash
docker exec quantum_redis redis-cli GET "quantum:policy:current"
```

### List All Keys
```bash
docker exec quantum_redis redis-cli KEYS "quantum:*"
```

### Check Redis Info
```bash
docker exec quantum_redis redis-cli INFO
```

### Monitor Events (Real-time)
```bash
docker exec quantum_redis redis-cli MONITOR
```

### Check Stream Length
```bash
docker exec quantum_redis redis-cli XLEN "quantum:events:ai.signal.generated"
```

---

## Migration Checklist

### From Legacy to v2

#### Logger
```python
# Before
import logging
logger = logging.getLogger(__name__)
logger.info("Message")

# After
from backend.core import get_logger
logger = get_logger(__name__, component="my_service")
logger.info("Message")  # Now with trace_id and structured context
```

#### PolicyStore
```python
# Before (in-memory)
from backend.services.policy_store import get_policy_store
policy_store = get_policy_store()
policy = policy_store.get_policy()  # Sync

# After (Redis-backed)
from backend.core import get_policy_store_v2
policy_store = get_policy_store_v2()
policy = await policy_store.get_policy()  # Async
```

#### Inter-Service Communication
```python
# Before (direct import)
from backend.services.risk_manager import evaluate_risk
result = evaluate_risk(signal)

# After (event-driven)
from backend.core import get_event_bus
event_bus = get_event_bus()
await event_bus.publish("ai.signal.generated", signal_data)
# Risk manager subscribes to this event
```

---

## Troubleshooting

### Logger Issues
**Problem:** `Logger._log() got an unexpected keyword argument`

**Solution:** Use f-strings, NOT keyword arguments:
```python
# ❌ WRONG
logger.info("Order placed", symbol=symbol)

# ✅ CORRECT
logger.info(f"Order placed: symbol={symbol}")
```

### PolicyStore Not Initialized
**Problem:** `get_policy_store_v2()` returns None

**Solution:** Check logs for v2 initialization errors:
```bash
docker compose logs backend | grep "Architecture v2"
```

### EventBus Not Receiving Events
**Problem:** Events published but not consumed

**Solution:**
1. Check subscription: `event_bus.subscribe("event.type", handler)`
2. Verify consumer started: `await event_bus.start()`
3. Check Redis logs: `docker compose logs redis`

### Health Endpoint 500 Error
**Problem:** `/api/v2/health` returns Internal Server Error

**Solution:**
1. Check if HealthChecker initialized: `docker compose logs backend | grep HealthChecker`
2. Verify Redis connectivity: `docker exec quantum_redis redis-cli PING`
3. Check backend logs for exceptions: `docker compose logs backend --tail 100`

---

**Quick Reference Version:** 1.0  
**Last Updated:** December 1, 2025  
**Architecture Version:** v2.0
