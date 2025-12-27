# ESS QUICK REFERENCE 🛡️

Emergency Stop System - Quick lookup for developers and operators

---

## 🚀 QUICK START

```python
from backend.core.safety.ess import EmergencyStopSystem
from backend.events.listeners.ess_listener import ESSEventListener

# Initialize ESS
ess = EmergencyStopSystem(policy_store, event_bus)

# Start listener
listener = ESSEventListener(ess, event_bus)
await listener.start()

# Check if orders allowed
can_execute = await ess.can_execute_orders()  # True or False
```

---

## 📊 STATES

| State           | Description                      | Can Execute? |
|-----------------|----------------------------------|--------------|
| `DISABLED`      | ESS turned off by policy         | ✅ Yes       |
| `ARMED`         | Normal monitoring                | ✅ Yes       |
| `TRIPPED`       | Emergency stop activated         | ❌ No        |
| `COOLING_DOWN`  | Waiting to re-arm                | ❌ No        |

---

## ⚙️ THRESHOLDS (PolicyStore)

```python
# Default values:
ess.enabled = True                      # Enable ESS
ess.max_daily_drawdown_pct = 5.0       # 5% max daily drawdown
ess.max_open_loss_pct = 10.0           # 10% max open loss
ess.max_execution_errors = 5            # 5 errors in 15 min
ess.cooldown_minutes = 15               # 15 min cooldown
ess.allow_manual_reset = True           # Allow operator reset
```

---

## 🔧 COMMON OPERATIONS

### Check Status
```python
status = ess.get_status()
print(status['state'])           # Current state
print(status['can_execute'])     # Can orders execute?
print(status['trip_reason'])     # Why tripped (if TRIPPED)
```

### Update Metrics
```python
await ess.update_metrics(
    daily_drawdown_pct=4.5,    # Current daily drawdown %
    open_loss_pct=7.0,         # Current open loss %
    execution_errors=2          # Error count in 15-min window
)
```

### Manual Reset (Operator)
```python
success = await ess.manual_reset(
    user="operator@example.com",
    reason="Issue resolved"
)
# Returns True if successful, False if not allowed
```

### Check Cooldown
```python
await ess.maybe_cooldown()  # Auto re-arms if cooldown expired
```

---

## 📡 EVENTS (EventBus)

### Published by ESS

**`ess.tripped`** - ESS activated
```json
{
  "reason": "Daily drawdown 6.0% exceeded threshold 5.0%",
  "metrics": {"daily_drawdown_pct": 6.0, ...},
  "timestamp": "2025-12-04T10:30:00Z"
}
```

**`ess.manual_reset`** - Operator reset
```json
{
  "user": "operator@example.com",
  "reason": "Issue resolved",
  "previous_state": "TRIPPED",
  "timestamp": "2025-12-04T10:45:00Z"
}
```

**`ess.rearmed`** - Auto re-arm after cooldown
```json
{
  "previous_state": "COOLING_DOWN",
  "cooldown_minutes": 15,
  "timestamp": "2025-12-04T10:45:00Z"
}
```

**`order.blocked_by_ess`** - Order blocked
```json
{
  "symbol": "BTCUSDT",
  "side": "long",
  "ess_state": "TRIPPED",
  "trip_reason": "Daily drawdown 6.0% exceeded threshold 5.0%",
  "timestamp": "2025-12-04T10:30:05Z"
}
```

### Consumed by ESS Listener

- `portfolio.pnl_update` → daily_drawdown_pct
- `risk.drawdown_update` → daily_drawdown_pct, open_loss_pct
- `execution.error` → execution_errors (15-min window)
- `risk.alert` → various metrics

---

## 🔍 DEBUGGING

### Check ESS Status
```python
status = ess.get_status()
print(f"State: {status['state']}")
print(f"Can Execute: {status['can_execute']}")
print(f"Trip Reason: {status.get('trip_reason', 'N/A')}")
print(f"Metrics: {status['metrics']}")
```

### Check Logs
```bash
# Look for these log patterns:
grep "ESS" logs/app.log

# Initialization:
[OK] Emergency Stop System available
[OK] Emergency Stop System initialized
[ESS] System ARMED and monitoring

# Trip:
[ESS ERROR] EMERGENCY STOP TRIGGERED: Daily drawdown 6.0% exceeded threshold 5.0%

# Order Block:
🛑 [ESS BLOCK] Order blocked by Emergency Stop System: BTCUSDT LONG
```

---

## 🚨 TROUBLESHOOTING

### ESS Not Initializing
**Problem:** No ESS logs at startup  
**Solution:**
1. Check `ESS_AVAILABLE` flag
2. Verify `PolicyStore` is available in `app_state`
3. Check `EventBus` is passed to `EventDrivenExecutor`

### ESS Not Tripping
**Problem:** Metrics exceed thresholds but ESS stays ARMED  
**Solution:**
1. Check `ess.enabled` policy is `True`
2. Verify metrics are being updated: `ess.metrics.last_updated`
3. Check threshold policies: `ess.max_daily_drawdown_pct`, etc.

### Orders Not Blocked
**Problem:** ESS is TRIPPED but orders still execute  
**Solution:**
1. Check `ess.can_execute_orders()` returns `False`
2. Verify ESS check is in executor before `submit_order`
3. Check logs for `[ESS BLOCK]` messages

### Manual Reset Fails
**Problem:** `manual_reset()` returns `False`  
**Solution:**
1. Check `ess.allow_manual_reset` policy is `True`
2. Verify ESS is in TRIPPED or COOLING_DOWN state
3. Check logs for denial reason

---

## 📝 CONFIGURATION RECIPES

### Conservative (Testnet)
```python
policy_store.set("ess.max_daily_drawdown_pct", 3.0)
policy_store.set("ess.max_open_loss_pct", 7.0)
policy_store.set("ess.max_execution_errors", 3)
policy_store.set("ess.cooldown_minutes", 30)
```

### Moderate (Production)
```python
policy_store.set("ess.max_daily_drawdown_pct", 5.0)
policy_store.set("ess.max_open_loss_pct", 10.0)
policy_store.set("ess.max_execution_errors", 5)
policy_store.set("ess.cooldown_minutes", 15)
```

### Aggressive (High Risk)
```python
policy_store.set("ess.max_daily_drawdown_pct", 10.0)
policy_store.set("ess.max_open_loss_pct", 20.0)
policy_store.set("ess.max_execution_errors", 10)
policy_store.set("ess.cooldown_minutes", 5)
```

### Disable ESS
```python
policy_store.set("ess.enabled", False)
# ESS state → DISABLED, orders always allowed
```

---

## 🎯 USE CASES

### Use Case 1: Flash Crash Protection
**Scenario:** Market drops 15% in 1 minute  
**ESS Response:**
1. `daily_drawdown_pct` exceeds threshold
2. ESS trips to TRIPPED state
3. All new orders blocked
4. Existing positions remain (exit logic separate)

### Use Case 2: Exchange Issues
**Scenario:** Exchange API errors on every order  
**ESS Response:**
1. `execution_errors` accumulate in 15-min window
2. ESS trips when count exceeds threshold
3. Orders blocked, preventing more errors
4. Auto re-arm after cooldown (exchange recovered)

### Use Case 3: Operator Intervention
**Scenario:** Suspicious market activity detected  
**Operator Action:**
1. Check ESS status
2. If needed, adjust thresholds to trip ESS
3. After resolution, manual reset to resume

---

## 📞 SUPPORT

**Documentation:** `SPRINT1_D3_ESS_IMPLEMENTATION_COMPLETE.md`  
**Tests:** `tests/unit/test_ess_sprint1_d3.py`  
**Source:** `backend/core/safety/ess.py`  
**Listener:** `backend/events/listeners/ess_listener.py`

---

*Quick Reference v1.0 - December 4, 2025*
