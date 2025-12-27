# Risk Gate Integration Guide

## EPIC-RISK3-EXEC-001: Enforce Global Risk v3 in Execution

**Status:** Risk gate implemented ✅ | Integration into order flow pending ⏳

---

## 🎯 Integration Point

**Function Added:** `enforce_risk_gate()` in `backend/services/execution/execution.py` (line ~1383)

### Where to Call

The risk gate should be called **AFTER** account/exchange routing and **BEFORE** any order submission.

### Typical Flow

```python
# 1. Account routing (EPIC-MT-ACCOUNTS-001)
account_name = select_account_for_signal(...)

# 2. Exchange routing (EPIC-EXCH-ROUTING-001)
exchange_name = select_exchange_for_strategy(...)

# 3. Build order request
order_request = {
    "symbol": "BTCUSDT",
    "side": "BUY",
    "size": 1000.0,
    "leverage": 2.0,
}

# 4. ✨ NEW: Enforce risk gate (EPIC-RISK3-EXEC-001)
try:
    order_request = await enforce_risk_gate(
        account_name=account_name,
        exchange_name=exchange_name,
        strategy_id=strategy_id,
        order_request=order_request,
    )
except OrderBlockedByRiskGate as e:
    logger.error(f"Order blocked: {e}")
    return  # Skip order placement

# 5. Submit order
client = get_exchange_client_for_account(account_name, exchange_name)
result = await safe_executor.place_order_with_safety(...)
```

---

## 📝 Integration Examples

### Example 1: Event-Driven Executor

**File:** `backend/services/execution/event_driven_executor.py`

**Location:** Inside order submission logic (where `SafeOrderExecutor` is called)

```python
async def _place_order_internal(
    self,
    account_name: str,
    exchange_name: str,
    strategy_id: str,
    symbol: str,
    side: str,
    size: float,
    leverage: float = 1.0,
) -> OrderResult:
    # Build order request
    order_request = {
        "symbol": symbol,
        "side": side,
        "size": size,
        "leverage": leverage,
    }
    
    # ✨ NEW: Enforce risk gate
    from backend.services.execution.execution import enforce_risk_gate
    
    try:
        order_request = await enforce_risk_gate(
            account_name=account_name,
            exchange_name=exchange_name,
            strategy_id=strategy_id,
            order_request=order_request,
        )
    except Exception as e:
        logger.error(f"[RISK-GATE] Order blocked: {e}")
        return OrderResult(success=False, error_message=str(e))
    
    # Get client
    client = self._get_exchange_client(account_name, exchange_name)
    
    # Submit order with safety
    result = await self.safe_executor.place_order_with_safety(
        submit_func=client.futures_create_order,
        order_params=order_request,
        symbol=symbol,
        side=side,
        ...
    )
    
    return result
```

### Example 2: Hybrid TP/SL Module

**File:** `backend/services/execution/hybrid_tpsl.py`

**Location:** Before entry order placement

```python
async def place_entry_with_hybrid_tpsl(
    client,
    symbol: str,
    side: str,
    quantity: float,
    strategy_id: str,
    account_name: str,  # Add if not present
    exchange_name: str = "binance",  # Add if not present
    leverage: float = 1.0,
    ...
) -> bool:
    # Build entry order request
    entry_order = {
        "symbol": symbol,
        "side": side,
        "size": quantity,
        "leverage": leverage,
        "type": "MARKET",
    }
    
    # ✨ NEW: Enforce risk gate
    from backend.services.execution.execution import enforce_risk_gate
    
    try:
        entry_order = await enforce_risk_gate(
            account_name=account_name,
            exchange_name=exchange_name,
            strategy_id=strategy_id,
            order_request=entry_order,
        )
    except Exception as e:
        logger.error(f"[RISK-GATE] Entry order blocked: {e}")
        return False
    
    # Submit entry order
    result = await safe_executor.place_order_with_safety(
        submit_func=lambda **params: client._signed_request("POST", "/fapi/v1/order", params),
        order_params=entry_order,
        ...
    )
    
    # ... (TP/SL logic continues)
```

---

## 🔧 Required Parameters

### enforce_risk_gate() Parameters

```python
account_name: str       # "PRIVATE_MAIN", "PRIVATE_AGGRO", etc.
exchange_name: str      # "binance", "bybit", "okx", etc.
strategy_id: str        # "neo_scalper_1", "trend_rider_5", etc.
order_request: dict     # {"symbol": "BTCUSDT", "side": "BUY", "size": 1000.0, "leverage": 2.0}
```

### order_request Fields (minimum)

```python
{
    "symbol": str,      # Trading pair (e.g., "BTCUSDT")
    "side": str,        # "BUY" or "SELL"
    "size": float,      # Position size in quote currency (USD)
    "leverage": float,  # Leverage multiplier (e.g., 2.0 for 2x)
}
```

---

## ⚠️ Error Handling

### OrderBlockedByRiskGate Exception

When an order is blocked, `enforce_risk_gate()` raises `OrderBlockedByRiskGate`:

```python
try:
    order_request = await enforce_risk_gate(...)
except OrderBlockedByRiskGate as e:
    # Log the block reason
    logger.warning(f"Order blocked: {e}")
    
    # Emit event (optional)
    await event_bus.publish("order.blocked", {
        "reason": str(e),
        "account": account_name,
        "strategy": strategy_id,
    })
    
    # Return early (skip order)
    return None
```

### Scale-Down Behavior

When risk gate scales down an order, it modifies `order_request["size"]`:

```python
# Original size
original_size = order_request["size"]  # e.g., 1000.0

# After enforce_risk_gate()
order_request = await enforce_risk_gate(...)

# Scaled size (if decision was "scale_down")
scaled_size = order_request["size"]  # e.g., 500.0 (50% scale)

# Log shows: [RISK-GATE] 📉 Order SCALED DOWN by risk gate
```

---

## 🚀 Initialization

### Risk Gate Startup

The risk gate must be initialized at application startup:

```python
# In main.py or execution service startup
from backend.risk.risk_gate_v3 import init_risk_gate, RiskStateFacade
from backend.services.risk.emergency_stop_system import EmergencyStopSystem

# Initialize Risk v3 facade
risk_facade = RiskStateFacade(risk_api_url="http://localhost:8001")

# Get ESS instance
ess = EmergencyStopSystem(...)  # Or retrieve from global state

# Initialize global risk gate
init_risk_gate(risk_facade=risk_facade, ess=ess)

logger.info("[STARTUP] Risk Gate v3 initialized")
```

**⚠️ Important:** If risk gate is not initialized, `enforce_risk_gate()` will block ALL orders with reason `"risk_gate_not_initialized"`.

---

## 📊 Logging Examples

### Log Output Examples

**Order Allowed:**
```
[RISK-GATE] ✅ Order ALLOWED by risk gate
  reason: all_risk_checks_passed
  risk_level: INFO
  account: PRIVATE_MAIN
  exchange: binance
  strategy: neo_scalper_1
  symbol: BTCUSDT
```

**Order Blocked (ESS):**
```
[RISK-GATE] ❌ Order BLOCKED by risk gate
  reason: ess_trading_halt_active
  risk_level: None
  ess_active: True
  account: PRIVATE_MAIN
  exchange: binance
  strategy: neo_scalper_1
  symbol: BTCUSDT
```

**Order Blocked (Global Risk CRITICAL):**
```
[RISK-GATE] ❌ Order BLOCKED by risk gate
  reason: global_risk_critical: Leverage exceeded 5x; Drawdown > 10%
  risk_level: CRITICAL
  ess_active: False
  account: PRIVATE_MAIN
  exchange: binance
  strategy: neo_scalper_1
  symbol: BTCUSDT
```

**Order Scaled Down:**
```
[RISK-GATE] 📉 Order SCALED DOWN by risk gate
  reason: single_trade_risk_too_high
  scale_factor: 0.5
  original_size: 1000.0
  scaled_size: 500.0
  account: PRIVATE_MAIN
  exchange: binance
  strategy: neo_scalper_1
  symbol: BTCUSDT
```

---

## 🧪 Testing Integration

### Manual Test

```python
# Test risk gate directly
from backend.services.execution.execution import enforce_risk_gate

order_request = {
    "symbol": "BTCUSDT",
    "side": "BUY",
    "size": 1000.0,
    "leverage": 2.0,
}

try:
    result = await enforce_risk_gate(
        account_name="PRIVATE_MAIN",
        exchange_name="binance",
        strategy_id="neo_scalper_1",
        order_request=order_request,
    )
    print(f"✅ Order allowed: {result}")
except Exception as e:
    print(f"❌ Order blocked: {e}")
```

### Integration Test Checklist

- [ ] Risk gate initialized at startup
- [ ] `enforce_risk_gate()` called before every order
- [ ] Blocked orders are logged with reason
- [ ] Scaled-down orders have reduced size
- [ ] ESS halt blocks ALL orders
- [ ] Global Risk CRITICAL blocks orders
- [ ] Capital profile limits enforced
- [ ] No order can bypass risk gate

---

## 📚 Related Documentation

- **Implementation:** `EPIC_RISK3_EXEC_001_SUMMARY.md`
- **Quick Reference:** `RISK_GATE_V3_QUICKREF.md`
- **Code:** `backend/risk/risk_gate_v3.py`
- **Tests:** `tests/risk/test_risk_gate_v3.py` (13/13 passing ✅)

---

## 🎯 Next Steps

1. **Find order submission points** in:
   - `backend/services/execution/event_driven_executor.py`
   - `backend/services/execution/hybrid_tpsl.py`
   - `backend/services/execution/smart_execution.py`
   - Any other files calling `SafeOrderExecutor.place_order_with_safety()`

2. **Add risk gate call** before each order submission using examples above

3. **Test integration** with:
   - Normal orders (should pass)
   - ESS active (should block)
   - High leverage (should block)
   - CRITICAL risk state (should block)

4. **Initialize at startup** in `main.py` or service initialization

5. **Monitor logs** for risk gate decisions during operation

---

**EPIC-RISK3-EXEC-001:** ✅ Risk Gate Implemented | ⏳ Integration Pending
