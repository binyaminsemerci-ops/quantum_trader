# Root Cause: Why Close Order Never Executed

## Evidence

### Apply-Layer Logs
```
BTCUSDT: Plan XXX already executed (duplicate)
BTCUSDT: Result published (executed=False, error=None)
```

Pattern: Plans keep being marked "duplicate", results show `executed=False`

### Code Analysis
P3.4 (`reconcile_engine/main.py`) **NEVER writes trading plans**.

- ✓ Sets HOLD
- ✓ Detects drift
- ✓ Logs errors
- ❌ **Does NOT write RECONCILE_CLOSE or any action plan**

### Trading Plans Source
Plans come from:
1. P1 (ML strategy)
2. External signals
3. **NOT from P3.4**

## The Design Bug (Circular Dependency)

```
Flow Right Now:
─────────────

HOLD set by P3.4 ──→ P3.3 blocks all EXECUTE
                         ↓
                    System waits for position close
                         ↓
                    But who closes the position? ← NOBODY
                    (P3.4 doesn't create plans,
                     P1 blocked by HOLD)
                         ↓
                    System stuck forever ⚠️
```

**This is a real design issue, not operator error.**

---

## Correct Fix: "Reconcile-Close Bypass"

### What's Needed

When P3.4 detects drift and sets HOLD:
1. Generate a special `RECONCILE_CLOSE` plan (from P3.4, not P1)
2. Apply Layer recognizes this plan type
3. Bypasses normal permits but enforces strict guardrails:
   - ✅ Only `reduceOnly=true` (never increase position)
   - ✅ Only `type=MARKET` (fast execution)
   - ✅ `qty = min(abs(exchange_amt - ledger_amt), abs(exchange_amt))`
   - ✅ Only for symbol with active HOLD
   - ✅ Separate metrics: `reconcile_close_count`
   - ✅ Logs with `reconcile_action=true`

### Safe Guardrails (Fail-Closed)

```python
# Safe reconcile close logic:

if plan.decision == "RECONCILE_CLOSE":
    # Validate tight constraints
    assert plan.reduceOnly == True           # Never add position
    assert plan.side == "close"              # Only closing
    assert plan.qty <= abs(exchange_position) # Can't exceed exposure
    assert plan.symbol in HOLD_SYMBOLS       # Only affected symbol
    
    # Execute MARKET order only
    assert plan.type in ["MARKET"]
    
    # Log for audit
    metrics.reconcile_close_attempts += 1
    logger.info(f"[RECONCILE] Closing {qty} {symbol}")
    
    # After successful close:
    # P3.4 detects exchange=0.0, releases HOLD automatically
    # P3.3 detects no HOLD, switches back to ALLOW
```

---

## Immediate Fix (Testnet)

### Option 1: Manual Close (What You Need to Do Now)

```bash
# Close position outside the system
# Via Binance testnet UI or direct API

curl -X POST https://testnet.binance.vision/api/v3/order \
  -d "symbol=BTCUSDT&side=SELL&type=MARKET&quantity=0.007&reduceOnly=true"

# Then verify:
sleep 3
redis-cli GET quantum:reconcile:hold:BTCUSDT
# Expected: (nil) or 0
```

### Option 2: Code Fix (Permanent)

Build "Reconcile-Close Plan Generator" in P3.4:

**File:** `microservices/reconcile_engine/main.py` (add method)

```python
def generate_reconcile_close_plan(self, symbol: str, qty: float):
    """
    Generate a special RECONCILE_CLOSE plan when drift detected.
    
    Guardrails:
    - Only reduceOnly closes
    - Only MARKET orders
    - Never increases position
    """
    
    plan = {
        "plan_id": self._generate_id(),
        "symbol": symbol,
        "decision": "RECONCILE_CLOSE",  # Special type
        "side": "close",
        "qty": qty,
        "type": "MARKET",
        "reduceOnly": True,
        "reason": f"drift_detected:{self.reason}",
        "timestamp": int(time.time()),
    }
    
    # Publish to trading.plan stream
    self.redis.xadd(
        "quantum:stream:trading.plan",
        plan,
        id="*"
    )
    
    logger.info(f"[RECONCILE] Generated close plan: {plan['plan_id']}")
    self.metrics.reconcile_close_plans += 1
```

**File:** `microservices/apply_layer/main.py` (add handler)

```python
def handle_reconcile_close(self, plan):
    """
    Execute reconcile-close plan with strict guardrails.
    Bypasses normal permits (fail-safe exception).
    """
    
    # Validate reconcile constraints
    if plan["decision"] != "RECONCILE_CLOSE":
        return False
    
    if not plan.get("reduceOnly"):
        logger.error(f"[DENY] Reconcile plan not reduceOnly: {plan}")
        return False
    
    if plan.get("type") not in ["MARKET"]:
        logger.error(f"[DENY] Reconcile plan not MARKET: {plan}")
        return False
    
    # Check HOLD is actually active for this symbol
    hold_key = f"quantum:reconcile:hold:{plan['symbol']}"
    if not self.redis.exists(hold_key):
        logger.warning(f"[SKIP] HOLD not active, no close needed: {plan}")
        return False
    
    # Execute with monitoring
    logger.info(f"[EXECUTE] Reconcile close {plan['qty']} {plan['symbol']}")
    
    try:
        order = self.exchange_api.place_order(
            symbol=plan["symbol"],
            side="SELL" if plan["side"] == "close" else "BUY",
            type="MARKET",
            quantity=plan["qty"],
            reduce_only=True
        )
        
        self.metrics.reconcile_close_executed += 1
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Reconcile close failed: {e}")
        return False
```

---

## Why This Is Better Than Manual Close

| Aspect | Manual Close | Reconcile-Close Plan |
|--------|-------------|----------------------|
| Speed | Manual, slow | Automatic, 1-2s |
| Safety | Manual error risk | Tight guardrails |
| Consistency | One-off | Automated every time |
| Audit Trail | Missing | Full logged |
| Testnet | OK | Better for production |

---

## Implementation Order

1. **Today (Testnet):** Manual close via Binance UI
2. **This week:** Add reconcile_close_plan generator to P3.4
3. **This week:** Add handler to Apply Layer
4. **This week:** Test with drift scenario
5. **Next release:** Merge to production

---

## Summary

**Current Bug:** P3.4 HOLD blocks system, but has no way to resolve it.  
**Root Cause:** P3.4 never generates close plans (design gap).  
**Immediate Fix:** Close manually on testnet.  
**Permanent Fix:** Add "Reconcile-Close Plan" type with strict fail-safe guardrails.

This prevents the deadlock without opening up unsafe trading paths.
