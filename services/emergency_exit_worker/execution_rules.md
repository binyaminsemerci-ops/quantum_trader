# Emergency Exit Worker - Execution Rules

## Order Type

**ALWAYS:** Market order with reduce-only

```python
order_params = {
    "symbol": position.symbol,
    "side": "SELL" if position.amount > 0 else "BUY",
    "type": "MARKET",
    "quantity": abs(position.amount),
    "reduceOnly": True
}
```

## Execution Sequence

```
┌─────────────────────────────────────────────────────┐
│            PANIC CLOSE EXECUTION FLOW               │
├─────────────────────────────────────────────────────┤
│                                                     │
│  1. Receive system.panic_close event               │
│     │                                               │
│  2. Validate trigger (source, timestamp)           │
│     │                                               │
│  3. Fetch ALL open positions from exchange         │
│     │                                               │
│  4. For EACH position (parallel where possible):   │
│     ├─► Send MARKET close order                    │
│     ├─► Log result (success/fail)                  │
│     └─► Continue immediately (no wait)             │
│     │                                               │
│  5. Publish panic_close.completed                  │
│     │                                               │
│  6. HALT all trading services                      │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## Failure Handling

| Failure Type | Response |
|--------------|----------|
| One position fails | Log error, continue with next |
| Exchange timeout | Log error, continue with next |
| Rate limit | Short pause, retry same position once |
| All positions fail | Log CRITICAL, keep system halted |

## NO RETRY LOGIC

```python
# WRONG - Do not do this
for attempt in range(3):
    try:
        close_position(symbol)
        break
    except:
        continue

# CORRECT - Fire and forget
try:
    close_position(symbol)
    log_success(symbol)
except Exception as e:
    log_failure(symbol, e)
# Continue to next position regardless
```

## Order Parameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| type | MARKET | Guaranteed fill |
| reduceOnly | True | Cannot open new positions |
| closePosition | True (if supported) | Close entire position |
| timeInForce | None | N/A for market orders |

## Parallel Execution

If exchange supports:
- Send up to 5 close orders in parallel
- Do not wait for confirmations
- Log all results

If exchange requires sequential:
- Send one at a time
- Max 100ms between orders
- Continue regardless of response

## Logging Requirements

Every panic_close execution MUST log:

```json
{
  "event": "panic_close_execution",
  "trigger_source": "risk_kernel",
  "trigger_reason": "Account drawdown > 15%",
  "timestamp_start": 1707840000.123,
  "timestamp_end": 1707840002.456,
  "positions_found": 5,
  "positions_closed": 4,
  "positions_failed": 1,
  "failed_symbols": ["XRPUSDT"],
  "total_notional_usd": 15000,
  "execution_time_ms": 2333
}
```

## Post-Execution State

```python
SYSTEM_STATE = {
    "trading_enabled": False,
    "position_opening": False,
    "position_closing": True,  # Manual close still allowed
    "requires_manual_reset": True,
    "last_panic_close": timestamp,
    "panic_reason": reason
}
```

## Exchange-Specific Notes

### Binance Futures
- Use `futures_create_order` with `reduceOnly=true`
- Quantity must match position size exactly
- Side opposite to position direction

### General
- Always fetch fresh position data before closing
- Do not rely on cached position state
- Verify position exists before sending order
