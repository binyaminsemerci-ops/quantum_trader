# Emergency Exit Worker

> **LAST LINE OF DEFENSE** - When intelligence is irrelevant

This service exists solely to close **ALL** open positions during catastrophic or uncertain system states.

## Principles

1. **Market orders only** - No limits, no conditions
2. **Reduce-only** - Cannot open new positions
3. **No strategy** - Ignores signals, AI, PnL
4. **No intelligence** - Pure mechanical execution
5. **Survival > Cost** - Accepts slippage to exit

## Trigger

**Event:** `system.panic_close`

**Authorized Publishers (ONLY):**
- Risk Kernel
- Exit Brain (fatal health-failure only)
- Ops/Manual (rare, logged)

## What Happens

```
1. Emergency Exit Worker receives event
2. Reads all open positions from exchange
3. For each position:
   - MARKET order
   - reduceOnly=true
   - closePosition=true
4. No retry logic on individual failures
5. If one fails → continue with next
6. Publish panic_close.completed
```

## Critical Rule

> **If Emergency Exit Worker fails → ALL trading remains stopped**
> No restart before manual inspection.

## Files

| File | Purpose |
|------|---------|
| `policy_ref.md` | Policy references and authorization |
| `trigger_conditions.md` | When panic_close can be triggered |
| `execution_rules.md` | How positions are closed |
| `emergency_exit_worker.py` | Main service implementation |
| `tests/` | Critical test scenarios |

## Testing Requirements (MVP)

1. **Idempotency** - panic_close sent 2× → no errors
2. **Partial failure** - 1 symbol fails → others still close
3. **Latency stress** - Worker continues without waiting for confirmations

## Redis Streams

| Stream | Direction | Purpose |
|--------|-----------|---------|
| `quantum:stream:system.panic_close` | IN | Trigger event |
| `quantum:stream:panic_close.completed` | OUT | Completion confirmation |

## DO NOT

- ❌ Add retry logic
- ❌ Add optimization
- ❌ Add conditions
- ❌ Add intelligence
- ❌ Wait for confirmations before next position
