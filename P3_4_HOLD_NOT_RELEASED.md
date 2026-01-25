# P3.4 HOLD Status: NOT RELEASED (Diagnosis)

## Current Evidence

| Check | Status | Value |
|-------|--------|-------|
| **HOLD Key** | 🔴 Still Active | `quantum:reconcile:hold:BTCUSDT = 1` |
| **HOLD TTL** | 🔴 Refreshing | `TTL = 300s` (continuously renewed) |
| **P3.4 Drift Detection** | 🔴 Ongoing | `exchange=LONG(0.007) ledger=(0.0)` |
| **Drift Counter** | 🔴 Incrementing | 41,513 detections (still growing) |
| **P3.3 Status** | 🔴 Still Blocking | DENY with `reconcile_required_qty_mismatch` |
| **Apply.Result** | ⚠️ DRY_RUN only | `CLOSE_PARTIAL_75` in would_execute, never executed |
| **Exchange Position** | 🔴 STILL 0.007 LONG | Position not actually closed |

## Root Cause

**The position was NOT actually closed on the exchange.**

Evidence:
- P3.4 logs: `exchange=LONG(0.007) ledger=(0.0)` (still seeing position)
- Apply.result: `CLOSE_PARTIAL_75` marked `would_execute` but never actual `executed=True`
- No filled order found in stream
- **Conclusion:** Close order was generated but not submitted/filled

## Why This Happened (Most Likely Scenarios)

### Scenario A: Close Order Generated But Not Submitted
- Apply Layer might have created the order in DRY_RUN mode
- Or order was blocked by permits/safety checks
- Or API call failed silently

### Scenario B: Exchange API Connection Issue
- Binance testnet might have rejected the close
- Network/auth issue on order submission
- Order was created but cancelled

### Scenario C: Permit Blocking the Close
- If close order needs permits too, it might be blocked by HOLD itself
- (Circular: HOLD blocks everything, including close actions)

## What P3.4 Sees Right Now

```
Loop iteration (every 1s):
  1. Fetch exchange position: LONG 0.007 BTC
  2. Fetch ledger position: 0.0 BTC
  3. Compare: 0.007 != 0.0 → SIDE MISMATCH
  4. Check: diff (0.005) > tolerance (0.001) → TRUE
  5. Action: Set HOLD (key is refreshed every second)
  6. Repeat...
```

## Next Steps to Actually Close

### Option 1: Direct Testnet Close (via API)
```bash
# If you have direct access to testnet:
# Close via Binance testnet directly (not through system)
# This breaks the circular dependency

curl -X POST https://testnet.binance.vision/api/v3/order \
  -d "symbol=BTCUSDT&side=SELL&type=MARKET&quantity=0.007&reduceOnly=true" \
  -H "X-MBX-APIKEY: ..."
```

Then wait 3s and verify HOLD releases.

### Option 2: Check Apply Layer Execution Log
```bash
# Why didn't the CLOSE_PARTIAL_75 actually execute?
journalctl -u quantum-apply-layer --since "30 minutes ago" --no-pager | grep -E "CLOSE|order|error|submitted|filled|BTCUSDT" | tail -50

# Look for:
#  - "order submitted"
#  - "order filled"
#  - "order rejected"
#  - "API error"
```

### Option 3: Check if Close Is Blocked by Permits
```bash
# If apply-layer CLOSE also needs permits, check:
redis-cli GET quantum:permit:p33:<PLAN_ID_OF_CLOSE>

# If missing → close can't get through
# Need to either:
#  a) Override permits for reconcile actions
#  b) Close directly outside system
```

## Recommended Fix Path

**Best approach for testnet:** Direct API close

1. **Close position on Binance testnet directly** (outside system)
   - Avoids permit/HOLD circular dependency
   - Simpler, faster verification
   - Exchange will show 0.0 LONG immediately

2. **Verify HOLD auto-releases**
   ```bash
   sleep 3
   redis-cli GET quantum:reconcile:hold:BTCUSDT
   # Should show: (nil) or 0
   ```

3. **Check P3.3 returns to ALLOW**
   ```bash
   journalctl -u quantum-position-state-brain --since "30 seconds ago" --no-pager | tail -10
   # Should show ALLOW instead of DENY
   ```

## Prevention (for Future)

Once this is resolved, implement:

1. **Reconcile-close special handling**
   - P3.4 detects drift → writes RECONCILE_CLOSE action
   - Apply Layer executes close WITHOUT permit requirements
   - (Can't block reconciliation with permits)

2. **Observability improvement**
   - Log why close didn't execute
   - Track order submission vs execution separately
   - Alert on "close order generated but not filled"

---

**Action:** Close 0.007 LONG directly on Binance testnet  
**Then:** Run `redis-cli GET quantum:reconcile:hold:BTCUSDT` to verify release  
**Expected:** Key returns (nil) or 0, P3.3 switches to ALLOW
