# Exit Brain V3 LIVE Mode Activation Runbook

**Status**: Phase 2B - LIVE Mode Ready  
**Date**: December 10, 2025  
**Safety Level**: 🔴 HIGH - AI will control real exits

---

## 📋 Overview

This document provides the **safe, controlled activation path** for Exit Brain V3 LIVE mode, where AI takes full control of exit order placement.

### Three-Layer Safety System

Exit Brain V3 uses **three independent toggles** for maximum safety:

1. **EXIT_MODE**: Which system owns exits (LEGACY or EXIT_BRAIN_V3)
2. **EXIT_EXECUTOR_MODE**: Executor behavior (SHADOW or LIVE)
3. **EXIT_BRAIN_V3_LIVE_ROLLOUT**: Additional kill-switch (DISABLED or ENABLED)

**For AI to place real orders, ALL THREE must align:**
```
EXIT_MODE=EXIT_BRAIN_V3
EXIT_EXECUTOR_MODE=LIVE
EXIT_BRAIN_V3_LIVE_ROLLOUT=ENABLED
```

If ANY toggle is wrong, the system automatically falls back to safe mode.

---

## 🎯 Three Operational States

### State A: Pure Legacy (Baseline)

**Configuration:**
```bash
EXIT_MODE=LEGACY
EXIT_EXECUTOR_MODE=<ignored>
EXIT_BRAIN_V3_LIVE_ROLLOUT=<ignored>
```

**Behavior:**
- ✅ Traditional position_monitor + hybrid_tpsl control exits
- ✅ Exit Brain executor OFF (not started)
- ✅ Gateway allows all legacy modules through
- ✅ No AI involvement in exits

**Use When:**
- Initial production deployment
- Rollback from any issue
- Testing non-exit features

---

### State B: AI Shadow Mode (Observation)

**Configuration:**
```bash
EXIT_MODE=EXIT_BRAIN_V3
EXIT_EXECUTOR_MODE=SHADOW
EXIT_BRAIN_V3_LIVE_ROLLOUT=DISABLED  # Or ENABLED, doesn't matter in SHADOW
```

**Behavior:**
- ✅ Exit Brain executor RUNNING
- ✅ AI monitors positions and decides what it WOULD do
- ✅ Decisions logged to console + `backend/data/exit_brain_shadow.jsonl`
- ✅ **NO ORDERS PLACED** by AI
- ✅ Legacy modules still control real exits
- ✅ Gateway logs ownership conflicts (soft warnings)

**Use When:**
- Evaluating AI decision quality
- Comparing AI vs legacy exits
- Building confidence before LIVE
- Extended testing period (24-48h recommended)

**Key Logs to Monitor:**
```
[EXIT_BRAIN_SHADOW] BTCUSDT long size=1.5000 pnl=+2.34% type=move_sl ...
[EXIT_GUARD] 🚨 OWNERSHIP CONFLICT: Legacy module 'position_monitor' ...
```

---

### State C: AI LIVE Mode (Full Control)

**Configuration:**
```bash
EXIT_MODE=EXIT_BRAIN_V3
EXIT_EXECUTOR_MODE=LIVE
EXIT_BRAIN_V3_LIVE_ROLLOUT=ENABLED  # 🔴 This is the critical one!
```

**Behavior:**
- 🔴 Exit Brain executor RUNNING in LIVE MODE
- 🔴 AI monitors positions and **PLACES REAL ORDERS**
- 🔴 Orders routed through `exit_order_gateway` as module `exit_executor`
- 🔴 Legacy modules **HARD BLOCKED** by gateway (orders rejected)
- 🔴 AI is the **SINGLE MUSCLE** for exits
- ✅ Gateway enforces ownership boundaries
- ✅ Metrics track blocked legacy attempts

**Use When:**
- Shadow mode analysis confirms AI improves outcomes
- Team ready for AI-controlled exits
- Monitoring and rollback plan in place

**Key Logs to Monitor:**
```
[EXIT_MODE] 🔴 EXIT BRAIN V3 LIVE MODE ACTIVE 🔴 AI executor will place real orders. Legacy exit modules will be blocked.
[EXIT_BRAIN_LIVE] Executing partial_close for BTCUSDT long: Partial TP hit: pnl=5.42%, taking 50% profit
[EXIT_GUARD] 🛑 BLOCKED: Legacy module 'position_monitor' attempted to place sl ...
[EXIT_METRICS] 🛑 Blocked 12 legacy module orders in EXIT_BRAIN_V3 LIVE mode (expected behavior).
```

---

## 🚀 Activation Sequence (Step-by-Step)

### Step 1: Shadow Mode Validation (24-48h)

**Goal**: Verify AI decisions are sensible before giving control.

1. **Set environment variables:**
   ```bash
   export EXIT_MODE=EXIT_BRAIN_V3
   export EXIT_EXECUTOR_MODE=SHADOW
   export EXIT_BRAIN_V3_LIVE_ROLLOUT=DISABLED
   export EXIT_BRAIN_V3_ENABLED=true  # For consistency
   ```

2. **Update .env file** (if using):
   ```
   EXIT_MODE=EXIT_BRAIN_V3
   EXIT_EXECUTOR_MODE=SHADOW
   EXIT_BRAIN_V3_LIVE_ROLLOUT=DISABLED
   EXIT_BRAIN_V3_ENABLED=true
   ```

3. **Restart backend:**
   ```bash
   # Stop backend
   pkill -f "python.*main.py"  # Or use your stop script
   
   # Start backend
   python backend/main.py
   ```

4. **Verify activation:**
   ```bash
   # Check logs
   tail -f backend/logs/quantum_trader.log | grep EXIT_BRAIN
   
   # Or use diagnostic endpoint
   curl http://localhost:8000/health/exit_brain_status | jq
   
   # Or use CLI tool
   python backend/tools/print_exit_status.py
   ```

   **Expected output:**
   ```
   [EXIT_MODE] Configuration loaded: EXIT_MODE=EXIT_BRAIN_V3, EXIT_EXECUTOR_MODE=SHADOW, EXIT_BRAIN_V3_LIVE_ROLLOUT=DISABLED
   [EXIT_BRAIN_EXECUTOR] Initialized in SHADOW MODE - Config: EXIT_MODE=EXIT_BRAIN_V3, EXIT_EXECUTOR_MODE=SHADOW, EXIT_BRAIN_V3_LIVE_ROLLOUT=DISABLED
   [EXIT_BRAIN_EXECUTOR] Starting monitoring loop (interval=10.0s, mode=SHADOW)
   ```

5. **Monitor for 24-48 hours:**
   
   Watch for:
   - `[EXIT_BRAIN_SHADOW]` logs showing AI decisions
   - `[EXIT_GUARD]` warnings about legacy module conflicts
   - Shadow log file: `backend/data/exit_brain_shadow.jsonl`

6. **Analyze shadow logs:**
   ```python
   # Quick analysis
   import pandas as pd
   import json
   
   # Load shadow logs
   logs = []
   with open('backend/data/exit_brain_shadow.jsonl') as f:
       for line in f:
           logs.append(json.loads(line))
   
   df = pd.DataFrame(logs)
   
   # Decision type distribution
   print(df['decision.decision_type'].value_counts())
   
   # Average confidence by decision type
   print(df.groupby('decision.decision_type')['decision.confidence'].mean())
   
   # Symbols with emergency exit suggestions
   emergency = df[df['decision.decision_type'] == 'full_exit_now']
   print(f"Emergency exits suggested for: {emergency['symbol'].unique()}")
   ```

7. **Evaluate criteria:**
   - ✅ No executor crashes
   - ✅ AI decisions look reasonable
   - ✅ Shadow logs populated correctly
   - ✅ No obvious bugs in decision logic

**If criteria NOT met**: Stay in SHADOW, fix issues, restart Step 1.

**If criteria met**: Proceed to Step 2.

---

### Step 2: Pre-LIVE Dry Run (Optional but Recommended)

**Goal**: Test LIVE config without actually enabling LIVE behavior.

1. **Update environment:**
   ```bash
   export EXIT_MODE=EXIT_BRAIN_V3
   export EXIT_EXECUTOR_MODE=LIVE  # Changed!
   export EXIT_BRAIN_V3_LIVE_ROLLOUT=DISABLED  # Still disabled (safety)
   ```

2. **Restart backend.**

3. **Verify behavior:**
   ```bash
   python backend/tools/print_exit_status.py
   ```

   **Expected:**
   - `Operational State: EXIT_BRAIN_V3_SHADOW` (NOT LIVE)
   - `EXIT_EXECUTOR_MODE: LIVE`
   - `EXIT_BRAIN_V3_LIVE_ROLLOUT: DISABLED`
   - Executor still logs to shadow, doesn't place orders

   **Log should show:**
   ```
   [EXIT_BRAIN_EXECUTOR] ⚠️  EXIT_EXECUTOR_MODE=LIVE but EXIT_BRAIN_V3_LIVE_ROLLOUT=DISABLED. Forcing SHADOW mode for safety.
   ```

4. **Confirm fallback logic works**: Even with `EXIT_EXECUTOR_MODE=LIVE`, the system stays in SHADOW because `EXIT_BRAIN_V3_LIVE_ROLLOUT=DISABLED`.

**This validates the kill-switch works correctly.**

---

### Step 3: Enable LIVE Mode 🔴

**Goal**: Give AI full control of exit orders.

⚠️ **CRITICAL**: Only proceed if:
- Step 1 completed successfully (24-48h shadow validation)
- AI decisions confirmed to improve outcomes
- Team monitoring and ready for rollback
- Off-hours or low-volume period preferred

1. **Enable the rollout flag:**
   ```bash
   export EXIT_MODE=EXIT_BRAIN_V3
   export EXIT_EXECUTOR_MODE=LIVE
   export EXIT_BRAIN_V3_LIVE_ROLLOUT=ENABLED  # 🔴 The critical toggle!
   ```

2. **Update .env file:**
   ```
   EXIT_MODE=EXIT_BRAIN_V3
   EXIT_EXECUTOR_MODE=LIVE
   EXIT_BRAIN_V3_LIVE_ROLLOUT=ENABLED
   EXIT_BRAIN_V3_ENABLED=true
   ```

3. **Restart backend:**
   ```bash
   # Stop backend
   pkill -f "python.*main.py"
   
   # Start backend (with close monitoring)
   python backend/main.py
   ```

4. **Verify LIVE activation:**
   ```bash
   # Check logs immediately
   tail -f backend/logs/quantum_trader.log | grep EXIT
   
   # Check diagnostic endpoint
   curl http://localhost:8000/health/exit_brain_status | jq
   
   # Or CLI tool
   python backend/tools/print_exit_status.py
   ```

   **Expected output:**
   ```
   [EXIT_MODE] 🔴 EXIT BRAIN V3 LIVE MODE ACTIVE 🔴 AI executor will place real orders. Legacy exit modules will be blocked.
   [EXIT_BRAIN_EXECUTOR] 🔴 LIVE MODE ACTIVE 🔴 - AI will place real orders via exit_order_gateway. Legacy modules will be blocked.
   [EXIT_BRAIN_EXECUTOR] Starting monitoring loop (interval=10.0s, mode=LIVE)
   ```

   **Diagnostic should show:**
   ```
   🔴 Operational State: EXIT_BRAIN_V3_LIVE
   📋 Configuration:
     EXIT_MODE: EXIT_BRAIN_V3
     EXIT_EXECUTOR_MODE: LIVE
     EXIT_BRAIN_V3_LIVE_ROLLOUT: ENABLED
     Live Mode Active: True
   ```

5. **Monitor closely for first 1-2 hours:**
   
   **Watch for:**
   - `[EXIT_BRAIN_LIVE]` logs showing order execution
   - `[EXIT_GUARD] ✅ Exit Brain module 'exit_executor' placing ...`
   - `[EXIT_GUARD] 🛑 BLOCKED: Legacy module ...` (expected, shows blocking works)
   - `[EXIT_METRICS]` summaries showing executor orders + blocked legacy

   **Example healthy logs:**
   ```
   [EXIT_BRAIN_LIVE] Executing move_sl for BTCUSDT long: SL adjustment (breakeven): pnl=2.10%
   [EXIT_GUARD] ✅ Exit Brain module 'exit_executor' placing sl for BTCUSDT (mode=LIVE).
   [EXIT_GATEWAY] 📤 Submitting sl order: module=exit_executor, symbol=BTCUSDT, type=STOP_MARKET
   [EXIT_GATEWAY] ✅ Order placed successfully: module=exit_executor, symbol=BTCUSDT, order_id=12345678, kind=sl
   
   [EXIT_GUARD] 🛑 BLOCKED: Legacy module 'position_monitor' attempted to place sl for ETHUSDT in EXIT_BRAIN_V3 LIVE mode. Executor is the single MUSCLE. Order NOT sent to exchange.
   ```

6. **Check metrics periodically:**
   ```bash
   curl http://localhost:8000/health/exit_brain_status | jq '.metrics'
   ```

   **Healthy metrics:**
   ```json
   {
     "total_exit_orders": 45,
     "blocked_legacy_orders": 12,
     "ownership_conflicts": 12,
     "orders_by_module": {
       "exit_executor": 33,
       "position_monitor": 0,  // Blocked
       "hybrid_tpsl": 0        // Blocked
     },
     "orders_by_kind": {
       "sl": 15,
       "tp": 12,
       "partial_tp": 6
     }
   }
   ```

   **Red flags:**
   - `exit_executor` count is 0 (executor not working)
   - `blocked_legacy_orders` is 0 but legacy modules have counts (blocking not working)
   - High ownership conflicts but low blocked count (logic error)

7. **Extended monitoring (6-12 hours):**
   - Compare AI exits vs shadow log predictions
   - Monitor position PnL trajectories
   - Check for any errors or exceptions
   - Verify no dual positions (old bug)

---

### Step 4: Emergency Rollback 🚨

If ANY issue occurs (executor crash, bad decisions, unexpected behavior), **immediately rollback**.

#### Quick Rollback Option A: Kill-Switch (Fastest)

1. **Disable rollout flag:**
   ```bash
   export EXIT_BRAIN_V3_LIVE_ROLLOUT=DISABLED
   ```

2. **Restart backend:**
   ```bash
   pkill -f "python.*main.py"
   python backend/main.py
   ```

3. **Verify:**
   ```bash
   python backend/tools/print_exit_status.py
   ```

   **Should show:** `EXIT_BRAIN_V3_SHADOW` (forced fallback)

**Executor continues running but in SHADOW mode. Legacy modules take over exits.**

#### Quick Rollback Option B: Back to SHADOW

1. **Change executor mode:**
   ```bash
   export EXIT_EXECUTOR_MODE=SHADOW
   ```

2. **Restart backend.**

3. **Verify shadow mode active.**

#### Full Rollback Option C: Back to LEGACY

1. **Switch to LEGACY mode:**
   ```bash
   export EXIT_MODE=LEGACY
   ```

2. **Restart backend.**

3. **Verify:**
   - Executor NOT started
   - Legacy modules controlling exits
   - No EXIT_BRAIN logs

#### Rollback Validation Checklist

After any rollback:
- ✅ Check operational state: NOT LIVE
- ✅ Verify open positions have TP/SL protection
- ✅ Monitor for 30+ minutes to confirm stability
- ✅ Review logs to understand what triggered rollback
- ✅ Fix issue before attempting LIVE again

---

## 🔍 Verification & Monitoring

### Health Check Endpoint

**URL**: `GET /health/exit_brain_status`

**Response:**
```json
{
  "timestamp": "2025-12-10T22:15:00Z",
  "config": {
    "exit_mode": "EXIT_BRAIN_V3",
    "exit_executor_mode": "LIVE",
    "exit_brain_live_rollout": "ENABLED",
    "live_mode_active": true
  },
  "executor": {
    "running": true,
    "effective_mode": "LIVE",
    "last_decision_timestamp": "2025-12-10T22:14:55Z"
  },
  "metrics": {
    "total_exit_orders": 67,
    "blocked_legacy_orders": 18,
    "ownership_conflicts": 18,
    "orders_by_module": {
      "exit_executor": 49
    },
    "orders_by_kind": {
      "sl": 20,
      "tp": 18,
      "partial_tp": 11
    }
  },
  "operational_state": "EXIT_BRAIN_V3_LIVE"
}
```

### CLI Diagnostic Tool

**Usage:**
```bash
python backend/tools/print_exit_status.py
```

**Output:**
```
============================================================
EXIT BRAIN V3 STATUS
============================================================

🔴 Operational State: EXIT_BRAIN_V3_LIVE

📋 Configuration:
  EXIT_MODE: EXIT_BRAIN_V3
  EXIT_EXECUTOR_MODE: LIVE
  EXIT_BRAIN_V3_LIVE_ROLLOUT: ENABLED
  Live Mode Active: True

🤖 Executor:
  Running: True
  Effective Mode: LIVE
  Last Decision: 2025-12-10T22:14:55Z

📊 Metrics:
  Total Exit Orders: 67
  Blocked Legacy Orders: 18
  Ownership Conflicts: 18

  Orders by Module:
    exit_executor: 49

  Orders by Kind:
    sl: 20
    tp: 18
    partial_tp: 11

============================================================

⚠️  WARNING: EXIT BRAIN V3 LIVE MODE ACTIVE
  AI executor is placing real orders.
  Legacy exit modules are blocked.

✅ Blocked 18 legacy module orders (expected in LIVE mode)
```

### Key Logs to Monitor

#### Startup Logs
```
[EXIT_MODE] Configuration loaded: EXIT_MODE=EXIT_BRAIN_V3, EXIT_EXECUTOR_MODE=LIVE, EXIT_BRAIN_V3_LIVE_ROLLOUT=ENABLED
[EXIT_MODE] 🔴 EXIT BRAIN V3 LIVE MODE ACTIVE 🔴 AI executor will place real orders. Legacy exit modules will be blocked.
[EXIT_BRAIN_EXECUTOR] Initialized in LIVE MODE - Config: EXIT_MODE=EXIT_BRAIN_V3, EXIT_EXECUTOR_MODE=LIVE, EXIT_BRAIN_V3_LIVE_ROLLOUT=ENABLED
[EXIT_BRAIN_EXECUTOR] 🔴 LIVE MODE ACTIVE 🔴 - AI will place real orders via exit_order_gateway. Legacy modules will be blocked.
```

#### Runtime Logs (SHADOW)
```
[EXIT_BRAIN_SHADOW] BTCUSDT long size=1.5000 pnl=+2.34% type=move_sl new_sl=$96500.0000 sl_reason=breakeven reason='SL adjustment (breakeven): pnl=2.34%' conf=0.80
[EXIT_GUARD] 🚨 OWNERSHIP CONFLICT: Legacy module 'position_monitor' placing sl (STOP_MARKET) for BTCUSDT in EXIT_BRAIN_V3 mode.
```

#### Runtime Logs (LIVE)
```
[EXIT_BRAIN_LIVE] Executing move_sl for BTCUSDT long: SL adjustment (breakeven): pnl=2.10%
[EXIT_GUARD] ✅ Exit Brain module 'exit_executor' placing sl for BTCUSDT (mode=LIVE).
[EXIT_GATEWAY] 📤 Submitting sl order: module=exit_executor, symbol=BTCUSDT, type=STOP_MARKET
[EXIT_GATEWAY] ✅ Order placed successfully: module=exit_executor, symbol=BTCUSDT, order_id=12345678, kind=sl

[EXIT_GUARD] 🛑 BLOCKED: Legacy module 'position_monitor' attempted to place sl for ETHUSDT in EXIT_BRAIN_V3 LIVE mode. Executor is the single MUSCLE. Order NOT sent to exchange.
```

#### Metrics Logs
```
[EXIT_METRICS] Summary: total_orders=67, conflicts=18, blocked=18, mode=EXIT_BRAIN_V3 (LIVE)
[EXIT_METRICS] Orders by module: {'exit_executor': 49}
[EXIT_METRICS] Orders by kind: {'sl': 20, 'tp': 18, 'partial_tp': 11}
[EXIT_METRICS] 🛑 Blocked 18 legacy module orders in EXIT_BRAIN_V3 LIVE mode (expected behavior).
```

---

## 📝 Environment Variable Reference

### Complete .env Example (LEGACY)
```bash
# Exit Brain v3 - LEGACY Mode
EXIT_MODE=LEGACY
EXIT_EXECUTOR_MODE=SHADOW
EXIT_BRAIN_V3_LIVE_ROLLOUT=DISABLED
EXIT_BRAIN_V3_ENABLED=false
```

### Complete .env Example (SHADOW)
```bash
# Exit Brain v3 - SHADOW Mode (AI observes, legacy controls)
EXIT_MODE=EXIT_BRAIN_V3
EXIT_EXECUTOR_MODE=SHADOW
EXIT_BRAIN_V3_LIVE_ROLLOUT=DISABLED
EXIT_BRAIN_V3_ENABLED=true
```

### Complete .env Example (LIVE) 🔴
```bash
# Exit Brain v3 - LIVE Mode (AI controls, legacy blocked)
EXIT_MODE=EXIT_BRAIN_V3
EXIT_EXECUTOR_MODE=LIVE
EXIT_BRAIN_V3_LIVE_ROLLOUT=ENABLED  # 🔴 CRITICAL!
EXIT_BRAIN_V3_ENABLED=true
```

### Variable Descriptions

| Variable | Values | Description |
|----------|--------|-------------|
| `EXIT_MODE` | `LEGACY` \| `EXIT_BRAIN_V3` | Which system owns exits |
| `EXIT_EXECUTOR_MODE` | `SHADOW` \| `LIVE` | Executor behavior (only matters if EXIT_MODE=EXIT_BRAIN_V3) |
| `EXIT_BRAIN_V3_LIVE_ROLLOUT` | `DISABLED` \| `ENABLED` | Safety kill-switch for LIVE mode |
| `EXIT_BRAIN_V3_ENABLED` | `true` \| `false` | Legacy flag for consistency checks |

---

## ✅ Success Criteria

### For SHADOW Mode (Step 1)
- ✅ Executor starts without errors
- ✅ `[EXIT_BRAIN_SHADOW]` logs appear every ~10s when positions open
- ✅ Shadow log file (`backend/data/exit_brain_shadow.jsonl`) populated
- ✅ AI decisions look reasonable (no obvious bugs)
- ✅ No executor crashes over 24-48h period

### For LIVE Mode (Step 3)
- ✅ Health endpoint shows `"operational_state": "EXIT_BRAIN_V3_LIVE"`
- ✅ `[EXIT_BRAIN_LIVE]` logs show order execution
- ✅ `[EXIT_GUARD] ✅` logs confirm executor orders allowed
- ✅ `[EXIT_GUARD] 🛑 BLOCKED` logs confirm legacy orders blocked
- ✅ Metrics show `exit_executor` as dominant module
- ✅ Metrics show `blocked_legacy_orders` > 0
- ✅ No dual positions (LONG + SHORT on same symbol)
- ✅ All positions have TP/SL protection
- ✅ PnL trajectory matches or exceeds legacy performance

---

## ⚠️ Known Gotchas & Troubleshooting

### Issue: Executor starts but mode is still SHADOW
**Symptom:** Logs show SHADOW despite setting LIVE config.

**Cause:** `EXIT_BRAIN_V3_LIVE_ROLLOUT` still `DISABLED`.

**Fix:**
```bash
export EXIT_BRAIN_V3_LIVE_ROLLOUT=ENABLED
# Restart backend
```

### Issue: Legacy modules still placing orders in LIVE mode
**Symptom:** No blocked orders in metrics.

**Cause:** Gateway blocking logic not triggered.

**Check:**
1. Verify `is_exit_brain_live_fully_enabled()` returns True
2. Check gateway logs for `[EXIT_GUARD] 🛑 BLOCKED` messages
3. Verify module names match `LEGACY_MODULES` list

**Fix:** Check `backend/services/execution/exit_order_gateway.py` blocking logic.

### Issue: Executor crashes on startup
**Symptom:** Executor starts then immediately stops.

**Check:**
1. Binance API credentials set correctly
2. Position source (Binance client) initialized
3. ExitBrainAdapter dependencies available
4. Check full stack trace in logs

**Fix:** Address specific error in stack trace.

### Issue: No shadow logs appearing
**Symptom:** Executor running but no `[EXIT_BRAIN_SHADOW]` logs.

**Cause:** No open positions, or positions not detected.

**Check:**
1. Are there actually open positions?
2. Check `_get_open_positions()` returns non-empty list
3. Verify Binance API connectivity

**Fix:** Open test position or fix position detection.

### Issue: Health endpoint 500 error
**Symptom:** `/health/exit_brain_status` returns 500.

**Cause:** Diagnostic module import error or app.state not available.

**Check:**
1. `backend/diagnostics/exit_brain_status.py` exists
2. No import errors in that module
3. Backend fully started

**Fix:** Check error in backend logs, fix import issues.

---

## 📞 Support & Emergency Contacts

### Rollback Priority
If issues occur:
1. **Immediate**: Set `EXIT_BRAIN_V3_LIVE_ROLLOUT=DISABLED` (fastest)
2. **Quick**: Set `EXIT_EXECUTOR_MODE=SHADOW`
3. **Full**: Set `EXIT_MODE=LEGACY`

### Log Files
- Main log: `backend/logs/quantum_trader.log`
- Shadow log: `backend/data/exit_brain_shadow.jsonl`

### Monitoring Commands
```bash
# Real-time logs
tail -f backend/logs/quantum_trader.log | grep EXIT

# Health check
curl http://localhost:8000/health/exit_brain_status | jq

# CLI status
python backend/tools/print_exit_status.py

# Metrics summary (search logs)
grep "EXIT_METRICS" backend/logs/quantum_trader.log | tail -20
```

---

## 📚 Additional Resources

- **Architecture Doc**: `AI_EXIT_BRAIN_PHASE2A_COMPLETE.md`
- **Config Module**: `backend/config/exit_mode.py`
- **Gateway**: `backend/services/execution/exit_order_gateway.py`
- **Executor**: `backend/domains/exits/exit_brain_v3/dynamic_executor.py`
- **Diagnostics**: `backend/diagnostics/exit_brain_status.py`

---

**Last Updated**: December 10, 2025  
**Version**: 1.0  
**Status**: Production Ready
