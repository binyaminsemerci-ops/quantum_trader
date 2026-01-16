# RL v3 Live Orchestrator - Production Verification

## Patches Applied

### ✅ PATCH 1: PolicyStore Configuration
**File**: `data/policy_snapshot.json`
```json
"rl_v3_live": {
  "enabled": true,
  "mode": "SHADOW",
  "min_confidence": 0.6,
  "max_trades_per_hour": 10,
  "max_size_pct": 0.15,              // Max 15% portfolio per trade
  "max_margin_alloc_pct": 0.25,      // Max 25% margin allocation
  "liq_buffer_pct": 0.20,            // 20% buffer before liquidation
  "max_loss_per_trade_pct": 0.02,    // Max 2% loss per trade
  "check_open_positions": true,      // Prevent double intents
  "promotion_requires_ack": true     // Require ACK before PRIMARY/HYBRID
}
```

### ✅ PATCH 2: RiskGuard - evaluate_trade_intent()
**File**: `backend/services/risk/risk_guard.py`

**NEW METHOD**: `async def evaluate_trade_intent(trade_intent, trace_id)`

**Checks**:
- ✅ Input validation (symbol, side, size_pct)
- ✅ Size limit from policy (max_size_pct)
- ✅ Leverage limit from active risk profile
- ✅ Liquidation buffer (margin_usage vs liq_buffer_pct)
- ✅ Structured logging with trace_id

**CRITICAL**: RL v3 does NO futures math. All risk calculations delegated to RiskGuard.

### ✅ PATCH 3: RL v3 Live Orchestrator - Production Ready
**File**: `backend/services/ai/rl_v3_live_orchestrator.py`

**PRODUCTION GUARDS**:
- **GUARD A**: Check open position per symbol → prevent double intent
- **GUARD B**: Size limit check (size_pct <= max_size_pct)
- **GUARD C**: Rate limiting (max_trades_per_hour)

**SHADOW PROTECTION**:
- SHADOW mode NEVER publishes trade.intent
- Records metrics only for analysis
- Explicit log: "🔒 SHADOW mode - recording metrics only, NO trade intent published"

**PROMOTION SAFETY**:
- PRIMARY/HYBRID requires `_promotion_acked = True`
- Promotion blocked until RiskGuard+ExitBrain verify system
- Method: `async def promote_to_live(mode)` with ACK flow

**CONFIG LOGGING**:
- Logs config source: "📋 Config loaded from policy" vs "default"
- Shows mode, enabled, max_size_pct at startup

**OPEN INTENT TRACKING**:
- `_open_intents` set tracks symbols with active intents
- Auto-cleanup after 60s timeout
- Manual clear via `clear_open_intent(symbol)`

---

## Verification Commands

### ✅ CHECK 1: RL v3 Reads Policy Config

```powershell
# Check that config is loaded from policy (not default)
journalctl -u quantum_backend.service 2>&1 | Select-String -Pattern "Config loaded from policy|Config loaded from default" | Select-Object -Last 3
```

**Expected Output**:
```
📋 Config loaded from policy: mode=SHADOW, enabled=True, max_size_pct=0.15
```

**FAIL if**: Shows "Config loaded from default" → policy not loaded

---

### ✅ CHECK 2: SHADOW Mode Does NOT Publish Intents

```powershell
# Verify SHADOW mode logging
journalctl -u quantum_backend.service 2>&1 | Select-String -Pattern "SHADOW mode.*NO trade intent|SHADOW mode - recording metrics only" | Select-Object -Last 5
```

**Expected Output**:
```
🔒 SHADOW mode - recording metrics only, NO trade intent published
```

**Then verify NO trade.intent published**:
```powershell
# Should be ZERO results in SHADOW mode
journalctl -u quantum_backend.service 2>&1 | Select-String -Pattern "Trade intent published" | Measure-Object
```

**Expected**: Count = 0 (no intents published in SHADOW)

**FAIL if**: Any "Trade intent published" messages in SHADOW mode

---

### ✅ CHECK 3: Guards Prevent Invalid Trades

```powershell
# Check GUARD A: Open position prevention
journalctl -u quantum_backend.service 2>&1 | Select-String -Pattern "GUARD A.*Open intent already exists" | Select-Object -Last 3

# Check GUARD B: Size limit enforcement
journalctl -u quantum_backend.service 2>&1 | Select-String -Pattern "GUARD B.*exceeds policy limit" | Select-Object -Last 3

# Check GUARD C: Rate limiting
journalctl -u quantum_backend.service 2>&1 | Select-String -Pattern "GUARD C.*Rate limit exceeded" | Select-Object -Last 3
```

**Expected**: Guards trigger when limits exceeded (if traffic exists)

**FAIL if**: Trades bypass guards (no guard logs when they should trigger)

---

### ✅ CHECK 4: RiskGuard Integration Works

```powershell
# Verify RiskGuard.evaluate_trade_intent() is called
journalctl -u quantum_backend.service 2>&1 | Select-String -Pattern "Trade intent approved|RiskGuard denied" | Select-Object -Last 5
```

**Expected Output** (when in PRIMARY mode with traffic):
```
[RiskGuard] ✅ Trade intent approved: BTCUSDT LONG 0.10 5x
```

**FAIL if**: No RiskGuard logs → integration broken

---

### ✅ CHECK 5: Promotion Safety Blocks PRIMARY

```powershell
# Check promotion lock is active
journalctl -u quantum_backend.service 2>&1 | Select-String -Pattern "PROMOTION BLOCKED.*requires ACK" | Select-Object -Last 3
```

**Expected Output** (if attempted to go PRIMARY without ACK):
```
🚨 PROMOTION BLOCKED: mode=PRIMARY requires ACK from RiskGuard+ExitBrain
```

**To manually promote** (ONLY after verification):
```python
# In Python shell:
orchestrator = app.state.rl_v3_live_orchestrator
orchestrator._promotion_acked = True  # Manual ACK
success, msg = await orchestrator.promote_to_live("PRIMARY")
print(msg)
```

---

## Quick Health Check (All-in-One)

```powershell
# Run all checks in sequence
Write-Host "`n=== CHECK 1: Config Source ===`n"
journalctl -u quantum_backend.service 2>&1 | Select-String -Pattern "Config loaded from" | Select-Object -Last 1

Write-Host "`n=== CHECK 2: SHADOW Protection ===`n"
journalctl -u quantum_backend.service 2>&1 | Select-String -Pattern "SHADOW mode.*NO trade intent" | Select-Object -Last 3
journalctl -u quantum_backend.service 2>&1 | Select-String -Pattern "Trade intent published" | Measure-Object | Select-Object -ExpandProperty Count | ForEach-Object { Write-Host "Intents Published: $_" }

Write-Host "`n=== CHECK 3: Guards Active ===`n"
journalctl -u quantum_backend.service 2>&1 | Select-String -Pattern "GUARD [ABC]" | Select-Object -Last 5

Write-Host "`n=== CHECK 4: RiskGuard Integration ===`n"
journalctl -u quantum_backend.service 2>&1 | Select-String -Pattern "evaluate_trade_intent|Trade intent approved" | Select-Object -Last 3

Write-Host "`n=== CHECK 5: Orchestrator Status ===`n"
journalctl -u quantum_backend.service 2>&1 | Select-String -Pattern "RL v3 Live Orchestrator started" | Select-Object -Last 1
```

---

## Post-Verification Actions

### If All Checks Pass ✅

1. **Monitor SHADOW mode for 2-4 hours**:
   ```powershell
   # Watch decision quality
   docker logs -f quantum_backend 2>&1 | Select-String "rl_v3_orchestrator"
   ```

2. **Analyze shadow metrics**:
   - Check confidence distribution
   - Verify action decisions align with signals
   - Confirm no crashes or exceptions

3. **Prepare for promotion**:
   - Set `_promotion_acked = True` manually
   - Test with 1 symbol first (limit exposure)
   - Monitor execution closely

### If Any Check Fails ❌

1. **STOP**: Do not promote to PRIMARY/HYBRID
2. **Debug**: Check specific failure pattern
3. **Fix**: Apply targeted patch
4. **Re-verify**: Run all checks again

---

## Critical Safety Rules

### 🚨 NEVER:
- Skip verification before PRIMARY promotion
- Disable guards or safety checks
- Set mode=PRIMARY without _promotion_acked
- Run PRIMARY without monitoring

### ✅ ALWAYS:
- Monitor SHADOW for 2+ hours first
- Verify all 5 checks pass
- Have manual kill switch ready
- Watch RiskGuard denials closely

---

## Emergency Rollback

If RL v3 Live causes issues in PRIMARY/HYBRID:

```python
# Instant rollback to SHADOW
orchestrator = app.state.rl_v3_live_orchestrator
policy = await orchestrator.policy_store.get_policy()
policy.rl_v3_live["mode"] = "SHADOW"
await orchestrator.policy_store.update_policy(policy)
orchestrator._config_cache = {}  # Force reload
```

**OR** via PolicyStore directly:
```json
# Edit data/policy_snapshot.json
"rl_v3_live": {
  "mode": "SHADOW",  // ← Change back to SHADOW
  ...
}
```

Then restart: `docker restart quantum_backend`

---

## Summary

**Production-ready when**:
- ✅ All 5 verification checks pass
- ✅ SHADOW mode runs clean for 2+ hours
- ✅ No guard violations or RiskGuard denials
- ✅ Manual ACK given after team review

**Blocking issues**:
- ❌ Config not loaded from policy
- ❌ Trade intents published in SHADOW
- ❌ Guards not triggering
- ❌ RiskGuard integration broken

**Current status**: SHADOW mode active, awaiting verification results.

