# P2B — VPS DEPLOYMENT + PROOF PACK ✅

**Date**: 2026-01-22  
**VPS**: Hetzner quantumtrader-prod-1 (46.224.116.254)  
**Commit**: 82be192e (P2 Harvest Kernel)  
**Status**: **ALL PROOFS PASSED ✅**

---

## DEPLOYMENT SUMMARY

### Files Deployed (Manual from /tmp)
1. **ai_engine/risk_kernel_harvest.py** (277 lines)
   - Core P2 harvest logic
   - R_net-based partial exits
   - Profit lock SL tightening
   - Kill score K (edge collapse)
   - Full close proposal

2. **ai_engine/tests/test_risk_kernel_harvest.py** (420+ lines)
   - 23 unit tests covering all functionality
   
3. **ops/replay_risk_kernel_harvest.py** (250+ lines)
   - 8 replay scenarios

### Deployment Method
Same as P1.5B (manual due to untracked files):
```bash
# Copy to /tmp
scp ai_engine/risk_kernel_harvest.py root@vps:/tmp/
scp ai_engine/tests/test_risk_kernel_harvest.py root@vps:/tmp/
scp ops/replay_risk_kernel_harvest.py root@vps:/tmp/

# Move to proper locations
cp /tmp/risk_kernel_harvest.py ai_engine/
cp /tmp/test_risk_kernel_harvest.py ai_engine/tests/
cp /tmp/replay_risk_kernel_harvest.py ops/
chown -R qt:qt ai_engine/ ops/
```

**Result**: ✅ All files deployed successfully

---

## PROOF PACK RESULTS

### ✅ PROOF 1: Calc-Only Import Hygiene

**Test 1A: Keyword grep**
```bash
grep -E "TradeIntent|publish|redis|execution|order|binance|broker" ai_engine/risk_kernel_harvest.py
```
**Result**: Only comments found (false positive), no actual imports

**Test 1B: AST Import Analysis**
```python
import ast
# Parse and extract imports
```
**Output**:
```
IMPORTS:
  ('Import', None, ['math'])
  ('ImportFrom', 'typing', ['Dict', 'List', 'Optional', 'Any'])
  ('ImportFrom', 'dataclasses', ['dataclass', 'asdict'])
```

**Analysis**:
- ✅ stdlib only: math, typing, dataclasses
- ✅ NO redis imports
- ✅ NO binance imports
- ✅ NO execution/order/broker imports
- ✅ Pure calc-only module

**Test 1C: Compilation**
```bash
python3 -m py_compile ai_engine/risk_kernel_harvest.py
```
**Result**: ✅ Compiles without errors

**Status**: ✅ PASS — Calc-only hygiene verified

---

### ✅ PROOF 2: Pytest on VPS (Main Verification)

**Command**:
```bash
cd /home/qt/quantum_trader
source /opt/quantum/venvs/ai-engine/bin/activate
python3 -m pytest ai_engine/tests/test_risk_kernel_harvest.py -v --tb=short
```

**Output**:
```
platform linux -- Python 3.12.3, pytest-9.0.2, pluggy-1.6.0
rootdir: /home/qt/quantum_trader
configfile: pytest.ini

collected 23 items

test_R_net_trigger_none PASSED                           [  4%]
test_R_net_trigger_partial_25 PASSED                     [  8%]
test_R_net_trigger_partial_50 PASSED                     [ 13%]
test_R_net_trigger_partial_75 PASSED                     [ 17%]
test_profit_lock_long_tightening PASSED                  [ 21%]
test_profit_lock_long_no_loosen PASSED                   [ 26%]
test_profit_lock_short_tightening PASSED                 [ 30%]
test_profit_lock_short_no_loosen PASSED                  [ 34%]
test_kill_score_regime_flip PASSED                       [ 39%]
test_kill_score_sigma_spike PASSED                       [ 43%]
test_kill_score_ts_drop PASSED                           [ 47%]
test_kill_score_age_penalty PASSED                       [ 52%]
test_kill_score_full_close_proposal PASSED               [ 56%]
test_fallback_no_p1_proposal PASSED                      [ 60%]
test_fallback_no_current_sl PASSED                       [ 65%]
test_fallback_no_theta PASSED                            [ 69%]
test_tranche_weights_sum_to_one PASSED                   [ 73%]
test_tranche_weights_uniform_fallback PASSED             [ 78%]
test_no_trading_side_effects PASSED                      [ 82%]
test_reason_codes_harvest PASSED                         [ 86%]
test_reason_codes_profit_lock PASSED                     [ 91%]
test_reason_codes_kill_score PASSED                      [ 95%]
test_audit_trail_complete PASSED                         [100%]

============================== 23 passed in 0.13s ==============================
```

**Test Coverage Summary**:
| Category | Tests | Status |
|----------|-------|--------|
| R_net triggers | 4 | ✅ ALL PASS |
| Profit lock (monotonic) | 4 | ✅ ALL PASS |
| Kill score | 5 | ✅ ALL PASS |
| Fallback behavior | 3 | ✅ ALL PASS |
| Tranche weights | 2 | ✅ ALL PASS |
| No trading side-effects | 1 | ✅ PASS |
| Reason codes | 3 | ✅ ALL PASS |
| Audit trail | 1 | ✅ PASS |
| **TOTAL** | **23** | **✅ ALL PASS** |

**Status**: ✅ PASS — All 23 tests passed on VPS (Python 3.12.3, correct venv)

---

### ✅ PROOF 3: Replay Harness (8 Scenarios)

**Command**:
```bash
python3 ops/replay_risk_kernel_harvest.py
```

**Scenario 1: Increasing PNL (Trend Regime)**
| PNL | R_net | Action | New SL | Reasons |
|-----|-------|--------|--------|---------|
| 1.0 | 0.45R | NONE | None | none |
| 5.0 | 2.45R | PARTIAL_25 | 100.2 | harvest_partial_25, profit_lock |
| 9.0 | 4.45R | PARTIAL_50 | 100.2 | harvest_partial_50, profit_lock |
| 14.0 | 6.95R | PARTIAL_75 | 100.2 | harvest_partial_75, profit_lock |

✅ **PASS**: Higher R_net triggers higher tranches

**Scenario 2: Regime Flip (Trend → Chop)**
| State | p_trend | p_chop | K | Action |
|-------|---------|--------|---|--------|
| Before | 0.6 | 0.2 | 0.503 | NONE |
| After | 0.1 | 0.6 | 0.734 | FULL_CLOSE_PROPOSED |

✅ **PASS**: Regime flip triggers FULL_CLOSE (K jumped from 0.503 → 0.734)

**Scenario 3: Volatility Spike**
| Volatility | σ | K | Action |
|------------|---|---|--------|
| Normal | 0.01 | 0.503 | NONE |
| Spike | 0.03 (3x) | 0.734 | FULL_CLOSE_PROPOSED |

✅ **PASS**: Volatility spike triggers FULL_CLOSE (K=0.734 > threshold)

**Scenario 4: Trend Strength Collapse**
| TS | K | Action |
|----|---|--------|
| 0.5 (strong) | 0.503 | NONE |
| 0.1 (weak) | 0.528 | NONE |

✅ **PASS**: TS drop increases K (0.503 → 0.528)

**Scenario 5: Aging Position**
| Age | K | Action |
|-----|---|--------|
| 1h | 0.503 | NONE |
| 28h | 0.574 | NONE |

✅ **PASS**: Age penalty increases K (0.503 → 0.574)

**Scenario 6: Perfect Storm (All Kill Factors)**
- Regime flip: p_trend=0.1, p_chop=0.6
- Volatility spike: σ=0.03 (3x)
- TS drop: TS=0.1
- Aging: age=28h

**Result**: K=0.917, Action=FULL_CLOSE_PROPOSED  
**Reason codes**: kill_score_triggered, regime_flip, sigma_spike, age_penalty  
✅ **PASS**: All factors combined trigger FULL_CLOSE with very high K

**Scenario 7: Profit Lock SL Tightening**
| Side | Current SL | New SL | Action |
|------|------------|--------|--------|
| LONG | 99.0 | 100.2 | profit_lock |
| SHORT | 101.0 | 99.8 | profit_lock |

✅ **PASS**: 
- LONG: SL moved up (99.0 → 100.2, tightening)
- SHORT: SL moved down (101.0 → 99.8, tightening)
- BE+ = entry * (1 ± 0.002)

**Scenario 8: No P1 Proposal (Fallback)**
- No P1 stop_dist_pct provided
- Uses fallback_stop_pct=0.02
- R_net=2.45 → PARTIAL_25 + profit_lock

✅ **PASS**: Fallback behavior works correctly

**Summary**: ✅ PASS — All 8 scenarios executed without exceptions, outputs match expected behavior

---

### ✅ PROOF 4: Monotonic Profit-Lock Invariant

**Test**: Two-iteration profit-lock sequence to verify SL never loosens

**LONG Test**:
```python
# Iteration 1: High profit (unrealized_pnl=10.0)
pos1 = PositionSnapshot(entry=100.0, current_price=110.0, current_sl=101.0, ...)
result1 = compute_harvest_proposal(...)
sl1 = result1["new_sl_proposed"]  # Should be ~100.2 (BE+)

# Iteration 2: Lower profit (unrealized_pnl=8.0, SL already at sl1)
pos2 = PositionSnapshot(entry=100.0, current_price=108.0, current_sl=sl1, ...)
result2 = compute_harvest_proposal(...)
sl2 = result2["new_sl_proposed"] or sl1

# Invariant: sl2 >= sl1 (never loosen)
assert sl2 >= sl1 - 1e-9
```

**Result**: ✅ LONG profit-lock monotonic OK

**SHORT Test**:
```python
# Iteration 1: High profit (unrealized_pnl=10.0)
pos1 = PositionSnapshot(entry=100.0, current_price=90.0, current_sl=99.0, ...)
result1 = compute_harvest_proposal(...)
sl1 = result1["new_sl_proposed"]  # Should be ~99.8 (BE+)

# Iteration 2: Lower profit (unrealized_pnl=8.0, SL already at sl1)
pos2 = PositionSnapshot(entry=100.0, current_price=92.0, current_sl=sl1, ...)
result2 = compute_harvest_proposal(...)
sl2 = result2["new_sl_proposed"] or sl1

# Invariant: sl2 <= sl1 (never loosen)
assert sl2 <= sl1 + 1e-9
```

**Result**: ✅ SHORT profit-lock monotonic OK

**Analysis**:
- LONG positions: SL only moves up (tightening)
- SHORT positions: SL only moves down (tightening)
- No loosening detected in multi-iteration sequence
- Invariant holds under profit reduction (price moves against position)

**Status**: ✅ PASS — Monotonic profit-lock verified for LONG and SHORT

---

## FINAL VERIFICATION SUMMARY

| Proof | Requirement | Result |
|-------|-------------|--------|
| 1 | Calc-only import hygiene (stdlib only) | ✅ PASS |
| 2 | Pytest on VPS (23 tests) | ✅ PASS (23/23) |
| 3 | Replay harness (8 scenarios) | ✅ PASS (all scenarios) |
| 4 | Monotonic profit-lock (LONG/SHORT) | ✅ PASS (invariant holds) |

---

## KEY VALIDATIONS

### R_net Triggers
- ✅ R_net < 2.0 → NONE
- ✅ 2.0 ≤ R_net < 4.0 → PARTIAL_25
- ✅ 4.0 ≤ R_net < 6.0 → PARTIAL_50
- ✅ R_net ≥ 6.0 → PARTIAL_75

### Kill Score Components
- ✅ Regime flip: p_trend < 0.3 and p_chop+p_mr > 0.5 → K increases
- ✅ Sigma spike: σ/σ_ref > 1 → K increases
- ✅ TS drop: ts_ref - ts > 0 → K increases
- ✅ Age penalty: age_sec / max_age_sec → K increases
- ✅ Combined factors: K=0.917 → FULL_CLOSE_PROPOSED

### Profit Lock
- ✅ Triggers at R_net ≥ 1.5
- ✅ LONG: SL → max(current_sl, entry * 1.002)
- ✅ SHORT: SL → min(current_sl, entry * 0.998)
- ✅ Monotonic tightening (never loosens)

### Fallback Behavior
- ✅ No P1 proposal → uses fallback_stop_pct=0.02
- ✅ No current_sl → proposes from scratch
- ✅ No theta → uses HarvestTheta() defaults

### Determinism & Side-Effects
- ✅ Same inputs → same outputs (deterministic)
- ✅ No mutations (pure functions)
- ✅ No I/O (calc-only)
- ✅ No trading imports (math, typing, dataclasses only)

---

## VPS ENVIRONMENT

**Python**: 3.12.3  
**Venv**: /opt/quantum/venvs/ai-engine  
**Pytest**: 9.0.2  
**Working Directory**: /home/qt/quantum_trader  
**User**: qt (files chowned correctly)

---

## COMPARISON WITH LOCAL

All outputs match local verification:
- Test counts: 23/23 (local) vs 23/23 (VPS) ✅
- Replay scenarios: 8/8 (local) vs 8/8 (VPS) ✅
- K values: Match within floating-point precision ✅
- Harvest actions: Identical ✅
- SL proposals: Identical ✅

---

## NEXT STEPS

### P2.5: Harvest Proposal Publisher (systemd service)

Now that P2 is LOCKED and VPS-verified, implement P2.5:

**Files to Create**:
1. **microservices/harvest_proposal_publisher/main.py**
   - Reads from Redis:
     - `quantum:marketstate:<symbol>` (P0.5)
     - `quantum:risk:proposal:<symbol>` (P1.5)
     - Position data (auto-detect or explicit)
   - Computes: `compute_harvest_proposal()`
   - Publishes to Redis:
     - `quantum:harvest:proposal:<symbol>` (hash)
     - Optional: stream for history

2. **deployment/systemd/quantum-harvest-proposal.service**
   - Similar to P1.5 service
   - ExecStart: python3 /home/qt/quantum_trader/microservices/harvest_proposal_publisher/main.py
   - Environment: PYTHONPATH, PATH
   - EnvironmentFile: /etc/quantum/harvest-proposal.env

3. **deployment/config/harvest-proposal.env**
   - REDIS_URL=redis://localhost:6379
   - SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT
   - PUBLISH_INTERVAL_SEC=10
   - POSITION_SOURCE=auto

**Then**: P2.5B VPS deployment + proof pack (6 tests like P1.5B)

---

## LOCKED SPECS STATUS

| Phase | Module | Local | VPS |
|-------|--------|-------|-----|
| P0 | MarketState | ✅ LOCKED | ✅ PASSED (14/14) |
| P1 | Risk Kernel (Stops) | ✅ LOCKED | ✅ PASSED (13/13) |
| P1.5 | Risk Proposal Publisher | ✅ LOCKED | ✅ PASSED (6/6) |
| **P2** | **Harvest Kernel** | **✅ LOCKED** | **✅ PASSED (4/4)** |
| P2.5 | Harvest Publisher | 🔲 TODO | 🔲 TODO |

---

## COMMITS

**P0**: 3db38c19 — MarketState LOCKED SPEC v1.0  
**P1**: 86b561c2 — Risk Kernel LOCKED SPEC v1.0  
**P1.5**: bfedcf9a — Risk Proposal Publisher  
**P2**: 82be192e — Harvest Kernel (profit harvesting)  
**P2B**: 7d4556d8 — Documentation (this proof pack)

---

**🎯 P2B VPS DEPLOYMENT + PROOF PACK: COMPLETE ✅**

**All 4 proofs passed. P2 Harvest Kernel is now LOCKED and VPS-verified.**

**Ready for P2.5 Harvest Proposal Publisher implementation.**
