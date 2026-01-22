# P2 HARVEST KERNEL — IMPLEMENTATION COMPLETE ✅

**Date**: 2026-01-22  
**Commit**: 82be192e  
**Status**: LOCAL VERIFIED — Ready for VPS deployment (P2B)

---

## WHAT WAS BUILT

P2 Harvest Kernel is a **calc-only profit harvesting proposal engine** that suggests:

1. **Partial exits** (25%, 50%, 75%) based on normalized risk-adjusted returns (R_net)
2. **Profit lock** SL tightening (move to BE+ when profitable, never loosen)
3. **Kill score** K (edge collapse detection via regime flip, volatility spike, TS drop, age)
4. **Full close proposal** when K exceeds threshold

**NO trading side-effects. NO orders. NO execution.**

---

## FILES CREATED

### 1. Core Module (277 lines)
**`ai_engine/risk_kernel_harvest.py`**
- `HarvestTheta` dataclass: 24 tunables (triggers, weights, thresholds)
- `PositionSnapshot` dataclass: position state
- `MarketState` dataclass: regime probabilities
- `P1Proposal` dataclass: optional P1 stop_dist_pct
- `compute_harvest_proposal()`: main entry point
- Helper functions: risk_unit, R_net, harvest_action, profit_lock_sl, kill_score, tranche_weights

### 2. Test Suite (420+ lines, 24 tests)
**`ai_engine/tests/test_risk_kernel_harvest.py`**
- R_net triggers (4 tests): NONE → PARTIAL_25 → PARTIAL_50 → PARTIAL_75
- Profit lock monotonic tightening (4 tests): LONG/SHORT, tighten/no-loosen
- Kill score (6 tests): regime flip, sigma spike, TS drop, age penalty, full close, combined
- Fallback behavior (3 tests): no P1, no current_sl, no theta
- Tranche weights (2 tests): sum to 1.0, uniform fallback
- No trading side-effects (1 test): deterministic, pure dict
- Reason codes (3 tests): harvest, profit_lock, kill_score
- Audit trail (1 test): complete inputs + intermediates

### 3. Replay Scenarios (250+ lines, 8 scenarios)
**`ops/replay_risk_kernel_harvest.py`**
1. Increasing PNL (trend regime): 4 levels → triggers higher tranches
2. Regime flip (trend → chop): K increases from 0.503 → 0.734 → FULL_CLOSE_PROPOSED
3. Volatility spike: σ 3x → K=0.734 → FULL_CLOSE_PROPOSED
4. TS collapse: TS 0.5 → 0.1 → K increases
5. Aging position: 1h → 28h → K increases
6. Perfect storm: all factors → K=0.917 → FULL_CLOSE_PROPOSED
7. Profit lock: LONG/SHORT SL tightening (99.0→100.2, 101.0→99.8)
8. No P1 fallback: uses fallback_stop_pct=0.02

### 4. Documentation (400+ lines)
**`P2_HARVEST_KERNEL.md`**
- Purpose, inputs, formulas, outputs
- Reason codes table
- Replay scenario results
- Test coverage summary
- Validation checklist (all passed ✅)
- Usage example
- Interface contract
- Next steps (P2B VPS deployment)

---

## CORE FORMULAS

### R_net (Normalized Risk-Adjusted Return)
```python
risk_unit = entry_price * stop_dist_pct  # From P1 or fallback
cost_est = (cost_bps / 10000) * entry_price
R_net = (unrealized_pnl - cost_est) / risk_unit
```

### Harvest Triggers
- **R_net ≥ 2.0**: PARTIAL_25
- **R_net ≥ 4.0**: PARTIAL_50
- **R_net ≥ 6.0**: PARTIAL_75

### Profit Lock (Monotonic Tightening)
```python
if R_net >= 1.5:
    LONG: new_sl = max(current_sl, entry_price * 1.002)
    SHORT: new_sl = min(current_sl, entry_price * 0.998)
    # Only propose if tightens
```

### Kill Score K (Edge Collapse)
```python
z = (k_regime_flip * regime_flip +
     k_sigma_spike * sigma_spike +
     k_ts_drop * ts_drop +
     k_age_penalty * age_penalty)
K = 1 / (1 + exp(-z))  # Sigmoid to [0,1]

if K >= 0.6:
    harvest_action = FULL_CLOSE_PROPOSED
```

---

## VERIFICATION RESULTS

### Local Replay Output (All Passed ✅)

**Scenario 1: Increasing PNL**
- Low profit (R=0.45): NONE
- Medium profit (R=2.45): PARTIAL_25 + profit_lock ✅
- High profit (R=4.45): PARTIAL_50 + profit_lock ✅
- Very high profit (R=6.95): PARTIAL_75 + profit_lock ✅

**Scenario 2: Regime Flip**
- Before (p_trend=0.6): K=0.503, NONE ✅
- After (p_trend=0.1, p_chop=0.6): K=0.734, FULL_CLOSE_PROPOSED ✅

**Scenario 3: Volatility Spike**
- Normal (σ=0.01): K=0.503, NONE ✅
- Spike (σ=0.03): K=0.734, FULL_CLOSE_PROPOSED ✅

**Scenario 4: TS Collapse**
- Strong (TS=0.5): K=0.503, NONE ✅
- Weak (TS=0.1): K=0.528, NONE ✅

**Scenario 5: Aging Position**
- Young (1h): K=0.503, NONE ✅
- Old (28h): K=0.574, NONE ✅

**Scenario 6: Perfect Storm**
- All factors: K=0.917, FULL_CLOSE_PROPOSED ✅
- Reason codes: kill_score_triggered, regime_flip, sigma_spike, age_penalty ✅

**Scenario 7: Profit Lock**
- LONG: SL 99.0 → 100.2 (BE+ tightening) ✅
- SHORT: SL 101.0 → 99.8 (BE+ tightening) ✅

**Scenario 8: No P1 Fallback**
- Uses fallback_stop_pct=0.02 ✅
- R=2.45 → PARTIAL_25 + profit_lock ✅

---

## DESIGN PATTERNS (MATCHES P1)

### 1. Dataclass Inputs
- Strongly typed via `@dataclass`
- Clear interface contracts
- Easy to serialize/deserialize

### 2. Pure Functions
- No side-effects
- Deterministic output
- No I/O, no state mutations

### 3. Audit Trail
- All inputs preserved in output
- All intermediates captured (K components, tranche weights)
- Full reproducibility

### 4. Reason Codes
- Human-readable labels
- Easy to filter/aggregate
- Supports dashboard display

### 5. Calc-Only Guarantees
- NO binance imports
- NO trade_intent imports
- NO execution/order modules
- Only: math, typing, dataclasses

---

## TUNABLE PARAMETERS (theta.harvest.*)

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `fallback_stop_pct` | 0.02 | Risk unit if P1 unavailable |
| `cost_bps` | 10.0 | Transaction cost (bps) |
| `T1_R` | 2.0 | Trigger PARTIAL_25 |
| `T2_R` | 4.0 | Trigger PARTIAL_50 |
| `T3_R` | 6.0 | Trigger PARTIAL_75 |
| `lock_R` | 1.5 | Trigger profit lock |
| `be_plus_pct` | 0.002 | BE+ buffer (0.2%) |
| `trend_min` | 0.3 | Min p_trend (regime flip) |
| `sigma_ref` | 0.01 | Reference volatility |
| `ts_ref` | 0.3 | Reference trend strength |
| `max_age_sec` | 86400.0 | Max position age (24h) |
| `kill_threshold` | 0.6 | Full close trigger |

**All tunables configurable at runtime** (no code changes needed)

---

## INTEGRATION WITH P1

P2 consumes P1 outputs:
- **stop_dist_pct**: Used for risk_unit calculation (preferred over fallback)
- **proposed_sl**: Not used by P2 (P2 computes own profit lock SL)

P2 can work standalone:
- If P1 not available: uses `fallback_stop_pct=0.02`
- All computations still work correctly

---

## OUTPUT STRUCTURE

```json
{
  "harvest_action": "PARTIAL_50",
  "new_sl_proposed": 100.2,
  "R_net": 4.45,
  "risk_unit": 2.0,
  "cost_est": 0.1,
  "kill_score": 0.503,
  "reason_codes": ["harvest_partial_50", "profit_lock"],
  "audit": {
    "position": {...},
    "market_state": {...},
    "p1_proposal": {...},
    "theta": {...},
    "tranche_weights": [0.333, 0.333, 0.333],
    "k_components": {
      "regime_flip": 0.0,
      "sigma_spike": 0.0,
      "ts_drop": 0.0,
      "age_penalty": 0.042
    }
  }
}
```

---

## NEXT STEPS

### P2B: VPS Deployment + Proof Pack

**Deploy files to VPS**:
1. `ai_engine/risk_kernel_harvest.py` → `/home/qt/quantum_trader/ai_engine/`
2. `ops/replay_risk_kernel_harvest.py` → `/home/qt/quantum_trader/ops/`
3. Ensure `__init__.py` exists in `ai_engine/`

**Run VPS proof pack**:
```bash
cd /home/qt/quantum_trader
/opt/quantum/venvs/ai-engine/bin/python3 ops/replay_risk_kernel_harvest.py
```

**Verification criteria**:
- All 8 scenarios execute ✅
- Outputs match local replay ✅
- No exceptions/errors ✅
- Calc-only verified (no trading imports) ✅

**Create report**: `P2B_VPS_PROOF_PACK_COMPLETE.md`

### P2.5: Harvest Publisher (systemd service)

After P2B passes, implement:
- `microservices/harvest_proposal_publisher/main.py`
- `deployment/systemd/quantum-harvest-proposal.service`
- `deployment/config/harvest-proposal.env`

**Inputs** (from Redis):
- `quantum:marketstate:<symbol>` (P0.5 publisher)
- `quantum:risk:proposal:<symbol>` (P1.5 publisher)
- Position data (auto-detect or explicit source)

**Outputs** (to Redis):
- `quantum:harvest:proposal:<symbol>`

**Publish interval**: 10s (configurable)

**Then**: P2.5B VPS deployment + proof pack (6 tests like P1.5B)

---

## LOCKED SPECS STATUS

| Phase | Module | Status | VPS Proof |
|-------|--------|--------|-----------|
| P0 | MarketState | ✅ LOCKED | ✅ PASSED (14/14) |
| P1 | Risk Kernel (Stops/Trailing) | ✅ LOCKED | ✅ PASSED (13/13) |
| P1.5 | Risk Proposal Publisher | ✅ LOCKED | ✅ PASSED (6/6) |
| **P2** | **Harvest Kernel** | **✅ LOCKED** | **⏳ PENDING** |
| P2.5 | Harvest Publisher | 🔲 TODO | 🔲 TODO |

---

## KEY INSIGHTS

### 1. R_net Normalization
- Different positions with different entry prices comparable
- Risk-adjusted returns account for stop distance
- Cost estimation makes P&L more realistic

### 2. Monotonic SL Tightening
- Prevents "loosening" mistakes
- Always moves SL in favor of position
- BE+ buffer provides safety margin

### 3. Kill Score Composition
- Multiple weak signals → strong signal
- Sigmoid smooths binary decisions
- Tunable weights allow domain expertise

### 4. Calc-Only Philosophy
- Proposals, not actions
- Human review possible (via dashboard)
- Safety gates applied later (P3+)

---

## COMMIT HISTORY

**P1**: 86b561c2 — Risk Kernel (stops/trailing)  
**P1.5**: bfedcf9a — Risk Proposal Publisher  
**P2**: 82be192e — Harvest Kernel (profit harvesting)

---

**🎯 P2 HARVEST KERNEL: IMPLEMENTATION COMPLETE ✅**

**Next Action**: Deploy to VPS and run P2B proof pack
