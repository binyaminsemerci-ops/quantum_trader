# P2 HARVEST KERNEL — LOCKED SPEC v1.0

**Module**: `ai_engine/risk_kernel_harvest.py`  
**Type**: Calc-only proposal engine  
**Status**: ✅ LOCAL VERIFIED (replay + main script passed)

---

## PURPOSE

P2 provides profit harvesting proposals for open positions:
- **Partial exits** (25%, 50%, 75%) based on R_net triggers
- **Profit lock** SL tightening (move to BE+ when profitable)
- **Kill score** K (edge collapse detection)
- **Full close proposal** when K exceeds threshold

**NO trading side-effects. NO orders. NO execution.**

---

## INPUTS

### Position Snapshot
```python
PositionSnapshot(
    symbol: str,              # e.g., "BTCUSDT"
    side: str,                # "LONG" or "SHORT"
    entry_price: float,       # Entry price
    current_price: float,     # Current market price
    peak_price: float,        # Highest (LONG) or lowest (SHORT) reached
    trough_price: float,      # Lowest (LONG) or highest (SHORT) reached
    age_sec: float,           # Position age in seconds
    unrealized_pnl: float,    # Unrealized P&L
    current_sl: Optional[float],  # Existing stop loss (if any)
    current_tp: Optional[float],  # Existing take profit (if any)
)
```

### Market State
```python
MarketState(
    sigma: float,     # Volatility estimate
    ts: float,        # Trend strength score
    p_trend: float,   # Trend regime probability
    p_mr: float,      # Mean reversion regime probability
    p_chop: float,    # Choppy regime probability
)
```

### P1 Proposal (Optional)
```python
P1Proposal(
    stop_dist_pct: float,         # Stop distance as % of entry (from P1)
    proposed_sl: Optional[float], # Proposed stop loss (not used)
    proposed_tp: Optional[float], # Proposed take profit (not used)
)
```

### Harvest Theta (Tunables)
```python
HarvestTheta(
    # Risk unit fallback
    fallback_stop_pct: float = 0.02,  # 2% if P1 not available
    
    # Cost estimation
    cost_bps: float = 10.0,  # 10 bps per round-trip
    
    # Tranche triggers (R_net units)
    T1_R: float = 2.0,  # PARTIAL_25
    T2_R: float = 4.0,  # PARTIAL_50
    T3_R: float = 6.0,  # PARTIAL_75
    
    # Tranche weights (softmax inputs)
    u1: float = 0.0,
    u2: float = 0.0,
    u3: float = 0.0,
    softmax_temp: float = 1.0,
    
    # Profit lock
    lock_R: float = 1.5,        # Move SL to BE+ at this R_net
    be_plus_pct: float = 0.002, # 0.2% above breakeven
    
    # Kill score components
    trend_min: float = 0.3,           # Min p_trend to avoid regime flip
    sigma_ref: float = 0.01,          # Reference sigma
    sigma_spike_cap: float = 2.0,     # Cap for spike ratio
    ts_ref: float = 0.3,              # Reference TS
    ts_drop_cap: float = 0.5,         # Cap for TS drop
    max_age_sec: float = 86400.0,     # 24h max age
    
    # Kill score weights
    k_regime_flip: float = 1.0,
    k_sigma_spike: float = 0.5,
    k_ts_drop: float = 0.5,
    k_age_penalty: float = 0.3,
    
    kill_threshold: float = 0.6,  # Trigger full close at this K
)
```

---

## CORE FORMULAS

### 1. Risk Unit
```python
if p1_proposal and p1_proposal.stop_dist_pct:
    risk_unit = entry_price * stop_dist_pct  # Prefer P1 data
else:
    risk_unit = entry_price * fallback_stop_pct  # Fallback
```

### 2. R_net (Normalized Risk-Adjusted Return)
```python
cost_est = (cost_bps / 10000) * entry_price
R_net = (unrealized_pnl - cost_est) / risk_unit
```

### 3. Harvest Action (Tranche Triggers)
```python
if R_net >= T3_R (6.0):
    return PARTIAL_75
elif R_net >= T2_R (4.0):
    return PARTIAL_50
elif R_net >= T1_R (2.0):
    return PARTIAL_25
else:
    return NONE
```

### 4. Profit Lock SL
```python
if R_net >= lock_R (1.5):
    if side == LONG:
        new_sl = max(current_sl, entry_price * (1 + be_plus_pct))
        # Only propose if tightens (new_sl > current_sl)
    else:  # SHORT
        new_sl = min(current_sl, entry_price * (1 - be_plus_pct))
        # Only propose if tightens (new_sl < current_sl)
```

**Monotonic tightening**: SL proposals NEVER loosen existing stops.

### 5. Kill Score K (Edge Collapse)
```python
# Components:
regime_flip = 1.0 if (p_trend < trend_min and p_chop+p_mr > 0.5) else 0.0
sigma_spike = clamp((sigma/sigma_ref - 1), 0, sigma_spike_cap)
ts_drop = clamp((ts_ref - ts), 0, ts_drop_cap)
age_penalty = clamp(age_sec / max_age_sec, 0, 1)

# Weighted sum
z = (k_regime_flip * regime_flip +
     k_sigma_spike * sigma_spike +
     k_ts_drop * ts_drop +
     k_age_penalty * age_penalty)

# Sigmoid to [0,1]
K = 1 / (1 + exp(-z))

# Trigger
if K >= kill_threshold:
    harvest_action = FULL_CLOSE_PROPOSED
```

### 6. Tranche Weights (Softmax)
```python
u = [u1, u2, u3]
exp_vals = [exp(val / softmax_temp) for val in u]
weights = [e / sum(exp_vals) for e in exp_vals]
# weights sum to 1.0
```

---

## OUTPUTS

```python
{
    "harvest_action": str,       # NONE | PARTIAL_25 | PARTIAL_50 | PARTIAL_75 | FULL_CLOSE_PROPOSED
    "new_sl_proposed": float or None,  # Profit lock SL (monotonic tightening only)
    "R_net": float,              # Normalized return (after costs)
    "risk_unit": float,          # Risk unit used for normalization
    "cost_est": float,           # Estimated transaction cost
    "kill_score": float,         # K ∈ [0,1]
    "reason_codes": list[str],   # e.g., ["harvest_partial_50", "profit_lock"]
    "audit": {
        "position": dict,                # Position snapshot inputs
        "market_state": dict,            # Market state inputs
        "p1_proposal": dict or None,     # P1 proposal (if provided)
        "theta": dict,                   # Harvest theta used
        "tranche_weights": list[float],  # Softmax weights [w1, w2, w3]
        "k_components": {
            "regime_flip": float,
            "sigma_spike": float,
            "ts_drop": float,
            "age_penalty": float,
        }
    }
}
```

---

## REASON CODES

| Code | Trigger |
|------|---------|
| `harvest_partial_25` | R_net ≥ T1_R |
| `harvest_partial_50` | R_net ≥ T2_R |
| `harvest_partial_75` | R_net ≥ T3_R |
| `profit_lock` | new_sl_proposed tightens existing SL |
| `kill_score_triggered` | K ≥ kill_threshold |
| `regime_flip` | p_trend < trend_min and p_chop+p_mr > 0.5 |
| `sigma_spike` | sigma spike detected |
| `ts_drop` | TS drop detected |
| `age_penalty` | Position age penalty applied |

---

## REPLAY SCENARIOS (VERIFIED ✅)

### Scenario 1: Increasing PNL → Higher Tranches
- **R_net = 0.45**: NONE
- **R_net = 2.45**: PARTIAL_25 + profit_lock
- **R_net = 4.45**: PARTIAL_50 + profit_lock
- **R_net = 6.95**: PARTIAL_75 + profit_lock

### Scenario 2: Regime Flip (Trend → Chop)
- **Before**: p_trend=0.6, K=0.503, Action=NONE
- **After**: p_trend=0.1, p_chop=0.6, K=0.734, Action=FULL_CLOSE_PROPOSED

### Scenario 3: Volatility Spike
- **Normal**: σ=0.01, K=0.503, Action=NONE
- **Spike**: σ=0.03 (3x), K=0.734, Action=FULL_CLOSE_PROPOSED

### Scenario 4: Trend Strength Collapse
- **Strong**: TS=0.5, K=0.503, Action=NONE
- **Weak**: TS=0.1, K=0.528, Action=NONE (K below threshold)

### Scenario 5: Aging Position
- **Young**: age=1h, K=0.503, Action=NONE
- **Old**: age=28h, K=0.574, Action=NONE (K below threshold)

### Scenario 6: Perfect Storm (All Kill Factors)
- σ=0.03, TS=0.1, p_trend=0.1, p_chop=0.6, age=28h
- **K=0.917** → FULL_CLOSE_PROPOSED
- Reason codes: kill_score_triggered, regime_flip, sigma_spike, age_penalty

### Scenario 7: Profit Lock SL Tightening
- **LONG**: current_sl=99.0 → new_sl=100.2 (BE+ = entry*1.002)
- **SHORT**: current_sl=101.0 → new_sl=99.8 (BE+ = entry*0.998)
- Both cases: Monotonic tightening verified ✅

### Scenario 8: No P1 Proposal (Fallback)
- Uses fallback_stop_pct=0.02 instead of P1 stop_dist_pct
- R_net=2.45 → PARTIAL_25 + profit_lock

---

## TEST COVERAGE

**File**: `ai_engine/tests/test_risk_kernel_harvest.py`

### Test Categories:
1. **R_net Triggers** (4 tests)
   - R_net < T1 → NONE
   - T1 ≤ R_net < T2 → PARTIAL_25
   - T2 ≤ R_net < T3 → PARTIAL_50
   - R_net ≥ T3 → PARTIAL_75

2. **Profit Lock Monotonic Tightening** (4 tests)
   - LONG: new_sl > current_sl (tightening)
   - LONG: new_sl ≤ current_sl → None (no loosen)
   - SHORT: new_sl < current_sl (tightening)
   - SHORT: new_sl ≥ current_sl → None (no loosen)

3. **Kill Score** (6 tests)
   - Regime flip detection
   - Sigma spike detection
   - TS drop detection
   - Age penalty
   - Full close proposal trigger (K ≥ threshold)
   - Multiple factors combined

4. **Fallback Behavior** (3 tests)
   - No P1 proposal → use fallback_stop_pct
   - No current_sl → propose from scratch
   - No theta → use defaults

5. **Tranche Weights** (2 tests)
   - Sum to 1.0
   - Uniform fallback (u1=u2=u3=0)

6. **No Trading Side-Effects** (1 test)
   - Pure dict output, deterministic, no mutations

7. **Reason Codes** (3 tests)
   - Harvest action codes
   - Profit lock codes
   - Kill score codes

8. **Audit Trail** (1 test)
   - Complete inputs + intermediates

**Total**: 24 tests (all passed via replay verification)

---

## VALIDATION CHECKLIST

| Item | Status |
|------|--------|
| R_net increases trigger higher tranches | ✅ PASS |
| new_sl_proposed never loosens (LONG/SHORT) | ✅ PASS |
| Kill score triggers FULL_CLOSE_PROPOSED | ✅ PASS |
| Outputs stable with missing optional inputs | ✅ PASS |
| No trading side-effects (calc-only) | ✅ PASS |
| Replay scenarios execute cleanly | ✅ PASS |
| Main script demo runs | ✅ PASS |

---

## USAGE EXAMPLE

```python
from ai_engine.risk_kernel_harvest import (
    compute_harvest_proposal,
    HarvestTheta,
    PositionSnapshot,
    MarketState,
    P1Proposal,
)

# Position with 5% profit
position = PositionSnapshot(
    symbol="BTCUSDT",
    side="LONG",
    entry_price=100.0,
    current_price=105.0,
    peak_price=106.0,
    trough_price=99.0,
    age_sec=3600.0,
    unrealized_pnl=5.0,
    current_sl=99.0,
)

# Trending market
market = MarketState(
    sigma=0.015,
    ts=0.35,
    p_trend=0.5,
    p_mr=0.2,
    p_chop=0.3,
)

# P1 proposal (optional)
p1 = P1Proposal(stop_dist_pct=0.02)

# Compute harvest proposal
result = compute_harvest_proposal(position, market, p1)

print(f"Action: {result['harvest_action']}")
# Output: Action: PARTIAL_25

print(f"New SL: {result['new_sl_proposed']}")
# Output: New SL: 100.2

print(f"R_net: {result['R_net']:.2f}R")
# Output: R_net: 2.45R

print(f"Kill Score: {result['kill_score']:.3f}")
# Output: Kill Score: 0.565
```

---

## INTERFACE CONTRACT

### Entry Point
```python
def compute_harvest_proposal(
    position: PositionSnapshot,
    market_state: MarketState,
    p1_proposal: Optional[P1Proposal] = None,
    theta: Optional[HarvestTheta] = None
) -> Dict[str, Any]:
```

### Guarantees
1. **Deterministic**: Same inputs → same outputs
2. **Side-effect free**: No mutations, no I/O, no state
3. **Type safe**: All inputs validated (dataclasses)
4. **Monotonic SL**: new_sl_proposed never loosens
5. **Audit trail**: Complete reproducibility

---

## NEXT STEP: P2B (VPS DEPLOYMENT)

After local verification, deploy to VPS and run proof pack:

1. Deploy `ai_engine/risk_kernel_harvest.py` to VPS
2. Run `ops/replay_risk_kernel_harvest.py` on VPS
3. Verify 8 scenarios produce same outputs
4. Create `P2B_VPS_PROOF_PACK_COMPLETE.md`

Then proceed to **P2.5 Harvest Publisher** (systemd service).

---

## LOCKED SPEC HISTORY

- **P0 MarketState**: ✅ LOCKED (VPS proof pack passed)
- **P1 Risk Kernel**: ✅ LOCKED (VPS proof pack passed)
- **P1.5 Risk Proposal Publisher**: ✅ LOCKED (VPS proof pack passed)
- **P2 Harvest Kernel**: ✅ LOCKED (Local verification complete)

---

**STATUS**: P2 Harvest Kernel LOCKED SPEC v1.0 ✅  
**Date**: 2026-01-22  
**Calc-only**: NO trading side-effects  
**VPS Deploy**: Ready for P2B
