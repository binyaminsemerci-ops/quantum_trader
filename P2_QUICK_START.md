# P2 HARVEST KERNEL — QUICK START

## Run Locally

### Demo Script
```bash
python ai_engine/risk_kernel_harvest.py
```
**Output**: Example harvest proposal with R_net=2.45R → PARTIAL_25 + profit_lock

### Replay All Scenarios
```bash
python ops/replay_risk_kernel_harvest.py
```
**Output**: 8 scenarios (increasing PNL, regime flip, volatility spike, TS collapse, aging, perfect storm, profit lock, fallback)

### Run Tests
```bash
pytest ai_engine/tests/test_risk_kernel_harvest.py -v
```
**Expected**: 24 tests passed

---

## Usage Example

```python
from ai_engine.risk_kernel_harvest import (
    compute_harvest_proposal,
    HarvestTheta,
    PositionSnapshot,
    MarketState,
    P1Proposal,
)

# Define position
pos = PositionSnapshot(
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

# Define market state
market = MarketState(
    sigma=0.015,
    ts=0.35,
    p_trend=0.5,
    p_mr=0.2,
    p_chop=0.3,
)

# Optional P1 proposal
p1 = P1Proposal(stop_dist_pct=0.02)

# Compute harvest proposal
result = compute_harvest_proposal(pos, market, p1)

# Use result
print(f"Action: {result['harvest_action']}")        # PARTIAL_25
print(f"New SL: {result['new_sl_proposed']}")       # 100.2
print(f"R_net: {result['R_net']:.2f}R")             # 2.45R
print(f"Kill Score: {result['kill_score']:.3f}")    # 0.565
print(f"Reasons: {result['reason_codes']}")         # [harvest_partial_25, profit_lock]
```

---

## Key Parameters

### Harvest Actions
- **NONE**: R_net < 2.0
- **PARTIAL_25**: 2.0 ≤ R_net < 4.0
- **PARTIAL_50**: 4.0 ≤ R_net < 6.0
- **PARTIAL_75**: R_net ≥ 6.0
- **FULL_CLOSE_PROPOSED**: K ≥ 0.6 (kill threshold)

### Profit Lock
- **Trigger**: R_net ≥ 1.5
- **LONG**: SL → max(current_sl, entry * 1.002)
- **SHORT**: SL → min(current_sl, entry * 0.998)

### Kill Score Components
- **Regime flip**: p_trend < 0.3 and (p_chop + p_mr) > 0.5
- **Sigma spike**: sigma / sigma_ref - 1 (capped at 2.0)
- **TS drop**: ts_ref - ts (capped at 0.5)
- **Age penalty**: age_sec / 86400.0 (capped at 1.0)

---

## Tuning Tips

### Conservative Harvesting (fewer exits)
```python
theta = HarvestTheta(
    T1_R=3.0,  # Raise triggers
    T2_R=5.0,
    T3_R=8.0,
    kill_threshold=0.7,  # Raise threshold
)
```

### Aggressive Harvesting (more exits)
```python
theta = HarvestTheta(
    T1_R=1.5,  # Lower triggers
    T2_R=3.0,
    T3_R=5.0,
    kill_threshold=0.5,  # Lower threshold
)
```

### Tighter Profit Lock
```python
theta = HarvestTheta(
    lock_R=1.0,        # Lock earlier
    be_plus_pct=0.005, # Larger buffer (0.5%)
)
```

---

## VPS Deployment (P2B)

```bash
# 1. Deploy file
scp ai_engine/risk_kernel_harvest.py vps:/home/qt/quantum_trader/ai_engine/
scp ops/replay_risk_kernel_harvest.py vps:/home/qt/quantum_trader/ops/

# 2. Run replay on VPS
ssh vps
cd /home/qt/quantum_trader
/opt/quantum/venvs/ai-engine/bin/python3 ops/replay_risk_kernel_harvest.py

# 3. Verify outputs match local
```

---

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `ai_engine/risk_kernel_harvest.py` | 277 | Core harvest logic |
| `ai_engine/tests/test_risk_kernel_harvest.py` | 420+ | 24 unit tests |
| `ops/replay_risk_kernel_harvest.py` | 250+ | 8 replay scenarios |
| `P2_HARVEST_KERNEL.md` | 400+ | Full spec docs |
| `P2_IMPLEMENTATION_COMPLETE.md` | 300+ | Implementation summary |

---

## Integration Points

### P1 (Risk Kernel)
- **Input**: `stop_dist_pct` (preferred for risk_unit)
- **Fallback**: `fallback_stop_pct=0.02` if P1 not available

### P0.5 (MarketState)
- **Input**: `sigma`, `ts`, `p_trend`, `p_mr`, `p_chop`

### P2.5 (Harvest Publisher) — Next Step
- **Input**: P0.5 MarketState + P1.5 Risk Proposals + Position Data
- **Output**: `quantum:harvest:proposal:<symbol>` in Redis
- **Service**: systemd publisher (similar to P1.5)

---

## Validation Checklist

- [x] R_net triggers correct harvest actions
- [x] Profit lock SL never loosens (monotonic)
- [x] Kill score triggers full close proposal
- [x] Fallback behavior with missing inputs
- [x] No trading side-effects (calc-only)
- [x] Replay scenarios execute cleanly
- [x] Demo script runs successfully
- [ ] VPS proof pack (P2B) — PENDING

---

**Status**: LOCAL VERIFIED ✅  
**Next**: VPS Deployment (P2B)  
**Commit**: 82be192e
