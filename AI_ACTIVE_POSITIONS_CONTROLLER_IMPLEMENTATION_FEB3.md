# 🎯 ACTIVE POSITIONS CONTROLLER - DYNAMIC SLOT MANAGEMENT
**Date:** February 3, 2026  
**Status:** ✅ IMPLEMENTED AND TESTED  
**Commit:** `655df4158`

---

## EXECUTIVE SUMMARY

Implemented **Active Positions Controller** - a dynamic slot management system that:

1. **No Hardcoded Symbols** - Reads universe from PolicyStore (Redis `quantum:policy:current`)
2. **Dynamic Slots** - Allocates 3-6 position slots based on market regime (not always fill 10)
3. **Capital Rotation** - Closes weakest position when better candidate arrives
4. **Correlation Caps** - Prevents >80% correlation with existing portfolio
5. **Margin Caps** - Blocks opens if margin usage >65%
6. **Fail-Closed** - Blocks all opens if PolicyStore unavailable

**Result:** Intelligent position management that adapts to market conditions and portfolio risk.

---

## THE PROBLEM: ALWAYS FILLING 10 SLOTS IS SUBOPTIMAL

### Before Implementation:
❌ **Fixed Slot Count:** Always tried to open 10 positions regardless of conditions  
❌ **No Regime Awareness:** Didn't adjust exposure based on market conditions  
❌ **No Rotation Logic:** Couldn't upgrade to better candidates when slots full  
❌ **No Correlation Check:** Could open 10 BTC-correlated positions  
❌ **No Risk Caps:** No margin usage limits  

### After Implementation:
✅ **Dynamic Slots:** 3-6 positions based on regime (trend_strong=6, chop/spike=3, moderate=4)  
✅ **Regime-Aware:** Reduces exposure in choppy/volatile markets  
✅ **Smart Rotation:** Closes weakest for significantly better candidates (15%+ improvement)  
✅ **Correlation Gating:** Blocks if >80% correlated with portfolio  
✅ **Margin Gating:** Blocks if total margin usage >65%  

---

## ARCHITECTURE

### Component Location
**File:** `backend/services/risk/active_positions_controller.py`  
**Integration:** `backend/services/risk/safety_governor.py`  
**Priority:** 2.5 (between Risk Manager and AI-HFOS)

### Data Flow
```
1. Intent arrives (NEW_TRADE or EXPAND_POSITION)
   ↓
2. SafetyGovernor calls Active Slots Controller
   ↓
3. Controller loads universe from PolicyStore
   ↓
4. Detects market regime (trend/chop/spike)
   ↓
5. Computes desired slots (3-6 based on regime)
   ↓
6. Checks: margin cap, correlation cap, slots available
   ↓
7. Decision: ALLOW, BLOCK, or ROTATION
   ↓
8. Returns to SafetyGovernor for final decision
```

---

## REGIME-BASED SLOT ALLOCATION

### Regime Detection Logic
```python
# Input: market_features dict
trend_strength: float [0..1]
atr_pct: float (volatility %)
momentum_consistency: float [0..1]

# Output: RegimeType + confidence
```

### Slot Allocation Rules

| Regime | Condition | Slots | Rationale |
|--------|-----------|-------|-----------|
| **TREND_STRONG** | `trend_strength >= 0.75`<br>`momentum_consistency >= 0.6` | **6** | Strong trends = more opportunities |
| **TREND_MODERATE** | `trend_strength >= 0.45` | **4** | Base case - balanced exposure |
| **RANGE** | `trend_strength < 0.4` | **4** | Neutral conditions |
| **CHOP** | `trend_strength < 0.3`<br>`momentum_consistency < 0.4` | **3** | Choppy = reduce exposure |
| **VOLATILITY_SPIKE** | `atr_pct > 2.5` | **3** | High volatility = defensive |
| **UNKNOWN** | Fallback | **4** | Conservative default |

**Hard Cap:** Always ≤10 (PolicyStore universe size)

---

## CAPITAL ROTATION MECHANISM

### When Slots Are Full

**Scenario:** 4/4 slots filled, new candidate arrives

**Decision Logic:**
1. Find weakest position by score
2. Compute score improvement: `(new_score - weakest_score) / weakest_score`
3. If improvement > 15%: **ROTATE** (close weakest, open new)
4. Else: **BLOCK** (candidate not good enough)

### Example Rotation
```
Current Portfolio:
1. BTCUSDT   score=75.0 ✅
2. ETHUSDT   score=72.0 ✅
3. SOLUSDT   score=70.0 ✅
4. BNBUSDT   score=60.0 ← WEAKEST

New Candidate:
- ADAUSDT    score=73.0
- Improvement: (73-60)/60 = 21.7% > 15% ✅

Decision: ROTATION_TRIGGERED
- Close: BNBUSDT (score=60.0)
- Open: ADAUSDT (score=73.0)

Log:
ROTATION_CLOSE weakest=BNBUSDT weakest_score=60.00 new_symbol=ADAUSDT 
               new_score=73.00 threshold=15.00% improvement=21.67%
```

---

## RISK GATES

### 1. Correlation Cap (Max 80%)

**Purpose:** Prevent portfolio concentration risk (all positions moving together)

**Mechanism:**
```python
1. Fetch candidate log-returns (1h klines, 100 periods)
2. For each existing position:
   - Align return series lengths
   - Compute Pearson correlation
3. Find max correlation across portfolio
4. If max_corr > 0.80: BLOCK
```

**Example Block:**
```
Portfolio: BTCUSDT returns=[0.01, -0.02, 0.03, ...]
Candidate: ETHUSDT returns=[0.01, -0.02, 0.03, ...]  ← 99.4% correlated

Decision: BLOCKED_CORRELATION
Log: CORR_BLOCK symbol=ETHUSDT corr=0.994 threshold=0.80
```

### 2. Margin Usage Cap (Max 65%)

**Purpose:** Prevent over-leveraging the portfolio

**Mechanism:**
```python
total_margin_usage_pct = sum(position.size_usd * position.leverage) / account_equity * 100

if total_margin_usage_pct >= 65.0:
    BLOCK with decision=BLOCKED_MARGIN
```

**Example Block:**
```
Current Margin Usage: 70.0%
Max Allowed: 65.0%

Decision: BLOCKED_MARGIN
Log: BLOCKED_MARGIN symbol=ETHUSDT margin_usage_pct=70.0 max_margin_pct=65.0
```

### 3. Policy Enforcement (Fail-Closed)

**Purpose:** Never trade if PolicyStore unavailable or symbol not in universe

**Mechanism:**
```python
1. Read: redis HGET quantum:policy:current universe_symbols
2. Parse: JSON array of 10 symbols
3. If missing or invalid: BLOCK with decision=BLOCKED_NO_POLICY
4. If symbol not in universe: BLOCK with decision=BLOCKED_NO_POLICY
```

**Example Block:**
```
PolicyStore Query: quantum:policy:current
Result: {} (empty)

Decision: BLOCKED_NO_POLICY
Log: BLOCKED_NO_POLICY symbol=BTCUSDT reason="quantum:policy:current missing or invalid"
```

---

## CONFIGURATION

### SlotConfig Parameters

Located in `backend/services/risk/active_positions_controller.py`:

```python
@dataclass
class SlotConfig:
    # Base slot allocation
    base_slots: int = 4
    regime_slots_trend_strong: int = 6
    regime_slots_range_chop: int = 3
    regime_slots_volatility_spike: int = 3
    hard_cap_slots: int = 10
    
    # Rotation settings
    rotation_switch_threshold: float = 0.15  # 15% better score
    
    # Risk caps
    max_correlation: float = 0.80
    max_margin_usage_pct: float = 65.0
    
    # Regime detection thresholds
    trend_strong_threshold: float = 0.75
    trend_moderate_threshold: float = 0.45
    volatility_spike_threshold: float = 2.5  # ATR multiplier
```

**To Adjust:**
1. Edit `backend/services/risk/active_positions_controller.py`
2. Modify `SlotConfig` class defaults
3. Or pass custom config when initializing controller

---

## INTEGRATION WITH SAFETY GOVERNOR

### Initialization

```python
# In SafetyGovernor.__init__()
if enable_active_slots and policy_store:
    from backend.services.risk.active_positions_controller import (
        ActivePositionsController,
        SlotConfig,
    )
    self.active_slots_controller = ActivePositionsController(
        policy_store=policy_store,
        config=SlotConfig(),
    )
```

### Subsystem Priority

```
Priority 1: Self-Healing (system health)
Priority 2: Risk Manager (drawdown, emergency brake)
Priority 2.5: ACTIVE SLOTS CONTROLLER ← NEW
Priority 3: AI-HFOS (supreme coordinator)
Priority 4: Portfolio Balancer (constraints)
Priority 5: Profit Amplification (opportunities)
```

### Decision Collection

```python
# In SafetyGovernor.evaluate_trade_request()
active_slots_input = await self.collect_active_slots_input_async(
    symbol=symbol,
    action=action,
    candidate_score=candidate_score,
    candidate_returns=candidate_returns,
    market_features=market_features,
    portfolio_positions=portfolio_positions,
    total_margin_usage_pct=total_margin_usage_pct,
    active_slots_controller=self.active_slots_controller,
)
```

---

## GREP-FRIENDLY LOGS

### Log Formats

**Slot Status:**
```bash
grep "ACTIVE_SLOTS" /var/log/quantum/governor.log
```
**Example:**
```
ACTIVE_SLOTS desired=6 open=3 available=3 regime=TREND_STRONG 
             regime_confidence=0.85 symbol=BTCUSDT
```

**Blocks - Slots Full:**
```bash
grep "BLOCKED_SLOTS_FULL" /var/log/quantum/governor.log
```
**Example:**
```
BLOCKED_SLOTS_FULL symbol=ADAUSDT score=73.00 slots=4/4
```

**Blocks - No Policy:**
```bash
grep "BLOCKED_NO_POLICY" /var/log/quantum/governor.log
```
**Example:**
```
BLOCKED_NO_POLICY symbol=BTCUSDT reason="quantum:policy:current missing or invalid"
```

**Blocks - Correlation:**
```bash
grep "CORR_BLOCK" /var/log/quantum/governor.log
```
**Example:**
```
CORR_BLOCK symbol=ETHUSDT corr=0.994 threshold=0.80
```

**Blocks - Margin:**
```bash
grep "BLOCKED_MARGIN" /var/log/quantum/governor.log
```
**Example:**
```
BLOCKED_MARGIN symbol=ETHUSDT margin_usage_pct=70.0 max_margin_pct=65.0
```

**Rotation:**
```bash
grep "ROTATION_CLOSE" /var/log/quantum/governor.log
```
**Example:**
```
ROTATION_CLOSE weakest=BNBUSDT weakest_score=60.00 new_symbol=ADAUSDT 
               new_score=73.00 threshold=15.00% improvement=21.67%
```

**Allows:**
```bash
grep "ACTIVE_SLOTS_ALLOW" /var/log/quantum/governor.log
```
**Example:**
```
ACTIVE_SLOTS_ALLOW symbol=BTCUSDT score=80.00 slots=1/6
```

---

## PROOF SCRIPT RESULTS

**File:** `test_active_positions_controller.py`

### Test Scenarios (All Passed ✅)

**Scenario 1: Slots Full → Block**
```
Portfolio: 4/4 positions (base_slots=4, TREND_MODERATE)
Candidate: ADAUSDT score=73.0 (not 15%+ better than weakest=68.0)
Result: BLOCKED_SLOTS_FULL ✅
```

**Scenario 2: Rotation Trigger**
```
Portfolio: 4/4 positions
Candidate: ADAUSDT score=73.0 (21.7% better than weakest=60.0)
Result: ROTATION_TRIGGERED (close BNBUSDT, open ADAUSDT) ✅
```

**Scenario 3: Correlation Block**
```
Portfolio: BTCUSDT (base returns)
Candidate: ETHUSDT (99.4% correlated with BTCUSDT)
Result: BLOCKED_CORRELATION ✅
```

**Scenario 4: Margin Block**
```
Current margin usage: 70.0%
Max allowed: 65.0%
Result: BLOCKED_MARGIN ✅
```

**Scenario 5: No Policy Block (Fail-Closed)**
```
PolicyStore: quantum:policy:current MISSING
Result: BLOCKED_NO_POLICY ✅
```

**Scenario 6: Regime Changes Slots**
```
TREND_STRONG: 6 slots ✅
CHOP: 3 slots ✅
VOLATILITY_SPIKE: 3 slots ✅
```

---

## DEPLOYMENT CHECKLIST

### Files Added/Modified

1. ✅ `backend/services/risk/active_positions_controller.py` (NEW - 700 lines)
2. ✅ `backend/services/risk/safety_governor.py` (MODIFIED - added integration)
3. ✅ `test_active_positions_controller.py` (NEW - proof script)

### Integration Points

1. ✅ **SafetyGovernor** - Priority 2.5 subsystem
2. ✅ **PolicyStore** - Reads `quantum:policy:current` universe
3. ✅ **Redis** - Direct HGET for universe symbols
4. ⏳ **Intent Bridge** - Needs to pass market_features, candidate_score, returns
5. ⏳ **Apply Layer** - Needs to pass portfolio_positions snapshot

### Environment Variables (Optional)

```bash
# Enable/disable Active Slots Controller
export GOVERNOR_ENABLE_ACTIVE_SLOTS=true  # default

# Override slot config (advanced)
export ACTIVE_SLOTS_BASE=4
export ACTIVE_SLOTS_TREND_STRONG=6
export ACTIVE_SLOTS_CHOP=3
export ROTATION_THRESHOLD=0.15
export MAX_CORRELATION=0.80
export MAX_MARGIN_PCT=65.0
```

---

## EXAMPLE USAGE

### Scenario: Market in TREND_STRONG, 3/6 Positions Open

**Current Portfolio:**
1. BTCUSDT - score=75.0, returns=[...], margin=10%
2. ETHUSDT - score=72.0, returns=[...], margin=10%
3. SOLUSDT - score=70.0, returns=[...], margin=10%

**New Intent:** Open ADAUSDT

**Market Features:**
```python
{
    "trend_strength": 0.80,  # Strong trend
    "atr_pct": 1.5,          # Normal volatility
    "momentum_consistency": 0.7  # High consistency
}
```

**Evaluation Flow:**

1. **Load Policy:** ✅ ADAUSDT in universe
2. **Detect Regime:** TREND_STRONG (slots=6)
3. **Check Slots:** 3/6 open → 3 available ✅
4. **Check Margin:** 30% < 65% ✅
5. **Check Correlation:** Fetch ADAUSDT returns, compute corr with portfolio
   - vs BTCUSDT: 0.35 ✅
   - vs ETHUSDT: 0.42 ✅
   - vs SOLUSDT: 0.28 ✅
   - Max correlation: 0.42 < 0.80 ✅

**Decision:** ALLOW

**Logs:**
```
ACTIVE_SLOTS desired=6 open=3 available=3 regime=TREND_STRONG 
             regime_confidence=0.85 symbol=ADAUSDT
ACTIVE_SLOTS_ALLOW symbol=ADAUSDT score=78.00 slots=4/6
```

---

## TESTING COMMANDS

### Local Test (Proof Script)
```bash
cd /home/qt/quantum_trader
python test_active_positions_controller.py
```

**Expected Output:**
```
✅ SCENARIO 1 PASSED: Slots full blocks new opens
✅ SCENARIO 2 PASSED: Rotation triggered for 20% better candidate
✅ SCENARIO 3 PASSED: High correlation blocks new position
✅ SCENARIO 4 PASSED: High margin usage blocks new position
✅ SCENARIO 5 PASSED: Missing policy blocks (fail-closed)
✅ SCENARIO 6 PASSED: Regime changes adjust slot count
✅ ALL TESTS PASSED
```

### Monitor Live Decisions
```bash
# Watch slot allocations
journalctl -u quantum-governor -f | grep ACTIVE_SLOTS

# Watch blocks
journalctl -u quantum-governor -f | grep "BLOCKED_"

# Watch rotations
journalctl -u quantum-governor -f | grep ROTATION_CLOSE

# Watch correlation blocks
journalctl -u quantum-governor -f | grep CORR_BLOCK
```

---

## STATISTICS AND METRICS

### Controller Stats (Available via .get_stats())

```python
{
    "total_decisions": 150,
    "allows": 75,
    "slots_full_blocks": 30,
    "no_policy_blocks": 0,
    "correlation_blocks": 20,
    "margin_blocks": 15,
    "rotations_triggered": 10,
    "current_state": {
        "desired_slots": 6,
        "open_positions_count": 4,
        "available_slots": 2,
        "regime": "TREND_STRONG",
        "regime_confidence": 0.85,
        "total_margin_usage_pct": 45.0,
        "policy_available": True,
    }
}
```

---

## BENEFITS SUMMARY

### Risk Management
✅ **Regime-Adaptive Exposure** - Reduces positions in choppy/volatile markets  
✅ **Correlation Diversity** - Prevents all positions moving together  
✅ **Margin Discipline** - Hard cap at 65% margin usage  
✅ **Fail-Closed** - No trading without valid policy  

### Performance Optimization
✅ **Capital Rotation** - Upgrades to better opportunities  
✅ **Selective Entry** - Only takes best candidates when slots limited  
✅ **Quality Over Quantity** - Doesn't force 10 positions  

### Operational Excellence
✅ **No Hardcoding** - 100% policy-driven  
✅ **Deterministic** - Clear decision logic  
✅ **Observable** - Grep-friendly logs  
✅ **Testable** - Comprehensive proof script  

---

## NEXT STEPS

### Integration Tasks (Required for Production)

1. **Intent Bridge Enhancement**
   - Add `candidate_score` computation
   - Add `candidate_returns` fetch from klines
   - Add `market_features` dict (trend_strength, atr_pct, momentum_consistency)
   - Pass to SafetyGovernor

2. **Apply Layer Enhancement**
   - Create `portfolio_positions` snapshot
   - Include: symbol, size_usd, leverage, unrealized_pnl_usd, entry_price, current_price, age_seconds, score
   - Optionally add `returns_series` for correlation checks
   - Compute `total_margin_usage_pct`
   - Pass to SafetyGovernor

3. **Rotation Execution**
   - When `ROTATION_TRIGGERED` decision:
     - Close position at `record.weakest_symbol`
     - Allow open of `record.symbol`
   - Log: `ROTATION_EXECUTED weakest=... new=...`

4. **Monitoring Dashboard**
   - Add slot utilization gauge (open/desired)
   - Add regime indicator
   - Add correlation heatmap
   - Add margin usage meter

### Optional Enhancements

1. **Machine Learning Score** - Replace manual score with ML model prediction
2. **Dynamic Thresholds** - Adjust rotation_threshold based on market volatility
3. **Symbol Scoring** - Add real-time quality scoring for all universe symbols
4. **Historical Analysis** - Track rotation success rate

---

## CONCLUSION

✅ **Active Positions Controller Implemented**

**Key Achievements:**
- Dynamic slot allocation (3-6 based on regime)
- Capital rotation (15%+ improvement threshold)
- Correlation gating (max 80%)
- Margin gating (max 65%)
- Policy-driven (no hardcoding)
- Fail-closed (blocks if no policy)
- Fully tested (6 scenarios, all pass)

**Production Status:** ✅ **READY FOR INTEGRATION**

**Next Action:** Integrate with Intent Bridge and Apply Layer to provide required inputs.

---
**END OF DOCUMENT**
