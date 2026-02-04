# 🎯 HARVEST & EXIT BEREGNINGSFORMLER
**Quantum Trader Position Harvesting & Exit Calculations**
*Generert: 2025-01-16*

---

## 📊 OVERSIKT

Dette dokumentet kartlegger **alle beregningsformler** relatert til:
1. **Profit Harvesting** (partial TP, R-basert)
2. **Exit Decisions** (kill_score, full close)
3. **Adaptive TP/SL Levels** (LSF-basert, 3-stage)
4. **Trailing Stop Logic**

**VIKTIG**: Dette er EXIT/HARVEST formler, IKKE entry-side formler (margin, leverage, initial TP/SL).

---

## 🧠 1. PROFIT HARVESTING ENGINE (P2)
**Fil**: `ai_engine/risk_kernel_harvest.py`  
**Ansvar**: Beregner når og hvor mye profitt som skal tas ut (R-basert)

### 1.1 Risk Unit Beregning
```python
# Prefererer P1 stop_dist_pct hvis tilgjengelig, ellers fallback
if p1_proposal and p1_proposal.stop_dist_pct:
    risk_unit = entry_price * p1_proposal.stop_dist_pct
else:
    risk_unit = entry_price * fallback_stop_pct  # Default: 0.02 (2%)

risk_unit = max(risk_unit, 1e-9)  # Unngå division by zero
```

**Eksempel**:
- Entry price: $100
- Stop distance: 2%
- Risk unit = $100 × 0.02 = $2.00

---

### 1.2 R_net Beregning (Profit i Risk Units)
```python
cost_est = (cost_bps / 10000.0) * entry_price  # cost_bps = 10 (0.1%)
R_net = (unrealized_pnl - cost_est) / risk_unit
```

**Eksempel**:
- Unrealized PnL: $5.00
- Cost estimate: $100 × 0.001 = $0.10
- Risk unit: $2.00
- **R_net = ($5.00 - $0.10) / $2.00 = 2.45R**

**Tolkning**:
- R_net = 1.0 → Profit = 1× initial risk
- R_net = 2.0 → Profit = 2× initial risk (double win)
- R_net = 4.0 → Profit = 4× initial risk (quadruple win)

---

### 1.3 Harvest Action Triggers (Trancher)
**Formula**: R_net threshold-based ladder

```python
if R_net >= T3_R:      # Default: 6.0R
    return "PARTIAL_75"  # Close 75% av position
elif R_net >= T2_R:    # Default: 4.0R
    return "PARTIAL_50"  # Close 50% av position
elif R_net >= T1_R:    # Default: 2.0R
    return "PARTIAL_25"  # Close 25% av position
else:
    return "NONE"        # Hold position
```

**Standard Harvest Ladder**:
| R_net | Action | Beskrivelse |
|-------|--------|-------------|
| < 2.0R | NONE | Vent på profit |
| 2.0R - 3.9R | PARTIAL_25 | Lukk 25% (lock in første gevinst) |
| 4.0R - 5.9R | PARTIAL_50 | Lukk 50% (lock in halvparten) |
| ≥ 6.0R | PARTIAL_75 | Lukk 75% (aggressive profit taking) |

**Eksempel** (2.45R fra over):
- R_net = 2.45R → **PARTIAL_25** triggered
- Close 25% av position for å låse inn $1.25 profit

---

### 1.4 Volatility-Adjusted Harvest Fractions
**Formål**: Justere harvest % basert på volatilitet

```python
risk_pct = (entry_risk / entry_price) * 100

if risk_pct < 1.0:
    volatility_scale = 1.4  # Low vol → Close 35% (25% × 1.4)
elif risk_pct > 2.5:
    volatility_scale = 0.6  # High vol → Close 15% (25% × 0.6)
else:
    volatility_scale = 1.0  # Normal ladder
```

**Effekt**:
- **Low volatility**: Close mer (35%, 70%, 105% = full)
- **High volatility**: Close mindre (15%, 30%, 45%)

---

### 1.5 Profit Lock SL (Move to Breakeven+)
**Formula**: Tighten SL til BE+ når R_net ≥ lock_R (default: 1.5R)

```python
if R_net >= lock_R:  # Default: 1.5R
    if side == "LONG":
        be_plus = entry_price * (1 + be_plus_pct)  # Default: 0.002 (0.2%)
        new_sl = max(current_sl, be_plus)  # Monotonic tightening
    else:  # SHORT
        be_plus = entry_price * (1 - be_plus_pct)
        new_sl = min(current_sl, be_plus)
```

**Eksempel** (LONG):
- Entry: $100
- Current SL: $98 (-2%)
- R_net = 1.8R (over 1.5R threshold)
- **New SL = $100.20** (BE + 0.2%)
- Risk eliminated, profit locked

---

## 🔴 2. KILL SCORE (Exit Decision Engine)
**Fil**: `ai_engine/risk_kernel_harvest.py` → `compute_kill_score()`  
**Ansvar**: Beregner edge collapse score → FULL_CLOSE_PROPOSED

### 2.1 Kill Score Components
**Formula**: Weighted sum → Sigmoid [0,1]

```python
# 1. Regime Flip (trend → chop/mr)
if p_trend < trend_min:  # Default: 0.3
    regime_flip = 1.0 if (p_chop + p_mr) > 0.5 else 0.0
else:
    regime_flip = 0.0

# 2. Sigma Spike (volatility explosion)
sigma_ratio = sigma / sigma_ref  # sigma_ref = 0.01
sigma_spike = max(0.0, min(sigma_ratio - 1.0, sigma_spike_cap))  # cap = 2.0

# 3. TS Drop (technical strength collapse)
ts_drop = max(0.0, min(ts_ref - ts, ts_drop_cap))  # ts_ref=0.3, cap=0.5

# 4. Age Penalty (position too old)
age_penalty = max(0.0, min(age_sec / max_age_sec, 1.0))  # max_age = 86400s (24h)
```

---

### 2.2 Kill Score Aggregation
**Formula**: Weighted sum → Sigmoid normalization

```python
z = (
    k_regime_flip  * regime_flip +    # Weight: 1.0
    k_sigma_spike  * sigma_spike +    # Weight: 0.5
    k_ts_drop      * ts_drop +        # Weight: 0.5
    k_age_penalty  * age_penalty      # Weight: 0.3
)

K = 1.0 / (1.0 + exp(-z))  # Sigmoid → [0,1]

if K >= kill_threshold:  # Default: 0.6
    harvest_action = "FULL_CLOSE_PROPOSED"
```

**Eksempel Scenario**:
```
p_trend = 0.2        → regime_flip = 1.0 (trend collapsed)
sigma = 0.025        → sigma_spike = 1.5 (2.5× baseline volatility)
ts = 0.1             → ts_drop = 0.2 (technical strength dropped)
age = 72000 sec      → age_penalty = 0.83 (20h old)

z = 1.0×1.0 + 0.5×1.5 + 0.5×0.2 + 0.3×0.83
  = 1.0 + 0.75 + 0.1 + 0.249
  = 2.099

K = 1 / (1 + exp(-2.099)) = 1 / (1 + 0.122) = 0.891

→ 0.891 > 0.6 → **FULL_CLOSE_PROPOSED**
```

**Reason Codes Generated**:
- `kill_score_triggered`
- `regime_flip` (p_trend < 0.3)
- `sigma_spike` (sigma_spike > 0.5)
- `age_penalty` (age > 12h)

---

## 📐 3. ADAPTIVE TP/SL LEVELS (LSF-basert)
**Fil**: `microservices/exitbrain_v3_5/adaptive_leverage_engine.py`  
**Ansvar**: Beregner 3-stage TP levels + adaptive SL basert på leverage

### 3.1 Leverage Sensitivity Factor (LSF)
**Formula**: Høyere leverage → Lavere LSF → Tighter TP/SL

```python
LSF = 1.0 / (1.0 + ln(leverage + 1))
```

**Eksempel Verdier**:
| Leverage | LSF | Effekt |
|----------|-----|--------|
| 1x | 0.590 | Wide TP/SL (low risk) |
| 5x | 0.359 | Moderate TP/SL |
| 10x | 0.294 | Tighter TP/SL |
| 20x | 0.246 | Very tight TP/SL |
| 50x | 0.201 | Ultra tight TP/SL |
| 100x | 0.176 | Extreme tight TP/SL |

**Intuisjon**: LSF går fra 0.59 (1x) til 0.18 (100x) → Høyere leverage krever tightere targets

---

### 3.2 Multi-Stage TP Calculation
**Formula**: LSF-scalede progressive targets

```python
tp1 = base_tp * (0.6 + LSF)
tp2 = base_tp * (1.2 + LSF / 2.0)
tp3 = base_tp * (1.8 + LSF / 4.0)
```

**Eksempel** (base_tp = 2.0%, leverage = 10x):
```
LSF = 0.294

TP1 = 0.02 * (0.6 + 0.294) = 0.02 * 0.894 = 1.79%
TP2 = 0.02 * (1.2 + 0.147) = 0.02 * 1.347 = 2.69%
TP3 = 0.02 * (1.8 + 0.074) = 0.02 * 1.874 = 3.75%
```

**Med modifiers** (funding, volatility, divergence):
```python
tp_scale = 1.0 + (funding_delta * 0.8)  # +funding → higher TP
sl_scale = 1.0 + (divergence * 0.4) + (volatility * 0.2)  # volatility → wider SL

tp1 *= tp_scale
tp2 *= tp_scale
tp3 *= tp_scale
sl *= sl_scale
```

**Final Clamps**:
```python
# Soft minimums
tp1 = max(tp1, 0.003)  # 0.3% minimum TP
sl = max(sl, 0.0015)   # 0.15% minimum SL

# Hard clamps for SL
sl = min(max(sl, 0.001), 0.02)  # SL ∈ [0.1%, 2.0%]
```

---

### 3.3 Adaptive SL Calculation
**Formula**: Leverage-inverted scaling

```python
sl = base_sl * (1.0 + (1.0 - LSF) * 0.8)
```

**Eksempel** (base_sl = 1.2%, leverage = 10x):
```
LSF = 0.294
sl = 0.012 * (1.0 + (1.0 - 0.294) * 0.8)
   = 0.012 * (1.0 + 0.565)
   = 0.012 * 1.565
   = 1.88%
```

**Clamp check**: 1.88% ∈ [0.1%, 2.0%] ✅

---

### 3.4 Harvest Scheme (Partial TP Allocations)
**Formula**: Leverage-tiered allocations

```python
if leverage <= 10:
    return [0.3, 0.3, 0.4]  # Conservative: Gradual profit taking
elif leverage <= 30:
    return [0.4, 0.4, 0.2]  # Aggressive: Front-load profit taking
else:
    return [0.5, 0.3, 0.2]  # Ultra-aggressive: Maximize early profits
```

**Eksempel** (10x leverage):
- TP1 hit: Close 30% av position
- TP2 hit: Close 30% av remaining position
- TP3 hit: Close 40% av remaining position

**Effekt**:
- Høyere leverage → Mer aggressive early profit taking
- Lower leverage → Gradual, patient profit taking

---

## 🎯 4. EXITBRAIN v3.5 INTEGRATION
**Fil**: `microservices/exitbrain_v3_5/exit_brain.py`  
**Ansvar**: Orchestrates adaptive levels + cross-exchange adjustments

### 4.1 Complete Exit Plan Formula
**Process**:
```python
# 1. Calculate LSF
lsf = compute_lsf(leverage)

# 2. Calculate adaptive TP/SL levels
adaptive_levels = adaptive_engine.compute_levels(
    base_tp_pct=base_tp,
    base_sl_pct=base_sl,
    leverage=leverage,
    volatility_factor=atr / 100.0,
    funding_delta=funding_rate,
    exchange_divergence=exch_divergence
)

# 3. Apply cross-exchange adjustments (Phase 4M+)
if cross_exchange_adjustments:
    tp_multiplier = adjustments.get("tp_multiplier", 1.0)
    sl_multiplier = adjustments.get("sl_multiplier", 1.0)
    base_tp *= tp_multiplier
    base_sl *= sl_multiplier

# 4. Apply safety limits
final_tp = max(min_tp_pct, min(max_tp_pct, base_tp))
final_sl = max(min_sl_pct, min(max_sl_pct, base_sl))

# 5. Trailing callback (if enabled)
trailing_callback = trailing_callback_pct if (atr < 2.0) else None
```

**Safety Ranges** (ExitBrain v3.5 defaults):
```python
min_tp_pct = 1.5%    # Minimum TP (was 0.5%)
max_tp_pct = 10.0%   # Maximum TP (was 3%)
min_sl_pct = 1.2%    # Minimum SL (was 0.3%)
max_sl_pct = 5.0%    # Maximum SL (was 1.5%)
trailing_callback_pct = 0.8%  # Trailing stop callback
```

**Trailing Logic**:
```python
use_trailing = (atr_value < 2.0)  # Disable if volatility > 2%
```

---

### 4.2 Complete Calculation Example
**Scenario**:
```
Symbol: BTCUSDT
Side: LONG
Confidence: 0.75
Entry Price: $50,000
Leverage: 15x
ATR: 1.2%
Funding Rate: +0.01%
Exchange Divergence: 0.05
```

**Step-by-Step**:

**1. Calculate LSF**:
```
LSF = 1 / (1 + ln(15 + 1)) = 1 / (1 + 2.773) = 0.265
```

**2. Calculate base TP/SL** (base_tp=2.0%, base_sl=1.2%):
```
TP1 = 0.02 * (0.6 + 0.265) = 1.73%
TP2 = 0.02 * (1.2 + 0.133) = 2.67%
TP3 = 0.02 * (1.8 + 0.066) = 3.73%
SL  = 0.012 * (1.0 + (1-0.265)*0.8) = 1.91%
```

**3. Apply modifiers**:
```
volatility_factor = 1.2 / 100 = 0.012
funding_delta = 0.0001
divergence = 0.05

tp_scale = 1.0 + (0.0001 * 0.8) = 1.00008  ≈ 1.0
sl_scale = 1.0 + (0.05 * 0.4) + (0.012 * 0.2) = 1.0224

TP1 = 1.73% (unchanged)
TP2 = 2.67% (unchanged)
TP3 = 3.73% (unchanged)
SL  = 1.91% * 1.0224 = 1.95%
```

**4. Clamp SL**:
```
SL = min(max(1.95%, 0.1%), 2.0%) = 1.95% ✅
```

**5. Harvest Scheme** (15x → tier 2):
```
harvest_scheme = [0.4, 0.4, 0.2]
→ Close 40% at TP1, 40% at TP2, 20% at TP3
```

**6. Calculate TP/SL Prices**:
```
TP1 = $50,000 * (1 + 0.0173) = $50,865
TP2 = $50,000 * (1 + 0.0267) = $51,335
TP3 = $50,000 * (1 + 0.0373) = $51,865
SL  = $50,000 * (1 - 0.0195) = $49,025
```

**7. Trailing Stop** (if ATR < 2%):
```
ATR = 1.2% < 2.0% → Trailing ENABLED
Callback = 0.8%
```

**FINAL EXIT PLAN**:
```
Leverage: 15x
TP1: $50,865 (1.73%) → Close 40%
TP2: $51,335 (2.67%) → Close 40% of remaining
TP3: $51,865 (3.73%) → Close 20% of remaining
SL: $49,025 (1.95%)
Trailing: Enabled (0.8% callback)
```

---

## 📊 5. HARVEST BRAIN STREAM INTEGRATION
**Fil**: `microservices/harvest_brain/harvest_brain.py`  
**Ansvar**: Listen to execution.result → Track positions → Publish harvest intents

### 5.1 Position Tracking from Executions
```python
# Ingest execution.result events
if status == "FILLED" or status == "PARTIAL":
    if symbol not in positions:
        # New position
        positions[symbol] = Position(
            symbol=symbol,
            side=side,
            qty=qty,
            entry_price=entry_price,
            current_price=price,
            entry_risk=abs(entry_price - stop_loss),
            stop_loss=stop_loss
        )
    else:
        # Update existing position
        pos.qty += qty if side == pos.side else -qty
        pos.current_price = price
        pos.unrealized_pnl = (price - entry_price) * qty  # LONG
```

### 5.2 R-Level Calculation
```python
def r_level(position) -> float:
    """Current profit in risk units"""
    if position.entry_risk <= 0:
        return 0.0
    return position.unrealized_pnl / position.entry_risk
```

### 5.3 Harvest Intent Publishing
```python
if r_level >= harvest_trigger:
    harvest_intent = HarvestIntent(
        intent_type='HARVEST_PARTIAL',
        symbol=symbol,
        side='SELL' if pos.side == 'LONG' else 'BUY',  # Exit side
        qty=pos.qty * fraction,  # 0.25, 0.50, 0.75
        reason=f'R={r_level:.2f}R >= trigger={harvest_trigger:.2f}R',
        r_level=r_level,
        reduce_only=True,
        dry_run=(harvest_mode == 'shadow')
    )
    
    redis.xadd('quantum:stream:trade.intent', asdict(harvest_intent))
```

**Shadow vs Live Mode**:
- `shadow`: Dry run logging (default)
- `live`: Publish to trade.intent for execution

---

## 🎯 6. KOMPLETT EKSEMPEL (Real Trade)
**Trade Setup**:
```
Symbol: ETHUSDT
Side: LONG
Entry: $3,000
Qty: 1.0 ETH
Leverage: 10x
Stop Loss: $2,940 (-2%)
Initial Risk: $60
```

**Exit Plan Calculation** (ExitBrain v3.5):
```
LSF = 1/(1+ln(11)) = 0.294
TP1 = 2% * (0.6 + 0.294) = 1.79% → $3,053.70
TP2 = 2% * (1.2 + 0.147) = 2.69% → $3,080.70
TP3 = 2% * (1.8 + 0.074) = 3.75% → $3,112.50
SL = 1.2% * (1 + 0.565) = 1.88% → $2,943.60
Harvest: [30%, 30%, 40%]
```

**Trade Evolution**:

**T+1h: Price = $3,054 (+1.8%)**
```
Unrealized PnL = ($3,054 - $3,000) * 1.0 = $54
R_net = ($54 - $3) / $60 = 0.85R
Action: NONE (< 2.0R threshold)
```

**T+4h: Price = $3,150 (+5.0%)**
```
Unrealized PnL = $150
R_net = ($150 - $3) / $60 = 2.45R
Action: **PARTIAL_25** triggered (2.45R >= 2.0R)
→ Close 0.25 ETH @ $3,150
→ Realized profit: $37.50
→ Remaining: 0.75 ETH
→ Move SL to BE+0.2% = $3,006
```

**T+8h: Price = $3,280 (+9.3%)**
```
Remaining PnL = ($3,280 - $3,000) * 0.75 = $210
R_net = $210 / $60 = 3.5R (on original risk unit)
Action: Still PARTIAL_25 tier (3.5R < 4.0R)
→ No new action (already closed 25%)
```

**T+12h: Price = $3,350 (+11.7%)**
```
Remaining PnL = ($3,350 - $3,000) * 0.75 = $262.50
R_net = $262.50 / $60 = 4.38R
Action: **PARTIAL_50** triggered (4.38R >= 4.0R)
→ Close 0.375 ETH @ $3,350 (50% of original)
→ Realized profit: $131.25 (additional)
→ Remaining: 0.375 ETH
```

**T+20h: Market Regime Change**
```
p_trend drops from 0.6 to 0.2
sigma spikes from 0.01 to 0.025
ts drops from 0.4 to 0.15
age = 72000 sec (20h)

Kill Score Calculation:
regime_flip = 1.0 (trend collapsed)
sigma_spike = 1.5 (2.5× baseline)
ts_drop = 0.15 (technical strength dropped)
age_penalty = 0.83 (20h old)

z = 1.0×1.0 + 0.5×1.5 + 0.5×0.15 + 0.3×0.83 = 2.024
K = 1/(1+exp(-2.024)) = 0.883

Action: **FULL_CLOSE_PROPOSED** (K=0.883 > 0.6 threshold)
→ Close remaining 0.375 ETH @ $3,320
→ Final realized profit: $120 (additional)

Total Realized: $37.50 + $131.25 + $120 = $288.75
Total R: $288.75 / $60 = 4.81R
```

**Summary**:
- **Entry**: 1.0 ETH @ $3,000 (10x leverage)
- **Partial Exit 1**: 0.25 ETH @ $3,150 (R=2.45R)
- **Partial Exit 2**: 0.375 ETH @ $3,350 (R=4.38R)
- **Full Close**: 0.375 ETH @ $3,320 (Kill Score=0.883)
- **Final Profit**: $288.75 (**4.81R**)

---

## 📋 7. TRIGGER SUMMARY TABLE

### Harvest Triggers (R-basert)
| Trigger | R_net Threshold | Action | Default % |
|---------|-----------------|--------|-----------|
| T1 | 2.0R | PARTIAL_25 | 25% |
| T2 | 4.0R | PARTIAL_50 | 50% |
| T3 | 6.0R | PARTIAL_75 | 75% |
| Profit Lock | 1.5R | Move SL to BE+0.2% | - |

### Kill Score Triggers
| Component | Weight | Range | Trigger Condition |
|-----------|--------|-------|-------------------|
| Regime Flip | 1.0 | [0,1] | p_trend < 0.3 |
| Sigma Spike | 0.5 | [0,2] | sigma > 2× baseline |
| TS Drop | 0.5 | [0,0.5] | ts < 0.3 |
| Age Penalty | 0.3 | [0,1] | age > 12h |
| **Total K** | - | [0,1] | K > 0.6 → FULL_CLOSE |

### Adaptive TP/SL Ranges
| Parameter | Soft Minimum | Hard Clamp | Typical Range |
|-----------|--------------|------------|---------------|
| TP1 | 0.3% | - | 1.5% - 3.0% |
| TP2 | 0.3% | - | 2.5% - 4.5% |
| TP3 | 0.3% | - | 3.5% - 6.0% |
| SL | 0.15% | [0.1%, 2.0%] | 1.2% - 2.0% |

---

## 🔍 8. FORMULA DEPENDENCIES

### Entry → Harvest Flow
```
TradingMathematician (entry.py)
  ↓ entry_price, stop_loss, qty
ExitBrain v3.5 (exit_brain.py)
  ↓ leverage, TP1/TP2/TP3, SL, harvest_scheme
Execution (auto_executor)
  ↓ execution.result stream
HarvestBrain (harvest_brain.py)
  ↓ Track position, calculate R_net
  ↓ trigger harvest at T1/T2/T3
P2 Harvest Kernel (risk_kernel_harvest.py)
  ↓ Compute kill_score
  ↓ Propose FULL_CLOSE if K > 0.6
Apply Layer
  ↓ Execute harvest/close intents
```

---

## 📌 9. KEY OBSERVATIONS

### 9.1 Design Philosophy
1. **R-based thinking**: All profit targets normalized to risk units
2. **Leverage-aware**: LSF adjusts TP/SL based on leverage exposure
3. **Multi-stage exits**: 3-tier TP ladder for gradual profit taking
4. **Kill score safety**: Automatic full close on edge collapse

### 9.2 Safety Mechanisms
1. **Hard clamps**: SL ∈ [0.1%, 2.0%], TP ≥ 0.3%
2. **Monotonic tightening**: SL only moves in protective direction
3. **Cost estimation**: 10 bps cost deducted from PnL
4. **Age penalty**: Old positions automatically flagged for exit

### 9.3 Adaptive Features
1. **Volatility adjustment**: Harvest fractions scale with risk_pct
2. **Funding adjustment**: TP targets increase with positive funding
3. **Divergence adjustment**: SL widens with cross-exchange divergence
4. **Trailing stops**: Enabled only when ATR < 2%

---

## ✅ VALIDERING

**Formler verificate mot**:
- ✅ `exitbrain_v3_5/exit_brain.py` (ExitBrain v3.5)
- ✅ `exitbrain_v3_5/adaptive_leverage_engine.py` (LSF, TP/SL)
- ✅ `harvest_brain/harvest_brain.py` (R-based harvesting)
- ✅ `ai_engine/risk_kernel_harvest.py` (Kill score, P2)

**Produksjonsstatus**:
- ExitBrain v3.5: ✅ ACTIVE
- Harvest Brain: ❌ BROKEN (execution.result stale)
- P2 Harvest Kernel: ✅ CALC-ONLY (pure functions)

---

**END OF HARVEST FORMULAS REPORT**
