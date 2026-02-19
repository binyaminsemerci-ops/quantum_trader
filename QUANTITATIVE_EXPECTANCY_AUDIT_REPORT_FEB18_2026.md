# QUANTITATIVE EXPECTANCY AUDIT REPORT
## Statistical Analysis of Live Trading System Performance

**Audit Date:** February 18, 2026  
**System:** Quantum Trader Production (Binance Futures Testnet)  
**Audit Type:** Statistical Expectancy Analysis  
**Auditor:** Forensic Quantitative Analyst  

---

## EXECUTIVE SUMMARY

### Audit Objective

Determine whether the trading strategy has positive expectancy through quantitative analysis of closed trade history.

### Critical Finding: DATA INSUFFICIENT ❌

**Sample Size:** 11 closed trades  
**Minimum Required:** 174 trades (95% confidence interval)  
**Data Completeness:** 6.3%  

**Status:** Statistical analysis **IMPOSSIBLE** with current sample size.

### What Can Be Determined

✅ System is **operationally functional** (executing and closing positions)  
⚠️  Early indicators **POTENTIALLY positive** (but not statistically significant)  
❌ **NO RELIABLE CONCLUSIONS** about system expectancy can be drawn  

---

## PHASE 1: TRADE HISTORY EXTRACTION

### Data Sources Examined

| Source | Status | Records Found |
|--------|--------|---------------|
| **Redis** `quantum:ledger:*` | ✅ CONNECTED | 2 ledgers, 11 trades |
| **SQLite** `backend/data/trades.db` | ❌ TABLE NOT FOUND | 0 trades |
| **Redis** `apply.result` stream | ✅ AVAILABLE | 10,022 events (mostly skipped) |

### Extraction Summary

```
Ledger Keys Scanned: 2
├─ quantum:ledger:RIVERUSDT (5 trades)
└─ quantum:ledger:[other symbol] (6 trades)

Total Closed Trades: 11
Data Fields Available:
├─ timestamp ✓
├─ pnl_usdt ✓
├─ pnl_pct ✓
├─ entry_price ❌ (not tracked per trade)
├─ exit_price ❌ (not tracked per trade)
├─ stop_loss_price ❌ (not tracked per trade)
├─ position_size ❌ (not tracked per trade)
└─ leverage ❌ (ledger-level only)
```

### Data Quality Assessment

| Requirement | Status | Impact |
|-------------|--------|--------|
| Minimum 200 trades | ❌ **FAIL** (11/200 = 5.5%) | Cannot establish statistical significance |
| Entry/Exit prices | ❌ **MISSING** | Cannot calculate R-multiples |
| Stop-loss prices | ❌ **MISSING** | Cannot determine risk per trade |
| Position sizing | ❌ **MISSING** | Cannot assess risk-adjusted returns |
| Leverage per trade | ❌ **MISSING** | Cannot analyze leverage sensitivity |

**Verdict Phase 1:** Data extraction successful but **data completeness insufficient** for rigorous analysis.

---

## PHASE 2: R-MULTIPLE CALCULATION

### Calculation Formula

**For LONG positions:**
```
R = (Exit Price - Entry Price) / (Entry Price - Stop Loss)
```

**For SHORT positions:**
```
R = (Entry Price - Exit Price) / (Stop Loss - Entry Price)
```

### Calculation Status

❌ **CANNOT BE PERFORMED**

**Reason:** Required data fields are not available in current logging infrastructure.

### Available Data Fields

Current trade history format stores only:
- `timestamp` — Trade completion time
- `pnl_usdt` — Realized PnL in USDT
- `pnl_pct` — Realized PnL as percentage

**Missing Critical Fields:**
- Entry price per trade
- Exit price per trade
- Stop-loss price per trade
- Position size per trade

### Alternative Metric: PnL Distribution

Since R-multiple calculation is impossible, we analyzed **absolute PnL distribution** instead:

```
Trade PnL Values (USDT):
[ +3.2, -4.1, +2.1, +8.5, -5.0, +12.3, +7.6, +9.1, +6.4, -3.2, +18.4 ]

Mean: $5.03 USDT
Std Dev: $6.77 USDT
Range: [-$5.00, +$18.40]
```

**Verdict Phase 2:** R-multiple analysis **NOT POSSIBLE** with current data structure. Fallback to absolute PnL analysis.

---

## PHASE 3: CORE METRICS

### Statistical Metrics Computed

| Metric | Value | Statistical Validity |
|--------|-------|---------------------|
| **Total Trades** | 11 | ❌ Insufficient |
| **Winning Trades** | 8 (72.7%) | ⚠️  Unreliable |
| **Losing Trades** | 3 (27.3%) | ⚠️  Unreliable |
| **Win Rate** | 72.7% | ⚠️  Unreliable |
| **Average Win** | $8.46 USDT | ⚠️  Unreliable |
| **Average Loss** | -$4.13 USDT | ⚠️  Unreliable |
| **Profit Factor** | 5.46 | ⚠️  Unreliable |
| **Expectancy** | $5.03 USDT/trade | ⚠️  Unreliable |
| **Total PnL** | $55.30 USDT | ✓ Factual |
| **Standard Deviation** | $6.77 USDT | ⚠️  Unreliable |
| **Standard Error** | $2.04 USDT | ⚠️  Unreliable |

### Statistical Confidence Calculation

**Required Sample Size for 95% Confidence:**

Using formula: `n = (Z × σ / E)²`  
Where:
- Z = 1.96 (95% confidence)
- σ = $6.77 (standard deviation)
- E = $1.01 (20% of mean, desired precision)

**Result:** n = **174 trades minimum**

**Current sample:** 11 trades = **6.3% of required minimum**

### Expectancy Interpretation

```
Expectancy Formula:
E = (Win% × Avg Win) - (Loss% × |Avg Loss|)
E = (0.727 × $8.46) - (0.273 × $4.13)
E = $6.15 - $1.13
E = $5.02 USDT per trade
```

**Confidence Interval (95%):**

With Standard Error = $2.04:
- Lower bound: $5.03 - (1.96 × $2.04) = **$1.03 USDT**
- Upper bound: $5.03 + (1.96 × $2.04) = **$9.03 USDT**

**Range includes positive values**, suggesting potential positive expectancy, **BUT margin of error is 80% of the mean**, indicating **extreme uncertainty**.

### Profit Factor Analysis

```
Profit Factor = Gross Profit / Gross Loss
             = $67.70 / $12.40
             = 5.46
```

**Interpretation:**
- PF > 1.0 suggests profitable system
- PF = 5.46 is **extremely high** (typical profitable systems: 1.5-2.5)
- **Likely distorted by small sample size**
- With 11 trades, even 1-2 outliers dominate the metric

### Verdict Phase 3

**Calculated Metrics:** ✅ COMPLETE  
**Statistical Validity:** ❌ **INVALID**  
**Reliability:** ⚠️  **UNRELIABLE** (sample 16× below minimum)

**Classification (Tentative):**
- Win rate 72.7% > 50% → Potentially directionally accurate
- Profit factor 5.46 > 1.0 → Potentially profitable
- Expectancy $5.03 > 0 → Potentially positive edge

**BUT:** All metrics have **confidence intervals wider than the metrics themselves**.

---

## PHASE 4: DISTRIBUTION ANALYSIS

### Histogram Request

**Cannot be generated:** Minimum 50-100 data points required for meaningful histogram.

With 11 trades, any binning approach produces mostly empty bins.

### Qualitative Distribution Observations

**Winners (8 trades):**
```
$18.40 (outlier, 1 trade)
$12.30
$9.10
$8.50
$7.60
$6.40
$3.20
$2.10
```

**Observations:**
- ✓ Winners range from **$2.10 to $18.40** (9× range)
- ✓ **No obvious capping** visible (but sample too small)
- ⚠️  One outlier ($18.40) represents 33% of total profit
- ⚠️  Removing outlier: Avg win = $7.14 (vs $8.46 with outlier)

**Losers (3 trades):**
```
-$5.00
-$4.10
-$3.20
```

**Observations:**
- ✓ Losses range from **-$3.20 to -$5.00** (1.6× range)
- ✓ Losses appear **contained** (no fat tails visible)
- ⚠️  Sample too small to confirm clustering
- ⚠️  Cannot determine if losses hit full stop-loss (-1R)

### Key Distribution Questions

| Question | Answer | Confidence |
|----------|--------|------------|
| **Are winners capped?** | NO visible cap | ⚠️  LOW (outlier exists) |
| **Are losses clustering at −1R?** | CANNOT DETERMINE | ❌ No R-data |
| **Are there fat tail losses beyond −1R?** | NO fat tails visible | ⚠️  LOW (only 3 losses) |
| **Is distribution symmetric?** | NO (positively skewed) | ⚠️  LOW |

### Fat Tail Risk Analysis

**Definition:** Losses exceeding -1.5R (or beyond stop-loss)

**Current Data:**
- Largest loss: -$5.00 USDT
- Without entry_risk data, cannot calculate R-multiple
- Cannot determine if -$5.00 exceeds expected stop-loss

**Proxy Analysis:**
- If typical position size is $100-200 USDT
- If stop-loss is 2% ($2-4 USDT)
- Then -$5.00 is **within or slightly beyond** expected SL range
- **But this is speculative without actual data**

### Verdict Phase 4

**Distribution Analysis:** ⚠️  **INCONCLUSIVE**

**Reasons:**
1. Sample size too small for histogram
2. Cannot calculate R-multiples
3. Cannot determine clustering patterns
4. Cannot assess tail risk quantitatively

**Observable Patterns:**
- ✓ Winners show variation (not obviously capped)
- ✓ Losses appear contained (no extreme outliers)
- ✓ Positive skew (more/larger wins than losses)

**But ALL observations lack statistical power.**

---

## PHASE 5: LEVERAGE SENSITIVITY

### Analysis Request

Recalculate expectancy assuming different leverage levels (10x, 5x) to determine if leverage amplifies structural loss.

### Challenge

**Leverage sensitivity analysis requires:**
1. Entry risk per trade (% of capital)
2. R-multiples per trade
3. Position sizing per trade

**Available data:**
- ❌ Entry risk not tracked
- ❌ R-multiples cannot be calculated
- ❌ Position sizing not tracked per trade
- ⚠️  Only total PnL in USDT available

### Alternative Approach: PnL% Estimation

**Assumptions (speculative):**
- Testnet account balance: ~$1000 USDT
- Current leverage: 30×
- Typical position size: 10-20% of capital ($100-200)
- Entry risk per trade: 2% of capital ($20)

**Expectancy as % of capital:**

With 30× leverage, $5.03 USDT profit per trade:
- As % of $1000 capital: **0.50%** per trade

**Projection at different leverage:**

| Leverage | Risk/Trade | Expectancy USDT | Expectancy % | Notes |
|----------|------------|-----------------|--------------|-------|
| **5×** | 2%/5 = 0.4% | $5.03/6 = $0.84 | 0.08% | Scaled linearly |
| **10×** | 2%/3 = 0.67% | $5.03/3 = $1.68 | 0.17% | Scaled linearly |
| **20×** | 2%/1.5 = 1.33% | $5.03/1.5 = $3.35 | 0.34% | Scaled linearly |
| **30×** | 2% | $5.03 | 0.50% | Current |

**Interpretation:**

IF system has positive expectancy:
- ✓ Lower leverage reduces profit per trade
- ✓ Higher leverage amplifies profit per trade
- ✓ Risk proportional to leverage (expected)

**BUT:** This assumes:
1. Position sizing scales linearly with leverage (unverified)
2. Entry risk remains constant (unverified)
3. Market impact unchanged (unlikely at higher size)
4. Expectancy remains constant across leverage (unlikely)

### Leverage Impact on Drawdown Risk

**What we CANNOT determine:**
- Maximum drawdown at current leverage
- Margin call proximity
- Liquidation risk exposure
- Volatility of equity curve

**What leverage WILL amplify:**
- ✓ Profit per trade (if positive expectancy)
- ✓ Loss per trade (if negative expectancy)
- ✓ Account volatility
- ✓ Ruin risk

### Verdict Phase 5

**Leverage Sensitivity Analysis:** ❌ **CANNOT BE PERFORMED**

**Reasons:**
1. No entry risk data
2. No position sizing data
3. No leverage tracking per trade
4. Account balance unknown (for % calculations)

**Speculative Conclusion:**
- IF expectancy is positive ($5.03/trade)
- THEN leverage amplifies gains
- BUT also amplifies variance and ruin risk

**Reliable analysis requires 200+ trades with full position data.**

---

## PHASE 6: VERDICT

### Classification Framework

**Expectancy Categories:**
- **A) POSITIVE EXPECTANCY:** E > 0 with statistical significance
- **B) NEUTRAL EXPECTANCY:** E ≈ 0 within confidence interval
- **C) NEGATIVE EXPECTANCY:** E < 0 with statistical significance

### Statistical Classification

**❌ CLASSIFICATION NOT POSSIBLE**

**Reason:** Sample size (n=11) is **16× below** minimum required for 95% confidence (n=174).

### Provisional Assessment (Statistically Invalid)

Based on **11 trades only** (unreliable):

**Metric-Based Indication:**
- Expectancy: +$5.03 USDT/trade → **POTENTIALLY POSITIVE**
- Profit Factor: 5.46 → **POTENTIALLY POSITIVE**
- Win Rate: 72.7% → **POTENTIALLY POSITIVE**

**Confidence Interval:**
- 95% CI: [$1.03, $9.03] → **Does not include zero**
- **BUT:** Margin of error (±$4.00) is 80% of mean
- **Interpretation:** "Positive somewhere between barely profitable and highly profitable"

### Provisional Classification

```
⚠️  TENTATIVELY CLASSIFIED AS:
    POTENTIALLY POSITIVE EXPECTANCY (UNCONFIRMED)
```

**Support:**
- All metrics point in positive direction
- Confidence interval excludes zero (barely)
- Profit factor substantially > 1.0

**Caveats:**
- Sample 16× too small
- One outlier (+$18.40) dominates results
- No R-multiple validation
- No risk-adjusted metrics
- Early data may not represent long-term behavior

### Root Cause Analysis (If Negative)

**NOT APPLICABLE** — Metrics indicate potential positive expectancy.

**However, IF future data reveals negative expectancy, likely causes would be:**

1. **Low Win Rate (<45%)**
   - Current: 72.7% (but unreliable)
   - Watch for regression to mean with more data
   
2. **Large Average Loss**
   - Current: -$4.13 vs +$8.46 win (ratio: 1:2.0)
   - Appears favorable (but unreliable)
   
3. **Small Average Win**
   - Current: +$8.46 USDT
   - Relative to losses: 2.0× larger (favorable)
   - May indicate early exit (harvest ladder effect from architectural audit)
   
4. **Fat-Tail Loss**
   - Current: Max loss -$5.00 (contained)
   - No fat tails visible (but only 3 losses observed)
   
5. **Exit Asymmetry**
   - Winners: 8 trades, range $2.10-$18.40 (variation suggests not heavily capped)
   - Losers: 3 trades, range -$3.20 to -$5.00 (clustering suggests SL working)
   - **MAY have harvest ladder effect** (per architectural audit finding)
   - But cannot confirm without more data

### Issue Priority (Speculative)

**IF negative expectancy emerges with more data, likely culprit:**

1. **EXIT ASYMMETRY** (harvest ladder closes winners early)
   - Architectural audit found: 50% closed by 1.0R
   - Current data shows variation, but sample too small
   
2. **POSITION SIZING** (too small for edge to overcome fees)
   - Average trade ~$5 profit on unknown position size
   - If positions are $10-20 (per architectural audit), ROI ≈ 25-50% per trade (unlikely)
   - More likely: positions $100-200, ROI ≈ 2.5-5% (reasonable)
   
3. **REGIME DEPENDENCY** (system may work only in certain market conditions)
   - 11 trades insufficient to assess across regimes
   - Need minimum 50+ trades per regime

### Honest Assessment

**What We Know:**
✓ System is operational (11 trades completed)  
✓ Short-term PnL is positive (+$55.30 total)  
✓ Early metrics trend positive (but not significant)  

**What We Do NOT Know:**
❌ Long-term expectancy  
❌ Risk-adjusted returns  
❌ Distribution characteristics  
❌ Leverage sensitivity  
❌ Regime stability  
❌ Whether profit is skill or luck  

**Statistical Truth:**

With 11 trades, we have **94% uncertainty**. The profit could be:
- Random variance (pure luck)
- Early cherry-picked results
- System warming up (not yet stable)
- Genuine edge (but unproven)

**We cannot distinguish these scenarios with current data.**

---

## CONCLUSIONS

### Primary Finding

**INSUFFICIENT DATA** — Statistical analysis cannot be completed with n=11 trades.

### Secondary Finding

**PRELIMINARY INDICATORS POSITIVE** — Early data suggests potential edge, but lacks statistical significance.

### Tertiary Finding

**INFRASTRUCTURE GAP** — Trade logging does not capture entry/exit/stop-loss data required for R-multiple analysis.

---

## RECOMMENDATIONS

### Immediate Actions

1. **CONTINUE TRADING** — Accumulate minimum 200 closed trades before drawing conclusions
2. **ENABLE DETAILED LOGGING** — Modify trade_history_logger to capture:
   - Entry price per trade
   - Exit price per trade
   - Stop-loss price per trade
   - Position size per trade
   - Leverage per trade
3. **VERIFY LOGGER SERVICE** — Ensure trade_history_logger is running and functional
4. **MONITOR LEDGER GROWTH** — Track trade accumulation weekly

### Analysis Timeline

| Milestone | Trades Required | Est. Time | Action |
|-----------|----------------|-----------|---------|
| **Preliminary Signal** | 30 trades | ~1-2 weeks | Rerun with 95% CI ≥ ±50% |
| **Initial Assessment** | 100 trades | ~1 month | Rerun with 95% CI ≥ ±30% |
| **Statistical Validation** | 200 trades | ~2 months | Rerun with 95% CI ≥ ±20% |
| **Robust Conclusion** | 500+ trades | ~4-6 months | Full audit with regime analysis |

### Do NOT Do

❌ **DO NOT** tune system parameters based on current data  
❌ **DO NOT** increase leverage based on early positive results  
❌ **DO NOT** draw firm conclusions about system quality  
❌ **DO NOT** use current metrics for risk sizing  
❌ **DO NOT** assume positive expectancy is confirmed  

### Do Continue

✅ **DO** monitor for consistent PnL growth  
✅ **DO** track trade count toward 200-trade milestone  
✅ **DO** log full trade details for future analysis  
✅ **DO** rerun audit monthly as data accumulates  
✅ **DO** maintain conservative risk management regardless of early results  

---

## APPENDIX: DATA INTEGRITY NOTES

### Redis Ledger Structure (Actual)

```json
{
  "total_pnl_usdt": "55.3",
  "total_trades": "11",
  "winning_trades": "8",
  "losing_trades": "3",
  "total_volume_usdt": "3000.0",
  "trade_history": [
    {
      "timestamp": 1770098629.59,
      "pnl_usdt": 3.2,
      "pnl_pct": 0.3
    },
    ...
  ]
}
```

### Required Schema Enhancement

```json
{
  "trade_history": [
    {
      "timestamp": 1770098629.59,
      "symbol": "BTCUSDT",
      "side": "LONG",
      "entry_price": 50000.00,
      "exit_price": 50500.00,
      "stop_loss": 49500.00,
      "qty": 0.010,
      "leverage": 30,
      "pnl_usdt": 5.00,
      "pnl_pct": 1.00,
      "entry_risk_usd": 5.00,
      "r_multiple": 1.00
    }
  ]
}
```

### Implementation Path

1. Update `microservices/trade_history_logger/trade_history_logger.py`
2. Modify `update_ledger_history()` function to capture full trade details
3. Source data from `apply.result` stream + position snapshots
4. Backfill entry_price/stop_loss from ledger context
5. Calculate R-multiple on write

---

## AUDIT ATTESTATION

**Audit Scope:** Statistical expectancy analysis of closed trade history  
**Data Sources:** Redis ledger (2 symbols, 11 trades)  
**Analysis Method:** Classical expectancy calculation, confidence intervals  
**Limitations:** Sample size 16× below statistical minimum  

**Conclusion:** Analysis **INCONCLUSIVE** due to insufficient data.

**Recommendation:** Rerun audit after accumulating 200+ closed trades.

**Auditor Note:** Current positive indicators are **encouraging but not probative**. System requires extended operational period to validate expectancy claims.

---

**Report ID:** QUANT-EXPECT-AUDIT-2026-02-18  
**Version:** 1.0  
**Status:** INCOMPLETE — DATA INSUFFICIENT  
**Next Scheduled Audit:** After 100 trades accumulated  

**END OF REPORT**
