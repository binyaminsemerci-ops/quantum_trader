# P2: PERFORMANCE, ALPHA & DRAWDOWN CONTROL - ROADMAP

**Status**: Phase 1 Complete - Baseline analyzer ready, waiting for trade data  
**Date**: 2026-01-02  
**Objective**: Øke risk-justert avkastning (Sharpe/Sortino) + redusere drawdown + bevise hva som faktisk tjener penger

---

## 🎯 SUCCESS CRITERIA

**Win Condition** (need ONE of these):
- Sharpe Ratio: +20% improvement
- Max Drawdown: -20% improvement
- Win Rate: +10% improvement (cap at 65%)
- Profit Factor: +25% improvement

**Current Baseline**: ⚠️ NO TRADES YET (see PERFORMANCE_BASELINE.md)

---

## 🚧 CURRENT BLOCKER: NO TRADE DATA

**Finding**: Database exists (quantum_trader.db, 2.1MB) but tables `trade_logs` and `execution_journal` are **EMPTY**.

**Reason**: System has completed:
- ✅ P1-B: Ops Hardening (container health, disk cleanup)
- ✅ P1-C Phase A: Preflight (all 16 checks passed)

But has NOT yet executed:
- ⏸️ Phase B: Shadow Mode (paper trading with mainnet data)
- ⏸️ Phase C: Live Small (real micro-notional trades)

---

## 🔀 TWO PATHS FORWARD

### PATH A: Shadow Mode First (RECOMMENDED)
**Purpose**: Generate data WITHOUT risk, validate full pipeline

1. **Run Phase B (Shadow Mode)**:
   ```bash
   wsl bash scripts/go_live_shadow.sh
   ```
   - Duration: 60 minutes
   - Uses: MAINNET data, PAPER_TRADING=true
   - Validates: Intent processing, order decisions, execution logic
   - Output: GO_LIVE_SHADOW_PROOF.md with detailed logs

2. **Extract Shadow Data**:
   - Parse "WOULD_SUBMIT" logs from auto-executor
   - Extract: symbol, side, confidence, regime, intent type
   - Create: shadow_trades.csv for analysis

3. **Run Baseline Analysis**:
   ```bash
   python3 scripts/analyze_performance_baseline.py
   ```
   - Will show: What WOULD have happened
   - Metrics: Confidence distribution, regime distribution, symbol coverage

4. **Analyze & Optimize**:
   - Step 2: Confidence filtering (would we trade too much at low confidence?)
   - Step 4: Regime filtering (would we lose money in certain regimes?)
   - Step 5: Exit optimization (would trailing stops improve?)

5. **Implement Optimizations**:
   - Add filters based on shadow data findings
   - Re-run shadow mode to validate improvements
   - Compare: Before vs After metrics

6. **Then Phase C (Live Small)**:
   - Once shadow mode shows positive expectancy
   - Run with extreme safety: 50 USDT max, 1 position, 2x leverage
   - Validate: Real execution matches shadow predictions

**Pros**:
- ✅ Zero risk data generation
- ✅ Full pipeline validation
- ✅ Can iterate quickly (no real money at stake)
- ✅ Find bugs before live trading

**Cons**:
- ⚠️ Shadow data != real data (no slippage, no liquidity issues)
- ⚠️ 60-90 minutes to generate data

---

### PATH B: Live Small Immediately
**Purpose**: Get real data fast, prove system works end-to-end

1. **Run Phase C (Live Small)**:
   ```bash
   wsl bash scripts/go_live_live_small.sh
   ```
   - Safety: 50 USDT max, 1 position, 2x leverage, 60s cooldown
   - Requires: Manual confirmation (type 'LIVE')
   - Duration: Run until 3-5 trades executed (could take hours/days)

2. **Wait for Trades**:
   - Monitor: Grafana dashboard + logs
   - Expect: Very few trades due to extreme restrictions
   - Risk: Real money, but micro amounts

3. **Run Baseline Analysis**:
   ```bash
   python3 scripts/analyze_performance_baseline.py
   ```
   - Will show: Real performance metrics
   - Metrics: Sharpe, Sortino, Max DD, Winrate, Profit Factor

4. **Analyze & Optimize**:
   - Attribution: Which trades made money?
   - Confidence: Did high confidence trades outperform?
   - Regime: Which regimes were profitable?

5. **Scale Up Gradually**:
   - If positive expectancy: Increase notional to 100 USDT
   - If negative expectancy: Fix issues, test again at 50 USDT

**Pros**:
- ✅ Real data (includes slippage, fees, latency)
- ✅ Proves system works end-to-end with real money
- ✅ Fastest path to real performance metrics

**Cons**:
- ⚠️ Takes time (low trade frequency with extreme restrictions)
- ⚠️ Real money at risk (even if micro amounts)
- ⚠️ Less data for optimization (trades are rare)

---

## 🎬 RECOMMENDED SEQUENCE

**BEST PRACTICE**: Shadow First, then Live Small

```
Day 1 (TODAY):
  1. Run Phase B (Shadow Mode) - 60 min
  2. Analyze shadow data
  3. Identify 1-2 quick wins (e.g., min confidence threshold)
  4. Implement + re-test in shadow mode

Day 2:
  5. If shadow mode shows positive expectancy → Phase C (Live Small)
  6. Run overnight, wait for 3-5 real trades
  
Day 3:
  7. Analyze real trade data
  8. Compare shadow vs real (calibration check)
  9. Decide on scaling or further optimization
```

---

## 📋 P2 STEP-BY-STEP PLAN

Once we have data (from shadow or live), follow this sequence:

### Step 1: PERFORMANCE_BASELINE.md ✅ READY
- **Status**: Analyzer created, waiting for data
- **Input**: trade_logs or execution_journal or shadow logs
- **Output**: Baseline metrics document

### Step 2: Trade Attribution Layer
- **Script**: `scripts/analyze_trade_attribution.py`
- **Analysis**: 
  * PnL by symbol (which assets make money?)
  * PnL by confidence bucket (does higher confidence = higher PnL?)
  * PnL by regime (trend vs chop vs high vol)
  * PnL by strategy (if tagged)
  * PnL by exit reason (TP vs SL vs trailing vs time)
- **Output**: `TRADE_ATTRIBUTION_REPORT.md`

### Step 3: Confidence Filtering Hypothesis
- **Question**: "Do we trade too much with low confidence?"
- **Analysis**:
  * 0.50-0.60: Trade count, PnL, expectancy
  * 0.60-0.70: Trade count, PnL, expectancy
  * 0.70-0.80: Trade count, PnL, expectancy
  * 0.80+: Trade count, PnL, expectancy
- **Hypothesis Test**: 
  * If 0.50-0.60 has negative expectancy → Set MIN_CONFIDENCE=0.60
  * If 0.60-0.70 has negative expectancy → Set MIN_CONFIDENCE=0.70
- **Output**: `CONFIDENCE_FILTER_ANALYSIS.md`

### Step 4: Regime-Aware Trading
- **Question**: "Which regimes have negative expectancy?"
- **Analysis**:
  * PnL in trending regime
  * PnL in sideways/chop regime
  * PnL in high volatility regime
  * PnL in low volatility regime
- **Hypothesis Test**:
  * If sideways has negative expectancy → Add SKIP_SIDEWAYS filter
  * If high vol has negative expectancy → Add MAX_VOL_PERCENTILE
- **Output**: `REGIME_FILTER_ANALYSIS.md`

### Step 5: Exit Optimization
- **Question**: "Are we exiting too early or too late?"
- **Analysis**:
  * MFE (Max Favorable Excursion) vs Exit Price
  * MAE (Max Adverse Excursion) vs Stop Loss
  * Trailing stop effectiveness
  * TP vs SL ratio
- **Hypothesis Test**:
  * If MFE >> Exit Price → Consider wider TPs or trailing
  * If MAE < SL often → Consider tighter stops
- **Output**: `EXIT_OPTIMIZATION_ANALYSIS.md`

### Step 6: Portfolio Correlation
- **Question**: "Do we cluster losses when multiple positions are open?"
- **Analysis**:
  * Drawdown events with 1 position vs 2+ positions
  * Symbol correlation during drawdowns
  * Leverage exposure during drawdowns
- **Hypothesis Test**:
  * If multi-position DD > 1.5x single-position DD → Add position correlation filter
- **Output**: `PORTFOLIO_CORRELATION_ANALYSIS.md`

### Step 7: A/B Testing Framework
- **Purpose**: Test optimizations without full rollout
- **Implementation**:
  * Create `config/ab_test_config.yaml`
  * Add A/B test flag to intent processor
  * Log which group each trade belongs to
  * Compare Group A (baseline) vs Group B (optimized)
- **Output**: `AB_TEST_FRAMEWORK_GUIDE.md`

### Step 8: Results Documentation
- **Final Report**: `P2_PERFORMANCE_OPTIMIZATION_REPORT.md`
- **Contents**:
  * Before/After metrics
  * Each optimization tested
  * Statistical significance
  * Recommendations for Phase D (scaling)

---

## 🚨 CRITICAL RULES

1. **NO CODE CHANGES WITHOUT BASELINE**: Always measure before optimizing
2. **ONE CHANGE AT A TIME**: Never stack multiple optimizations (can't attribute impact)
3. **STATISTICAL SIGNIFICANCE**: Need minimum 30 trades for confidence, 100+ for robust conclusions
4. **SHADOW BEFORE LIVE**: Test all changes in shadow mode first
5. **PROOF DOCUMENTS**: Every optimization gets a markdown report with proof

---

## ⏭️ IMMEDIATE NEXT ACTION

**CHOICE POINT**: 

**Option A (Recommended)**: Start with Shadow Mode
```bash
cd /home/qt/quantum_trader
bash scripts/go_live_shadow.sh
```
Then analyze logs and iterate.

**Option B**: Start with Live Small (if you want real data immediately)
```bash
cd /home/qt/quantum_trader
bash scripts/go_live_live_small.sh
# Type 'LIVE' when prompted
```
Then wait for trades and analyze.

**Which path do you want to take?**

---

## 📝 NOTES

- **No data = No optimization**: Can't improve what we can't measure
- **Shadow mode is safe**: Zero risk, full validation
- **Live Small is proof**: Real money, real results, but slow
- **P2 is iterative**: Not a one-shot deal, continuous improvement

---

**Status**: Waiting for decision on Path A (Shadow) vs Path B (Live Small)

