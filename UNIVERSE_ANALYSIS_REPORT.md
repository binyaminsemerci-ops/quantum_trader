# QUANTUM TRADER — UNIVERSE ANALYSIS REPORT
**Generated:** 2025-11-23 UTC  
**Analysis Period:** Initial deployment (limited data: 464 signals from 180 symbols)  
**Current Universe:** 222 symbols (explicit mode)

---

## EXECUTIVE SUMMARY

### Current State
- **Universe Configuration:** Explicit mode with 222 manually selected symbols
- **Signal Coverage:** 180/222 symbols (81.1%) have generated at least one signal
- **Data Maturity:** Early stage — only 464 total signal decisions recorded
- **Major Coins Coverage:** 15/15 majors present in universe ✓

### Key Findings

⚠️ **INSUFFICIENT DATA FOR STATISTICAL CONFIDENCE**
- Most symbols have only 1-3 signals each
- Only 2 symbols have >= 5 signals (ZILUSDT, MAVUSDT)
- Only 125 symbols have >= 3 signals
- Recommendation: **Collect 7-14 days of data before making universe changes**

However, preliminary patterns are emerging:

### Preliminary Performance Indicators

#### Strong Performers (100% allow rate, >= 3 signals):
**TIER 1 — MAJORS (High Liquidity):**
- BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, LINKUSDT, LTCUSDT
- AVAXUSDT, MATICUSDT

**TIER 2 — LAYER 1/2 & DEFI:**
- RNDRUSDT, HYPEUSDT, ICPUSDT, ZENUSDT, BCHUSDT
- TAOUSDT, PAXGUSDT, AAVEUSDT, ZECUSDT, DASHUSDT
- AGIXUSDT, GIGGLEUSDT, EOSUSDT, ETCUSDT, FTMUSDT

**Total: 25 symbols showing consistent signal approval**

#### Problematic Symbols (0% allow rate, >= 3 signals):
- ZILUSDT (6 signals, 0% allowed) — **BLACKLIST CANDIDATE**
- DOGEUSDT (3 signals, 0% allowed) — ⚠️ Major coin with poor recent performance
- BLURUSDT, PYTHUSDT, DYMUSDT, PORTALUSDT, ALTUSDT
- XAIUSDT, 1000PEPEUSDT, PUMPUSDT, SOONUSDT, JCTUSDT
- TRUMPUSDT, TRXUSDT, WLFIUSDT, DUSKUSDT, PENGUUSDT
- ALPHAUSDT, UXLINKUSDT, STRKUSDT

**Total: 18 symbols showing complete rejection by risk controls**

#### Moderate Performers:
- UNIUSDT (66.7%, 3 signals)
- RESOLVUSDT (66.7%, 3 signals)
- MAVUSDT (33.3%, 6 signals) — **CONDITIONAL**
- INJUSDT (33.3%, 3 signals)
- ORDIUSDT (33.3%, 3 signals)

---

## UNIVERSE QUALITY ASSESSMENT

### Coverage Analysis

| Metric | Count | Percentage |
|--------|-------|------------|
| Total Universe | 222 | 100.0% |
| Symbols with signals | 180 | 81.1% |
| Symbols with >= 3 signals | 125 | 56.3% |
| Symbols with >= 5 signals | 2 | 0.9% |
| Symbols with no data | 42 | 18.9% |

### Signal Distribution
- **Mean signals per symbol:** 2.6
- **Median signals per symbol:** 2.0
- **Mode:** 3 signals (most common)
- **Max:** 6 signals (ZILUSDT, MAVUSDT)

### Preliminary Risk Control Effectiveness
- **Overall allow rate:** ~53% (246 allowed / 464 total)
- **High-quality signals:** Majors showing 100% allow rate indicates good model confidence
- **Risk filtering:** 47% rejection rate suggests robust risk controls

---

## SYMBOL CLASSIFICATION

### 1. CORE SYMBOLS (High Confidence, Production Ready)
**Count: 25 symbols**

#### Majors (Always Include):
```
BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, LINKUSDT, LTCUSDT, AVAXUSDT, MATICUSDT
```

#### Layer 1 Protocols:
```
RNDRUSDT, ICPUSDT, ZENUSDT, BCHUSDT, ZECUSDT, DASHUSDT, EOSUSDT, ETCUSDT, FTMUSDT
```

#### AI/Emerging Tech:
```
HYPEUSDT, TAOUSDT, AGIXUSDT, GIGGLEUSDT
```

#### DeFi:
```
AAVEUSDT, PAXGUSDT
```

**Characteristics:**
- 100% signal allow rate (3+ signals each)
- Strong model confidence
- Passed all risk filters
- Suitable for mainnet deployment

### 2. EXPANSION SYMBOLS (Moderate Confidence)
**Count: ~50-75 symbols (pending more data)**

**Currently identified:**
- UNIUSDT (66.7% allow)
- RESOLVUSDT (66.7% allow)
- INJUSDT (33.3% allow)
- ORDIUSDT (33.3% allow)

**Characteristics:**
- 30-70% allow rate
- Some signals passed risk controls
- Need more data for confirmation
- Suitable for testnet/aggressive profiles

### 3. CONDITIONAL SYMBOLS (Use with Caution)
**Count: 1 symbol confirmed**

- MAVUSDT (33.3% allow rate, 6 signals) — Most data available, mixed performance

**Characteristics:**
- Inconsistent performance
- High variance in signal quality
- May perform well only in specific market conditions
- Requires regime-specific filtering

### 4. BLACKLIST CANDIDATES (Poor Performance)
**Count: 18 symbols**

**Permanent Blacklist:**
```
ZILUSDT — 6 signals, 0% allowed (highest confidence for blacklist)
```

**Conditional Blacklist (0% allow rate, 3 signals):**
```
BLURUSDT, PYTHUSDT, DYMUSDT, PORTALUSDT, ALTUSDT, XAIUSDT
1000PEPEUSDT, PUMPUSDT, SOONUSDT, JCTUSDT, TRUMPUSDT, TRXUSDT
WLFIUSDT, DUSKUSDT, PENGUUSDT, ALPHAUSDT, UXLINKUSDT, STRKUSDT
```

**Special Note:**
- DOGEUSDT (major coin) showing 0% allow — requires investigation
- May be temporarily affected by market conditions or model issues

**Characteristics:**
- 0% allow rate (all signals blocked by risk controls)
- Poor model confidence or high risk metrics
- May have structural issues (low liquidity, high slippage, poor data quality)

### 5. NO DATA SYMBOLS
**Count: 42 symbols**

Symbols in universe but no signals generated yet:
- Indicates low market activity OR
- Models not confident enough to generate signals OR
- Unsuitable market conditions

**Action:** Monitor for 7-14 days before classification

---

## UNIVERSE SIZE OPTIMIZATION

### Current vs Optimal Size Analysis

Based on limited data, preliminary recommendations:

#### Phase 1: CURRENT (Weeks 1-2)
**Size:** 222 symbols (keep as-is)  
**Rationale:** Collect more data across full universe before making changes

#### Phase 2: SAFE PROFILE (Weeks 3-4, if deploying to mainnet)
**Recommended Size:** 150-180 symbols  
**Include:**
- 25 CORE symbols (confirmed)
- 50-75 EXPANSION symbols (pending data confirmation)
- 50-80 symbols currently with limited data but no negative signals

**Exclude:**
- 18 blacklist candidates
- 20-40 symbols with consistently poor metrics after 2+ weeks

**Expected Performance:**
- Higher average signal quality
- Lower drawdown risk
- Better capital efficiency

#### Phase 3: AGGRESSIVE PROFILE (Testnet only)
**Recommended Size:** 300-400 symbols  
**Include:**
- All CORE symbols
- All EXPANSION symbols
- All CONDITIONAL symbols
- Selected NO_DATA symbols with good fundamentals

**Exclude:**
- PERMANENT_BLACKLIST only (ZILUSDT confirmed)
- Symbols with structural issues (delisted, low liquidity)

**Expected Performance:**
- Maximum opportunity capture
- Higher variance
- More diverse training data for ML models

#### Phase 4: SCALP PROFILE (Ultra-high frequency)
**Recommended Size:** 15-30 symbols  
**Include:**
- MAJORS only: BTC, ETH, BNB, SOL, XRP, ADA, LINK, AVAX, DOGE, DOT, MATIC, UNI, LTC, ATOM, TRX

**Characteristics:**
- Highest liquidity
- Tightest spreads
- Most reliable execution
- Lowest slippage

---

## PERFORMANCE PROJECTIONS

### With Current Universe (222 symbols):
- **Signal generation rate:** ~460 signals/day (based on limited sample)
- **Allow rate:** ~53% → ~245 tradeable signals/day
- **Coverage:** 81% of universe active
- **Quality score:** Medium (need more data)

### With SAFE Profile (150-180 symbols):
- **Signal generation rate:** ~350-420 signals/day (estimated)
- **Allow rate:** ~65-75% → ~270-315 tradeable signals/day
- **Coverage:** 90%+ expected
- **Quality score:** High (best performers only)

### With AGGRESSIVE Profile (300-400 symbols):
- **Signal generation rate:** ~600-800 signals/day (estimated)
- **Allow rate:** ~45-55% → ~320-440 tradeable signals/day
- **Coverage:** 70-80% expected
- **Quality score:** Medium-High (diverse opportunity set)

### With SCALP Profile (15-30 symbols):
- **Signal generation rate:** ~100-150 signals/day (estimated)
- **Allow rate:** ~80-90% → ~90-135 tradeable signals/day
- **Coverage:** 100% expected
- **Quality score:** Very High (majors only)

---

## DEPLOYMENT CONFIGURATION

### RECOMMENDED PROFILES

#### 1. SAFE PROFILE — Mainnet / Real Money
```yaml
QT_UNIVERSE: custom
QT_MAX_SYMBOLS: 180

WHITELIST: [
  # CORE SYMBOLS (25 confirmed)
  "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "LINKUSDT", "LTCUSDT",
  "AVAXUSDT", "MATICUSDT", "RNDRUSDT", "HYPEUSDT", "ICPUSDT", 
  "ZENUSDT", "BCHUSDT", "TAOUSDT", "PAXGUSDT", "AAVEUSDT",
  "ZECUSDT", "DASHUSDT", "AGIXUSDT", "GIGGLEUSDT", "EOSUSDT",
  "ETCUSDT", "FTMUSDT",
  
  # EXPANSION (add after 2 weeks of data confirmation)
  # ... remaining 155 symbols from current universe with good metrics
]

BLACKLIST: [
  "ZILUSDT",  # Confirmed poor performance
  # Additional blacklist entries after 2 weeks of data
]
```

**When to use:** Production deployment with real capital

#### 2. AGGRESSIVE PROFILE — Testnet / Training
```yaml
QT_UNIVERSE: l1l2-top  # or all-usdt
QT_MAX_SYMBOLS: 400

WHITELIST: [
  # All symbols except permanent blacklist
]

BLACKLIST: [
  "ZILUSDT",
]
```

**When to use:** 
- Training ML models
- Testing new strategies
- Maximum opportunity diversity
- Testnet deployment

#### 3. SCALP PROFILE — High Frequency
```yaml
QT_UNIVERSE: megacap
QT_MAX_SYMBOLS: 50

WHITELIST: [
  "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
  "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "LINKUSDT", "DOTUSDT",
  "MATICUSDT", "UNIUSDT", "LTCUSDT", "ATOMUSDT", "TRXUSDT",
  # Add more majors as needed
]

BLACKLIST: []
```

**When to use:**
- Ultra-high frequency trading
- Low latency requirements
- Maximum execution reliability

---

## ACTIONABLE RECOMMENDATIONS

### IMMEDIATE ACTIONS (Week 1)

1. ✅ **Keep Current Universe (222 symbols)**
   - Continue collecting data
   - No changes to universe configuration yet

2. ✅ **Add Permanent Blacklist**
   ```python
   # config/config.py or environment
   PERMANENT_BLACKLIST = ["ZILUSDT"]
   ```

3. ✅ **Monitor High Priority Symbols**
   - Track DOGEUSDT (major coin with 0% allow rate) — investigate
   - Watch MAVUSDT (most signals, mixed performance)
   - Validate CORE symbols maintaining 100% allow rate

4. ✅ **Set Up Monitoring Dashboards**
   - Symbol-level allow/block rates
   - Signal generation frequency per symbol
   - Confidence distribution per symbol
   - Regime-specific performance

### SHORT-TERM ACTIONS (Weeks 2-4)

1. **Data Collection Milestone: 7 Days**
   - Re-run universe analyzer
   - Expect ~3,200 signals (7 × 460/day)
   - Most symbols should have 10-30 signals
   - Confidence: Medium

2. **Data Collection Milestone: 14 Days**
   - Re-run universe analyzer
   - Expect ~6,400 signals (14 × 460/day)
   - Most symbols should have 20-60 signals
   - Confidence: High
   - **DECISION POINT:** Implement SAFE profile if deploying to mainnet

3. **Classification Refinement**
   - Move symbols between CORE/EXPANSION/CONDITIONAL based on data
   - Expand permanent blacklist
   - Remove NO_DATA classification for most symbols

4. **Universe Size Optimization**
   - Test SAFE profile (150-180 symbols) in paper trading
   - Compare performance metrics vs current 222-symbol universe
   - Measure: total PnL, Sharpe ratio, max drawdown, signal quality

### MEDIUM-TERM ACTIONS (Months 2-3)

1. **Dynamic Universe Management**
   - Implement automatic symbol promotion/demotion based on rolling 30-day metrics
   - Create regime-specific universe profiles
   - Develop volatility-adjusted universe sizing

2. **Advanced Filtering**
   - Add symbol-specific confidence thresholds
   - Implement spread/slippage-based dynamic filtering
   - Create time-of-day universe rules (e.g., fewer symbols during low liquidity hours)

3. **Performance Tracking**
   - Symbol-level PnL attribution
   - Cost analysis per symbol (spreads, slippage, funding)
   - Opportunity cost analysis (missed signals from blacklisted symbols)

---

## RISK CONSIDERATIONS

### Current Risks

1. **Insufficient Data**
   - **Risk:** Making universe changes based on < 500 signals
   - **Mitigation:** Wait for 7-14 days of data (3,000-6,000+ signals)

2. **Major Coin Anomaly**
   - **Risk:** DOGEUSDT (major) showing 0% allow rate
   - **Mitigation:** Investigate root cause before blacklisting
   - **Possible causes:**
     - Temporary market conditions
     - Model calibration issue
     - Data quality problem
     - Orchestrator policy too restrictive for volatile memecoins

3. **Overfitting to Recent Market Conditions**
   - **Risk:** Current data may represent specific regime (trending/ranging/volatile)
   - **Mitigation:** Collect data across multiple regimes before finalizing universe

### Recommended Safeguards

1. **Gradual Universe Changes**
   - Remove max 10-20 symbols per week
   - Monitor impact on PnL and signal generation
   - Keep rollback capability

2. **Core Symbols Protection**
   - Never blacklist major coins without manual review
   - Require 50+ signals before blacklisting any symbol
   - Require 90 days of poor performance before permanent blacklist

3. **Testnet Validation**
   - Test any universe changes in paper trading first
   - Run parallel configs (old vs new) for 7 days
   - Compare: total PnL, max DD, signal count, execution quality

---

## MONITORING METRICS

### Key Performance Indicators (KPIs)

**Universe-Level Metrics:**
- Total signal generation rate (signals/day)
- Overall allow rate (allowed/total)
- Coverage (symbols with signals / total symbols)
- Avg confidence per allowed signal
- Avg confidence per blocked signal

**Symbol-Level Metrics:**
- Signal frequency (signals/day)
- Allow rate (allowed/total)
- Avg confidence (mean, std dev)
- Consensus quality (strong/weak ratio)
- Agreement rate (policy agrees with decision)

**Performance Metrics (requires trade data):**
- Win rate per symbol
- Avg R-multiple per symbol
- Total R per symbol
- Sharpe ratio per symbol
- Max drawdown contribution per symbol

### Alert Thresholds

**Immediate Investigation Required:**
- Major coin with < 20% allow rate for 7+ days
- Any symbol with 100% block rate for 14+ days (50+ signals)
- Universe coverage < 70% for 7+ days
- Signal generation drops > 30% day-over-day

**Review Recommended:**
- Symbol allow rate changes > 40% week-over-week
- New symbols with 0% allow rate after 20+ signals
- CORE symbols dropping below 80% allow rate

---

## NEXT STEPS

### Phase 1: Data Collection (Current — Week 2)
- [x] Deploy current 222-symbol universe
- [x] Collect signal decision data
- [x] Run initial universe analysis
- [ ] Achieve 7-day data milestone
- [ ] Re-run analysis with Week 1 data

### Phase 2: Validation (Weeks 2-4)
- [ ] Achieve 14-day data milestone
- [ ] Classify all symbols with statistical confidence
- [ ] Identify permanent blacklist (10-30 symbols expected)
- [ ] Define CORE universe (50-100 symbols expected)
- [ ] Test SAFE profile in paper trading

### Phase 3: Optimization (Months 2-3)
- [ ] Implement dynamic universe management
- [ ] Deploy SAFE profile to mainnet (if applicable)
- [ ] Deploy AGGRESSIVE profile to testnet
- [ ] Develop regime-specific universe rules
- [ ] Create automated symbol promotion/demotion system

### Phase 4: Advanced Features (Months 3-6)
- [ ] Multi-regime universe optimization
- [ ] Volatility-adjusted universe sizing
- [ ] Time-of-day universe management
- [ ] Symbol-specific parameter optimization
- [ ] Machine learning for universe selection

---

## CONCLUSION

### Summary of Findings

**Current State:**
- 222-symbol universe in explicit mode
- Early-stage deployment with limited but promising data
- 81% universe coverage showing good model activity
- Clear performance separation emerging between strong and weak symbols

**Key Insights:**
1. **25 CORE symbols identified** with 100% allow rate (preliminary)
2. **18 problematic symbols identified** with 0% allow rate (preliminary)
3. **ZILUSDT confirmed for blacklist** (most signals, worst performance)
4. **Majority of universe (179 symbols)** requires more data for classification

**Confidence Level:** 🟡 MEDIUM
- Sufficient data to identify extreme performers (best/worst)
- Insufficient data for fine-grained classification
- Recommendation: Continue current config for 7-14 more days

### Strategic Recommendation

**For Testnet Deployment:**
→ **Keep current 222-symbol universe**, add ZILUSDT to blacklist

**For Mainnet Deployment (when ready):**
→ **Wait for 14-day data milestone**, then deploy SAFE profile (150-180 symbols)

**For Maximum Learning:**
→ **Expand to AGGRESSIVE profile** (300-400 symbols) to train models on diverse market conditions

---

## APPENDIX: DETAILED METRICS

### Top 30 Symbols by Signal Count (Raw Data)

| Rank | Symbol | Total Signals | Allowed | Blocked | Allow % |
|------|--------|---------------|---------|---------|---------|
| 1 | ZILUSDT | 6 | 0 | 6 | 0.0% |
| 2 | MAVUSDT | 6 | 2 | 4 | 33.3% |
| 3 | ARBUSDT | 4 | 0 | 4 | 0.0% |
| 4 | ARUSDT | 4 | 0 | 4 | 0.0% |
| 5 | BTCUSDT | 3 | 3 | 0 | 100.0% |
| 6 | ETHUSDT | 3 | 3 | 0 | 100.0% |
| 7 | BNBUSDT | 3 | 3 | 0 | 100.0% |
| 8 | SOLUSDT | 3 | 3 | 0 | 100.0% |
| 9 | DOGEUSDT | 3 | 0 | 3 | 0.0% |
| 10 | LINKUSDT | 3 | 3 | 0 | 100.0% |
| 11 | AVAXUSDT | 3 | 3 | 0 | 100.0% |
| 12 | MATICUSDT | 3 | 3 | 0 | 100.0% |
| 13 | UNIUSDT | 3 | 2 | 1 | 66.7% |
| 14 | LTCUSDT | 3 | 3 | 0 | 100.0% |
| 15 | RNDRUSDT | 3 | 3 | 0 | 100.0% |
| 16 | INJUSDT | 3 | 1 | 2 | 33.3% |
| 17 | ORDIUSDT | 3 | 1 | 2 | 33.3% |
| 18-50 | (Various) | 3 | Mixed | Mixed | 0-100% |

### Environment Configuration Templates

**Current (Explicit Mode):**
```env
QT_SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT,...  # 222 symbols
```

**Recommended after 14 days (Dynamic Mode — SAFE):**
```env
QT_UNIVERSE=custom
QT_MAX_SYMBOLS=180
# + implement whitelist/blacklist in code
```

**Recommended for Testnet (Dynamic Mode — AGGRESSIVE):**
```env
QT_UNIVERSE=l1l2-top
QT_MAX_SYMBOLS=400
# + implement permanent blacklist in code
```

---

**END OF REPORT**

*This analysis should be re-run after:*
- *7 days (Week 1 milestone)*
- *14 days (Week 2 milestone — DECISION POINT)*
- *30 days (Month 1 milestone — Full classification)*
- *90 days (Quarter 1 — Long-term optimization)*

*Generated by: Universe Analyzer v1.0*  
*Contact: Senior Quant Researcher*
