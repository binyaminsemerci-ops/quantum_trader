# UNIVERSE SELECTOR AGENT — Deployment Guide

**Version:** 1.0  
**Date:** November 23, 2025  
**Status:** ✅ Deployed and Operational

---

## 🎯 AGENT MISSION

The Universe Selector Agent is an **autonomous AI subsystem** that continuously evaluates and optimizes the trading universe based on:

- Symbol performance metrics (win rate, R-multiples, PnL)
- Execution quality (slippage, spread, disallow rates)
- Market regime adaptation (trending, ranging, mixed)
- Volatility behavior (normal, high, extreme)
- Stability and reliability scores

**Key Principle:** The agent **NEVER modifies code** — it only analyzes data and generates recommendations.

---

## 📊 CURRENT STATUS

### Initial Run Results (Nov 23, 2025)

**Data Available:**
- ✅ 1,161 signal records from 194 symbols
- ❌ No trade data yet (system just deployed)
- 🟡 Data confidence: **LOW**

**Classification Results:**
- **CORE:** 0 symbols (need trade data)
- **EXPANSION:** 0 symbols (need trade data)
- **CONDITIONAL:** 0 symbols (need trade data)
- **BLACKLIST:** 156 symbols (high disallow rate, no trade proof)
- **INSUFFICIENT_DATA:** 62 symbols (< 5 signals)

**Top 10 Symbols by Signal Quality (Signal-Only Analysis):**
1. MATICUSDT (Quality: 0.0702)
2. SOLUSDT (Quality: 0.0696)
3. DGBUSDT (Quality: 0.0696)
4. LINKUSDT (Quality: 0.0691)
5. AGIXUSDT (Quality: 0.0691)
6. AUDIOUSDT (Quality: 0.0689)
7. YFIUSDT (Quality: 0.0684)
8. BNBUSDT (Quality: 0.0683)
9. EOSUSDT (Quality: 0.0680)
10. FTMUSDT (Quality: 0.0680)

**⚠️ Critical Note:** Without trade data, the agent cannot compute:
- Win rates
- R-multiples
- Profitability scores
- Stability scores
- True performance classifications

**Recommendation:** Run agent again after **7-14 days** when trade data is available.

---

## 🏗️ AGENT ARCHITECTURE

### Input Sources

```
1. Universe Snapshot (/app/data/universe_snapshot.json)
   └─ Current universe configuration
   └─ Symbol list
   └─ Generation timestamp

2. Policy Observations (/app/data/policy_observations/*.jsonl)
   └─ Signal decisions (allowed/blocked)
   └─ Confidence scores
   └─ Regime tags
   └─ Volatility levels
   └─ Disallow reasons

3. Trade Logs (/app/data/trades/*.jsonl)
   └─ Entry/exit prices
   └─ R-multiples
   └─ PnL
   └─ Slippage
   └─ Spreads
   └─ Exit reasons
   └─ Holding times
```

### Feature Engineering Pipeline

```
For each symbol:

1. SIGNAL FEATURES
   ├─ Total signals
   ├─ Allow rate
   ├─ Disallow rate
   ├─ Avg confidence
   ├─ Confidence std dev
   └─ Regime/vol distribution

2. TRADE FEATURES (requires trade data)
   ├─ Win rate
   ├─ Avg R-multiple
   ├─ Total R
   ├─ R std dev
   ├─ Avg slippage
   ├─ Avg spread
   ├─ Regime-specific R
   └─ Vol-specific R

3. COMPOSITE SCORES
   ├─ Stability Score = f(avg_r, winrate, variance, costs, consistency)
   ├─ Quality Score = f(stability, profitability, reliability, adaptability, execution)
   ├─ Profitability Score = f(avg_r, total_r)
   └─ Reliability Score = f(allow_rate, confidence, consistency)
```

### Classification Logic

```
BLACKLIST (Exclude completely):
  ├─ total_r < -0.5 AND winrate < 35%
  ├─ OR avg_r < 0.1
  ├─ OR disallow_rate > 50%
  └─ OR stability < 0.05

CORE (Must-trade, production-ready):
  ├─ stability >= 0.20
  ├─ quality >= 0.25
  ├─ winrate >= 45%
  ├─ disallow_rate <= 25%
  └─ avg_r >= 0.5

EXPANSION (Good performers, testnet-ready):
  ├─ stability >= 0.10
  ├─ quality >= 0.15
  ├─ winrate >= 35%
  ├─ disallow_rate <= 40%
  └─ avg_r >= 0.3

CONDITIONAL (Regime-specific):
  ├─ Profitable in TRENDING but not RANGING
  ├─ OR profitable in NORMAL vol but not EXTREME
  ├─ OR profitable but unstable
  └─ Requires situational filtering

INSUFFICIENT_DATA:
  └─ < 5 signals OR < 3 trades
```

---

## 🚀 OPERATIONAL USAGE

### Running the Agent

**Manual Execution:**
```bash
# From host
docker exec quantum_backend python /app/universe_selector_agent.py

# From inside container
python /app/universe_selector_agent.py
```

**Output Location:**
```
/app/data/universe_selector_output.json
```

**Copy Output to Host:**
```bash
docker cp quantum_backend:/app/data/universe_selector_output.json ./
```

### Recommended Execution Schedule

| **Phase** | **Frequency** | **Purpose** |
|-----------|---------------|-------------|
| **Week 1** | Daily | Monitor signal accumulation |
| **Week 2-4** | Every 3 days | Track early trade performance |
| **Month 2+** | Weekly | Optimize universe based on mature data |
| **Quarter 2+** | Bi-weekly | Long-term optimization |

### Data Requirements for Reliable Output

| **Metric** | **Minimum** | **Good** | **Excellent** |
|------------|-------------|----------|---------------|
| **Signals per symbol** | 5 | 20 | 50+ |
| **Trades per symbol** | 3 | 10 | 30+ |
| **Total signals** | 1,000 | 5,000 | 10,000+ |
| **Total trades** | 100 | 500 | 1,000+ |
| **Data confidence** | LOW | MEDIUM | HIGH |

**Current Status:** 1,161 signals, 0 trades → **LOW confidence**

---

## 📋 OUTPUT SCHEMA

### JSON Structure

```json
{
  "generated_at": "2025-11-23T00:00:00Z",
  "agent_version": "1.0",
  "data_confidence": "LOW|MEDIUM|HIGH|VERY_HIGH",
  
  "market_state": {
    "regime": "TRENDING|RANGING|MIXED|UNKNOWN",
    "volatility_level": "NORMAL|HIGH|EXTREME|UNKNOWN"
  },
  
  "current_universe": {
    "mode": "explicit",
    "symbol_count": 222,
    "generated_at": "..."
  },
  
  "classifications": {
    "CORE": {
      "count": 25,
      "symbols": ["BTCUSDT", "ETHUSDT", ...],
      "description": "..."
    },
    "EXPANSION": {...},
    "CONDITIONAL": {...},
    "BLACKLIST": {...},
    "INSUFFICIENT_DATA": {...}
  },
  
  "recommendations": {
    "SAFE": {
      "recommended_size": 180,
      "description": "Conservative - Real money / Mainnet",
      "include": [...],
      "exclude": [...],
      "qt_universe": "custom",
      "qt_max_symbols": 180
    },
    "AGGRESSIVE": {...},
    "EXPERIMENTAL": {...}
  },
  
  "deltas": {
    "SAFE": {
      "to_add": [...],
      "to_remove": [...],
      "to_keep": [...],
      "add_count": 10,
      "remove_count": 42,
      "keep_count": 180
    },
    ...
  },
  
  "performance_curves": {
    "quality_vs_size": [
      {"size": 50, "avg_quality": 0.45},
      {"size": 100, "avg_quality": 0.42},
      ...
    ],
    "stability_vs_size": [...],
    "cumulative_r_vs_size": [...]
  },
  
  "symbol_scores": {
    "BTCUSDT": {
      "quality_score": 0.85,
      "stability_score": 0.78,
      "profitability_score": 0.92,
      "reliability_score": 0.88,
      "winrate": 0.65,
      "avg_r": 1.2,
      "total_r": 15.6,
      "allow_rate": 0.92,
      "total_signals": 45,
      "total_trades": 28
    },
    ...
  },
  
  "summary": {
    "total_symbols_analyzed": 222,
    "symbols_with_data": 194,
    "symbols_with_trades": 0,
    "core_count": 0,
    "expansion_count": 0,
    "conditional_count": 0,
    "blacklist_count": 156,
    "insufficient_data_count": 62
  }
}
```

---

## 🎯 INTERPRETING AGENT RECOMMENDATIONS

### Profile Selection Guide

#### **SAFE Profile** — Production / Mainnet
**When to use:**
- Deploying to mainnet with real capital
- Risk-averse trading
- Proven track record required

**Characteristics:**
- Size: 150-200 symbols
- Includes: CORE + top EXPANSION
- Excludes: BLACKLIST + CONDITIONAL
- Expected: High stability, lower variance

**Deployment:**
```yaml
QT_UNIVERSE: custom
QT_MAX_SYMBOLS: 180
# Implement whitelist from agent output
```

#### **AGGRESSIVE Profile** — Testnet / Training
**When to use:**
- Testnet deployment
- ML model training
- Maximum opportunity capture

**Characteristics:**
- Size: 300-400 symbols
- Includes: CORE + EXPANSION + some CONDITIONAL
- Excludes: BLACKLIST only
- Expected: Higher variance, more signals

**Deployment:**
```yaml
QT_UNIVERSE: l1l2-top
QT_MAX_SYMBOLS: 400
# Implement blacklist from agent output
```

#### **EXPERIMENTAL Profile** — Research / Development
**When to use:**
- Strategy research
- Market structure analysis
- Maximum data collection

**Characteristics:**
- Size: 500-600 symbols
- Includes: Everything except BLACKLIST
- Expected: Maximum diversity, highest variance

**Deployment:**
```yaml
QT_UNIVERSE: all-usdt
QT_MAX_SYMBOLS: 600
# Implement minimal blacklist
```

---

## 🔄 DELTA IMPLEMENTATION WORKFLOW

### Step 1: Review Agent Output
```bash
# Copy output from container
docker cp quantum_backend:/app/data/universe_selector_output.json ./

# Parse recommendations
cat universe_selector_output.json | jq '.recommendations.SAFE'
```

### Step 2: Analyze Deltas
```json
{
  "deltas": {
    "SAFE": {
      "to_add": ["SYMBOL1", "SYMBOL2", ...],     // New symbols to include
      "to_remove": ["SYMBOL3", "SYMBOL4", ...],  // Symbols to exclude
      "to_keep": ["BTCUSDT", "ETHUSDT", ...],    // Existing symbols to retain
      "add_count": 15,
      "remove_count": 37,
      "keep_count": 185
    }
  }
}
```

### Step 3: Validate Changes
**Pre-deployment checklist:**
- [ ] Review all symbols in `to_remove` list
- [ ] Check if any majors (BTC, ETH, BNB, SOL) are being removed
- [ ] Verify `to_add` symbols have sufficient data
- [ ] Confirm blacklist symbols are truly poor performers
- [ ] Check deltas don't reduce universe below minimum (50 symbols)

### Step 4: Test in Paper Trading
```bash
# Create test configuration with agent recommendations
# Run for 7 days in paper trading mode
# Compare metrics vs current universe
```

### Step 5: Deploy to Production
```yaml
# docker-compose.yml or environment config
QT_UNIVERSE: custom  # or l1l2-top, all-usdt
QT_MAX_SYMBOLS: 180  # from agent recommendation

# Implement whitelist/blacklist in code
# Restart backend: docker-compose restart backend
```

---

## ⚠️ CRITICAL WARNINGS

### 1. Data Quality Requirements
**DO NOT deploy agent recommendations without:**
- ✅ At least 5 signals per symbol
- ✅ At least 3 trades per symbol
- ✅ Data confidence level: MEDIUM or higher
- ✅ 7+ days of continuous trading data

**Current status:** ❌ No trade data → **WAIT for data accumulation**

### 2. Major Coin Protection
**NEVER blacklist major coins without manual review:**
- BTC, ETH, BNB, SOL, XRP, ADA, DOT, AVAX, MATIC, LINK

**If agent recommends blacklisting a major:**
1. Investigate root cause (model issue, data quality, temporary conditions)
2. Review 30+ days of performance
3. Get manual approval before exclusion

### 3. Universe Size Limits
**Maintain minimum universe size:**
- **Absolute minimum:** 50 symbols
- **Recommended minimum:** 100 symbols (for diversity)
- **Safe range:** 150-300 symbols
- **Maximum:** 600 symbols (execution quality degrades beyond this)

### 4. Gradual Changes
**Implement changes gradually:**
- Week 1: Add/remove max 10% of universe
- Week 2: Add/remove another 10% if Week 1 successful
- Month 1: Full transition to recommended universe

**Avoid:** Changing 50%+ of universe at once

---

## 📊 MONITORING & VALIDATION

### Key Metrics to Track

**Before Deployment:**
```
- Current universe size
- Current allow rate
- Current avg confidence
- Current signals/day
- Current (if available) win rate, avg R, total PnL
```

**After Deployment:**
```
- New universe size (change %)
- New allow rate (vs baseline)
- New avg confidence (vs baseline)
- New signals/day (vs baseline)
- New win rate (vs baseline)
- New avg R (vs baseline)
- New total PnL (vs baseline)
- Max drawdown change
```

**Success Criteria:**
- ✅ Allow rate increases OR stays stable
- ✅ Avg confidence increases OR stays stable
- ✅ Win rate increases (if trade data available)
- ✅ Avg R increases (if trade data available)
- ✅ Max drawdown decreases OR stays stable
- ✅ Signal count >= 80% of previous (acceptable decrease for quality)

### Rollback Conditions

**Immediate rollback if:**
- ❌ Allow rate drops > 20%
- ❌ Win rate drops > 15% (over 7 days)
- ❌ Avg R drops > 20% (over 7 days)
- ❌ Max drawdown increases > 30%
- ❌ Signal count drops > 50%

---

## 🔧 AGENT CUSTOMIZATION

### Adjusting Classification Thresholds

**File:** `universe_selector_agent.py`

**CORE Thresholds (Lines 40-46):**
```python
CORE_THRESHOLDS = {
    'min_stability': 0.20,      # ↑ Stricter = fewer CORE symbols
    'min_quality': 0.25,         # ↑ Stricter = fewer CORE symbols
    'min_winrate': 0.45,         # ↑ Stricter = fewer CORE symbols
    'max_disallow_rate': 0.25,   # ↓ Stricter = fewer CORE symbols
    'min_avg_r': 0.5             # ↑ Stricter = fewer CORE symbols
}
```

**EXPANSION Thresholds (Lines 48-54):**
```python
EXPANSION_THRESHOLDS = {
    'min_stability': 0.10,       # More lenient than CORE
    'min_quality': 0.15,
    'min_winrate': 0.35,
    'max_disallow_rate': 0.40,
    'min_avg_r': 0.3
}
```

**BLACKLIST Thresholds (Lines 56-62):**
```python
BLACKLIST_THRESHOLDS = {
    'max_total_r': -0.5,         # ↓ More negative = stricter blacklist
    'max_winrate': 0.35,         # ↓ Lower = stricter blacklist
    'max_avg_r': 0.1,            # ↓ Lower = stricter blacklist
    'min_disallow_rate': 0.50,   # ↑ Higher = stricter blacklist
    'max_stability': 0.05        # ↓ Lower = stricter blacklist
}
```

**Tuning Tips:**
- **Conservative (mainnet):** Increase CORE thresholds, decrease BLACKLIST thresholds
- **Aggressive (testnet):** Decrease CORE thresholds, increase BLACKLIST thresholds
- **Balanced:** Use default values

---

## 🚀 INTEGRATION WITH ORCHESTRATOR

### Automatic Universe Updates (Future Enhancement)

```python
# Pseudocode for automatic integration

class OrchestratorPolicy:
    
    def __init__(self):
        self.universe_agent = UniverseSelectorAgent()
        self.last_universe_update = None
        self.update_frequency = timedelta(days=7)
    
    def should_update_universe(self):
        """Check if universe update is due"""
        if self.last_universe_update is None:
            return False
        
        elapsed = datetime.now() - self.last_universe_update
        return elapsed >= self.update_frequency
    
    def update_universe(self):
        """Run agent and apply recommendations"""
        
        # Run agent
        output = self.universe_agent.run()
        
        # Check data confidence
        if output['data_confidence'] not in ['HIGH', 'VERY_HIGH']:
            logger.warning("Universe update skipped - insufficient data confidence")
            return
        
        # Get recommendation for current mode
        if self.mode == 'MAINNET':
            rec = output['recommendations']['SAFE']
        else:
            rec = output['recommendations']['AGGRESSIVE']
        
        # Apply deltas
        self.apply_universe_deltas(rec['include'], rec['exclude'])
        
        # Update timestamp
        self.last_universe_update = datetime.now()
        
        logger.info(f"Universe updated: {len(rec['include'])} symbols")
```

**Implementation Status:** 🟡 **Not yet implemented** (manual execution only)

---

## 📝 CHANGE LOG

### Version 1.0 (Nov 23, 2025)
- ✅ Initial agent deployment
- ✅ Signal-based feature engineering
- ✅ Trade-based performance analysis (when data available)
- ✅ 4-tier classification (CORE, EXPANSION, CONDITIONAL, BLACKLIST)
- ✅ 3 profile recommendations (SAFE, AGGRESSIVE, EXPERIMENTAL)
- ✅ Delta computation (add/remove/keep)
- ✅ Performance curve analysis
- ✅ JSON output generation

### Planned Enhancements (Version 2.0)
- [ ] Automatic periodic execution (cron-like scheduling)
- [ ] Regime-specific universe profiles
- [ ] Time-of-day universe adjustments
- [ ] Spread/liquidity-based dynamic filtering
- [ ] Integration with Orchestrator for auto-updates
- [ ] A/B testing framework for universe changes
- [ ] Machine learning for universe optimization

---

## 🎯 QUICK START GUIDE

### For New Deployments (Day 1-7)

```bash
# 1. Run agent to establish baseline
docker exec quantum_backend python /app/universe_selector_agent.py

# 2. Copy output
docker cp quantum_backend:/app/data/universe_selector_output.json ./

# 3. Review output
cat universe_selector_output.json | jq '.summary'

# 4. EXPECTED RESULT: LOW confidence, most symbols in INSUFFICIENT_DATA
# → WAIT for trade data accumulation
```

### For Established Systems (Week 2+)

```bash
# 1. Run agent weekly
docker exec quantum_backend python /app/universe_selector_agent.py

# 2. Copy and review output
docker cp quantum_backend:/app/data/universe_selector_output.json ./

# 3. If data confidence is MEDIUM or HIGH:
#    - Review classifications
#    - Analyze deltas
#    - Test recommended universe in paper trading
#    - Deploy after validation

# 4. If data confidence is still LOW:
#    - WAIT for more data
#    - Re-run in 7 days
```

---

## 📞 SUPPORT & TROUBLESHOOTING

### Common Issues

**Issue: All symbols classified as BLACKLIST**
- **Cause:** No trade data available
- **Solution:** Wait for trade data accumulation (7-14 days)

**Issue: Agent output shows 0 CORE symbols**
- **Cause:** Insufficient profitable trades
- **Solution:** Check trading strategy performance, adjust thresholds

**Issue: Data confidence always LOW**
- **Cause:** Not enough signals/trades per symbol
- **Solution:** Run system longer, increase signal generation

**Issue: Agent recommends removing major coins**
- **Cause:** Temporary poor performance or data anomaly
- **Solution:** Manual review required, do NOT auto-apply

### Debug Commands

```bash
# Check data availability
docker exec quantum_backend ls -lh /app/data/policy_observations/
docker exec quantum_backend ls -lh /app/data/trades/

# Check signal count
docker exec quantum_backend bash -c "wc -l /app/data/policy_observations/signals_*.jsonl"

# View agent output summary
cat universe_selector_output.json | jq '.summary'

# List BLACKLIST symbols
cat universe_selector_output.json | jq '.classifications.BLACKLIST.symbols'

# Check top performers
cat universe_selector_output.json | jq '.symbol_scores | to_entries | sort_by(.value.quality_score) | reverse | .[0:10]'
```

---

## 🎓 BEST PRACTICES

1. **Run agent weekly** after initial 14-day data collection period
2. **Always review output** before applying recommendations
3. **Test changes in paper trading** for 7 days before production
4. **Never change > 20% of universe** at once
5. **Protect major coins** from automatic blacklisting
6. **Monitor performance metrics** closely after universe changes
7. **Keep rollback plan** ready (previous universe snapshot)
8. **Document all changes** with timestamps and rationale

---

**END OF GUIDE**

*For questions or issues, consult:*
- *UNIVERSE_ANALYSIS_REPORT.md (comprehensive analysis)*
- *UNIVERSE_ANALYSIS_SUMMARY.md (quick reference)*
- *universe_selector_output.json (latest agent output)*

*Next recommended action:* **Wait 7-14 days for trade data, then re-run agent**
