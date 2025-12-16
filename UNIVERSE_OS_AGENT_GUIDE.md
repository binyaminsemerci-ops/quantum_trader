# UNIVERSE OS AGENT — Complete Operational Guide

**The Autonomous AI Operating System for Trading Universe Management**

---

## 🎯 MISSION

UNIVERSE OS AGENT is **NOT a simple function** — it is a **full AI operating system** responsible for the entire trading symbol universe lifecycle:

```
BUILD → ANALYZE → VISUALIZE → RANK → SELECT → 
SCHEDULE → SNAPSHOT → OPTIMIZE → INTEGRATE
```

---

## 🏗️ ARCHITECTURE

### Core Responsibilities

1. **Universe Discovery** — Load and merge all data sources
2. **Universe Quality Analysis** — Compute 30+ metrics per symbol
3. **Symbol-level Health Diagnostics** — Performance, costs, stability
4. **Universe Ranking + Optimization** — Performance curves, diminishing returns
5. **Core/Expansion/Blacklist Classification** — 4-tier intelligent categorization
6. **Dynamic Universe Construction** — SAFE, AGGRESSIVE, EXPERIMENTAL profiles
7. **Snapshot Writer** — Generate complete universe configs
8. **Universe Delta Engine** — Track add/remove/keep changes
9. **Visualizations** — Charts, plots, heatmaps (optional)
10. **Scheduler** — Periodic universe optimization
11. **Reporting + Recommendations** — Executive summaries, actionable insights
12. **Integration** — OrchestratorPolicy & Risk Manager compatible

---

## 📊 DATA INPUTS

### A) Universe Snapshot (`universe_snapshot.json`)
- Last generated universe configuration
- Symbol count, mode, QT_MAX_SYMBOLS
- Current symbol list

### B) Selector Output (`universe_selector_output.json`)
- Previous Universe Selector Agent results
- Historical classifications
- Legacy recommendations

### C) Trade Logs (`/app/data/trades/*.jsonl`)
```json
{
  "symbol": "BTCUSDT",
  "entry_timestamp": "...",
  "exit_timestamp": "...",
  "R": 1.5,
  "pnl": 150.0,
  "slippage": 0.001,
  "spread": 0.0005,
  "exit_reason": "TP_HIT",
  "holding_time_seconds": 3600,
  "regime_tag": "TRENDING_STRONG",
  "vol_level": "NORMAL"
}
```

### D) Policy Observations (`/app/data/policy_observations/*.jsonl`)
```json
{
  "symbol": "ETHUSDT",
  "decision": "TRADE_ALLOWED",
  "confidence": 0.75,
  "regime": "TRENDING",
  "volatility": "NORMAL",
  "timestamp": "..."
}
```

### E) OrchestratorPolicy Parameters
- Current regime_tag
- Current vol_level
- Disallowed symbols list
- Risk profile

### F) Exchange Market Metadata (optional)
- Volume data
- Liquidity estimates

---

## 🔬 SYMBOL FEATURES ENGINE

### 30+ Computed Metrics Per Symbol

#### Performance Metrics
```python
winrate               # Win percentage
avg_R                 # Average R-multiple
median_R              # Median R-multiple
total_R               # Cumulative R
std_R                 # R volatility
max_R                 # Best trade
min_R                 # Worst trade
profit_factor         # Gross win / gross loss
trade_count           # Total trades
R_per_trade           # R efficiency
```

#### Cost Metrics
```python
avg_slippage          # Average execution cost
avg_spread            # Average bid-ask spread
max_slippage          # Worst slippage event
rollover_cost         # Overnight costs (if available)
```

#### Stability Metrics
```python
volatility_score      # std_R (R variance)
stability_score       # (avg_R * winrate) / (costs + variance)
consistency_score     # 1 / variance(R_series)
```

#### Regime Metrics
```python
trending_R            # Performance in TRENDING regime
ranging_R             # Performance in RANGING regime
mixed_R               # Performance in MIXED regime
regime_dependency_score  # How much performance varies by regime
```

#### Volatility Metrics
```python
high_vol_R            # Performance in HIGH/EXTREME volatility
extreme_vol_R         # Performance in EXTREME volatility
normal_vol_R          # Performance in NORMAL/LOW volatility
```

#### Policy Metrics
```python
disallow_rate         # % of signals blocked by policy
avg_confidence        # Average signal confidence
signal_count          # Total signals generated
allow_count           # Signals allowed
disallow_count        # Signals blocked
```

#### Composite Scores
```python
quality_score         # Overall symbol quality (0-1)
profitability_score   # Profit generation ability (0-1)
reliability_score     # Signal consistency (0-1)
```

---

## 🎯 CLASSIFICATION SYSTEM

### CORE (Top 50-150 symbols)
**Production-ready, must-trade symbols**

Thresholds:
- `stability_score >= 0.20`
- `quality_score >= 0.25`
- `winrate >= 0.45` (45%)
- `disallow_rate <= 0.25` (25%)
- `avg_R >= 0.5`

Characteristics:
- Stable performance
- Consistently profitable
- Low spread & slippage
- High stability_score
- Low disallow_rate

### EXPANSION (150-350 symbols)
**Good performers for aggressive profiles**

Thresholds:
- `stability_score >= 0.10`
- `quality_score >= 0.15`
- `winrate >= 0.35` (35%)
- `disallow_rate <= 0.40` (40%)
- `avg_R >= 0.3`

Characteristics:
- Profitable with some volatility
- Acceptable slippage
- Good for testnet/training

### CONDITIONAL (Situational)
**Profitable in specific conditions**

Criteria:
- Profitable only in TRENDING (`trending_R > 0.5`)
- Profitable only in NORMAL volatility (`normal_vol_R > 0.5`)
- Profitable but unstable (`avg_R > 0.3` but low `stability_score`)

Use Case:
- Regime-specific strategies
- Advanced conditional logic
- Research & backtesting

### BLACKLIST (Exclude)
**Poor performers to avoid**

Criteria (any of):
- `total_R < -0.5` AND `winrate < 0.35`
- `avg_R < 0.1`
- `disallow_rate > 0.50`
- `stability_score < 0.05` (with `trade_count >= 10`)

Characteristics:
- Consistently negative performance
- High slippage or spread
- High disallow_rate
- Unstable

### INSUFFICIENT_DATA
**Need more data**

Criteria:
- `trade_count < 3`
- `signal_count < 5`

Action: Wait for more data accumulation

---

## 🚀 UNIVERSE PROFILES

### SAFE Profile (150-200 symbols)
**Production & Mainnet Trading**

Composition:
- All CORE symbols
- Top EXPANSION symbols (sorted by quality_score)
- Exclude CONDITIONAL and BLACKLIST

Target Size: 150-200 symbols

Risk Level: **LOW**

Use Cases:
- Real money trading
- Mainnet deployment
- Conservative strategies

Expected:
- High stability
- Low slippage
- Consistent performance

### AGGRESSIVE Profile (300-400 symbols)
**Testnet & Training**

Composition:
- All CORE symbols
- All EXPANSION symbols
- Some CONDITIONAL symbols
- Exclude BLACKLIST only

Target Size: 300-400 symbols

Risk Level: **MEDIUM**

Use Cases:
- ML model training
- Testnet deployment
- Maximum opportunity capture

Expected:
- More volatility
- Higher signal volume
- Broader diversification

### EXPERIMENTAL Profile (500-600 symbols)
**Research & Data Collection**

Composition:
- CORE + EXPANSION + CONDITIONAL
- INSUFFICIENT_DATA symbols
- Exclude BLACKLIST only

Target Size: 500-600 symbols

Risk Level: **HIGH**

Use Cases:
- Strategy research
- AI learning & exploration
- Maximum data collection

Expected:
- Maximum volatility
- All market conditions
- Broadest coverage

---

## 📈 OPTIMIZATION METHODOLOGY

### Performance Curves

Universe OS Agent computes performance vs universe size for:

```
Sizes tested: 50, 100, 150, 200, 300, 400, 500, 600
```

Metrics tracked:
- **Avg R** — Average R-multiple
- **Avg Winrate** — Win percentage
- **Avg Quality** — Quality score
- **Avg Stability** — Stability score
- **Total R** — Cumulative R
- **Avg Slippage** — Execution costs

Analysis:
- **Diminishing returns** — Where does quality plateau?
- **Slippage impact** — How does size affect costs?
- **Regime sensitivity** — Performance across conditions

Recommendation:
- **Optimal size** — Best risk/reward balance
- **Safe ceiling** — Max size before quality degrades

---

## 🔄 DELTA ENGINE

### Delta Computation

Compares **new recommended universe** vs **current universe**:

```python
{
  "to_add": [...]        # Symbols to add
  "to_remove": [...]     # Symbols to remove
  "to_keep": [...]       # Symbols to keep
  "net_change": +/- N    # Net change count
}
```

### Symbol Movements Tracking

```python
{
  "promoted_to_core": [...]       # Moved to CORE
  "demoted_from_core": [...]      # Removed from CORE
  "added_to_blacklist": [...]     # New blacklist entries
  "removed_from_blacklist": [...] # Blacklist removals
}
```

### Immediate Actions

**Immediate Removals** (high-conviction blacklist):
- `total_R < -1.0`
- `winrate < 0.30`
- `trade_count >= 5`

**Immediate Additions** (high-quality CORE):
- Not in current universe
- `quality_score > 0.30`
- `trade_count >= 5`

**Watch List** (borderline symbols):
- EXPANSION or CONDITIONAL tier
- `quality_score > 0.20`
- `trade_count < 10` (need more data)

---

## 🕒 SCHEDULER & OPERATING MODES

### Environment Variables

```bash
# Operating mode
UNIVERSE_OS_MODE=OBSERVE          # OBSERVE | FULL

# Execution interval
UNIVERSE_OS_INTERVAL_HOURS=4      # Run every N hours

# Visualization
UNIVERSE_OS_VISUALIZE=false       # true | false
```

### OBSERVE Mode (Default)
**Safe monitoring mode**

Actions:
- ✅ Analyze all data
- ✅ Compute features
- ✅ Classify symbols
- ✅ Generate profiles
- ✅ Compute deltas
- ✅ Write snapshots
- ✅ Generate reports
- ❌ **NO changes applied to trading universe**

Use Case:
- Initial deployment
- Validation runs
- Auditing changes before applying

### FULL Mode
**Autonomous operation mode**

Actions:
- ✅ All OBSERVE mode actions
- ✅ **Update universe configs**
- ✅ **Prepare new universe for next restart**
- ✅ **Apply whitelist/blacklist**

Use Case:
- Production automation
- Continuous optimization
- After validation period

⚠️ **WARNING:** FULL mode modifies production configs. Use only after thorough testing.

---

## 📋 EXECUTION COMMANDS

### Run Universe OS Agent (Manual)

```bash
# In container
docker exec quantum_backend python /app/universe_os_agent.py

# Copy outputs
docker cp quantum_backend:/app/data/universe_os_snapshot.json ./
docker cp quantum_backend:/app/data/universe_delta_report.json ./
```

### View Snapshots

```bash
# Full snapshot
cat universe_os_snapshot.json | jq '.'

# Classification summary
cat universe_os_snapshot.json | jq '.classifications'

# Profile recommendations
cat universe_os_snapshot.json | jq '.profiles'

# Deltas
cat universe_os_snapshot.json | jq '.deltas'

# Top performers
cat universe_os_snapshot.json | jq '.performance_curves'
```

### View Delta Report

```bash
# Full report
cat universe_delta_report.json | jq '.'

# Immediate actions
cat universe_delta_report.json | jq '.recommendations'

# Symbol movements
cat universe_delta_report.json | jq '.symbol_movements'
```

---

## 📊 OUTPUT FILES

### `universe_os_snapshot.json`
Complete universe state with:
- Data confidence level
- Current universe info
- Classifications (CORE, EXPANSION, CONDITIONAL, BLACKLIST)
- Universe profiles (SAFE, AGGRESSIVE, EXPERIMENTAL)
- Performance curves
- Deltas
- Recommendations

### `universe_delta_report.json`
Detailed change tracking:
- Deltas by profile
- Symbol movements (promotions/demotions)
- Immediate actions (removals, additions, watch list)

### `universe_charts/` (if visualization enabled)
- `pnl_vs_universe_size.png`
- `cumulative_R_curve.png`
- `symbol_performance_scatter.png`
- `slippage_heatmap.png`
- `spread_distribution.png`
- `regime_performance_bars.png`
- `core_vs_blacklist_comparison.png`

---

## 🎯 DATA CONFIDENCE LEVELS

| Confidence | Signals | Trades | Action |
|------------|---------|--------|--------|
| **LOW** | < 1,000 | < 100 | ⏸️ WAIT — Collect more data |
| **MEDIUM** | 1,000-5,000 | 100-500 | ⚠️ REVIEW — Preliminary recommendations |
| **HIGH** | 5,000-10,000 | 500-1,000 | ✅ DEPLOY — Reliable recommendations |
| **VERY_HIGH** | 10,000+ | 1,000+ | ✅ DEPLOY — High confidence |

**Current Status:** LOW (need trade data accumulation)

---

## 🚦 DEPLOYMENT WORKFLOW

### Week 1: Initial Baseline (Current)
```
✅ Universe OS Agent deployed
✅ Initial run completed
✅ Baseline established

📊 Status:
   - 218 symbols analyzed
   - 0 trades (too early)
   - 0 signals loaded
   - Data confidence: LOW

⏸️ Action: WAIT for data accumulation
```

### Week 2: First Real Analysis (7-14 days)
```
1. Collect 100+ trades across symbols
2. Collect 1,000+ signals
3. Re-run Universe OS Agent
4. Review classifications (expect CORE/EXPANSION/BLACKLIST splits)
5. Check data confidence (target: MEDIUM)

If confidence >= MEDIUM:
  → Proceed to Week 3
Else:
  → Continue collecting data
```

### Week 3: Validation (Paper Trading)
```
1. Run Universe OS Agent
2. Extract recommended profile (SAFE or AGGRESSIVE)
3. Review deltas:
   - Are removals justified?
   - Are additions high-quality?
   - Are majors protected?
4. Deploy to paper trading for 7 days
5. Compare metrics vs baseline:
   - Allow rate
   - Win rate
   - Avg R
   - Signal count
   - Slippage
```

### Week 4: Production Deployment
```
If paper trading validates:
  1. Review delta report one final time
  2. Check immediate actions list
  3. Update QT_UNIVERSE in config
  4. Update QT_MAX_SYMBOLS
  5. Implement whitelist/blacklist
  6. Restart backend
  7. Monitor closely for 72 hours
  8. Track all metrics
  9. Be ready to rollback

Else:
  → Adjust thresholds or wait for more data
```

---

## ⚠️ CRITICAL WARNINGS

### ❌ DO NOT Deploy If:
- [ ] Data confidence is LOW
- [ ] Agent recommends removing 3+ major coins without clear reason
- [ ] Agent recommends removing > 40% of current universe
- [ ] Trade data shows < 3 trades per symbol on average
- [ ] Classification shows 0 CORE symbols
- [ ] Blacklist contains > 50% of universe

### ⚠️ Manual Review Required If:
- [ ] Any major coin (BTC, ETH, BNB, SOL, XRP, ADA) in BLACKLIST
- [ ] CORE count < 20 symbols
- [ ] BLACKLIST count > 100 symbols
- [ ] Recommended universe size < 100 symbols
- [ ] Deltas show > 30% change from current

### ✅ Safe to Deploy If:
- [x] Data confidence >= MEDIUM
- [x] CORE count >= 20 symbols
- [x] BLACKLIST count < 30% of universe
- [x] All majors in CORE or EXPANSION
- [x] Deltas show < 25% change
- [x] Paper trading results positive

---

## 🔧 AGENT CUSTOMIZATION

### Threshold Tuning

Edit `universe_os_agent.py`:

```python
# CORE classification (line ~520)
if (f.stability_score >= 0.20 and    # ← Adjust
    f.quality_score >= 0.25 and      # ← Adjust
    f.winrate >= 0.45 and            # ← Adjust
    f.disallow_rate <= 0.25 and      # ← Adjust
    f.avg_R >= 0.5):                 # ← Adjust
    return "CORE", ...

# EXPANSION classification (line ~528)
if (f.stability_score >= 0.10 and    # ← Adjust
    f.quality_score >= 0.15 and      # ← Adjust
    f.winrate >= 0.35 and            # ← Adjust
    f.disallow_rate <= 0.40 and      # ← Adjust
    f.avg_R >= 0.3):                 # ← Adjust
    return "EXPANSION", ...

# BLACKLIST classification (line ~510)
if f.total_R < -0.5 and f.winrate < 0.35:  # ← Adjust
    return "BLACKLIST", ...
```

### Feature Weight Tuning

```python
# Profitability score (line ~370)
score = 0.5 * avg_r_component + \      # ← Adjust weights
        0.3 * total_r_component + \
        0.2 * pf_component

# Reliability score (line ~380)
score = 0.4 * allow_rate_component + \  # ← Adjust weights
        0.3 * confidence_component + \
        0.3 * consistency_component

# Quality score (line ~390)
score = 0.4 * f.profitability_score + \  # ← Adjust weights
        0.3 * f.reliability_score + \
        0.3 * min(1, f.stability_score)
```

---

## 🔗 ORCHESTRATOR INTEGRATION

### Current State
Universe OS Agent runs **independently** and outputs recommendations.

### Future Integration (Planned)

```python
# In orchestrator.py
from universe_os_agent import UniverseOSAgent

class OrchestratorPolicy:
    def __init__(self):
        self.universe_os = UniverseOSAgent()
        
    def periodic_universe_check(self):
        """Run Universe OS Agent on schedule"""
        if self.should_run_universe_os():
            snapshot = self.universe_os.run()
            
            if snapshot['data_confidence'] in ['HIGH', 'VERY_HIGH']:
                self.apply_universe_changes(snapshot)
    
    def apply_universe_changes(self, snapshot):
        """Apply Universe OS recommendations"""
        deltas = snapshot['deltas']['SAFE']
        
        # Remove blacklist
        for symbol in deltas['to_remove']:
            self.blacklist_symbol(symbol)
        
        # Add high-quality
        for symbol in deltas['to_add']:
            self.whitelist_symbol(symbol)
        
        # Update config
        self.update_universe_config(
            snapshot['recommendations']['recommended_profile']
        )
```

---

## 📊 MONITORING & KPIs

### Key Metrics to Track

**After Universe OS Agent Run:**
- Data confidence level
- Classification counts (CORE/EXPANSION/BLACKLIST)
- Recommended universe size
- Expected R & winrate for profiles

**After Universe Change Deployment:**
- Actual allow rate vs expected
- Actual win rate vs expected
- Actual avg R vs expected
- Signal count change
- Slippage impact
- Trade frequency
- PnL delta

### Weekly Checklist

```
□ Run Universe OS Agent
□ Review data confidence (target: MEDIUM → HIGH → VERY_HIGH)
□ Check classification distribution
□ Review top 20 symbols by quality
□ Review bottom 20 symbols (blacklist candidates)
□ Analyze deltas (if any major changes, investigate)
□ Check for major coins in BLACKLIST (manual review)
□ Compare metrics vs last week
□ Document any anomalies
```

---

## 🚨 ROLLBACK PROCEDURE

If universe change causes issues:

```bash
# 1. Identify issue
docker logs quantum_backend --since 1h | grep ERROR

# 2. Revert universe snapshot
docker exec quantum_backend cp /app/data/universe_snapshot.json.backup \
                                /app/data/universe_snapshot.json

# 3. Revert config
# Restore previous QT_SYMBOLS or QT_UNIVERSE in docker-compose.yml

# 4. Restart backend
docker-compose restart backend

# 5. Verify rollback
docker exec quantum_backend cat /app/data/universe_snapshot.json | jq '.symbol_count'

# 6. Document incident
echo "$(date): Rollback performed - [reason]" >> universe_changes.log
```

---

## 📝 BEST PRACTICES

1. **Wait for sufficient data** — Don't deploy with LOW confidence
2. **Validate in paper trading** — Test for 7 days before production
3. **Protect major coins** — Never auto-blacklist BTC, ETH, BNB, SOL, XRP, ADA
4. **Change gradually** — Max 20% of universe per week
5. **Monitor closely** — Track all metrics for 72 hours post-deployment
6. **Document everything** — Keep change log with timestamps
7. **Keep rollback ready** — Always backup previous universe
8. **Review manually** — Don't blindly apply agent recommendations
9. **Run weekly** — After Week 2, run Universe OS Agent every Monday
10. **Tune thresholds** — Adjust classification criteria based on strategy

---

## 🎓 TROUBLESHOOTING

### Issue: "All symbols in INSUFFICIENT_DATA"
**Cause:** No trade or signal data available  
**Fix:** Wait for data accumulation (7-14 days)

### Issue: "0 CORE symbols"
**Cause:** High thresholds OR poor trading performance  
**Fix:**
1. Check trading strategy performance
2. Review threshold settings (lower if needed)
3. Investigate market conditions

### Issue: "Major coin in BLACKLIST"
**Cause:** Temporary poor performance OR data anomaly  
**Fix:**
1. Review last 30 days performance for that coin
2. Check if model calibration issue
3. Manual override — DO NOT auto-blacklist majors

### Issue: "Performance curves not computed"
**Cause:** Insufficient trade data  
**Fix:** Wait for 100+ trades, then re-run

### Issue: "Data confidence stuck at LOW"
**Cause:** Insufficient signal/trade generation  
**Fix:**
1. Increase confidence threshold in trading strategy
2. Verify executor is running
3. Check signal generation rate (target: 300-500/day)

---

## 📞 DECISION TREE

```
Run Universe OS Agent
    │
    ├─ Data Confidence < MEDIUM?
    │   └─ YES → WAIT (collect more data)
    │   └─ NO → Continue
    │
    ├─ CORE count < 20?
    │   └─ YES → INVESTIGATE (tune thresholds?)
    │   └─ NO → Continue
    │
    ├─ Any major in BLACKLIST?
    │   └─ YES → MANUAL REVIEW required
    │   └─ NO → Continue
    │
    ├─ Deltas show > 30% change?
    │   └─ YES → GRADUAL deployment
    │   └─ NO → Continue
    │
    ├─ Deploy to paper trading
    │   │
    │   ├─ Results positive after 7 days?
    │   │   └─ YES → Deploy to production
    │   │   └─ NO → Reject, investigate
    │   │
    │   └─ Monitor production 72 hours
    │       │
    │       ├─ Metrics stable/improved?
    │       │   └─ YES → ✅ SUCCESS
    │       │   └─ NO → ⚠️ ROLLBACK
```

---

## ⏱️ EXPECTED TIMELINE

```
Day 1:    Universe OS Agent deployed, baseline established
Day 7:    ~100-300 trades, ~3,200 signals (MEDIUM confidence possible)
Day 14:   ~300-600 trades, ~6,400 signals (HIGH confidence likely)
Day 21:   Paper trading with recommended universe
Day 28:   Production deployment (if validated)
Day 35+:  Weekly optimization cycles
```

---

## 🎯 SUCCESS CRITERIA

### Week 2 Milestone
- [ ] Data confidence: MEDIUM or higher
- [ ] CORE: 20-50 symbols
- [ ] EXPANSION: 50-100 symbols
- [ ] BLACKLIST: < 30 symbols
- [ ] No major coins in BLACKLIST

### Week 4 Milestone
- [ ] Paper trading validates recommendations
- [ ] Allow rate: stable or improved
- [ ] Win rate: >= baseline
- [ ] Avg R: >= baseline
- [ ] Slippage: acceptable levels

### Month 2 Milestone
- [ ] Data confidence: HIGH or VERY_HIGH
- [ ] Universe optimized for strategy
- [ ] Weekly Universe OS runs automated
- [ ] Performance improvement confirmed
- [ ] Blacklist effectively reducing losses

---

## 🚀 NEXT STEPS

### Immediate (Today)
✅ Universe OS Agent deployed  
✅ Initial baseline established  
✅ Documentation complete

### Week 1
- Collect trade data (target: 100+ trades)
- Collect signal data (target: 1,000+ signals)
- Monitor system health
- No agent runs needed

### Week 2 (DECISION POINT)
- Run Universe OS Agent
- Review data confidence (target: MEDIUM)
- Analyze classifications
- Test recommendations in paper trading

### Month 1
- Weekly agent runs
- Monitor classification changes
- Track blacklist growth
- Compare profile performance
- Deploy validated universe

### Quarter 1
- Bi-weekly agent runs
- Implement auto-execution (FULL mode)
- Build regime-specific universes
- Integrate with Orchestrator
- A/B testing framework

---

## 📚 RELATED DOCUMENTATION

- **UNIVERSE_SELECTOR_AGENT_GUIDE.md** — Original agent guide (legacy)
- **UNIVERSE_SELECTOR_PLAYBOOK.md** — Quick reference (legacy)
- **UNIVERSE_ANALYSIS_REPORT.md** — Manual analysis
- **UNIVERSE_DEPLOYMENT_CONFIG.json** — Config templates

---

**END OF UNIVERSE OS AGENT OPERATIONAL GUIDE**

*This is the autonomous AI operating system for your trading universe.*

*Run command: `docker exec quantum_backend python /app/universe_os_agent.py`*

*Next milestone: Week 2 — Re-run after 100+ trades collected*

---

**Version:** 2.0  
**Date:** November 23, 2025  
**Status:** DEPLOYED & OPERATIONAL
