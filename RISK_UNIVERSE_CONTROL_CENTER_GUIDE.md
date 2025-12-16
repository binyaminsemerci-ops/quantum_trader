# RISK & UNIVERSE CONTROL CENTER OS — COMPLETE OPERATIONAL GUIDE

**Version:** 3.0  
**Date:** November 23, 2025  
**Status:** ✅ DEPLOYED & OPERATIONAL

---

## 🎯 MISSION

**MAINTAIN A SAFE, PROFITABLE, STABLE, SELF-OPTIMIZING TRADING UNIVERSE WITHOUT HUMAN INTERVENTION.**

The Risk & Universe Control Center OS is the **autonomous, high-level supervisory AI system** responsible for overseeing, evaluating, optimizing, protecting, and managing the full trading universe and risk environment in Quantum Trader.

---

## 🏗️ WHAT IS THIS?

This is **NOT a single module** — it is a **FULL OPERATING SYSTEM** governing:

- ✅ Universe Selection & Optimization
- ✅ Universe Health Monitoring & Performance Analysis
- ✅ Symbol Classification & Toxicity Detection
- ✅ Risk Oversight & Emergency Braking
- ✅ Real-time Monitoring & Scheduler Orchestration
- ✅ Visual Intelligence & Snapshot/Delta Engines
- ✅ Orchestrator + Risk Manager Integration

---

## 📊 DATA INPUTS (READ-ONLY)

The Control Center OS loads and analyzes data from multiple sources:

### 1. Universe Snapshot (`/app/data/universe_snapshot.json`)
- Current universe configuration
- Symbol list (222 symbols in current deployment)
- Universe mode (explicit/optimized)

### 2. Universe Selector Output (`/app/data/universe_selector_output.json`)
- Legacy selector agent decisions (optional)
- Historical classification reference

### 3. Trade Data (`/app/data/trades/*.jsonl`)
- Complete trade history per symbol
- Performance metrics: R-multiples, PnL, winrate
- Cost metrics: slippage, spread
- Regime and volatility tags
- Entry/exit timestamps

### 4. Signal Data (`/app/data/policy_observations/*.jsonl`)
- Policy decisions (TRADE_ALLOWED / TRADE_DISALLOWED)
- Confidence scores
- Disallow reasons
- Decision timestamps

### 5. Orchestrator Policy (Runtime State)
- `regime_tag`: Current market regime
- `vol_level`: Current volatility level
- `disallowed_symbols`: Currently blacklisted symbols
- `allow_new_trades`: Global trading flag
- `risk_profile`: Current risk mode

### 6. Exchange Metadata (Optional)
- Real-time spread
- Real-time slippage
- Liquidity levels
- Orderbook depth

---

## 🔬 SYMBOL HEALTH ENGINE (30+ Metrics Per Symbol)

The Control Center computes comprehensive health profiles for every symbol:

### Performance Metrics (10)
```python
- winrate              # Win rate: wins / total_trades
- avg_R                # Average R-multiple per trade
- total_R              # Cumulative R-multiple
- median_R             # Median R-multiple
- variance_R           # Variance of R-multiples
- profit_factor        # Gross wins / Gross losses
- trade_count          # Number of completed trades
```

### Cost Behavior Metrics (6)
```python
- avg_slippage         # Average slippage per trade
- avg_spread           # Average spread per trade
- max_slippage         # Maximum slippage observed
- max_spread           # Maximum spread observed
- slippage_spikes      # Count of slippage > emergency threshold
- spread_explosions    # Count of spread > emergency threshold
```

### Regime Profile Metrics (4)
```python
- trending_R           # Avg R in TRENDING regimes
- ranging_R            # Avg R in RANGING regimes
- mixed_R              # Avg R in MIXED regimes
- regime_dependency    # Std dev of regime R-multiples
```

### Volatility Profile Metrics (4)
```python
- high_vol_R           # Avg R in HIGH/EXTREME volatility
- extreme_vol_R        # Avg R in EXTREME volatility
- normal_vol_R         # Avg R in NORMAL/LOW volatility
- vol_sensitivity      # Std dev of volatility R-multiples
```

### Policy Alignment Metrics (3)
```python
- disallow_rate        # % of signals disallowed
- avg_confidence       # Average policy confidence
- confidence_std       # Std dev of confidence scores
```

### Composite Scores (3)
```python
- stability_score = (avg_R * winrate) / (spread + slippage + variance_R + ε)
- quality_score = 0.4 * profitability + 0.3 * reliability + 0.3 * stability
- toxicity_score = mean([slippage_spikes_rate, spread_explosions_rate, 
                         disallow_rate_penalty, negative_R_penalty])
```

---

## 🎯 SYMBOL CLASSIFICATION (4 Tiers)

Every symbol is classified into one of four tiers:

### CORE (Most Desirable)
**Criteria:**
- `stability_score >= 0.20`
- `quality_score >= 0.25`
- `winrate >= 0.45`
- Low cost behavior
- Predictable performance

**Characteristics:**
- Stable, consistent winners
- Low slippage and spread
- High policy confidence
- Safe for production

**Use Case:** Mainnet, real money trading

---

### EXPANSION (Good Performers)
**Criteria:**
- `stability_score >= 0.10`
- `quality_score >= 0.15`
- `winrate >= 0.35`
- Profitable but higher variance

**Characteristics:**
- Good performance with volatility
- Higher costs than CORE
- Regime sensitive
- Broader opportunity set

**Use Case:** Testnet, aggressive profiles

---

### CONDITIONAL (Regime-Specific)
**Criteria:**
- `trending_R > 0.5` OR
- `normal_vol_R > 0.5` OR
- Good in specific regimes only

**Characteristics:**
- Only profitable in certain conditions
- Regime-dependent winners
- Volatile in wrong regimes

**Use Case:** Regime-aware trading, experimental

---

### BLACKLIST (Exclude)
**Criteria:**
- `toxicity_score > 0.5` OR
- `avg_R < -0.3` (with 5+ trades) OR
- `disallow_rate > 35%` OR
- Multiple slippage/spread explosions

**Characteristics:**
- Consistently toxic
- High cost behavior
- Unpredictable losses
- Policy frequently disallows

**Use Case:** NEVER trade these symbols

---

## 🚀 UNIVERSE OPTIMIZATION ENGINE

The Control Center generates three optimized universe profiles:

### SAFE PROFILE (Mainnet Production)
**Composition:** CORE + top EXPANSION  
**Target Size:** 150–200 symbols  
**Risk Level:** LOW  

**Selection Criteria:**
1. All CORE symbols (highest quality)
2. Top EXPANSION symbols by `quality_score`
3. Exclude all BLACKLIST
4. Minimize cost exposure

**Expected Characteristics:**
- High stability
- Low slippage/spread
- Consistent winrate (45%+)
- Predictable performance
- Low variance

**Use Case:** Real money, mainnet, conservative

---

### AGGRESSIVE PROFILE (Testnet/Training)
**Composition:** CORE + EXPANSION + top CONDITIONAL  
**Target Size:** 250–400 symbols  
**Risk Level:** MEDIUM  

**Selection Criteria:**
1. All CORE symbols
2. All EXPANSION symbols
3. Top 20 CONDITIONAL symbols
4. Exclude BLACKLIST only

**Expected Characteristics:**
- Broader coverage
- Higher signal volume
- More volatility exposure
- Regime diversity
- Higher opportunity count

**Use Case:** Testnet, ML training, max opportunity

---

### EXPERIMENTAL PROFILE (Research)
**Composition:** All except BLACKLIST  
**Target Size:** 400–600 symbols  
**Risk Level:** HIGH  

**Selection Criteria:**
1. All classified symbols
2. All INSUFFICIENT_DATA symbols
3. Exclude BLACKLIST only
4. Maximum breadth

**Expected Characteristics:**
- Maximum diversity
- Highest volatility
- Broadest coverage
- Most AI learning data
- Unpredictable performance

**Use Case:** Research, AI training, data collection

---

## 🛑 EMERGENCY BRAKE MODULE

The Control Center continuously monitors for emergency conditions and triggers protective actions:

### Emergency Triggers

#### Symbol-Level Triggers
- **Slippage Spikes:** 3+ trades with slippage > 1%
- **Spread Explosions:** 3+ trades with spread > 0.5%
- **Loss Streak:** 5+ consecutive losing trades
- **Toxicity:** `toxicity_score > 0.7`

#### Universe-Level Triggers
- **Severe Drawdown:** Cumulative R drawdown < -15%
- **Cost Explosion:** 10+ symbols with cost spikes
- **Winrate Collapse:** Rolling winrate < 35%
- **Multiple Toxic Symbols:** 5+ symbols with high toxicity

### Emergency Actions

#### Symbol-Level Actions
```python
WATCH         # Monitor closely, no restrictions
PAUSE_SYMBOL  # Stop new trades, exit existing
REDUCE_RISK   # Lower position size
BLACKLIST     # Permanently exclude
```

#### Global-Level Actions
```python
NORMAL               # No restrictions
NO_NEW_TRADES        # Block all new entries
REDUCE_GLOBAL_RISK   # Lower all position sizes
DEFENSIVE_EXIT_MODE  # Exit all positions safely
SAFE_UNIVERSE_MODE   # Switch to SAFE profile
EMERGENCY_BRAKE      # Full trading halt
```

### Emergency Response Structure
```json
{
  "triggered": true,
  "severity": "CRITICAL",
  "reason": "Severe drawdown: -18.5",
  "recommended_action": "DEFENSIVE_EXIT_MODE",
  "affected_symbols": ["BTCUSDT", "ETHUSDT"],
  "duration_hours": 48,
  "timestamp": "2025-11-23T00:00:00Z"
}
```

---

## 🕒 SCHEDULER ENGINE

The Control Center operates in two modes:

### OBSERVE MODE (Default)
**Behavior:**
- ✅ Analyze all data
- ✅ Compute symbol health
- ✅ Classify symbols
- ✅ Generate universe profiles
- ✅ Monitor for emergencies
- ✅ Write snapshots and deltas
- ❌ NO AUTOMATIC CHANGES to trading universe

**Use Case:** Validation, testing, monitoring

**Configuration:**
```bash
UNIVERSE_OS_MODE=OBSERVE
```

---

### FULL_AUTONOMY MODE (Production)
**Behavior:**
- ✅ All OBSERVE mode actions
- ✅ Update universe configurations
- ✅ Apply blacklist changes
- ✅ Update risk profile
- ✅ Trigger emergency brakes
- ✅ Integrate with Orchestrator

**Use Case:** Autonomous operation, production

**Configuration:**
```bash
UNIVERSE_OS_MODE=FULL_AUTONOMY
```

---

## 📋 EXECUTION COMMANDS

### Run Control Center OS (Manual)
```bash
docker exec quantum_backend python /app/risk_universe_control_center.py
```

### Run with Environment Variables
```bash
docker exec -e UNIVERSE_OS_MODE=OBSERVE \
            -e UNIVERSE_OS_INTERVAL_HOURS=4 \
            -e EMERGENCY_SLIPPAGE_THRESHOLD=0.01 \
            quantum_backend python /app/risk_universe_control_center.py
```

### View Health Report
```bash
docker cp quantum_backend:/app/data/universe_health_report.json ./
cat universe_health_report.json | jq '.universe_health'
```

### View Control Snapshot
```bash
docker cp quantum_backend:/app/data/universe_control_snapshot.json ./
cat universe_control_snapshot.json | jq '.classifications'
```

### View Delta Report
```bash
docker cp quantum_backend:/app/data/universe_delta.json ./
cat universe_delta.json | jq
```

### Check Emergency Status
```bash
docker cp quantum_backend:/app/data/emergency_brake_status.json ./
cat emergency_brake_status.json | jq '.triggered'
```

---

## 📊 OUTPUT FILES

### 1. Universe Health Report (`universe_health_report.json`)
**Contents:**
- Overall universe health status
- Health score (0–1)
- Performance metrics (cumulative R, winrate, drawdown)
- Cost metrics (slippage, spread)
- Universe-level alerts
- Recommended global action

**Size:** ~2.5 KB

---

### 2. Universe Control Snapshot (`universe_control_snapshot.json`)
**Contents:**
- All symbol classifications (CORE/EXPANSION/CONDITIONAL/BLACKLIST)
- Complete symbol health profiles (30+ metrics per symbol)
- Three universe profiles (SAFE/AGGRESSIVE/EXPERIMENTAL)
- Generation timestamp
- Operating mode

**Size:** ~215 KB (with full symbol health data)

---

### 3. Universe Delta Report (`universe_delta.json`)
**Contents:**
- Symbols to add
- Symbols to remove
- Symbols to keep
- Net change count
- Comparison with current universe

**Size:** ~5.6 KB

---

### 4. Emergency Brake Status (`emergency_brake_status.json`)
**Contents:**
- Triggered flag (true/false)
- Severity level
- Reason for trigger
- Recommended action
- Affected symbols
- Duration (hours)

**Size:** ~2 KB

---

## 🔗 ORCHESTRATOR INTEGRATION

The Control Center produces recommendations for OrchestratorPolicy:

### Integration Output
```json
{
  "allow_new_trades": true,
  "risk_profile": "NORMAL",
  "disallowed_symbols": ["BTCUSDT", "ETHUSDT"],
  "universe_change_required": false,
  "recommended_universe": "current",
  "emergency_override": false
}
```

### Integration Fields

#### `allow_new_trades` (bool)
- `true`: Normal operation
- `false`: Emergency brake triggered, block new entries

#### `risk_profile` (string)
- `"NORMAL"`: Standard risk
- `"REDUCED"`: Lower position sizes
- `"DEFENSIVE"`: Exit mode, no new trades

#### `disallowed_symbols` (list)
- Symbols currently in BLACKLIST
- Should be excluded from all trading

#### `universe_change_required` (bool)
- `true`: Switch universe profile recommended
- `false`: Continue with current universe

#### `recommended_universe` (string)
- `"current"`: No change
- `"SAFE"`: Switch to SAFE profile
- `"AGGRESSIVE"`: Switch to AGGRESSIVE profile

#### `emergency_override` (bool)
- `true`: Emergency condition, ignore normal logic
- `false`: Normal operation

---

## ⚠️ CRITICAL WARNINGS & GUARDRAILS

### 1. Data Requirements
- **Minimum for meaningful classifications:** 100+ trades per symbol
- **Minimum for confidence:** 1,000+ signals per symbol
- **Initial deployment:** Expect all INSUFFICIENT_DATA

### 2. Emergency Brake Safety
- **NEVER override emergency brake without investigation**
- **Always check `emergency_brake_status.json` before trading**
- **Respect `duration_hours` — wait full period**

### 3. Mode Switching
- **Test in OBSERVE mode first** (7–14 days minimum)
- **Validate all outputs before FULL_AUTONOMY**
- **Monitor closely for first 48 hours after mode switch**

### 4. Universe Profile Changes
- **NEVER switch profiles during active trades**
- **Test new profile in paper trading first**
- **Monitor for 24–48 hours after switch**

### 5. Blacklist Decisions
- **Investigate before removing from blacklist**
- **Require 3x trade count after improvements**
- **Monitor closely if re-enabled**

---

## 🚦 DEPLOYMENT WORKFLOW

### Week 1: Data Collection (Current Stage)
**Status:** ✅ COMPLETE  
**Actions:**
- ✅ Control Center OS deployed
- ✅ Initial run completed
- ⏳ Collecting trade data (target: 100+ trades)
- ⏳ Collecting signal data (target: 1,000+ signals)

**Expected Output:**
- All symbols: INSUFFICIENT_DATA
- All profiles: Empty or minimal
- Emergency brake: Not triggered

---

### Week 2: First Classification (DECISION POINT)
**Status:** ⏳ PENDING  
**Trigger:** 100+ trades, 1,000+ signals collected  
**Actions:**
1. Re-run Control Center OS
2. Review classifications (expect 20–50 CORE, 50–100 EXPANSION)
3. Analyze symbol health profiles
4. Validate emergency brake logic

**Decision:**
- ✅ If classifications look reasonable → Week 3
- ❌ If still insufficient data → Wait another 7 days

---

### Week 3: Paper Trading Validation
**Status:** ⏳ PENDING  
**Actions:**
1. Deploy AGGRESSIVE profile to paper trading
2. Monitor for 7 days:
   - Allow rate vs baseline
   - Win rate vs baseline
   - Avg R vs baseline
   - Emergency triggers
3. Compare actual performance to expected profile performance

**Decision:**
- ✅ If paper trading validates → Week 4
- ❌ If issues found → Investigate and re-test

---

### Week 4: Production Deployment
**Status:** ⏳ PENDING  
**Actions:**
1. Deploy SAFE profile to production
2. Monitor closely for 72 hours
3. Track all metrics (PnL, winrate, slippage, drawdown)
4. Keep rollback plan ready

**Success Criteria:**
- No emergency brakes triggered
- Winrate >= 45%
- Avg R >= 0.5
- Slippage < 0.5%

---

### Month 2+: Continuous Optimization
**Status:** ⏳ PENDING  
**Actions:**
- Weekly Control Center OS runs (every Monday)
- Monitor classification changes
- Track blacklist growth
- Compare profile performance
- Adjust thresholds as needed
- Consider FULL_AUTONOMY mode

---

## 🔧 AGENT CUSTOMIZATION

All classification thresholds can be adjusted via environment variables:

### Core Classification
```bash
CORE_MIN_STABILITY=0.20      # Minimum stability score
CORE_MIN_QUALITY=0.25        # Minimum quality score
CORE_MIN_WINRATE=0.45        # Minimum win rate (45%)
```

### Emergency Thresholds
```bash
EMERGENCY_SLIPPAGE_THRESHOLD=0.01   # 1%
EMERGENCY_SPREAD_THRESHOLD=0.005    # 0.5%
EMERGENCY_LOSS_STREAK=5             # 5 consecutive losses
EMERGENCY_DD_THRESHOLD=-0.15        # -15% drawdown
```

### Blacklist Criteria
```bash
BLACKLIST_MAX_DISALLOW=0.35         # 35% max disallow rate
```

### Scheduler
```bash
UNIVERSE_OS_MODE=OBSERVE            # OBSERVE | FULL_AUTONOMY
UNIVERSE_OS_INTERVAL_HOURS=4        # Run every 4 hours
```

---

## 📊 MONITORING & KPIs

### Symbol-Level KPIs
- **Health Status Distribution:** % in HEALTHY/MODERATE/CRITICAL
- **Classification Stability:** Tier changes per week
- **Toxicity Trends:** Symbols entering/leaving BLACKLIST
- **Cost Behavior:** Slippage/spread trends per symbol

### Universe-Level KPIs
- **Overall Health Score:** Target > 0.6 (HEALTHY)
- **Cumulative R:** Track growth rate
- **Rolling Winrate:** Target > 45%
- **Drawdown:** Keep above -10%
- **Emergency Frequency:** Target 0 per month

### Profile Performance KPIs
- **SAFE vs AGGRESSIVE:** Compare R, winrate, slippage
- **Profile Stability:** % of symbols staying in profile
- **Expected vs Actual:** Profile predictions vs reality

---

## 🚨 ROLLBACK PROCEDURE

### When to Rollback
1. **Emergency brake triggered repeatedly** (3+ times in 24h)
2. **Severe drawdown** (> -15%)
3. **Classification chaos** (symbols rapidly changing tiers)
4. **Performance degradation** (> 20% drop in key metrics)

### Rollback Steps
1. **Immediate:** Set `UNIVERSE_OS_MODE=OBSERVE` (stop auto-changes)
2. **Backup:** Save current `universe_control_snapshot.json`
3. **Restore:** Apply previous working universe configuration
4. **Investigate:** Analyze what went wrong
5. **Re-test:** Validate fixes in paper trading before retry

---

## 📝 BEST PRACTICES

### 1. Always Start in OBSERVE Mode
Never deploy in FULL_AUTONOMY without extensive validation.

### 2. Validate Before Acting
Review all outputs before applying recommendations.

### 3. Monitor Emergency Status Daily
Check `emergency_brake_status.json` every day.

### 4. Track Classification Changes
Monitor symbols moving between tiers weekly.

### 5. Respect Data Requirements
Don't make decisions with <100 trades per symbol.

### 6. Test Profile Changes in Paper Trading
Never deploy untested profiles to production.

### 7. Keep Historical Snapshots
Archive snapshots weekly for analysis.

### 8. Investigate Blacklist Additions
Understand why symbols become toxic.

### 9. Adjust Thresholds Gradually
Make small incremental changes to parameters.

### 10. Maintain Human Oversight
Even in FULL_AUTONOMY, review decisions regularly.

---

## 🎓 TROUBLESHOOTING

### Problem: All Symbols Classified as INSUFFICIENT_DATA
**Cause:** Not enough trade data yet  
**Solution:** Wait for 100+ trades per symbol (7–14 days)

---

### Problem: Emergency Brake Triggered Immediately
**Cause:** Thresholds too strict OR genuine emergency  
**Solution:**
1. Check actual slippage/spread values
2. Adjust thresholds if false positive
3. Investigate if real emergency

---

### Problem: Classifications Unstable (Symbols Jumping Tiers)
**Cause:** Insufficient data OR high volatility  
**Solution:**
1. Increase minimum trade count requirement
2. Use longer rolling windows
3. Add hysteresis (tier change cooldown)

---

### Problem: SAFE Profile Empty
**Cause:** No symbols meet CORE criteria  
**Solution:**
1. Check if thresholds too strict
2. Review CORE_MIN_* parameters
3. Wait for more data

---

### Problem: High Toxicity Scores Across Board
**Cause:** Market-wide issues OR threshold misconfiguration  
**Solution:**
1. Check if global market volatility spike
2. Review cost data (slippage/spread)
3. Adjust toxicity calculation if needed

---

## 📞 SUPPORT & MAINTENANCE

### Daily Tasks
- [ ] Check emergency brake status
- [ ] Monitor trade generation
- [ ] Review any new alerts

### Weekly Tasks
- [ ] Run Control Center OS
- [ ] Review classification changes
- [ ] Analyze symbol health trends
- [ ] Check blacklist additions

### Monthly Tasks
- [ ] Compare profile performance
- [ ] Adjust thresholds if needed
- [ ] Archive historical snapshots
- [ ] Review rollback procedures

---

## 🌟 WHAT MAKES THIS SPECIAL

This is **NOT a simple classifier or monitor** — it is a **FULL OPERATING SYSTEM** for universe and risk governance:

✅ **30+ metrics per symbol** (not 5–10)  
✅ **Multi-source data fusion** (trades, signals, policy, exchange)  
✅ **4-tier classification** with intelligent reasoning  
✅ **3 dynamic profiles** (SAFE/AGGRESSIVE/EXPERIMENTAL)  
✅ **Emergency brake system** with graduated responses  
✅ **Scheduler integration** (OBSERVE/FULL_AUTONOMY modes)  
✅ **Orchestrator-ready** (seamless integration)  
✅ **Complete lifecycle management** (discovery → monitoring → protection)  

---

## 📚 RELATED DOCUMENTATION

- **UNIVERSE_OS_AGENT_GUIDE.md** — Detailed Universe OS Agent guide
- **UNIVERSE_MANAGEMENT_INDEX.md** — Complete documentation index
- **AI_TRADING_ARCHITECTURE.md** — Overall system architecture
- **MONITORING_GUIDE.md** — Monitoring best practices

---

## ✅ DEPLOYMENT CHECKLIST

- [x] Control Center OS deployed to container
- [x] Initial run completed successfully
- [x] All output files generated
- [x] Documentation created
- [ ] Week 1 data collection (100+ trades)
- [ ] Week 2 re-run and validation
- [ ] Week 3 paper trading test
- [ ] Week 4 production deployment
- [ ] Month 2+ continuous optimization

---

**Version:** 3.0  
**Last Updated:** November 23, 2025  
**Status:** ✅ DEPLOYED & OPERATIONAL (OBSERVE MODE)  
**Next Milestone:** Week 2 Re-run (7–14 days)
