# UNIVERSE SELECTOR AGENT — Operational Playbook

**Quick Reference for Daily Operations**

---

## 🎯 WHAT IS THIS AGENT?

An **autonomous AI system** that:
- Analyzes 15+ performance metrics per symbol
- Classifies symbols into performance tiers
- Recommends optimal universe configurations
- Generates add/remove deltas for implementation
- **NEVER modifies code** — only provides recommendations

---

## ⚡ QUICK START

### Run Agent (Manual)
```bash
docker exec quantum_backend python /app/universe_selector_agent.py
```

### View Output
```bash
# Copy from container
docker cp quantum_backend:/app/data/universe_selector_output.json ./

# View summary
cat universe_selector_output.json | jq '.summary'

# View classifications
cat universe_selector_output.json | jq '.classifications | to_entries[] | {category: .key, count: .value.count}'

# View top 10 symbols
cat universe_selector_output.json | jq '.symbol_scores | to_entries | sort_by(.value.quality_score) | reverse | .[0:10]'
```

---

## 📊 INTERPRETING OUTPUT

### Data Confidence Levels

| Confidence | Signals | Trades | Action |
|------------|---------|--------|--------|
| **LOW** | < 1,000 | < 100 | ⏸️ WAIT — Collect more data |
| **MEDIUM** | 1,000-5,000 | 100-500 | ⚠️ REVIEW — Preliminary recommendations |
| **HIGH** | 5,000-10,000 | 500-1,000 | ✅ DEPLOY — Reliable recommendations |
| **VERY_HIGH** | 10,000+ | 1,000+ | ✅ DEPLOY — High confidence |

**Current Status:** LOW (1,161 signals, 0 trades)

### Symbol Classifications

```
CORE (Must-trade)
├─ Thresholds:
│  ├─ Win rate >= 45%
│  ├─ Avg R >= 0.5
│  ├─ Stability >= 0.20
│  ├─ Quality >= 0.25
│  └─ Disallow rate <= 25%
└─ Use: Production/Mainnet

EXPANSION (Good performers)
├─ Thresholds:
│  ├─ Win rate >= 35%
│  ├─ Avg R >= 0.3
│  ├─ Stability >= 0.10
│  ├─ Quality >= 0.15
│  └─ Disallow rate <= 40%
└─ Use: Testnet/Aggressive

CONDITIONAL (Situational)
├─ Criteria:
│  ├─ Profitable in specific regimes
│  ├─ Profitable in specific volatility
│  └─ Profitable but unstable
└─ Use: Advanced strategies only

BLACKLIST (Exclude)
├─ Criteria:
│  ├─ Total R < -0.5 AND win rate < 35%
│  ├─ OR Avg R < 0.1
│  ├─ OR Disallow rate > 50%
│  └─ OR Stability < 0.05
└─ Use: Never trade

INSUFFICIENT_DATA
└─ < 5 signals OR < 3 trades
```

---

## 🚀 DEPLOYMENT WORKFLOW

### Phase 1: Initial Baseline (Week 1)
```
1. Run agent daily to monitor data accumulation
2. Track signal count growth
3. Wait for first trades to appear
4. DO NOT deploy recommendations yet
```

### Phase 2: Preliminary Analysis (Week 2)
```
1. Run agent every 3 days
2. Check if data confidence reaches MEDIUM
3. Review classifications:
   - Are CORE symbols reasonable?
   - Are BLACKLIST symbols truly poor?
   - Are majors protected?
4. If confidence >= MEDIUM:
   → Proceed to Phase 3
   Else:
   → Continue collecting data
```

### Phase 3: Validation (Week 3)
```
1. Run agent
2. Extract recommended universe for target profile:
   - SAFE (mainnet)
   - AGGRESSIVE (testnet)
3. Compare deltas:
   - Review all symbols in "to_remove"
   - Review all symbols in "to_add"
   - Check if any majors are being removed (FLAG for review)
4. Deploy to paper trading for 7 days
5. Compare metrics:
   - Allow rate (vs baseline)
   - Win rate (vs baseline)
   - Avg R (vs baseline)
   - Signal count (vs baseline)
```

### Phase 4: Production Deployment (Week 4)
```
1. If paper trading results are positive:
   → Deploy to production
   Else:
   → Adjust thresholds or wait for more data

2. Implementation:
   a. Update QT_UNIVERSE in config
   b. Update QT_MAX_SYMBOLS
   c. Implement whitelist/blacklist in code
   d. Restart backend
   e. Monitor closely for 72 hours

3. Post-deployment:
   - Track all key metrics
   - Be ready to rollback if issues
   - Document changes with timestamps
```

---

## 🎯 PROFILE SELECTION MATRIX

| Scenario | Profile | Size | Risk | Use Case |
|----------|---------|------|------|----------|
| **Mainnet deployment** | SAFE | 150-200 | Low | Real money |
| **Testnet training** | AGGRESSIVE | 300-400 | Medium | ML training |
| **Research mode** | EXPERIMENTAL | 500-600 | High | Data collection |
| **HFT strategies** | SAFE + Filter | 50-100 | Low | Majors only |
| **Conservative** | SAFE | 100-150 | Very Low | Risk-averse |

---

## ⚠️ RED FLAGS & WARNINGS

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

## 📋 WEEKLY CHECKLIST

### Every Monday (Week 2+)
```
□ Run Universe Selector Agent
□ Copy output to host
□ Review data confidence level
□ Check classification counts
□ Review top 10 and bottom 10 symbols
□ Compare vs last week's output
□ Document any anomalies
```

### Every Month (Month 2+)
```
□ Full delta analysis
□ Test recommended universe in paper trading
□ Deploy changes if validated
□ Update documentation with changes
□ Backup previous universe configuration
```

### Every Quarter (Quarter 2+)
```
□ Full system audit
□ Threshold tuning (if needed)
□ Performance attribution analysis
□ Agent enhancement planning
```

---

## 🔧 TROUBLESHOOTING

### Issue: "All symbols in BLACKLIST"
**Cause:** No trade data available  
**Fix:** Wait for trade data (7-14 days)

### Issue: "0 CORE symbols"
**Cause:** High thresholds OR poor trading performance  
**Fix:** 
1. Check trading strategy performance
2. Review threshold settings
3. Investigate if market conditions are unusual

### Issue: "Major coin in BLACKLIST"
**Cause:** Temporary poor performance OR data anomaly  
**Fix:**
1. Review last 30 days of performance for that coin
2. Check if it's a model calibration issue
3. Do NOT auto-blacklist — manual override required

### Issue: "Data confidence stuck at LOW"
**Cause:** Insufficient signal/trade generation  
**Fix:**
1. Increase confidence threshold in trading strategy
2. Check if executor is running properly
3. Verify signal generation rate (target: 300-500/day)

---

## 📊 PERFORMANCE TRACKING

### Metrics to Monitor After Universe Change

| Metric | Baseline | New | Change | Status |
|--------|----------|-----|--------|--------|
| Universe size | 222 | ? | ? | ? |
| Allow rate | ~53% | ? | ? | ? |
| Avg confidence | 0.50 | ? | ? | ? |
| Signals/day | 460 | ? | ? | ? |
| Win rate | ? | ? | ? | ? |
| Avg R | ? | ? | ? | ? |
| Total PnL | ? | ? | ? | ? |

**Update weekly after each agent run**

---

## 🚨 ROLLBACK PROCEDURE

If universe change causes issues:

```bash
# 1. Identify last known good universe
docker exec quantum_backend cat /app/data/universe_snapshot.json > universe_backup.json

# 2. Revert docker-compose.yml or config
# Restore previous QT_SYMBOLS or QT_UNIVERSE settings

# 3. Restart backend
docker-compose restart backend

# 4. Verify rollback
docker logs quantum_backend --since 1m | grep "UNIVERSE"

# 5. Document incident
echo "Rollback performed on $(date): [reason]" >> universe_changes.log
```

---

## 📝 CHANGE LOG TEMPLATE

```
Date: 2025-XX-XX
Profile: SAFE | AGGRESSIVE | EXPERIMENTAL
Action: DEPLOY | ROLLBACK | TEST
Universe Size: XXX → YYY
Symbols Added: [LIST]
Symbols Removed: [LIST]
Reason: [DESCRIPTION]
Data Confidence: LOW | MEDIUM | HIGH | VERY_HIGH
Paper Trading Results: [SUMMARY]
Decision: APPROVED | REJECTED | DEFERRED
Approved By: [NAME]
```

---

## 🎓 BEST PRACTICES SUMMARY

1. **Wait for data** — Don't deploy with LOW confidence
2. **Validate in paper trading** — Test for 7 days before production
3. **Protect majors** — Never auto-blacklist BTC, ETH, BNB, SOL, etc.
4. **Change gradually** — Max 20% of universe per week
5. **Monitor closely** — Track all metrics for 72 hours post-deployment
6. **Document everything** — Keep change log with timestamps and rationale
7. **Keep rollback ready** — Always have previous universe backed up
8. **Review manually** — Don't blindly apply agent recommendations
9. **Run weekly** — After Week 2, run agent every Monday
10. **Tune thresholds** — Adjust classification criteria based on strategy

---

## 📞 DECISION TREE

```
Run Agent
    │
    ├─ Data Confidence < MEDIUM?
    │   └─ YES → WAIT, collect more data
    │   └─ NO → Continue
    │
    ├─ CORE count < 20?
    │   └─ YES → INVESTIGATE, may need threshold tuning
    │   └─ NO → Continue
    │
    ├─ Any major in BLACKLIST?
    │   └─ YES → MANUAL REVIEW required
    │   └─ NO → Continue
    │
    ├─ Deltas show > 30% change?
    │   └─ YES → GRADUAL deployment (multiple weeks)
    │   └─ NO → Continue
    │
    ├─ Deploy to paper trading
    │   │
    │   ├─ Results positive after 7 days?
    │   │   └─ YES → Deploy to production
    │   │   └─ NO → Reject changes, investigate
    │   │
    │   └─ Monitor production for 72 hours
    │       │
    │       ├─ Metrics stable/improved?
    │       │   └─ YES → SUCCESS
    │       │   └─ NO → ROLLBACK
```

---

## 🔗 RELATED DOCUMENTATION

- **UNIVERSE_SELECTOR_AGENT_GUIDE.md** — Complete technical guide
- **UNIVERSE_ANALYSIS_REPORT.md** — Comprehensive manual analysis
- **UNIVERSE_ANALYSIS_SUMMARY.md** — Quick reference for current universe
- **UNIVERSE_DEPLOYMENT_CONFIG.json** — Deployment configuration templates

---

## ⏱️ EXPECTED TIMELINE

```
Day 1:    Agent deployed, baseline established
Day 7:    ~3,200 signals, 100-300 trades (MEDIUM confidence possible)
Day 14:   ~6,400 signals, 300-600 trades (HIGH confidence likely)
Day 21:   Paper trading with recommended universe
Day 28:   Production deployment (if validated)
Day 35+:  Weekly optimization cycles
```

---

**END OF PLAYBOOK**

*Keep this document handy for daily operations*

*Run agent: `docker exec quantum_backend python /app/universe_selector_agent.py`*

*Next action: Wait for Week 2 data milestone, then re-run agent*
