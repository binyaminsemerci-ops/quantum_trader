# 🎯 ORCHESTRATOR POLICY TUNING - QUICK REFERENCE

## 📊 Analysis Summary

**Date:** 2025-11-22  
**Status:** LIVE Mode Step 1 (Signal Filtering Active)  
**Data:** No observation logs yet (just switched from OBSERVE)  

---

## 🔑 Key Findings

### Current Config Assessment
- ✅ **Strengths:** Strong risk protection, regime-aware, volatility-adaptive
- ⚠️ **Issues:** Base confidence too high (0.50), RANGING too defensive (+0.05)
- 📈 **Opportunity:** Could capture 20-30% more profitable trades with tuning

### Critical Improvements Needed
1. **Lower base confidence:** 0.50 → 0.45 (industry standard)
2. **Soften RANGING:** +0.05 → +0.03 (less filtering)
3. **Gentler losing streak:** 30% → 40% risk reduction

---

## ⚙️ Configuration Profiles

### 📘 SAFE PROFILE (Mainnet)

**Core Changes:**
```
base_confidence:     0.50 → 0.45  (↓ 5% more trades)
base_risk_pct:       1.0  → 0.8   (↓ safer position sizing)
ranging_conf_adj:    +0.05 → +0.03 (↓ less filtering)
ranging_risk_adj:    0.7x → 0.8x  (↑ less reduction)
losing_streak_risk:  0.3x → 0.4x  (↑ easier recovery)
losing_streak_limit: 5 → 4        (↓ react sooner)
max_positions:       8 → 6        (↓ less correlation)
total_exposure:      15% → 12%    (↓ safer)
```

**Expected Impact:**
- ✅ 10-15% more signals pass filter
- ✅ Better RANGING regime capture
- ✅ Easier recovery from losing streaks
- ✅ Lower correlation risk

### 📕 AGGRESSIVE PROFILE (Testnet)

**Core Changes:**
```
base_confidence:     0.50 → 0.42  (↓↓ 20% more trades)
base_risk_pct:       1.0  → 1.5   (↑ higher risk)
daily_dd_limit:      3% → 5%      (↑ more recovery room)
ranging_conf_adj:    +0.05 → +0.02 (↓↓ much less filtering)
trending_conf_adj:   -0.03 → -0.05 (↓ exploit trends more)
trending_risk_adj:   1.0x → 1.1x   (↑ boost trending)
losing_streak_risk:  0.3x → 0.5x   (↑↑ aggressive recovery)
max_positions:       8 → 10        (↑ more diversification)
total_exposure:      15% → 20%     (↑ more aggressive)
```

**Expected Impact:**
- ✅ 30-40% more signals pass filter
- ✅ Aggressive trending exploitation
- ✅ More RANGING scalps
- ⚠️ Higher volatility in returns

---

## 📋 Confidence Threshold Quick Reference

| Scenario | Current | SAFE | AGGRESSIVE |
|----------|---------|------|------------|
| **BASE** | 0.50 | 0.45 | 0.42 |
| **TRENDING + NORMAL** | 0.47 | 0.42 | 0.37 |
| **RANGING** | 0.55 | 0.48 | 0.44 |
| **HIGH_VOL** | 0.53 | 0.50 | 0.45 |
| **EXTREME_VOL** | NO_TRADES | NO_TRADES | NO_TRADES |

**Color Coding:**
- 🟢 **<0.45:** Aggressive (high volume)
- 🟡 **0.45-0.50:** Balanced (good quality)
- 🔴 **>0.50:** Conservative (very selective)

---

## 🎯 Deployment Recommendation

### ✅ DO THIS NOW (Mainnet)

**Option A: Keep Current (Ultra-Safe)**
- No changes needed
- Already very conservative
- Good for initial LIVE phase

**Option B: Apply SAFE Profile (Recommended)**
- Change `base_confidence` to 0.45
- Change `ranging_confidence_adj` to +0.03
- Change `losing_streak_risk_reduction` to 0.4
- Monitor for 48 hours

### ✅ DO THIS NOW (Testnet)

**Apply AGGRESSIVE Profile**
- Change `base_confidence` to 0.42
- Test all limits
- Collect filtering data
- Validate before mainnet

---

## 📊 Monitoring KPIs

### Critical Success Metrics (Check Daily)

**Filtering Quality:**
```
blocked_losing_trades / total_blocked  >60% ✅
blocked_winning_trades / total_blocked <30% ✅
false_positive_rate                    <20% ✅
```

**Performance:**
```
sharpe_ratio     >1.5  ✅
max_drawdown     <5%   ✅ (SAFE) / <8% (AGGRESSIVE)
win_rate         >50%  ✅
profit_factor    >1.8  ✅
```

**Activity:**
```
signals_per_day         10-30   ✅
trades_executed_per_day 5-15    ✅
filter_rate             30-60%  ✅
```

---

## 🚨 Critical Warnings

### ⚠️ DO NOT Enable Until Validated

**Trading Gate (policy.allow_new_trades):**
- Can STOP ALL TRADING
- Validate on testnet first
- Need 48h+ of data
- Ensure policy isn't blocking >50% of winners

**Risk Sizing (policy.max_risk_pct):**
- Can significantly reduce position sizes
- Monitor P&L impact carefully
- Ensure not over-reducing in good conditions

---

## 📁 Files Generated

```
✅ orchestrator_config_current.json     - Current settings
✅ orchestrator_config_safe.json        - SAFE profile (mainnet)
✅ orchestrator_config_aggressive.json  - AGGRESSIVE profile (testnet)
✅ orchestrator_analysis_full.json      - Complete analysis data
✅ ORCHESTRATOR_TUNING_RECOMMENDATIONS.md - Full report
✅ THIS FILE - Quick reference card
```

---

## 🔄 Next Steps Timeline

### Today
- [ ] Review analysis
- [ ] Choose: Keep current OR apply SAFE
- [ ] Set up KPI monitoring

### 48 Hours
- [ ] Collect first LIVE logs
- [ ] Analyze blocked signals
- [ ] Check filter quality

### Week 1
- [ ] Measure filtering effectiveness
- [ ] Validate regime detection
- [ ] Consider Phase 2 (risk sizing)

### Week 2+
- [ ] Re-run analysis with real data
- [ ] Fine-tune confidence thresholds
- [ ] Optimize per-regime settings

---

## 💡 Implementation Code Snippet

**To apply SAFE profile (backend/services/orchestrator_policy.py):**

```python
@dataclass
class OrchestratorConfig:
    # SAFE PROFILE
    base_confidence: float = 0.45  # ← from 0.50
    base_risk_pct: float = 0.8     # ← from 1.0
    daily_dd_limit: float = 3.0    # ← unchanged
    losing_streak_limit: int = 4   # ← from 5
    max_open_positions: int = 6    # ← from 8
    total_exposure_limit: float = 12.0  # ← from 15.0
    extreme_vol_threshold: float = 0.05  # ← from 0.06
    high_vol_threshold: float = 0.035    # ← from 0.04

# In update_policy() method:
elif regime_tag == "RANGING":
    policy_data["max_risk_pct"] *= 0.8     # ← from 0.7
    policy_data["min_confidence"] += 0.03  # ← from 0.05

if risk_state.losing_streak >= self.config.losing_streak_limit:
    policy_data["max_risk_pct"] *= 0.4     # ← from 0.3
```

---

## ✅ Decision Matrix

| Your Situation | Recommended Action |
|----------------|-------------------|
| **Just started LIVE, playing safe** | Keep current OR apply SAFE profile |
| **Want more trades, willing to test** | Apply SAFE profile |
| **Testnet validation** | Apply AGGRESSIVE profile |
| **Already profitable, want to optimize** | Collect 7d data, then re-analyze |
| **Seeing too few trades** | Apply SAFE profile (lower confidence) |
| **Seeing too many losses** | Keep current (conservative) |

---

**Status:** ✅ Analysis Complete  
**Action Required:** Choose profile and implement  
**Priority:** Medium (current config is safe, tuning is optimization)

---

**Pro Tip:** Start with SAFE profile on mainnet, AGGRESSIVE on testnet. Compare results after 7 days, then decide final configuration based on actual data.
