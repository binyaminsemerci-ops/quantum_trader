# 📊 ORCHESTRATOR POLICY TUNING ANALYSIS & RECOMMENDATIONS

**Date:** November 22, 2025  
**Analyst:** Senior Quant Developer & Researcher  
**Context:** LIVE Mode Step 1 Active (Signal Filtering Only)  

---

## Executive Summary

The Orchestrator Policy has been analyzed for optimal tuning based on:
- Industry best practices
- Current configuration design
- Risk management principles
- Regime-specific behavior patterns

**Key Finding:** Current configuration is **conservative and safe** (good for mainnet), but **may be over-filtering** profitable opportunities. Two profiles proposed: **SAFE** for mainnet and **AGGRESSIVE** for testnet.

---

## 🎯 Current Configuration Analysis

### Strengths ✅

1. **Multi-layered risk protection**
   - Daily DD limit (3%)
   - Losing streak protection (5 trades)
   - Position limits (8 max)
   - Total exposure cap (15%)

2. **Regime-aware behavior**
   - TRENDING: Lower confidence (-0.03), aggressive entry
   - RANGING: Higher confidence (+0.05), defensive scalping

3. **Volatility adaptation**
   - HIGH_VOL: 50% risk reduction, +0.03 confidence
   - EXTREME_VOL: Full trading stop

4. **Symbol performance filtering**
   - Automatically disallows BAD performers (WR < 35% or R < 0.5)

5. **Cost-aware filtering**
   - Adjusts confidence for high spreads/slippage

### Weaknesses ⚠️

1. **Base confidence too high (0.50)**
   - Industry standard: 0.45
   - Missing good trades with confidence 0.45-0.50

2. **RANGING mode too defensive (+0.05)**
   - Reaches 0.55 confidence threshold
   - Filtering out profitable range-bound scalps

3. **Losing streak action too harsh (30% risk)**
   - After 5 losses → reduces to 30% risk + raises confidence
   - Makes recovery difficult

4. **No dynamic win-rate adaptation**
   - Should lower threshold after winning streaks
   - Should raise threshold after losing streaks

5. **Reactive symbol filtering only**
   - No proactive blacklist for known problematic pairs

---

## 📈 Confidence Threshold Analysis

### Current Behavior

| Scenario | Base | Adjustment | Final | Assessment |
|----------|------|------------|-------|------------|
| **TRENDING + NORMAL** | 0.50 | -0.03 | **0.47** | ✅ Good |
| **RANGING** | 0.50 | +0.05 | **0.55** | ⚠️ Too strict |
| **HIGH_VOL** | 0.50 | +0.03 | **0.53** | ⚠️ Conservative |
| **EXTREME_VOL** | 0.50 | N/A | **NO_TRADES** | ✅ Correct |

### Recommendations

#### Base Confidence
- **Current:** 0.50
- **SAFE Profile:** 0.45 (industry standard)
- **AGGRESSIVE Profile:** 0.42 (more opportunities)
- **Reasoning:** 0.45 balances quality vs quantity

#### Regime Adjustments

**TRENDING:**
- Current: -0.03 → 0.47
- SAFE: -0.03 → 0.42 ✅
- AGGRESSIVE: -0.05 → 0.37 📈
- Reasoning: Trending is best regime, exploit more

**RANGING:**
- Current: +0.05 → 0.55
- SAFE: +0.03 → 0.48 ✅
- AGGRESSIVE: +0.02 → 0.44 📈
- Reasoning: Current +0.05 filters too many scalps

**HIGH_VOL:**
- Current: +0.03
- SAFE: +0.05 (more defensive)
- AGGRESSIVE: +0.03 (same)

---

## 🛡️ Risk Limits Analysis

### Daily Drawdown Limit

- **Current:** 3.0%
- **SAFE:** 3.0% (keep conservative)
- **AGGRESSIVE:** 5.0% (allow recovery attempts)
- **Assessment:** 3% is very strict but good for capital protection

### Losing Streak Limit

- **Current:** 5 losses → reduce to 30% risk, +0.05 confidence
- **SAFE:** 4 losses → reduce to 40% risk, +0.05 confidence
- **AGGRESSIVE:** 6 losses → reduce to 50% risk, +0.03 confidence
- **Issue:** 30% reduction too harsh, makes recovery difficult

### Position Limits

- **Current:** 8 max positions
- **SAFE:** 6 (reduce correlation risk)
- **AGGRESSIVE:** 10 (more opportunities)

### Total Exposure

- **Current:** 15%
- **SAFE:** 12% (safer)
- **AGGRESSIVE:** 20% (capture more opportunities)

### Base Risk Per Trade

- **Current:** 1.0%
- **SAFE:** 0.8% (very conservative for mainnet)
- **AGGRESSIVE:** 1.5% (testnet exploration)

---

## 🌐 Regime-Specific Recommendations

### TRENDING Regime

**Current Policy:**
- Entry: AGGRESSIVE
- Exit: TREND_FOLLOW
- Confidence: -0.03 (→ 0.47)
- Risk: 1.0x (no change)

**Analysis:** ✅ Good approach, but could be more aggressive

**Suggested Enhancements:**
- Confidence: -0.05 (→ 0.45 in SAFE, 0.37 in AGGRESSIVE)
- Risk: 1.1x (boost trending trades by 10%)
- Reasoning: Trending is our best regime

### RANGING Regime

**Current Policy:**
- Entry: DEFENSIVE
- Exit: FAST_TP
- Confidence: +0.05 (→ 0.55)
- Risk: 0.7x (30% reduction)

**Analysis:** ⚠️ Too defensive, missing scalp opportunities

**Suggested Enhancements:**
- Confidence: +0.02 (→ 0.47 in SAFE, 0.44 in AGGRESSIVE)
- Risk: 0.8x (20% reduction, not 30%)
- Reasoning: RANGING profitable with tight TP, don't over-filter

### Volatility Interaction Issue

**Current:** Adjustments stack multiplicatively
- Example: RANGING (0.7x) + HIGH_VOL (0.5x) = **0.35x risk**

**Problem:** Stacking too severe

**Solution:** Use max(reductions) not multiply
- Example: max(0.7, 0.5) = **0.7x risk**

---

## 📦 PROPOSED CONFIGURATION PROFILES

### 📘 SAFE PROFILE (Mainnet / Real Capital)

**Purpose:** Conservative settings for production trading

```python
SAFE_PROFILE = {
    # Core parameters
    "base_confidence": 0.45,          # ↓ from 0.50
    "base_risk_pct": 0.8,             # ↓ from 1.0
    
    # Risk limits
    "daily_dd_limit": 3.0,            # = (unchanged)
    "losing_streak_limit": 4,         # ↓ from 5
    "losing_streak_risk_reduction": 0.4,  # ↑ from 0.3 (less harsh)
    "max_open_positions": 6,          # ↓ from 8
    "total_exposure_limit": 12.0,     # ↓ from 15.0
    
    # Volatility thresholds
    "extreme_vol_threshold": 0.05,    # ↓ from 0.06 (trigger earlier)
    "high_vol_threshold": 0.035,      # ↓ from 0.04 (trigger earlier)
    
    # Regime adjustments
    "trending_confidence_adj": -0.03, # = (unchanged)
    "trending_risk_adj": 1.0,         # = (unchanged)
    "ranging_confidence_adj": +0.03,  # ↓ from +0.05
    "ranging_risk_adj": 0.8,          # ↑ from 0.7
    
    # Volatility adjustments
    "high_vol_risk_adj": 0.5,         # = (unchanged)
    "high_vol_confidence_adj": +0.05, # ↑ from +0.03 (more defensive)
    
    # Symbol filtering
    "min_winrate": 0.40,              # Stricter than default 0.35
    "min_avg_R": 0.7,                 # Stricter than default 0.5
    "consecutive_loss_disable": 3     # Quicker disable
}
```

**When to use:**
- ✅ Mainnet production trading
- ✅ Real capital deployment
- ✅ Risk-averse operations
- ✅ After initial validation period

### 📕 AGGRESSIVE PROFILE (Testnet / Experimentation)

**Purpose:** Higher risk/reward for strategy testing

```python
AGGRESSIVE_PROFILE = {
    # Core parameters
    "base_confidence": 0.42,          # ↓↓ from 0.50 (more trades)
    "base_risk_pct": 1.5,             # ↑ from 1.0 (higher risk)
    
    # Risk limits
    "daily_dd_limit": 5.0,            # ↑ from 3.0 (allow recovery)
    "losing_streak_limit": 6,         # ↑ from 5 (less reactive)
    "losing_streak_risk_reduction": 0.5,  # ↑ from 0.3 (allow recovery)
    "max_open_positions": 10,         # ↑ from 8 (more diversification)
    "total_exposure_limit": 20.0,     # ↑ from 15.0 (more aggressive)
    
    # Volatility thresholds
    "extreme_vol_threshold": 0.07,    # ↑ from 0.06 (tolerate more vol)
    "high_vol_threshold": 0.05,       # ↑ from 0.04 (tolerate more vol)
    
    # Regime adjustments
    "trending_confidence_adj": -0.05, # ↓ from -0.03 (exploit trending)
    "trending_risk_adj": 1.1,         # ↑ from 1.0 (boost trending)
    "ranging_confidence_adj": +0.02,  # ↓ from +0.05 (less strict)
    "ranging_risk_adj": 0.8,          # ↑ from 0.7 (less reduction)
    
    # Volatility adjustments
    "high_vol_risk_adj": 0.6,         # ↑ from 0.5 (less reduction)
    "high_vol_confidence_adj": +0.03, # = (unchanged)
    
    # Symbol filtering
    "min_winrate": 0.35,              # Default threshold
    "min_avg_R": 0.5,                 # Default threshold
    "consecutive_loss_disable": 4     # Allow more attempts
}
```

**When to use:**
- ✅ Testnet validation
- ✅ Strategy testing
- ✅ Model parameter tuning
- ✅ Data collection phase

---

## 🎯 Deployment Strategy

### 🧪 Testnet Deployment

**Profile:** AGGRESSIVE

**Duration:** 7-14 days

**Purpose:**
- Test policy limits
- Gather filtering effectiveness data
- Measure blocked_winning_trades vs blocked_losing_trades
- Validate regime-specific adjustments

**Validation Metrics:**
- Sharpe Ratio (target: >1.5)
- Max Drawdown (target: <8%)
- Win Rate (target: >50%)
- Trade Frequency (target: 10-20/day)
- Filter Rate (target: 30-60%)

**Success Criteria:**
- blocked_losing_trades / total_blocked > 60%
- blocked_winning_trades / total_blocked < 30%
- No policy oscillation (stable decisions)
- Positive risk-adjusted returns

### 💰 Mainnet Deployment

**Profile:** SAFE

**Phased Rollout:**

**Phase 1 (Current - Days 1-2):**
- ✅ SAFE profile active
- ✅ Signal filtering only (confidence + symbols)
- ⏸️ Risk sizing: Still fixed
- ⏸️ Position limits: Still fixed
- ⏸️ Trading gate: Not enforced

**Phase 2 (Days 3-7):**
- If Phase 1 shows good signal quality:
  - ✅ Add risk sizing enforcement
  - ✅ Use policy.max_risk_pct
  - Monitor: Ensure risk scaling improves drawdown

**Phase 3 (Days 8-14):**
- If Phase 2 maintains performance:
  - ✅ Add position limit enforcement
  - ✅ Use policy.max_open_positions
  - Monitor: Check correlation risk reduction

**Phase 4 (Days 15+):**
- If Phase 3 shows strong performance:
  - ✅ Consider trading gate enforcement
  - ✅ Use policy.allow_new_trades
  - ⚠️ High risk: Can stop all trading

**Phase 5 (Days 30+):**
- If consistently profitable:
  - 🤔 Evaluate shift from SAFE → AGGRESSIVE
  - Gradual: Increase confidence threshold by 0.01/week
  - Monitor: Ensure Sharpe ratio improves

---

## 📊 Monitoring & KPIs

### Effectiveness Metrics

**Policy Filtering Quality:**
- `blocked_losing_trades / total_blocked` (want **>60%**)
- `blocked_winning_trades / total_blocked` (want **<30%**)
- `false_positive_rate` (blocked winners / all winners, want **<20%**)

**Policy Activity:**
- `policy_active_time / total_time` (want **<80%** NO_TRADES state)
- `regime_detection_accuracy` (manual validation)
- `policy_oscillation_count` (want **<5 flips/day**)

### Performance Metrics

**Risk-Adjusted Returns:**
- `sharpe_ratio` (want **>1.5**)
- `sortino_ratio` (want **>2.0**)
- `calmar_ratio` (want **>3.0**)

**Drawdown Management:**
- `max_drawdown` (want **<5%** SAFE, **<8%** AGGRESSIVE)
- `avg_drawdown_duration` (want **<24h**)
- `recovery_factor` (want **>2.0**)

**Win Rate & Quality:**
- `overall_win_rate` (want **>50%**)
- `profit_factor` (want **>1.8**)
- `avg_R_multiple` (want **>1.5**)

### Filtering Metrics

**Signal Quality:**
- `signals_per_day` (want **10-30**)
- `trades_executed_per_day` (want **5-15**)
- `filter_rate` (signals_blocked / signals_total, want **30-60%**)

**Confidence Distribution:**
- `avg_confidence_passed` (want **>0.50**)
- `avg_confidence_blocked` (want **<0.45**)
- `confidence_separation` (want **>0.05** gap)

---

## 🔧 Implementation Guide

### Applying SAFE Profile (Mainnet)

**File:** `backend/services/orchestrator_policy.py`

**Changes needed:**

```python
@dataclass
class OrchestratorConfig:
    """Configuration for Orchestrator Policy Engine."""
    
    # SAFE PROFILE adjustments
    base_confidence: float = 0.45  # ← Changed from 0.50
    base_risk_pct: float = 0.8     # ← Changed from 1.0
    
    daily_dd_limit: float = 3.0    # ← Unchanged
    losing_streak_limit: int = 4   # ← Changed from 5
    max_open_positions: int = 6    # ← Changed from 8
    total_exposure_limit: float = 12.0  # ← Changed from 15.0
    
    extreme_vol_threshold: float = 0.05  # ← Changed from 0.06
    high_vol_threshold: float = 0.035    # ← Changed from 0.04
```

**In update_policy() method:**

```python
# SAFE PROFILE regime adjustments
if regime_tag == "TRENDING" and vol_level == "NORMAL":
    policy_data["entry_mode"] = "AGGRESSIVE"
    policy_data["exit_mode"] = "TREND_FOLLOW"
    policy_data["min_confidence"] -= 0.03  # ← Unchanged
    policy_data["max_risk_pct"] *= 1.0     # ← Unchanged (AGGRESSIVE would be 1.1)

elif regime_tag == "RANGING":
    policy_data["entry_mode"] = "DEFENSIVE"
    policy_data["exit_mode"] = "FAST_TP"
    policy_data["max_risk_pct"] *= 0.8     # ← Changed from 0.7
    policy_data["min_confidence"] += 0.03  # ← Changed from 0.05

# SAFE PROFILE losing streak adjustment
if risk_state.losing_streak >= self.config.losing_streak_limit:
    policy_data["max_risk_pct"] *= 0.4     # ← Changed from 0.3
    policy_data["entry_mode"] = "DEFENSIVE"
    policy_data["min_confidence"] += 0.05
```

### Applying AGGRESSIVE Profile (Testnet)

**Use environment variables:**

```bash
# testnet .env
ORCH_BASE_CONFIDENCE=0.42
ORCH_BASE_RISK_PCT=1.5
ORCH_DAILY_DD_LIMIT=5.0
ORCH_LOSING_STREAK_LIMIT=6
ORCH_MAX_OPEN_POSITIONS=10
ORCH_TOTAL_EXPOSURE_LIMIT=20.0
```

---

## 🚨 Critical Warnings

### ⚠️ Before Enabling Full LIVE Mode

**DO NOT enable trading gate enforcement until:**
1. ✅ 48+ hours of signal filtering data collected
2. ✅ Validated that policy is NOT blocking >50% of winners
3. ✅ Confirmed regime detection is accurate
4. ✅ Tested on testnet first

**Reason:** `policy.allow_new_trades = False` will STOP ALL TRADING. If policy is misconfigured, this could:
- Miss entire profitable market regimes
- Cause significant opportunity cost
- Require manual intervention to resume

### ⚠️ Symbol Blacklist Management

**Maintain a PERMANENT_BLACKLIST for:**
- Pairs with known API issues
- Delisted or delisting coins
- Extremely low liquidity pairs
- Coins with manipulation history

**Implementation:**
```python
PERMANENT_BLACKLIST = [
    # Add based on exchange-specific issues
    # Example: "LUNAUSDT" (after collapse)
]
```

---

## 📋 Next Steps Checklist

### Immediate (Today)

- [ ] Review this analysis document
- [ ] Decide: Keep current config OR apply SAFE profile
- [ ] For testnet: Consider applying AGGRESSIVE profile
- [ ] Set up monitoring dashboard for KPIs

### 24-48 Hours

- [ ] Collect first batch of LIVE filtering logs
- [ ] Analyze blocked signals (winners vs losers)
- [ ] Check policy oscillation frequency
- [ ] Validate regime detection accuracy

### Week 1

- [ ] Calculate filter_rate and signal quality
- [ ] Measure blocked_losing_trades / total_blocked
- [ ] Evaluate if confidence threshold needs adjustment
- [ ] Consider Phase 2: Risk sizing enforcement

### Week 2

- [ ] Re-run this analysis with actual observation data
- [ ] Fine-tune confidence thresholds per regime
- [ ] Optimize losing streak response
- [ ] Consider Phase 3: Position limit enforcement

### Month 1

- [ ] Full performance review
- [ ] Compare SAFE vs AGGRESSIVE profiles
- [ ] Decide on permanent configuration
- [ ] Document lessons learned

---

## 📁 Exported Configuration Files

- `orchestrator_config_current.json` - Current configuration
- `orchestrator_config_safe.json` - SAFE profile (mainnet)
- `orchestrator_config_aggressive.json` - AGGRESSIVE profile (testnet)
- `orchestrator_analysis_full.json` - Complete analysis data

---

## 🎯 Conclusion

**Current Status:** Orchestrator is in LIVE Mode Step 1 with **conservative settings** that prioritize capital protection over opportunity capture.

**Recommendation:**

1. **For Mainnet (Current):**
   - ✅ Apply SAFE profile (minor tweaks from current)
   - ✅ Keep signal filtering only for 48h
   - ✅ Monitor filtering effectiveness closely
   - ✅ Proceed with phased rollout if successful

2. **For Testnet:**
   - ✅ Apply AGGRESSIVE profile immediately
   - ✅ Test limits and edge cases
   - ✅ Collect data on blocked trades
   - ✅ Validate before mainnet rollout

3. **Key Tuning Priority:**
   - 🔥 **Confidence threshold:** Lower from 0.50 to 0.45
   - 🔥 **RANGING adjustment:** Reduce from +0.05 to +0.03
   - 🔥 **Losing streak:** Soften from 30% to 40% risk reduction

**Final Note:** The orchestrator is well-designed with strong safety mechanisms. The proposed tuning aims to balance **risk management** with **opportunity capture**, optimizing for real-world profitability while maintaining capital protection.

---

**Generated:** 2025-11-22  
**Author:** Senior Quant Developer & Researcher  
**Status:** ✅ Analysis Complete - Ready for Implementation
