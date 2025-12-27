# SHADOW MODELS: TECHNICAL FRAMEWORK

**Module 5: Shadow Models - Section 2**

## Overview

Shadow models enable **zero-risk parallel testing** of challenger models against a production champion through statistical hypothesis testing, risk-adjusted performance comparison, and automatic promotion based on rigorous criteria.

---

## 1. CORE CONCEPTS

### 1.1 Model Roles

**Champion Model:**
- **Definition:** Current best model in production
- **Allocation:** 100% of live trades
- **Status:** Proven performer with historical track record
- **Promotion:** Previous challenger that passed statistical tests

**Challenger Model:**
- **Definition:** Candidate model being evaluated
- **Allocation:** 0% (shadow mode) or ε% (exploratory allocation)
- **Status:** Unproven, under evaluation
- **Promotion:** Must statistically outperform champion to replace it

**Archive Models:**
- **Definition:** Previously promoted models kept for rollback
- **Allocation:** 0%
- **Status:** Dormant, available for emergency restore
- **Retention:** Last 3 champions kept for 90 days

### 1.2 Shadow Testing Modes

**Mode 1: Pure Shadow (0% Allocation)**
- Challenger receives market data, generates predictions
- Predictions recorded but **never executed**
- Zero risk, zero capital impact
- Used for initial screening (0-500 trades)

**Mode 2: Exploratory Allocation (ε% = 1-5%)**
- Challenger executes small percentage of trades
- Used for final validation with real execution
- Minimal risk (max $500 capital exposure)
- Used when statistical tests nearly conclusive (450-500 trades)

**Mode 3: Gradual Promotion (Ramp-Up)**
- After promotion, gradually increase allocation: 10% → 50% → 100%
- Monitor for instability during ramp
- Abort ramp if performance degrades
- Full promotion after 100 stable trades

---

## 2. STATISTICAL TESTING FRAMEWORK

### 2.1 Hypothesis Testing

**Null Hypothesis (H₀):**
> The challenger model is NOT better than the champion model. Any observed difference is due to random chance.

**Alternative Hypothesis (H₁):**
> The challenger model IS better than the champion model. The observed difference is statistically significant.

**Significance Level:** α = 0.05 (5% false positive rate)

**Required Confidence:** 95% (p-value < 0.05)

### 2.2 Test 1: Two-Sample T-Test (Mean PnL Comparison)

**Objective:** Compare mean PnL per trade between champion and challenger.

**Formulation:**
```
t = (μ_challenger - μ_champion) / SE_diff

where:
  μ_champion = mean PnL per trade (champion)
  μ_challenger = mean PnL per trade (challenger)
  
  SE_diff = sqrt((σ²_champion / n_champion) + (σ²_challenger / n_challenger))
  
  σ²_champion = variance of champion PnL
  σ²_challenger = variance of challenger PnL
  
  n_champion = number of trades (champion)
  n_challenger = number of trades (challenger)
```

**Decision Rule:**
- **p-value < 0.05:** Reject H₀ → Challenger significantly better ✅
- **p-value ≥ 0.05:** Fail to reject H₀ → Insufficient evidence ❌

**Example:**
- Champion: mean=$50, std=$120, n=500
- Challenger: mean=$60, std=$125, n=500

```python
from scipy import stats

t_stat, p_value = stats.ttest_ind(
    challenger_pnls,  # 500 trades
    champion_pnls,    # 500 trades
    equal_var=False   # Welch's t-test (unequal variances)
)

# Result: t=2.89, p=0.003 → Reject H₀ (challenger better)
```

**Limitations:**
- Assumes PnL approximately normally distributed (usually OK with n≥500)
- Sensitive to outliers (one $5,000 trade can skew results)
- Doesn't account for risk (only mean, not volatility)

### 2.3 Test 2: Bootstrap Confidence Interval (Non-Parametric)

**Objective:** Estimate 95% confidence interval for PnL difference without distributional assumptions.

**Algorithm:**
1. Compute observed difference: Δ_obs = mean(challenger) - mean(champion)
2. For i=1 to 10,000 (bootstrap iterations):
   - Resample challenger PnLs with replacement (500 samples)
   - Resample champion PnLs with replacement (500 samples)
   - Compute difference: Δ_i = mean(resample_challenger) - mean(resample_champion)
3. Sort all Δ_i values
4. 95% CI: [Δ_2.5th_percentile, Δ_97.5th_percentile]

**Decision Rule:**
- **If 0 ∉ CI:** Reject H₀ → Difference statistically significant ✅
- **If 0 ∈ CI:** Fail to reject H₀ → Difference not significant ❌

**Example:**
- Champion: mean=$50, std=$120, n=500
- Challenger: mean=$60, std=$125, n=500
- Bootstrap 95% CI: [$8.20, $11.85]
- **0 is NOT in [$8.20, $11.85] → Reject H₀ ✅**

**Advantages:**
- No distributional assumptions
- Robust to outliers
- Provides intuitive interval estimate

**Python Implementation:**
```python
def bootstrap_ci(champion_pnls, challenger_pnls, n_bootstrap=10000, ci=0.95):
    """Compute bootstrap confidence interval for mean difference"""
    diffs = []
    
    n_champion = len(champion_pnls)
    n_challenger = len(challenger_pnls)
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        sample_champion = np.random.choice(champion_pnls, size=n_champion, replace=True)
        sample_challenger = np.random.choice(challenger_pnls, size=n_challenger, replace=True)
        
        # Compute difference
        diff = sample_challenger.mean() - sample_champion.mean()
        diffs.append(diff)
    
    # Compute percentiles
    alpha = 1 - ci
    lower = np.percentile(diffs, 100 * alpha / 2)
    upper = np.percentile(diffs, 100 * (1 - alpha / 2))
    
    return lower, upper, diffs
```

### 2.4 Test 3: Win Rate Comparison (Binomial Test)

**Objective:** Compare win rates (proportion of profitable trades).

**Formulation:**
```
Champion: k_champion wins out of n_champion trades → WR_champion = k_champion / n_champion
Challenger: k_challenger wins out of n_challenger trades → WR_challenger = k_challenger / n_challenger

Null hypothesis: WR_champion = WR_challenger
Alternative: WR_challenger > WR_champion
```

**Z-Test for Proportions:**
```
z = (WR_challenger - WR_champion) / SE_diff

where:
  SE_diff = sqrt(WR_pooled * (1 - WR_pooled) * (1/n_champion + 1/n_challenger))
  
  WR_pooled = (k_champion + k_challenger) / (n_champion + n_challenger)
```

**Decision Rule:**
- **p-value < 0.05:** Reject H₀ → Challenger WR significantly higher ✅
- **p-value ≥ 0.05:** Fail to reject H₀ → WR difference not significant ❌

**Example:**
- Champion: 280 wins / 500 trades = 56.0% WR
- Challenger: 290 wins / 500 trades = 58.0% WR
- Difference: +2.0pp

```python
from statsmodels.stats.proportion import proportions_ztest

count = np.array([290, 280])  # wins
nobs = np.array([500, 500])   # trades

z_stat, p_value = proportions_ztest(count, nobs, alternative='larger')

# Result: z=1.42, p=0.078 → Fail to reject H₀ (not significant at α=0.05)
# Note: 2pp difference with n=500 is marginally significant
```

**Minimum Detectable Effect (Power Analysis):**
```python
from statsmodels.stats.power import zt_ind_solve_power

# Calculate required sample size for 80% power to detect 2pp WR difference
n_required = zt_ind_solve_power(
    effect_size=0.02 / sqrt(0.56 * 0.44),  # Cohen's h
    alpha=0.05,
    power=0.8,
    alternative='larger'
)

# Result: n ≈ 1,200 trades needed for 80% power
```

**Implication:** For 2pp WR difference, need 1,200 trades for 80% confidence. With 500 trades, only ~50% power.

### 2.5 Test 4: Sharpe Ratio Comparison

**Objective:** Compare risk-adjusted returns (reward per unit of risk).

**Sharpe Ratio Definition:**
```
Sharpe = (μ_PnL - r_f) / σ_PnL

where:
  μ_PnL = mean PnL per trade
  σ_PnL = standard deviation of PnL
  r_f = risk-free rate (≈0 for intraday trading)
```

**Sharpe Ratio Difference Test (Jobson-Korkie Test):**
```
z = (Sharpe_challenger - Sharpe_champion) / SE_diff

where:
  SE_diff = sqrt((1 / n) * (2 - ρ²))
  
  ρ = correlation between champion and challenger PnLs (usually 0 for shadow testing)
  n = number of trades
```

**Decision Rule:**
- **z > 1.96 (p < 0.05):** Challenger Sharpe significantly higher ✅
- **z ≤ 1.96:** Sharpe difference not significant ❌

**Example:**
- Champion: Sharpe = 1.8
- Challenger: Sharpe = 2.1
- Difference: +0.3 (+17%)
- n = 500, ρ ≈ 0

```python
z = (2.1 - 1.8) / np.sqrt((1/500) * 2)
# z = 0.3 / 0.0632 = 4.75 → p < 0.0001 (highly significant)
```

### 2.6 Test 5: Maximum Drawdown Comparison

**Objective:** Compare worst-case loss (tail risk).

**Maximum Drawdown Definition:**
```
MDD = max(Peak - Trough) over all time windows

where:
  Peak = maximum cumulative PnL achieved before drawdown
  Trough = minimum cumulative PnL during drawdown
```

**Bootstrap Test for MDD Difference:**
1. Compute observed MDDs: MDD_champion, MDD_challenger
2. Bootstrap 10,000 times:
   - Resample champion trades → compute MDD_i^champion
   - Resample challenger trades → compute MDD_i^challenger
   - Compute difference: Δ_MDD_i = MDD_i^champion - MDD_i^challenger
3. 95% CI for Δ_MDD

**Decision Rule:**
- **If CI entirely positive (challenger MDD < champion MDD):** Challenger lower risk ✅
- **If CI includes 0:** MDD difference not significant ❌

**Example:**
- Champion: MDD = -$6,000 (12% of capital)
- Challenger: MDD = -$4,500 (9% of capital)
- Bootstrap 95% CI for difference: [$800, $2,200]
- **CI entirely positive → Challenger has significantly lower MDD ✅**

---

## 3. PROMOTION CRITERIA

### 3.1 Primary Criteria (All Must Pass)

**Criterion 1: Statistical Significance (p < 0.05)**
```python
# At least ONE of these tests must pass:
t_test_passed = (t_test_p_value < 0.05)
bootstrap_passed = (0 not in bootstrap_ci)
sharpe_test_passed = (sharpe_z_stat > 1.96)

statistical_significance = t_test_passed or bootstrap_passed or sharpe_test_passed
```

**Criterion 2: Risk-Adjusted Performance (Sharpe Ratio)**
```python
# Challenger Sharpe must be ≥ Champion Sharpe (no degradation)
sharpe_criterion = (sharpe_challenger >= sharpe_champion)
```

**Criterion 3: Minimum Sample Size**
```python
# Require at least 500 trades for promotion
sample_size_criterion = (n_trades >= 500)
```

**Criterion 4: Maximum Drawdown Constraint**
```python
# Challenger MDD must not be >20% worse than champion
mdd_criterion = (mdd_challenger <= mdd_champion * 1.20)
```

**Criterion 5: Win Rate Floor**
```python
# Challenger WR must be ≥50% (profitable after fees)
win_rate_criterion = (win_rate_challenger >= 0.50)
```

**Combined Promotion Decision:**
```python
promote = (
    statistical_significance and
    sharpe_criterion and
    sample_size_criterion and
    mdd_criterion and
    win_rate_criterion
)
```

### 3.2 Secondary Criteria (Nice-to-Have)

**Criterion 6: Consistency (Rolling WR Stability)**
```python
# Check if challenger WR stable over time (not just lucky streak)
rolling_window = 100  # trades
rolling_wrs = compute_rolling_wr(challenger_trades, window=rolling_window)

consistency_criterion = (rolling_wrs.std() < 0.05)  # WR variance <5pp
```

**Criterion 7: Correlation with Champion (Diversity)**
```python
# Prefer challengers with low correlation to champion (diversification)
correlation = np.corrcoef(champion_pnls, challenger_pnls)[0, 1]

diversity_criterion = (correlation < 0.7)  # Prefer uncorrelated strategies
```

**Criterion 8: Sortino Ratio (Downside Risk)**
```python
# Sortino focuses on downside deviation (losses only)
sortino_challenger = mean_pnl / std(pnls[pnls < 0])
sortino_champion = mean_pnl_champion / std(champion_pnls[champion_pnls < 0])

sortino_criterion = (sortino_challenger >= sortino_champion)
```

### 3.3 Promotion Scoring System

**Weighted Score (0-100):**
```python
def compute_promotion_score(challenger_stats, champion_stats):
    """Compute 0-100 promotion score"""
    score = 0
    
    # Statistical significance (30 points)
    if challenger_stats['p_value'] < 0.01:
        score += 30
    elif challenger_stats['p_value'] < 0.05:
        score += 20
    
    # Sharpe ratio improvement (25 points)
    sharpe_improvement = (challenger_stats['sharpe'] - champion_stats['sharpe']) / champion_stats['sharpe']
    score += min(25, max(0, sharpe_improvement * 100))
    
    # Win rate improvement (20 points)
    wr_improvement = (challenger_stats['win_rate'] - champion_stats['win_rate'])
    score += min(20, max(0, wr_improvement * 400))  # +5pp = 20 points
    
    # MDD improvement (15 points)
    mdd_improvement = (champion_stats['mdd'] - challenger_stats['mdd']) / champion_stats['mdd']
    score += min(15, max(0, mdd_improvement * 30))
    
    # Consistency (10 points)
    if challenger_stats['rolling_wr_std'] < 0.05:
        score += 10
    
    return score

# Promotion thresholds:
# - Score ≥ 70: Auto-promote (strong evidence)
# - Score 50-69: Manual review (moderate evidence)
# - Score < 50: Reject (insufficient evidence)
```

---

## 4. MULTI-ARMED BANDIT (THOMPSON SAMPLING)

### 4.1 Problem: Exploration vs Exploitation

**Exploration-Exploitation Tradeoff:**
- **Exploitation:** Use champion (known good model) for 100% of trades
- **Exploration:** Allocate small % to challengers to gather performance data

**Naive Approach (Fixed ε=5%):**
- Champion: 95% allocation
- Challenger: 5% allocation
- **Problem:** Wastes 5% on bad challengers, under-allocates to good challengers

**Thompson Sampling Approach (Adaptive):**
- Dynamically adjust allocation based on observed performance
- Good challengers get more allocation (faster testing)
- Bad challengers get less allocation (minimize losses)

### 4.2 Thompson Sampling Algorithm

**Setup:**
- K models: 1 champion + (K-1) challengers
- Each model i has unknown true mean PnL: μ_i
- Goal: Maximize total PnL while testing challengers

**Bayesian Belief (Gaussian Prior):**
```
For each model i, maintain belief: μ_i ~ N(m_i, s_i²)

where:
  m_i = posterior mean (estimated PnL)
  s_i² = posterior variance (uncertainty)
```

**Update Rule (After Observing PnL x from Model i):**
```
# Prior: μ_i ~ N(m_i, s_i²)
# Likelihood: x ~ N(μ_i, σ²)  (observed PnL)

# Posterior (Bayesian update):
s_i²_new = 1 / (1/s_i² + 1/σ²)
m_i_new = s_i²_new * (m_i/s_i² + x/σ²)

# Intuition:
# - If σ² (observation noise) is high → don't update much
# - If s_i² (prior uncertainty) is high → update a lot (learn quickly)
```

**Allocation Rule (Thompson Sampling):**
```python
# For each trade:
for i in range(K):
    # Sample from posterior
    sample_i = np.random.normal(m_i, s_i)

# Allocate trade to model with highest sample
selected_model = argmax(sample_i)
```

**Example (1 Champion + 2 Challengers):**

| Model | Posterior Mean (m) | Posterior Std (s) | Sample | Selected? |
|-------|-------------------|-------------------|--------|-----------|
| **Champion** | $50 | $10 | $48 | ❌ |
| **Challenger 1** | $55 | $20 | **$62** | ✅ |
| **Challenger 2** | $45 | $15 | $40 | ❌ |

**Allocation Frequency (First 500 Trades):**
- Champion: ~60% (known good, low uncertainty)
- Challenger 1: ~35% (promising, high uncertainty → explore more)
- Challenger 2: ~5% (underperforming, quickly deprioritized)

### 4.3 Regret Minimization

**Regret Definition:**
```
Regret(T) = T * μ* - Σ(μ_selected_t)

where:
  T = total trades
  μ* = true mean of best model
  μ_selected_t = true mean of model selected at trade t
```

**Thompson Sampling Regret Bound:**
```
E[Regret(T)] ≤ O(√(T * K * log(T)))

# Sublinear regret → allocation converges to best model
```

**Comparison to Naive ε-Greedy:**
- **ε-Greedy:** E[Regret(T)] = O(T) (linear regret, always wastes ε%)
- **Thompson Sampling:** E[Regret(T)] = O(√T) (sublinear, adaptively reduces waste)

---

## 5. CHAMPION DEGRADATION DETECTION

### 5.1 Drift Detection Integration (Module 3)

**Trigger:** Champion model starts drifting (performance degrades)

**Action with Shadow Models:**
1. **Module 3 Drift Detection** flags champion degradation
2. **Module 5 Shadow Manager** checks if any challenger is currently testing
3. **Decision Tree:**
   - **If challenger exists AND outperforming degraded champion:** Promote immediately
   - **If no challenger:** Deploy retrained champion as challenger (shadow test first)
   - **If challenger exists BUT also degrading:** Escalate to Module 3 retraining

**Example:**
- Champion XGBoost: WR 56% → 52% (drift detected)
- Challenger LightGBM: WR 55% (tested for 300 trades)
- **Decision:** Promote LightGBM immediately (55% > 52%, prevents further losses)

### 5.2 Rolling Performance Monitoring

**Exponentially Weighted Moving Average (EWMA):**
```python
# Track champion performance with EWMA (faster response to degradation)
alpha = 0.1  # smoothing factor

ewma_wr = alpha * current_wr + (1 - alpha) * previous_ewma_wr

# Alert if EWMA drops >3pp below historical baseline
if ewma_wr < baseline_wr - 0.03:
    logger.warning("Champion degradation detected")
    trigger_shadow_testing()
```

**Change Point Detection (CUSUM):**
```python
# Cumulative sum control chart for detecting sudden WR shifts
def cusum(win_rates, target, drift=0.01, threshold=5):
    """
    Detect change point in win rate sequence
    
    Args:
        win_rates: Rolling WR sequence
        target: Expected WR (baseline)
        drift: Allowable drift (0.01 = 1pp)
        threshold: CUSUM threshold (higher = less sensitive)
    """
    cusum_pos = 0
    cusum_neg = 0
    
    for wr in win_rates:
        deviation = wr - target
        cusum_pos = max(0, cusum_pos + deviation - drift)
        cusum_neg = max(0, cusum_neg - deviation - drift)
        
        if cusum_pos > threshold or cusum_neg > threshold:
            return True  # Change point detected
    
    return False
```

---

## 6. ROLLBACK STRATEGY

### 6.1 Rollback Triggers

**Trigger 1: Performance Degradation Post-Promotion**
```python
# Monitor new champion for first 100 trades
if trades_since_promotion <= 100:
    if current_wr < promoted_wr - 0.03:  # 3pp drop
        logger.critical("New champion degraded, rolling back")
        rollback_to_previous_champion()
```

**Trigger 2: Maximum Drawdown Exceeded**
```python
# Abort if new champion MDD exceeds 1.5x expected
if current_mdd > expected_mdd * 1.5:
    logger.critical("MDD exceeded, rolling back")
    rollback_to_previous_champion()
```

**Trigger 3: Manual Intervention**
```python
# Allow manual rollback via API
POST /api/shadow/rollback
{
    "model_name": "xgboost_v23",
    "reason": "Unexpected behavior in EURUSD"
}
```

### 6.2 Rollback Procedure

**Steps:**
1. **Pause Trading:** Halt new trades for 30 seconds
2. **Restore Archive:** Load previous champion from archive
3. **Reallocate:** Champion (previous) → 100%, Failed model → 0%
4. **Resume Trading:** Unpause trades
5. **Alert Team:** Notification with rollback details
6. **Post-Mortem:** Log failure event for analysis

**Rollback Speed:** <30 seconds (champion always kept in memory)

**Testing:** Rollback procedure tested monthly in staging environment

---

## 7. SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────┐
│                     SHADOW MODEL MANAGER                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │  Champion   │  │ Challenger1 │  │ Challenger2 │            │
│  │  (100%)     │  │   (0%)      │  │   (0%)      │            │
│  │             │  │             │  │             │            │
│  │ XGBoost     │  │ LightGBM    │  │  CatBoost   │            │
│  │ WR: 56%     │  │ WR: 58%     │  │ WR: 54%     │            │
│  │ Sharpe: 1.8 │  │ Sharpe: 2.1 │  │ Sharpe: 1.6 │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│         │                 │                 │                   │
│         └─────────────────┴─────────────────┘                   │
│                           │                                     │
│                  ┌────────▼────────┐                            │
│                  │  Performance    │                            │
│                  │  Tracker        │                            │
│                  │                 │                            │
│                  │ - PnL history   │                            │
│                  │ - Win rates     │                            │
│                  │ - Sharpe ratios │                            │
│                  │ - Drawdowns     │                            │
│                  └────────┬────────┘                            │
│                           │                                     │
│                  ┌────────▼────────┐                            │
│                  │  Statistical    │                            │
│                  │  Testing Engine │                            │
│                  │                 │                            │
│                  │ - T-tests       │                            │
│                  │ - Bootstrap CI  │                            │
│                  │ - Sharpe tests  │                            │
│                  │ - MDD analysis  │                            │
│                  └────────┬────────┘                            │
│                           │                                     │
│                  ┌────────▼────────┐                            │
│                  │  Promotion      │                            │
│                  │  Decision Engine│                            │
│                  │                 │                            │
│                  │ - Criteria check│                            │
│                  │ - Scoring system│                            │
│                  │ - Auto-promote  │                            │
│                  │ - Rollback logic│                            │
│                  └────────┬────────┘                            │
│                           │                                     │
│         ┌─────────────────┴─────────────────┐                   │
│         │                                   │                   │
│  ┌──────▼──────┐                   ┌────────▼────────┐          │
│  │  Promote    │                   │  Archive        │          │
│  │  Challenger1│                   │  Old Champion   │          │
│  │  → Champion │                   │  (Rollback)     │          │
│  └─────────────┘                   └─────────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. PERFORMANCE CONSIDERATIONS

### 8.1 Computational Overhead

| Operation | Latency | Frequency | Daily Cost |
|-----------|---------|-----------|------------|
| **Shadow Prediction** | 50ms | Per trade | $0.20 |
| **Performance Tracking** | 5ms | Per trade | $0.05 |
| **Statistical Testing** | 200ms | Per 100 trades | $0.10 |
| **Promotion Decision** | 500ms | Per 500 trades | $0.05 |
| **TOTAL** | **50ms** (non-blocking) | - | **$0.40/day** |

**Impact on Trading Latency:** None (shadow predictions asynchronous)

### 8.2 Storage Requirements

| Data | Size | Retention | Total |
|------|------|-----------|-------|
| **Shadow Predictions** | 1 KB/trade | 90 days | 18 MB |
| **Performance Metrics** | 500 B/trade | 365 days | 36 MB |
| **Statistical Tests** | 100 KB/test | 365 days | 12 MB |
| **Archive Models** | 50 MB/model | Last 3 | 150 MB |
| **TOTAL** | - | - | **~200 MB** |

**Storage Cost:** $0.02/month (negligible)

---

## 9. SUCCESS METRICS

### 9.1 Testing Efficiency

- **Mean time to promotion decision:** <3 weeks (500 trades)
- **False promotion rate:** <5% (bad model promoted)
- **Missed opportunity rate:** <10% (good model not promoted)

### 9.2 Performance Improvement

- **Win rate improvement (per promotion):** +1-2pp
- **Sharpe ratio improvement:** +10-20%
- **Max drawdown reduction:** -10-25%

### 9.3 System Reliability

- **Rollback frequency:** <1/year (promotions validated)
- **Rollback speed:** <30 seconds
- **Zero downtime:** 100% (champion always available)

---

**Module 5 Section 2: Technical Framework - COMPLETE ✅**

Next: Implementation (shadow_model_manager.py with A/B testing infrastructure, statistical testing, automatic promotion)
