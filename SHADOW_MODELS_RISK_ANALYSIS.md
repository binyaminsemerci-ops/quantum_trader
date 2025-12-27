# SHADOW MODELS: RISK ANALYSIS

**Module 5: Shadow Models - Section 5**

## Overview

While shadow models enable zero-risk testing, the **decision-making process** itself carries risks. This analysis identifies 6 major risks in the shadow model system and provides prevention strategies.

---

## RISK 1: FALSE PROMOTIONS FROM LUCKY STREAKS

### Description

A genuinely inferior model may appear better due to random chance (lucky streak over 500 trades).

**Example:**
- Champion: True WR 56%, observed 56.2% over 500 trades
- Challenger: True WR 54%, observed 58.1% over 500 trades (lucky streak)
- Statistical test: p=0.04 → PROMOTION ❌ (Type I error)
- Reality: Challenger actually worse, but got promoted due to variance

### Root Causes

1. **Sample size too small:** 500 trades may not be enough for stable estimates
2. **High variance:** Crypto markets have high PnL variance (σ=$120)
3. **Multiple comparisons:** Testing 10 challengers increases false positive rate
4. **Selection bias:** Only "lucky" models reach 500 trades (survivorship bias)

### Probability

- Type I error rate: α=0.05 (5% false positive rate)
- With 20 challengers/year: Expected 1 false promotion/year
- Cost: $10K-$30K if undetected for 1 month

### Prevention Strategies

#### 1. Minimum Sample Size Validation

```python
# Power analysis for detecting +2pp WR improvement
def calculate_required_sample_size(effect_size_pp, alpha=0.05, power=0.80):
    """Calculate minimum trades needed"""
    p1 = 0.56  # Champion WR
    p2 = p1 + (effect_size_pp / 100)  # Challenger WR
    
    pooled_p = (p1 + p2) / 2
    pooled_var = pooled_p * (1 - pooled_p)
    
    z_alpha = 1.96  # 95% confidence
    z_beta = 0.84   # 80% power
    
    n = 2 * ((z_alpha + z_beta)**2 * pooled_var) / ((p2 - p1)**2)
    
    return int(np.ceil(n))

# For +2pp effect: n = 1,200 trades (not 500)
# For +3pp effect: n = 550 trades
# For +5pp effect: n = 200 trades

# Recommendation: Use 500 trades for ≥3pp effects only
```

#### 2. Bootstrap Validation (Already Implemented)

Bootstrap provides non-parametric confidence interval:
- 10,000 resamples → 95% CI for mean difference
- More robust to outliers than t-test
- Lower false positive rate in practice

#### 3. Multiple Test Correction

When testing multiple challengers, adjust α:

```python
# Bonferroni correction
def bonferroni_correction(alpha, n_tests):
    """Adjust significance level for multiple testing"""
    return alpha / n_tests

# Example: Testing 3 challengers simultaneously
alpha_adjusted = bonferroni_correction(0.05, 3)  # 0.0167

# Use adjusted α in statistical tests
self.statistical_tester = StatisticalTester(alpha=alpha_adjusted)
```

#### 4. Require Multiple Criteria

Already implemented: ALL 5 primary criteria must pass:
- Statistical significance (any test passed)
- Sharpe ratio ≥ champion
- Sample size ≥ 500
- Max drawdown ≤ 1.20x champion
- Win rate ≥ 50%

This reduces false positive rate from 5% to ~1% (multiplicative effect).

#### 5. Promotion Score Threshold

Score ≥70 required for auto-promotion:
- Lucky model may pass statistical test (30 points)
- But unlikely to also have Sharpe improvement (25 points)
- AND WR improvement (20 points)
- AND MDD improvement (15 points)

Combined likelihood of false high score: <1%

#### 6. Post-Promotion Monitoring

First 100 trades monitored:
- Alert if WR drops >3pp from baseline
- Rollback if WR drops >5pp

Catches lucky models that revert to true WR post-promotion.

**Expected catch rate:** 80% of false promotions detected within 100 trades

---

## RISK 2: SAMPLE SIZE BIAS (INSUFFICIENT POWER)

### Description

Statistical tests fail to detect truly better models because sample size is too small (Type II error, false negative).

**Example:**
- Champion: 56% WR
- Challenger: 58% WR (truly better by +2pp)
- After 500 trades: Observed 57.2% (close to truth, but not significant)
- Statistical test: p=0.12 → REJECTION ❌ (Type II error)
- Reality: Missed opportunity to promote better model

### Root Causes

1. **Small effect size:** +2pp WR improvement is real but subtle
2. **High variance:** PnL variance σ=$120 masks small improvements
3. **Fixed sample size:** 500 trades regardless of effect size
4. **Conservative α:** α=0.05 reduces power for marginal improvements

### Probability

- Type II error rate (β): ~20-40% for +2pp improvements at n=500
- With 20 challengers/year: Expected 4-8 missed opportunities/year
- Cost: $20K-$80K in forgone improvements

### Prevention Strategies

#### 1. Adaptive Sample Size

Adjust minimum trades based on observed effect size:

```python
def adaptive_sample_size(challenger_wr, champion_wr, current_n):
    """Determine if more trades needed"""
    effect_size = challenger_wr - champion_wr
    
    if effect_size >= 0.05:  # ≥5pp improvement
        return 200  # Small sample sufficient
    elif effect_size >= 0.03:  # 3-5pp improvement
        return 500  # Standard sample
    elif effect_size >= 0.02:  # 2-3pp improvement
        return 1200  # Large sample needed
    else:  # <2pp improvement
        return 2000  # Very large sample or reject
    
# Example usage in promotion check
required_n = adaptive_sample_size(0.58, 0.56, 500)
if current_n < required_n:
    return "Need {required_n - current_n} more trades"
```

#### 2. Sequential Testing

Instead of fixed n=500, use sequential probability ratio test (SPRT):

```python
def sequential_test(champion_pnls, challenger_pnls, alpha=0.05, beta=0.10):
    """Stop early if evidence strong enough"""
    
    # Likelihood ratio
    lr = np.prod(challenger_pnls) / np.prod(champion_pnls)
    
    # Decision boundaries
    A = (1 - beta) / alpha  # Upper boundary (reject H0)
    B = beta / (1 - alpha)  # Lower boundary (accept H0)
    
    if lr >= A:
        return "PROMOTE", len(challenger_pnls)
    elif lr <= B:
        return "REJECT", len(challenger_pnls)
    else:
        return "CONTINUE", len(challenger_pnls)

# Can detect strong effects in 200-300 trades
# Weak effects may need 800-1,000 trades
```

#### 3. Bayesian Evidence Threshold

Use Bayes factor instead of p-value:

```python
def bayes_factor(champion_pnls, challenger_pnls):
    """Calculate Bayes factor for model comparison"""
    # Model 1: Challenger better (H1)
    # Model 0: No difference (H0)
    
    # Approximate via Savage-Dickey ratio
    # BF > 10 = strong evidence for H1
    # BF > 100 = decisive evidence for H1
    
    # (Simplified calculation)
    t_stat, p_value = ttest_ind(challenger_pnls, champion_pnls)
    n = len(challenger_pnls)
    
    # Approximate BF from t-statistic
    bf = np.exp(-0.5 * (t_stat**2 - 2*np.log(n)))
    
    return bf

# Promote if BF > 10 (strong evidence)
# More nuanced than binary p<0.05
```

#### 4. Effect Size Thresholds

Focus on practical significance, not just statistical:

```python
# Require minimum effect size for promotion
MIN_EFFECT_SIZES = {
    'win_rate': 0.02,      # +2pp minimum
    'sharpe_ratio': 0.15,  # +0.15 minimum
    'mean_pnl': 5.0        # +$5 minimum
}

# Check if challenger meets thresholds
def check_practical_significance(champion_metrics, challenger_metrics):
    wr_diff = challenger_metrics.win_rate - champion_metrics.win_rate
    sharpe_diff = challenger_metrics.sharpe_ratio - champion_metrics.sharpe_ratio
    pnl_diff = challenger_metrics.mean_pnl - champion_metrics.mean_pnl
    
    if wr_diff < MIN_EFFECT_SIZES['win_rate']:
        return False, "WR improvement too small"
    if sharpe_diff < MIN_EFFECT_SIZES['sharpe_ratio']:
        return False, "Sharpe improvement too small"
    if pnl_diff < MIN_EFFECT_SIZES['mean_pnl']:
        return False, "PnL improvement too small"
    
    return True, "Practical significance achieved"
```

#### 5. Power Analysis Dashboard

Monitor statistical power in real-time:

```python
def compute_power(champion_metrics, challenger_metrics, n_trades, alpha=0.05):
    """Calculate power to detect observed effect"""
    effect_size = (challenger_metrics.mean_pnl - champion_metrics.mean_pnl) / champion_metrics.std_pnl
    
    # Cohen's d standardized effect size
    d = effect_size
    
    # Power calculation (approximate)
    z_alpha = 1.96  # 95% confidence
    z_beta = z_alpha - d * np.sqrt(n_trades / 2)
    power = 1 - norm.cdf(z_beta)
    
    return power

# Display on dashboard:
# "Current power: 65% (need 800 trades for 80% power)"
```

---

## RISK 3: CHAMPION DEGRADATION UNDETECTED

### Description

Champion performance degrades gradually, but degradation is not detected quickly enough because focus is on challengers.

**Example:**
- Champion promoted with 58% WR (Week 1)
- Week 2-4: WR gradually drops to 54% (market shift)
- Shadow testing focused on new challengers (starting from 0 trades)
- By Week 8: Champion at 52% WR, but no challenger has 500 trades yet
- Result: Trading with degraded model for 6 weeks

### Root Causes

1. **Focus on challengers:** Post-promotion monitoring lasts only 100 trades
2. **Long challenger testing:** Takes 2-3 weeks to reach 500 trades
3. **Gradual drift:** 1pp/week degradation hard to detect in real-time
4. **No baseline comparison:** Champion not compared to historical performance

### Probability

- Likelihood: 20-30% of champions degrade within 3 months
- Detection delay: 2-6 weeks average
- Cost: $5K-$30K per degradation event

### Prevention Strategies

#### 1. Continuous Champion Monitoring (EWMA)

Already mentioned in technical framework, implement:

```python
class ChampionMonitor:
    def __init__(self, alpha=0.1, alert_threshold=0.03):
        self.ewma_wr = None
        self.baseline_wr = None
        self.alpha = alpha  # Smoothing factor
        self.alert_threshold = alert_threshold  # 3pp drop
    
    def initialize(self, baseline_wr):
        """Set baseline at promotion"""
        self.baseline_wr = baseline_wr
        self.ewma_wr = baseline_wr
    
    def update(self, current_wr):
        """Update EWMA after each trade"""
        if self.ewma_wr is None:
            self.ewma_wr = current_wr
        else:
            self.ewma_wr = self.alpha * current_wr + (1 - self.alpha) * self.ewma_wr
        
        # Check for degradation
        if self.ewma_wr < self.baseline_wr - self.alert_threshold:
            return "DEGRADATION_ALERT", self.ewma_wr
        
        return "OK", self.ewma_wr

# Usage in AITradingEngine
self.champion_monitor = ChampionMonitor()
self.champion_monitor.initialize(baseline_wr=0.58)

# After each trade
status, ewma_wr = self.champion_monitor.update(current_wr)
if status == "DEGRADATION_ALERT":
    logger.warning(f"Champion degraded: EWMA WR {ewma_wr:.2%} vs baseline {0.58:.2%}")
    # Escalate to drift detection (Module 3)
    self.drift_detector.trigger_retraining()
```

#### 2. CUSUM Change Point Detection

Detect sudden shifts in champion performance:

```python
class CUSUMDetector:
    def __init__(self, threshold=5.0, drift=0.5):
        self.threshold = threshold
        self.drift = drift
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
    
    def update(self, observation, baseline=0.0):
        """Update CUSUM statistics"""
        deviation = observation - baseline
        
        self.cusum_pos = max(0, self.cusum_pos + deviation - self.drift)
        self.cusum_neg = max(0, self.cusum_neg - deviation - self.drift)
        
        # Check for change point
        if self.cusum_pos > self.threshold:
            return "UPWARD_SHIFT", self.cusum_pos
        elif self.cusum_neg > self.threshold:
            return "DOWNWARD_SHIFT", self.cusum_neg
        
        return "OK", 0.0

# Detects persistent shifts faster than EWMA
# Typical detection time: 50-100 trades
```

#### 3. Integration with Module 3 (Drift Detection)

Champion degradation should trigger drift detection:

```python
def check_champion_health():
    """Check champion performance and trigger actions"""
    champion = self.shadow_manager.get_champion()
    metrics = self.shadow_manager.get_metrics(champion)
    
    # Check EWMA
    status, ewma_wr = self.champion_monitor.update(metrics.win_rate)
    
    if status == "DEGRADATION_ALERT":
        # Escalate to drift detector
        drift_detected = self.drift_detector.check_drift(
            model_name=champion,
            current_wr=metrics.win_rate
        )
        
        if drift_detected:
            # Retrain model
            logger.info("[Champion] Retraining due to degradation")
            retrained_model = self.retrain_model(champion)
            
            # Deploy retrained model as CHALLENGER
            self.deploy_challenger_model(
                model_name=f"{champion}_retrained",
                model_type=champion.split('_')[0],
                model_instance=retrained_model,
                description="Retrained due to drift"
            )
            
            # Test retrained version as challenger before replacing
            logger.info("[Champion] Testing retrained version as challenger")
```

#### 4. Historical Performance Comparison

Compare current metrics to historical baselines:

```python
def compare_to_baseline(current_metrics, baseline_metrics):
    """Compare current performance to promotion baseline"""
    
    comparisons = {
        'win_rate': (current_metrics.win_rate, baseline_metrics.win_rate),
        'sharpe_ratio': (current_metrics.sharpe_ratio, baseline_metrics.sharpe_ratio),
        'mean_pnl': (current_metrics.mean_pnl, baseline_metrics.mean_pnl)
    }
    
    degradations = []
    
    for metric, (current, baseline) in comparisons.items():
        pct_change = (current - baseline) / baseline
        
        if pct_change < -0.10:  # 10% worse
            degradations.append((metric, pct_change))
    
    if degradations:
        msg = f"[Champion] Degraded: {degradations}"
        logger.warning(msg)
        return True
    
    return False
```

#### 5. Proactive Challenger Deployment

Always have 1-2 challengers testing:

```python
def maintain_challenger_pipeline():
    """Ensure challengers always in testing"""
    challengers = self.shadow_manager.get_challengers()
    
    if len(challengers) < 2:
        # Deploy new challenger
        new_model = self.train_new_model()
        self.deploy_challenger_model(
            model_name=f"model_{datetime.now().strftime('%Y%m%d')}",
            model_type='xgboost',
            model_instance=new_model,
            description="Proactive challenger"
        )
        
        logger.info(f"[Pipeline] Deployed new challenger (total: {len(challengers)+1})")

# Run weekly
schedule.every().week.do(maintain_challenger_pipeline)
```

---

## RISK 4: A/B TEST INTERFERENCE

### Description

Multiple challengers testing simultaneously may interfere with each other if they are correlated or if Thompson sampling allocates non-zero allocation.

**Example:**
- Challenger A: 0% allocation (pure shadow)
- Challenger B: 0% allocation (pure shadow)
- Champion: 100% allocation
- BUT: If Challengers A and B are highly correlated (ρ=0.9), their errors are not independent
- Result: Both may fail/succeed together due to shared biases

### Root Causes

1. **Model correlation:** Challengers trained on same data with similar architectures
2. **Thompson sampling allocation:** ε% allocation to challengers (if exploratory mode used)
3. **Market regime:** All models may perform poorly in adverse conditions
4. **Shared features:** Challengers use same feature engineering pipeline

### Probability

- Likelihood: 10-20% of challengers highly correlated (ρ>0.7)
- Impact: Mild (reduces diversity, but doesn't cause catastrophic failure)
- Cost: $2K-$10K in missed opportunities (not testing diverse models)

### Prevention Strategies

#### 1. Diversity Criterion in Promotion

Already included as secondary criterion:

```python
def check_diversity(champion_preds, challenger_preds):
    """Check if challenger sufficiently different from champion"""
    correlation = np.corrcoef(champion_preds, challenger_preds)[0, 1]
    
    if correlation > 0.7:
        return False, f"High correlation: {correlation:.2f}"
    
    return True, f"Low correlation: {correlation:.2f}"

# In PromotionEngine.check_criteria():
diversity_ok = check_diversity(champion_preds, challenger_preds)
```

#### 2. Pure Shadow Mode (0% Allocation)

Default to 0% allocation for all challengers:

```python
# In ShadowModelManager.register_model()
if role == ModelRole.CHALLENGER:
    metadata.allocation = 0.0  # Pure shadow, no execution
```

Only use exploratory allocation (ε% = 1-5%) if explicitly enabled.

#### 3. Thompson Sampling with Independent Arms

Thompson sampling already treats each model as independent arm:

```python
def sample_allocation(self, model_names):
    """Sample from independent posteriors"""
    samples = {}
    
    for model_name in model_names:
        # Sample from INDEPENDENT posterior
        sample = np.random.normal(
            loc=self.means[model_name],
            scale=self.stds[model_name]
        )
        samples[model_name] = sample
    
    # Select best sample
    return max(samples, key=samples.get)
```

Correlation handled by Bayesian updates, not allocation.

#### 4. Limit Simultaneous Challengers

Restrict to max 3 challengers at once:

```python
def deploy_challenger_model(self, model_name, model_type, model_instance, description):
    """Deploy challenger (with limit check)"""
    challengers = self.shadow_manager.get_challengers()
    
    if len(challengers) >= 3:
        raise ValueError(f"Max 3 challengers allowed (current: {len(challengers)})")
    
    # ... register model ...
```

Reduces interference and compute cost.

#### 5. Architecture Diversity

Ensure challengers use different model types:

```python
ALLOWED_CHALLENGER_TYPES = ['xgboost', 'lightgbm', 'catboost', 'neural_network']

def check_challenger_diversity(current_challengers, new_type):
    """Ensure model type diversity"""
    current_types = [c.split('_')[0] for c in current_challengers]
    
    if new_type in current_types:
        logger.warning(f"Challenger type {new_type} already testing")
        return False
    
    return True
```

---

## RISK 5: ROLLBACK FAILURES

### Description

Rollback mechanism fails when needed, leaving degraded champion in production.

**Example:**
- New champion promoted (Week 1)
- Week 2: WR drops to 50% (>5pp degradation)
- Rollback triggered automatically
- BUT: Archive champion corrupted or deleted
- Result: Stuck with bad champion, no rollback possible

### Root Causes

1. **Archive corruption:** Checkpoint file corrupted or overwritten
2. **Missing archive:** Old champion deleted after promotion
3. **Rollback logic bug:** Exception in rollback procedure
4. **State inconsistency:** Models metadata out of sync with actual models

### Probability

- Likelihood: <1% (rare but catastrophic)
- Impact: High ($50K-$200K if undetected for 1 month)
- Cost per incident: $50K-$200K

### Prevention Strategies

#### 1. Multiple Archive Checkpoints

Keep last 3 champions (not just 1):

```python
class ShadowModelManager:
    def __init__(self, ...):
        self.archive_champions = []  # List of archived champions (max 3)
        self.max_archive_size = 3
    
    def promote_challenger(self, challenger_name, force=False):
        """Promote challenger and archive old champion"""
        old_champion = self.current_champion
        
        # Archive old champion
        archive_metadata = ModelMetadata(
            model_name=old_champion,
            role=ModelRole.ARCHIVE,
            allocation=0.0,
            deployed_at=datetime.now()
        )
        
        # Add to archive list
        self.archive_champions.append(archive_metadata)
        
        # Keep only last 3
        if len(self.archive_champions) > self.max_archive_size:
            self.archive_champions.pop(0)
        
        # ... promote challenger ...
    
    def rollback_to_previous_champion(self, reason="", version=-1):
        """Rollback to specific archive version"""
        if not self.archive_champions:
            logger.error("[Rollback] No archived champions available")
            return False
        
        # Get archive (default: most recent)
        try:
            archive = self.archive_champions[version]
        except IndexError:
            logger.error(f"[Rollback] Invalid version: {version}")
            return False
        
        # ... restore archive ...
```

#### 2. Checkpoint Validation

Validate checkpoint integrity before saving:

```python
def checkpoint(self):
    """Save state with validation"""
    checkpoint_data = {
        'models': {name: meta.to_dict() for name, meta in self.models.items()},
        'current_champion': self.current_champion,
        'archive_champions': [a.to_dict() for a in self.archive_champions],
        'promotion_history': [p.to_dict() for p in self.promotion_history],
        'timestamp': datetime.now().isoformat(),
        'version': '1.0'
    }
    
    # Validate before saving
    if not self._validate_checkpoint(checkpoint_data):
        logger.error("[Checkpoint] Validation failed, not saving")
        return False
    
    # Save to temp file first
    temp_path = self.checkpoint_path.with_suffix('.tmp')
    with open(temp_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    # Atomic rename
    temp_path.replace(self.checkpoint_path)
    
    # Create backup
    backup_path = self.checkpoint_path.with_suffix('.backup')
    shutil.copy(self.checkpoint_path, backup_path)
    
    logger.info("[Checkpoint] Saved and validated")
    return True

def _validate_checkpoint(self, data):
    """Validate checkpoint data integrity"""
    required_keys = ['models', 'current_champion', 'archive_champions', 'promotion_history']
    
    for key in required_keys:
        if key not in data:
            logger.error(f"[Checkpoint] Missing key: {key}")
            return False
    
    # Check current champion exists
    if data['current_champion'] not in data['models']:
        logger.error("[Checkpoint] Champion not in models")
        return False
    
    # Check archive champions exist
    for archive in data['archive_champions']:
        if archive['model_name'] not in data['models']:
            logger.error(f"[Checkpoint] Archive {archive['model_name']} not in models")
            return False
    
    return True
```

#### 3. Rollback Dry-Run

Test rollback procedure without executing:

```python
def test_rollback(self, version=-1):
    """Simulate rollback without actually executing"""
    if not self.archive_champions:
        return False, "No archived champions"
    
    try:
        archive = self.archive_champions[version]
    except IndexError:
        return False, f"Invalid version: {version}"
    
    # Check if archive model loadable
    if not self._can_load_model(archive.model_name):
        return False, f"Archive model {archive.model_name} not loadable"
    
    # Check metadata consistent
    if archive.model_name not in self.models:
        return False, f"Archive metadata missing for {archive.model_name}"
    
    return True, f"Rollback to {archive.model_name} ready"

# Run dry-run before actual rollback
success, message = self.shadow_manager.test_rollback()
if not success:
    logger.error(f"[Rollback] Dry-run failed: {message}")
    # Alert team for manual intervention
else:
    # Proceed with actual rollback
    self.shadow_manager.rollback_to_previous_champion()
```

#### 4. Champion Always in Memory

Keep champion model loaded at all times:

```python
class AITradingEngine:
    def __init__(self, ...):
        # Champion model always loaded
        self.champion_model_instance = None
        self.champion_backup_instance = None  # Backup copy
        
        # Load champion
        champion = self.shadow_manager.get_champion()
        self.champion_model_instance = self._load_model(champion)
        
        # Create backup copy
        self.champion_backup_instance = self._clone_model(self.champion_model_instance)
    
    def promote_challenger(self, challenger_name):
        """Promote challenger with backup"""
        # Backup current champion BEFORE promotion
        old_champion_backup = self._clone_model(self.champion_model_instance)
        
        # Promote in shadow manager
        success = self.shadow_manager.promote_challenger(challenger_name)
        
        if success:
            # Load new champion
            self.champion_model_instance = self._load_model(challenger_name)
            
            # Keep old champion as backup
            self.champion_backup_instance = old_champion_backup
        
        return success
```

#### 5. Monitoring and Alerts

Alert if rollback fails:

```python
def rollback_to_previous_champion(self, reason=""):
    """Rollback with failure alerting"""
    try:
        # ... rollback logic ...
        
        if success:
            logger.info(f"[Rollback] Success: {archive.model_name}")
            self.notification_service.send_alert(
                f"✅ Rollback successful: {archive.model_name}",
                priority='MEDIUM'
            )
            return True
        else:
            raise Exception("Rollback failed: unknown reason")
    
    except Exception as e:
        logger.error(f"[Rollback] FAILED: {e}")
        
        # CRITICAL ALERT
        self.notification_service.send_alert(
            f"🚨 ROLLBACK FAILED: {e}\n"
            f"Champion: {self.current_champion}\n"
            f"Reason: {reason}\n"
            f"Action: MANUAL INTERVENTION REQUIRED",
            priority='CRITICAL'
        )
        
        return False
```

---

## RISK 6: OVER-TESTING (TOO MANY CHALLENGERS)

### Description

Testing too many challengers simultaneously increases computational overhead, multiple comparison issues, and monitoring complexity.

**Example:**
- 5 challengers testing simultaneously
- Each needs 500 trades → 2,500 total shadow predictions
- Statistical tests: 5 comparisons → increased Type I error
- Monitoring: 5 dashboards to track
- Result: Overwhelmed system, higher false positive rate

### Root Causes

1. **No limits:** System allows unlimited challengers
2. **Eager testing:** Deploy every new model immediately
3. **Long testing period:** 500 trades takes 2-3 weeks
4. **No prioritization:** All challengers treated equally

### Probability

- Likelihood: 30-50% without limits
- Impact: Medium (performance degradation, confusion)
- Cost: $5K-$20K in compute + monitoring overhead

### Prevention Strategies

#### 1. Hard Limit on Challengers

Max 3 challengers at once (already suggested):

```python
MAX_CHALLENGERS = 3

def deploy_challenger_model(self, model_name, model_type, model_instance, description):
    """Deploy challenger with limit check"""
    challengers = self.shadow_manager.get_challengers()
    
    if len(challengers) >= MAX_CHALLENGERS:
        raise ValueError(
            f"Max {MAX_CHALLENGERS} challengers allowed. "
            f"Current challengers: {', '.join(challengers)}. "
            f"Remove or promote one before deploying new challenger."
        )
    
    # ... register model ...
```

#### 2. Prioritization Queue

Queue challengers and test in priority order:

```python
class ChallengerQueue:
    def __init__(self, max_active=3):
        self.queue = []  # (priority, model_name, model_instance)
        self.max_active = max_active
    
    def add(self, model_name, model_instance, priority='MEDIUM'):
        """Add to queue"""
        priorities = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        self.queue.append((priorities[priority], model_name, model_instance))
        self.queue.sort(reverse=True)  # Highest priority first
    
    def get_next(self):
        """Get next challenger to deploy"""
        if self.queue:
            return self.queue.pop(0)
        return None
    
    def deploy_next_if_space(self, shadow_manager):
        """Deploy next if slots available"""
        challengers = shadow_manager.get_challengers()
        
        if len(challengers) < self.max_active:
            next_challenger = self.get_next()
            if next_challenger:
                priority, name, instance = next_challenger
                # Deploy
                return True
        
        return False

# Usage:
challenger_queue = ChallengerQueue(max_active=3)
challenger_queue.add('xgboost_v2', xgb_model, priority='HIGH')
challenger_queue.add('lightgbm_v3', lgb_model, priority='LOW')
```

#### 3. Auto-Remove Failed Challengers

Remove challengers that fail criteria:

```python
def cleanup_failed_challengers(self):
    """Remove challengers that clearly won't promote"""
    challengers = self.shadow_manager.get_challengers()
    
    for challenger in challengers:
        trade_count = self.shadow_manager.get_trade_count(challenger)
        
        # After 300 trades, check if on track
        if trade_count >= 300:
            metrics = self.shadow_manager.get_metrics(challenger)
            champion_metrics = self.shadow_manager.get_metrics(
                self.shadow_manager.get_champion()
            )
            
            # If clearly worse, remove
            if metrics.win_rate < champion_metrics.win_rate - 0.03:  # 3pp worse
                logger.info(f"[Cleanup] Removing {challenger} (clearly inferior)")
                self.shadow_manager.remove_model(challenger)
                
                # Deploy next from queue
                self.challenger_queue.deploy_next_if_space(self.shadow_manager)

# Run every 100 trades
if self.trades_since_shadow_check % 100 == 0:
    self.cleanup_failed_challengers()
```

#### 4. Bonferroni Correction

Adjust α for multiple comparisons:

```python
def check_promotion_criteria_with_correction(self, challenger_name):
    """Check criteria with Bonferroni correction"""
    n_challengers = len(self.get_challengers())
    
    # Adjust α
    adjusted_alpha = self.alpha / n_challengers
    
    # Use adjusted α in statistical tests
    tester = StatisticalTester(alpha=adjusted_alpha, n_bootstrap=self.n_bootstrap)
    
    # ... run tests ...
```

#### 5. Monitoring Dashboard

Visual overview of all challengers:

```
CHALLENGER STATUS DASHBOARD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Champion: xgboost_v1 (WR: 56.2%, Sharpe: 1.85)

Challengers:
  1. lightgbm_v2     [▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░] 500/500 trades | Status: APPROVED (Score: 78.5)
  2. catboost_v1     [▓▓▓▓▓▓░░░░░░░░░░░░░░] 320/500 trades | Status: PENDING (WR: 57.8%)
  3. xgboost_v2      [▓▓▓░░░░░░░░░░░░░░░░░] 150/500 trades | Status: PENDING (WR: 54.2%)

Queue:
  - neural_net_v1 (Priority: HIGH)
  - ensemble_v3 (Priority: MEDIUM)

Actions:
  - lightgbm_v2 ready for promotion (auto-promote in 10 minutes)
  - catboost_v1 needs 180 more trades
  - xgboost_v2 underperforming (consider removing)
```

---

## MONITORING METRICS

Track these metrics to detect risks early:

### 1. Promotion Quality Metrics

```python
promotion_quality = {
    'false_promotion_rate': 0.05,      # Target: <5%
    'missed_opportunity_rate': 0.15,   # Target: <15%
    'average_improvement': 0.025,      # Target: >2pp WR
    'rollback_rate': 0.10              # Target: <10% of promotions
}
```

### 2. Champion Health Metrics

```python
champion_health = {
    'ewma_wr': 0.562,                  # vs baseline 0.580
    'cusum_statistic': 2.3,            # vs threshold 5.0
    'trades_since_promotion': 450,
    'degradation_alerts': 0            # Target: 0
}
```

### 3. Challenger Pipeline Metrics

```python
challenger_pipeline = {
    'active_challengers': 2,           # Target: 2-3
    'queue_length': 3,                 # Target: <5
    'avg_testing_time': 2.5,           # weeks (Target: <3)
    'compute_overhead': 0.08           # Target: <10%
}
```

### 4. Statistical Test Metrics

```python
test_quality = {
    'avg_p_value': 0.03,               # For promoted models
    'avg_power': 0.78,                 # Target: >0.80
    'bonferroni_correction': 0.0167,   # α adjusted for 3 challengers
    'bayes_factor_avg': 15.2           # Target: >10 (strong evidence)
}
```

---

## RISK MITIGATION SUMMARY

| Risk | Probability | Impact | Cost | Prevention | Detection Time |
|------|-------------|--------|------|------------|----------------|
| **False Promotions** | 5% | High | $10K-$30K | Bootstrap + Multi-criteria + Score≥70 + Post-monitoring | 100 trades (80% catch) |
| **Sample Size Bias** | 20-40% | Medium | $20K-$80K | Adaptive sample size + Sequential testing + Bayes factor | N/A (prevention only) |
| **Champion Degradation** | 20-30% | High | $5K-$30K | EWMA + CUSUM + Module 3 integration | 50-100 trades |
| **A/B Interference** | 10-20% | Low | $2K-$10K | Diversity criterion + Pure shadow + Max 3 challengers | N/A (prevention only) |
| **Rollback Failures** | <1% | Critical | $50K-$200K | 3 archives + Checkpoint validation + Dry-run + Alerts | Immediate (on failure) |
| **Over-Testing** | 30-50% | Medium | $5K-$20K | Max 3 challengers + Queue + Bonferroni + Auto-cleanup | N/A (prevention only) |

**Total Expected Annual Risk Cost (without mitigations):** $92K-$370K

**Total Expected Annual Risk Cost (with mitigations):** $10K-$40K

**Risk Reduction:** 82-89%

---

**Module 5 Section 5: Risk Analysis - COMPLETE ✅**

Next: Test Suite (Section 6)
