# COVARIATE SHIFT HANDLING: RISK ANALYSIS

**Module 4: Covariate Shift Handling - Section 5**

## Risk Matrix Summary

| Risk | Severity | Probability | Priority | Impact |
|------|----------|-------------|----------|--------|
| **Extreme Importance Weights** | HIGH | MEDIUM | HIGH | Unstable predictions, overfitting to outliers |
| **Domain Adaptation Artifacts** | MEDIUM | MEDIUM | MEDIUM | Feature distortion, spurious correlations |
| **OOD Miscalibration** | HIGH | LOW | MEDIUM | Overconfident predictions on unfamiliar data |
| **Computational Overhead** | MEDIUM | MEDIUM | MEDIUM | Slow adaptation, trading latency |
| **False Positive Shifts** | MEDIUM | HIGH | HIGH | Unnecessary adaptation, model instability |
| **Adaptation Failure** | CRITICAL | LOW | CRITICAL | Losses accumulate during shift |

---

## RISK 1: EXTREME IMPORTANCE WEIGHTS (HIGH SEVERITY, MEDIUM PROBABILITY)

### What Can Go Wrong

**Scenario:** KMM assigns weight=850 to a single outlier training sample, causing the model to overfit to that specific pattern.

**Example:**
- Training: 500 samples with volatility 2-4%, one outlier at 9%
- Current market: volatility now 8-10% (covariate shift)
- KMM: Assigns weight=850 to the 9% outlier, weight=0.1 to others
- Model predictions: Heavily biased toward that single outlier's pattern
- Result: 43% WR (model overfits to one sample)

**Impact:**
- Financial: -$15,000 over 5 days (poor predictions)
- Operational: Model behaves erratically, hard to debug
- Time: 8 hours to diagnose and fix

### Root Causes

1. **Insufficient training diversity:** Training data doesn't cover current distribution
2. **Poor kernel bandwidth:** `kernel_gamma` too high → sharp kernel → outlier dominance
3. **Loose weight bounds:** `COVARIATE_WEIGHT_BOUND=1000` allows extreme weights
4. **Small sample size:** 500 training samples insufficient for diverse coverage
5. **No weight clipping:** Post-KMM weights not constrained

### Prevention Strategies

**Strategy 1: Strict Weight Clipping**
```python
# After importance weight calculation
weights = np.clip(weights, 0.1, 10)  # Hard bounds

# Reject if stability score too high
stability = weights.max() / weights.mean()
if stability > 20:
    logger.warning(f"Weights unstable (stability={stability:.1f}), using uniform weights")
    weights = np.ones_like(weights)
```

**Strategy 2: Kernel Bandwidth Tuning**
```python
# Adaptive gamma based on data variance
median_distance = np.median(pairwise_distances(X_train))
kernel_gamma = 1 / (2 * median_distance ** 2)

# Lower gamma = smoother kernel = less extreme weights
```

**Strategy 3: Use Discriminator Instead of KMM**
```python
# Discriminator more stable for outliers
method = 'discriminator'  # Instead of 'kmm'
weights = self.discriminator_weights(X_train, X_test)

# Discriminator: logistic regression doesn't overweight single samples
```

**Strategy 4: Minimum Sample Coverage**
```python
# Require minimum training samples in current distribution range
def check_training_coverage(X_train, X_test):
    """Check if training data covers test range"""
    for feat_idx in range(X_train.shape[1]):
        train_min, train_max = X_train[:, feat_idx].min(), X_train[:, feat_idx].max()
        test_min, test_max = X_test[:, feat_idx].min(), X_test[:, feat_idx].max()
        
        # Check if test range outside training range
        if test_min < train_min * 0.8 or test_max > train_max * 1.2:
            logger.warning(f"Feature {feat_idx} OOD: train=[{train_min:.2f}, {train_max:.2f}], test=[{test_min:.2f}, {test_max:.2f}]")
            return False
    
    return True

# If coverage insufficient, escalate to retraining
if not check_training_coverage(X_train, X_test):
    logger.error("Training data doesn't cover current distribution → Escalate to retraining")
    return None  # Don't adapt, trigger retraining instead
```

**Strategy 5: Weight Smoothing**
```python
# Apply moving average to weights (temporal smoothing)
if hasattr(self, 'previous_weights') and self.previous_weights is not None:
    weights = 0.7 * self.previous_weights + 0.3 * weights  # EMA
self.previous_weights = weights
```

### Fallback Procedures

1. **Detect unstable weights:** Check stability score > 20
2. **Revert to uniform weights:** Use `weights = np.ones(n_train)`
3. **Flag for retraining:** Log event, schedule retraining via Module 3
4. **Manual review:** Alert analysts to investigate distribution gap

### Monitoring Metrics

- **Stability score:** max/mean ratio (target <10, alert >20)
- **Weight distribution:** Track percentiles (P10, P50, P90)
- **Daily:** Plot weight histogram, flag bimodal distributions
- **Weekly:** Check correlation between weight variance and performance

---

## RISK 2: DOMAIN ADAPTATION ARTIFACTS (MEDIUM SEVERITY, MEDIUM PROBABILITY)

### What Can Go Wrong

**Scenario:** CORAL transform distorts feature correlations, creating spurious patterns the model mistakes for signals.

**Example:**
- Training: RSI and MACD uncorrelated (r=0.02)
- CORAL: Aligns covariances → introduces correlation r=0.68
- Model: Learns spurious rule "when RSI high AND MACD high → buy"
- Result: False signals, 51% WR (barely profitable after fees)

**Impact:**
- Financial: -$8,000 over 10 days (false signals)
- Model reliability: Trust eroded, requires validation
- Time: 6 hours to diagnose correlation artifact

### Root Causes

1. **Feature correlations change:** Market dynamics shift correlation structure
2. **CORAL overfitting:** Matches covariances too precisely to test data
3. **Small test sample:** 100 samples insufficient for stable covariance estimation
4. **No validation:** Transformation not validated before deployment

### Prevention Strategies

**Strategy 1: Regularized CORAL**
```python
def coral_transform_regularized(X_train, X_test, lambda_reg=0.1):
    """CORAL with regularization to prevent overfitting"""
    Sigma_train = np.cov(X_train.T)
    Sigma_test = np.cov(X_test.T)
    
    d = Sigma_train.shape[0]
    
    # Add regularization (shrink toward identity)
    Sigma_train_reg = (1 - lambda_reg) * Sigma_train + lambda_reg * np.eye(d)
    Sigma_test_reg = (1 - lambda_reg) * Sigma_test + lambda_reg * np.eye(d)
    
    # Compute transformation
    A = Sigma_test_reg_sqrt @ Sigma_train_reg_inv_sqrt
    
    return X_train @ A.T, A
```

**Strategy 2: Validation Gate**
```python
# After transformation, validate on holdout set
X_train_adapted, A = self.coral_transform(X_train, X_test)

# Predict on holdout (last 50 samples of training)
X_holdout = X_train[-50:]
y_holdout = y_train[-50:]

predictions_before = model.predict(X_holdout)
predictions_after = model.predict(X_holdout @ A.T)

# Check if performance maintained
wr_before = (predictions_before == y_holdout).mean()
wr_after = (predictions_after == y_holdout).mean()

if wr_after < wr_before - 0.05:  # 5pp drop
    logger.warning(f"CORAL validation failed: WR {wr_before:.2%} → {wr_after:.2%}, reverting")
    return X_train, None  # Don't use transformation
```

**Strategy 3: Use Standardization Instead**
```python
# For moderate shifts, use simple standardization (less aggressive)
if shift_severity == 'moderate':
    method = 'standardize'  # Mean/std alignment only
else:
    method = 'coral'  # Full covariance alignment for severe shifts
```

**Strategy 4: Incremental Adaptation**
```python
# Blend original and transformed features
alpha = 0.5  # Blending factor
X_adapted_blended = alpha * X_train_adapted + (1 - alpha) * X_train

# Start conservative (alpha=0.3), increase if performance good
```

### Fallback Procedures

1. **Monitor WR post-adaptation:** Track for 50 trades
2. **Auto-revert if WR drops >3pp:** Disable transformation
3. **A/B test transformations:** 80% original, 20% transformed for first 100 trades
4. **Manual review:** Check feature correlation changes

### Monitoring Metrics

- **Correlation shift:** Track Δr for all feature pairs (alert if Δr > 0.5)
- **Feature distribution shift:** Check if transformed features still realistic
- **Win rate post-adaptation:** Monitor for 24 hours
- **Prediction variance:** Track if predictions become more/less uncertain

---

## RISK 3: OOD MISCALIBRATION (HIGH SEVERITY, LOW PROBABILITY)

### What Can Go Wrong

**Scenario:** Mahalanobis distance flags 80% of predictions as OOD, reducing all confidences to <0.30, causing the system to halt trading unnecessarily.

**Example:**
- Market volatility spikes from 3% to 12% (extreme event)
- Mahalanobis distance: D_M > 10 for 80% of samples
- Calibration: confidence 0.68 → 0.22 (overly conservative)
- Result: System stops trading (confidence threshold 0.65), misses opportunities

**Impact:**
- Financial: Opportunity cost $12,000 (missed trades during recovery)
- Operational: System offline for 8 hours until manual override
- Time: 4 hours to recalibrate thresholds

### Root Causes

1. **Extreme market events:** Unprecedented volatility (flash crash, news)
2. **OOD threshold too low:** `ood_threshold=0.7` too sensitive
3. **Training distribution narrow:** Trained only on low-volatility data
4. **No adaptive calibration:** Fixed λ=0.1 doesn't adjust to event severity

### Prevention Strategies

**Strategy 1: Adaptive OOD Threshold**
```python
# Adjust threshold based on historical OOD rates
historical_ood_rate = self._compute_historical_ood_rate()

if historical_ood_rate > 0.5:  # Consistently high OOD
    # Market has shifted permanently, update baseline
    logger.info("High OOD rate detected, updating training baseline")
    self._update_training_baseline(X_test)
else:
    # Temporary spike, use adaptive threshold
    ood_threshold_adaptive = min(0.9, self.ood_threshold + 0.2)
```

**Strategy 2: Tiered Confidence Reduction**
```python
# Don't reduce confidence uniformly, use tiers
if D_M_norm < 0.5:
    conf_adjusted = conf * 1.0  # No reduction (in-distribution)
elif D_M_norm < 0.7:
    conf_adjusted = conf * 0.9  # Minor reduction
elif D_M_norm < 0.85:
    conf_adjusted = conf * 0.7  # Moderate reduction
else:
    conf_adjusted = conf * 0.5  # Severe reduction (OOD)
```

**Strategy 3: Ensemble-Based OOD Detection**
```python
# Use ensemble disagreement as secondary OOD signal
ensemble_variance = predictions_ensemble.var()

# Only flag OOD if BOTH Mahalanobis AND ensemble agree
ood_flag = (D_M_norm > ood_threshold) AND (ensemble_variance > 0.1)
```

**Strategy 4: Minimum Confidence Floor**
```python
# Don't reduce confidence below minimum
conf_adjusted = max(conf * np.exp(-lambda_decay * D_M_norm), 0.40)

# Ensures system doesn't halt completely during extreme events
```

### Fallback Procedures

1. **Monitor OOD rate:** Alert if >50% OOD for 2+ hours
2. **Emergency baseline update:** Use recent 200 samples as new baseline
3. **Manual override:** Allow trading with reduced position sizes
4. **Escalate to retraining:** If OOD persists 24+ hours

### Monitoring Metrics

- **OOD rate:** Track hourly (alert >30%, critical >50%)
- **Confidence distribution:** Plot histogram (alert if median <0.50)
- **Mahalanobis distance:** Track P50, P95 (flag extreme outliers)
- **Trading activity:** Alert if trade frequency drops >50%

---

## RISK 4: COMPUTATIONAL OVERHEAD (MEDIUM SEVERITY, MEDIUM PROBABILITY)

### What Can Go Wrong

**Scenario:** MMD² calculation takes 2.5 seconds per check, causing 250ms latency per trade (every 100 trades triggers check).

**Example:**
- 100 trades → MMD² check triggered
- Compute K_train_train (500×500 kernel matrix): 1.2 seconds
- Compute K_test_test (100×100): 0.3 seconds
- Compute K_train_test (500×100): 0.6 seconds
- Total: 2.1 seconds per check
- Result: Trading paused during computation, missed opportunity

**Impact:**
- Financial: Missed 3 trades worth $4,500 (latency)
- Operational: System feels sluggish, user frustration
- Time: 6 hours to optimize kernel computation

### Root Causes

1. **Large kernel matrices:** O(n²) complexity for n=500 samples
2. **Every-100-trades check:** Frequent checks with full recomputation
3. **No caching:** Kernel matrices recomputed from scratch
4. **Blocking computation:** Check runs in main thread

### Prevention Strategies

**Strategy 1: Incremental MMD Update**
```python
# Maintain running kernel sums (avoid full recomputation)
class IncrementalMMD:
    def __init__(self):
        self.K_train_sum = 0
        self.K_test_running = []
        self.n_train = 0
        self.n_test = 0
    
    def update(self, x_new, is_test=True):
        """Incremental update with new sample"""
        if is_test:
            # Update K_test_test sum
            k_new = [self._kernel(x_new, x_test) for x_test in self.X_test_buffer]
            self.K_test_sum += sum(k_new) + self._kernel(x_new, x_new)
            self.n_test += 1
        
        # Update MMD² incrementally (O(n) instead of O(n²))
        self.mmd_squared = (self.K_train_sum / self.n_train**2) + \
                          (self.K_test_sum / self.n_test**2) - \
                          (2 * self.K_cross_sum / (self.n_train * self.n_test))

# 100x speedup for large n
```

**Strategy 2: Async Computation**
```python
# Run shift detection asynchronously (don't block trading)
import asyncio

async def _check_covariate_shift_async(self, model_name):
    """Non-blocking covariate shift check"""
    # Run in background task
    task = asyncio.create_task(
        self._compute_shift_metrics(model_name)
    )
    
    # Continue trading immediately, process result later
    return task

# In main loop
shift_task = await self._check_covariate_shift_async(model_name)
# ... continue trading ...
# Later: result = await shift_task
```

**Strategy 3: Sampling for Large Datasets**
```python
# Subsample training data for MMD computation
if X_train.shape[0] > 1000:
    sample_size = 300
    indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
    X_train_sample = X_train[indices]
else:
    X_train_sample = X_train

# Compute MMD on sample (10x faster)
mmd_squared = self.compute_mmd_squared(X_train_sample, X_test)
```

**Strategy 4: Use Faster KS Tests Only**
```python
# For real-time, skip MMD and use per-feature KS tests (O(n log n))
if real_time_mode:
    ks_results = self.compute_ks_tests(X_train, X_test, feature_names)
    severity = self._determine_severity_from_ks(ks_results)
    # 50x faster than MMD for high-dimensional data
```

### Fallback Procedures

1. **Latency monitoring:** Alert if check >500ms
2. **Reduce check frequency:** Change to every 200 trades (if latency high)
3. **Use KS-only mode:** Disable MMD/KL, use KS tests only
4. **Increase sample size threshold:** Only check if >150 trades (not 100)

### Monitoring Metrics

- **Computation time:** Track MMD, KL, KS times separately (target <200ms total)
- **Kernel matrix size:** Log n_train, n_test (alert if >1000)
- **Check frequency:** Track checks/hour (reduce if >10/hour)
- **Trading latency:** Monitor p99 latency (target <100ms)

---

## RISK 5: FALSE POSITIVE SHIFTS (MEDIUM SEVERITY, HIGH PROBABILITY)

### What Can Go Wrong

**Scenario:** Temporary volume spike (whale transaction) triggers "covariate shift" alert, causing unnecessary adaptation that destabilizes the model.

**Example:**
- Normal volume: 20k-40k
- Whale transaction: 180k volume (single trade)
- MMD²: 0.052 (SEVERE shift detected)
- Action: CORAL transformation applied + importance weighting
- 30 minutes later: Volume returns to 25k (whale exited)
- Result: Model adapted to outlier, now performs poorly on normal data (55% → 52% WR)

**Impact:**
- Financial: -$3,500 over 2 days (adapted to wrong distribution)
- Operational: Model instability, requires rollback
- Time: 4 hours to detect false positive and revert

### Root Causes

1. **Outliers in recent samples:** Single extreme value triggers shift
2. **Small test sample (100):** Insufficient to distinguish outlier from shift
3. **No temporal validation:** Immediate action without waiting for confirmation
4. **MMD sensitive to outliers:** Kernel-based metrics affected by extremes

### Prevention Strategies

**Strategy 1: Outlier Removal Before Shift Detection**
```python
# Remove outliers from test samples before computing metrics
def remove_outliers(X, n_sigma=3):
    """Remove samples beyond n_sigma standard deviations"""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    
    # Compute z-scores
    z_scores = np.abs((X - mean) / (std + 1e-8))
    
    # Keep samples within n_sigma
    mask = (z_scores < n_sigma).all(axis=1)
    
    return X[mask]

X_test_clean = remove_outliers(X_test, n_sigma=3)
mmd_squared = self.compute_mmd_squared(X_train, X_test_clean)
```

**Strategy 2: Temporal Confirmation (Holdout Period)**
```python
# Require shift to persist for 24 hours before adapting
if shift_detected:
    if model_name not in self.pending_shifts:
        self.pending_shifts[model_name] = {
            'timestamp': datetime.utcnow(),
            'severity': severity
        }
        logger.info(f"Shift detected, entering 24h holdout period")
        return None  # Don't adapt yet
    else:
        # Check if 24 hours elapsed
        elapsed = (datetime.utcnow() - self.pending_shifts[model_name]['timestamp']).total_seconds()
        if elapsed < 86400:  # <24 hours
            return None  # Still in holdout
        else:
            # Confirmed shift, proceed with adaptation
            logger.info("Shift confirmed after 24h holdout")
```

**Strategy 3: Multi-Metric Consensus**
```python
# Require BOTH MMD and KL to agree (reduce false positives)
mmd_severe = mmd_squared >= self.mmd_threshold_severe
kl_severe = kl_divergence >= self.kl_threshold_severe
ks_significant = len(significant_features) >= 3

# Only trigger if 2/3 metrics agree
if sum([mmd_severe, kl_severe, ks_significant]) >= 2:
    severity = ShiftSeverity.SEVERE
else:
    severity = ShiftSeverity.MODERATE  # Downgrade
```

**Strategy 4: Rolling Window Smoothing**
```python
# Compute metrics over last 3 windows (300 trades), not just 100
windows = [
    self.compute_mmd_squared(X_train, X_recent[-300:-200]),
    self.compute_mmd_squared(X_train, X_recent[-200:-100]),
    self.compute_mmd_squared(X_train, X_recent[-100:])
]

mmd_smoothed = np.mean(windows)  # Average over 3 windows
```

### Fallback Procedures

1. **Monitor post-adaptation performance:** Track WR for 50 trades
2. **Auto-revert if WR drops:** If WR < baseline - 3pp, revert adaptation
3. **Log false positives:** Track FP rate, adjust thresholds if >20%
4. **Manual review:** Alert analysts for severe shifts

### Monitoring Metrics

- **False positive rate:** FPs / total_alerts (target <15%)
- **Shift duration:** Track how long shifts persist (flag <2 hours as suspicious)
- **Revert frequency:** Track adaptations reverted (target <10%)
- **Correlation with performance:** Check if shift alerts predict WR drops

---

## RISK 6: ADAPTATION FAILURE (CRITICAL SEVERITY, LOW PROBABILITY)

### What Can Go Wrong

**Scenario:** Severe covariate shift + adaptation fails to improve performance, losses accumulate for 3 days before escalating to retraining.

**Example:**
- Day 1: Severe shift detected (MMD²=0.068), CORAL + importance weighting applied
- Day 1-3: WR remains at 49% despite adaptation (below breakeven after fees)
- Losses: $18,000 over 3 days
- Day 3: Manual intervention, escalate to retraining
- Issue: Adaptation insufficient for this magnitude of shift

**Impact:**
- Financial: -$18,000 (3 days of poor performance)
- Operational: Emergency retraining required, system instability
- Time: 12 hours for emergency retraining + deployment

### Root Causes

1. **Concept drift (not covariate shift):** P(Y|X) changed, not just P(X)
2. **Adaptation insufficient:** Importance weighting can't fix fundamental model issues
3. **No performance gate:** System doesn't halt/escalate when adaptation fails
4. **Delayed escalation:** 3-day wait before retraining triggered

### Prevention Strategies

**Strategy 1: Immediate Performance Check (Critical)**
```python
# After adaptation, track performance for next 50 trades
async def monitor_post_adaptation_performance(self, model_name):
    """Track WR after adaptation, revert if fails"""
    baseline_wr = self.get_baseline_wr(model_name)
    
    for i in range(50):
        # Wait for trade outcome
        outcome = await self.get_next_trade_outcome(model_name)
        
        # Compute running WR
        running_wr = self.compute_running_wr(model_name, window=20)
        
        # Check if adaptation failing
        if i >= 20 and running_wr < baseline_wr - 0.05:  # 5pp drop
            logger.error(
                f"[{model_name}] Adaptation FAILED: "
                f"WR {baseline_wr:.2%} → {running_wr:.2%}, REVERTING"
            )
            await self._revert_adaptation(model_name)
            await self._escalate_to_retraining(model_name)
            break
```

**Strategy 2: Concept Drift Detection (Before Adapting)**
```python
# Check if concept drift (P(Y|X) changed) vs covariate shift (P(X) changed)
def detect_concept_drift(X_train, y_train, X_test, y_test_proxy):
    """
    Concept drift check: does model performance drop on test data?
    If yes: concept drift → retrain
    If no: covariate shift → adapt
    """
    # Train model on X_train, y_train
    model_temp = train_model(X_train, y_train)
    
    # Evaluate on test data
    preds_test = model_temp.predict(X_test)
    wr_test = (preds_test == y_test_proxy).mean()
    
    # Compare to training performance
    preds_train = model_temp.predict(X_train)
    wr_train = (preds_train == y_train).mean()
    
    if wr_test < wr_train - 0.05:  # 5pp drop
        logger.warning("Concept drift detected (performance drop), escalating to retraining")
        return True  # Concept drift
    else:
        logger.info("Pure covariate shift (no performance drop), adapting")
        return False  # Covariate shift only
```

**Strategy 3: Emergency Halt for Critical Failures**
```python
# If WR <45% for 4+ hours post-adaptation, HALT trading
if running_wr < 0.45 and hours_since_adaptation > 4:
    logger.critical(f"[{model_name}] Adaptation failed, WR={running_wr:.2%}, HALTING TRADING")
    self.trading_enabled[model_name] = False
    
    # Trigger emergency retraining
    await self._emergency_retrain(model_name)
```

**Strategy 4: A/B Testing for Risky Adaptations**
```python
# For severe shifts, A/B test adapted vs original model
if severity == 'severe':
    # Route 20% traffic to adapted model, 80% to original
    if np.random.rand() < 0.2:
        prediction = adapted_model.predict(features)
    else:
        prediction = original_model.predict(features)
    
    # After 100 trades, compare performance
    if wr_adapted > wr_original:
        # Promote adapted model to 100%
        logger.info("Adapted model validated, promoting to 100%")
    else:
        # Revert to original, trigger retraining
        logger.warning("Adapted model underperforms, reverting and retraining")
```

### Fallback Procedures

1. **Real-time WR monitoring:** Check every 20 trades (not 100)
2. **Automatic revert:** If WR <50% for 50 trades, revert adaptation
3. **Emergency halt:** If WR <45% for 4 hours, halt trading
4. **Immediate retraining:** Escalate to Module 3 with high priority

### Monitoring Metrics (CRITICAL)

- **Post-adaptation WR:** Track every 10 trades for first 100 trades
- **Alert:** WR <52% for 30+ trades
- **Auto-revert:** WR <50% for 50+ trades
- **Emergency halt:** WR <45% for 4+ hours
- **Escalation to retraining:** Auto-trigger if revert happens

---

## TESTING CHECKLIST

### Unit Tests
- [ ] Extreme weight detection (stability >20 flagged)
- [ ] CORAL validation gate (rejects transformations with WR drop >5pp)
- [ ] OOD threshold adaptation (adjusts based on historical rate)
- [ ] Outlier removal (3-sigma filter works)
- [ ] Incremental MMD update (matches full computation)

### Integration Tests
- [ ] False positive handling (whale transaction doesn't trigger adaptation)
- [ ] Adaptation failure detection (reverts if WR drops)
- [ ] Concept drift vs covariate shift differentiation
- [ ] Emergency halt triggers at WR <45%
- [ ] A/B testing for severe shifts

### Scenario Tests
- [ ] Extreme weight scenario (stability >100 handled)
- [ ] CORAL artifact scenario (spurious correlations detected)
- [ ] OOD miscalibration scenario (80% OOD flagged appropriately)
- [ ] Computational overhead scenario (latency <500ms)
- [ ] False positive scenario (temporary spike ignored)
- [ ] Adaptation failure scenario (halt + retrain triggered)

### Stress Tests
- [ ] 1000 training samples (performance OK)
- [ ] 100 features (computation <1 second)
- [ ] 10 consecutive shifts (system stable)
- [ ] 50% OOD samples (doesn't halt unnecessarily)

---

## MONITORING FRAMEWORK

### Daily Metrics
- **Importance weight stability:** P50, P95, max/mean ratio
- **Adaptation frequency:** Count adaptations/day (target 0-2)
- **OOD rate:** Percentage flagged as OOD (target <20%)
- **Post-adaptation WR:** Track for 24 hours after each adaptation
- **Computation latency:** Drift check time (target <200ms)

### Weekly Metrics
- **False positive rate:** False alarms / total alerts (target <15%)
- **Revert rate:** Adaptations reverted / total (target <10%)
- **Escalation rate:** Escalations to retraining (target 1-2/month)
- **Weight distribution:** Check for bimodal or extreme distributions

### Monthly Metrics
- **Adaptation ROI:** Savings from avoiding retraining vs adaptation cost
- **Threshold calibration:** Review thresholds based on FP/FN rates
- **Compute efficiency:** Track latency trends, optimize if increasing
- **Model health:** Correlation between adaptations and WR stability

---

## SUCCESS CRITERIA

1. **Stability score <10:** 95% of adaptations have max/mean weight ratio <10
2. **False positive rate <15%:** Less than 15% of alerts are false positives
3. **Zero adaptation failures:** No adaptations result in >5pp WR drop for >4 hours
4. **Latency <200ms:** Drift detection completes in <200ms per check
5. **OOD rate <25%:** Less than 25% of predictions flagged as OOD
6. **Escalation rate <3/month:** Fewer than 3 escalations to retraining per month

---

**Module 4 Section 5: Risk Analysis - COMPLETE ✅**

Next: Test Suite, Benefits Analysis
