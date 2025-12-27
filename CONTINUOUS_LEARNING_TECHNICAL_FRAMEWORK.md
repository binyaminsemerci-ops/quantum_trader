# CONTINUOUS LEARNING: TECHNICAL FRAMEWORK

**Module 6: Continuous Learning - Section 2**

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONTINUOUS LEARNING SYSTEM                    │
└─────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
        ┌───────▼──────┐ ┌─────▼─────┐ ┌──────▼──────┐
        │ Performance  │ │  Feature  │ │   Model     │
        │  Monitor     │ │  Tracker  │ │ Versioning  │
        └──────┬───────┘ └─────┬─────┘ └──────┬──────┘
               │               │               │
               └───────────────┼───────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Retraining Engine  │
                    │  - Trigger Logic    │
                    │  - Data Collection  │
                    │  - Model Training   │
                    └──────────┬──────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
        ┌───────▼──────┐ ┌────▼────┐ ┌──────▼──────┐
        │   Online     │ │ Shadow  │ │   Archive   │
        │  Learning    │ │ Testing │ │  Manager    │
        └──────────────┘ └─────────┘ └─────────────┘
```

---

## 1. PERFORMANCE MONITORING

### 1.1 Exponentially Weighted Moving Average (EWMA)

Track model performance decay using EWMA:

**Formula:**
```
EWMA(t) = α × Performance(t) + (1 - α) × EWMA(t-1)
```

**Where:**
- α = smoothing factor (0.1 recommended)
- Performance(t) = current metric (WR, Sharpe, PnL)
- EWMA(t-1) = previous EWMA value

**Decay Detection:**
```
Decay = Baseline_WR - EWMA_WR

Trigger Retraining if:
  Decay > threshold (3pp for WR)
```

**Example:**
```python
# Baseline (promotion): 58% WR
# Week 1: 57% → EWMA = 0.1×57 + 0.9×58 = 57.9%
# Week 2: 56% → EWMA = 0.1×56 + 0.9×57.9 = 57.71%
# Week 3: 54% → EWMA = 0.1×54 + 0.9×57.71 = 57.34%
# Week 4: 53% → EWMA = 0.1×53 + 0.9×57.34 = 56.91%

# Decay = 58% - 56.91% = 1.09pp (no trigger yet)

# Week 5: 52% → EWMA = 0.1×52 + 0.9×56.91 = 56.42%
# Decay = 58% - 56.42% = 1.58pp (no trigger)

# Week 6: 50% → EWMA = 0.1×50 + 0.9×56.42 = 55.78%
# Decay = 58% - 55.78% = 2.22pp (no trigger)

# Week 7: 49% → EWMA = 0.1×49 + 0.9×55.78 = 55.10%
# Decay = 58% - 55.10% = 2.90pp (no trigger)

# Week 8: 48% → EWMA = 0.1×48 + 0.9×55.10 = 54.39%
# Decay = 58% - 54.39% = 3.61pp 🔥 TRIGGER RETRAINING!
```

**Advantages:**
- Robust to short-term variance
- Smooths out lucky/unlucky streaks
- Clear decay threshold

---

### 1.2 CUSUM (Cumulative Sum Control Chart)

Detect sudden performance shifts faster than EWMA:

**Formula:**
```
CUSUM⁺(t) = max(0, CUSUM⁺(t-1) + x(t) - k)
CUSUM⁻(t) = max(0, CUSUM⁻(t-1) - x(t) - k)
```

**Where:**
- x(t) = current performance - baseline
- k = slack parameter (0.5 recommended)
- CUSUM⁺ = upward shift detector
- CUSUM⁻ = downward shift detector

**Trigger:**
```
if CUSUM⁻(t) > h:  # h = threshold (5.0 recommended)
    trigger_retraining()
```

**Example:**
```python
# Baseline: 58% WR, k=0.5, h=5.0

# Trade results: [Win, Win, Loss, Loss, Loss, Loss, Loss]
# Observed WR: 58%, 58%, 54%, 52%, 50%, 48%, 46%

# x(t) = observed - baseline
# Trade 1: x=0, CUSUM⁻ = 0
# Trade 2: x=0, CUSUM⁻ = 0
# Trade 3: x=-4, CUSUM⁻ = max(0, 0-(-4)-0.5) = 3.5
# Trade 4: x=-6, CUSUM⁻ = max(0, 3.5-(-6)-0.5) = 9.0 > 5.0 🔥 TRIGGER!

# CUSUM detected shift in 4 trades
# EWMA would need 8-10 trades
```

**Advantages:**
- Faster detection (50-70% faster than EWMA)
- Detects sudden shifts
- Complements EWMA (use both)

---

### 1.3 Statistical Process Control (SPC)

Monitor multiple metrics simultaneously:

**Control Limits:**
```
UCL = μ + 3σ  # Upper Control Limit
LCL = μ - 3σ  # Lower Control Limit
```

**Where:**
- μ = baseline metric mean
- σ = baseline metric standard deviation

**Out-of-Control Rules:**
- 1 point beyond 3σ (UCL/LCL)
- 2 of 3 consecutive points beyond 2σ
- 4 of 5 consecutive points beyond 1σ
- 8 consecutive points on one side of center

**Example:**
```python
# Win Rate Control Chart
μ = 58%, σ = 3%
UCL = 58% + 3×3% = 67%
LCL = 58% - 3×3% = 49%

# Recent 10 trades WR: 54%, 52%, 50%, 48%, 47%, 45%, 44%, 43%, 42%, 41%

# Trade 8: 43% < 49% (LCL) 🔥 OUT OF CONTROL!
# Action: Trigger retraining

# Sharpe Ratio Control Chart
μ = 1.85, σ = 0.25
UCL = 1.85 + 3×0.25 = 2.60
LCL = 1.85 - 3×0.25 = 1.10

# Recent Sharpe: 1.80, 1.75, 1.60, 1.50, 1.40, 1.30, 1.20, 1.05 < LCL 🔥
```

---

## 2. FEATURE IMPORTANCE TRACKING

### 2.1 SHAP (SHapley Additive exPlanations)

Measure each feature's contribution to model predictions:

**Formula:**
```
φᵢ(f) = Σ [|S|!·(|N|-|S|-1)! / |N|!] · [f(S∪{i}) - f(S)]
      S⊆N\{i}
```

**Where:**
- φᵢ(f) = SHAP value for feature i
- N = set of all features
- S = subset of features
- f(S) = model prediction with features S

**Interpretation:**
- φᵢ > 0: Feature increases prediction
- φᵢ < 0: Feature decreases prediction
- |φᵢ| = feature importance

**Example:**
```python
# XGBoost prediction for BTCUSDT LONG

Features:
  - RSI: 65 → φ_RSI = +0.05 (bullish)
  - MACD: -0.02 → φ_MACD = -0.03 (bearish)
  - Volume: 1.2M → φ_Volume = +0.08 (bullish)
  - OB_Imbalance: 0.15 → φ_OB = +0.12 (bullish)

Prediction = base_value + Σφᵢ
           = 0.50 + (0.05 - 0.03 + 0.08 + 0.12)
           = 0.50 + 0.22
           = 0.72 (72% confidence LONG)

Feature Ranking by |φᵢ|:
  1. Order Book Imbalance: 0.12
  2. Volume: 0.08
  3. RSI: 0.05
  4. MACD: 0.03
```

---

### 2.2 Feature Drift Detection

Monitor feature importance changes over time:

**Jensen-Shannon Divergence:**
```
D_JS(P || Q) = 0.5 · D_KL(P || M) + 0.5 · D_KL(Q || M)

where M = 0.5 · (P + Q)
```

**Where:**
- P = baseline feature importance distribution
- Q = current feature importance distribution
- D_KL = Kullback-Leibler divergence

**Trigger:**
```
if D_JS > 0.3:
    # Significant feature shift
    trigger_retraining()
```

**Example:**
```python
# Month 1 (Baseline):
P = {
    'RSI': 0.30,
    'MACD': 0.25,
    'Volume': 0.20,
    'ATR': 0.15,
    'OB': 0.10
}

# Month 3 (Current):
Q = {
    'RSI': 0.15,      # Dropped 50%
    'MACD': 0.10,     # Dropped 60%
    'Volume': 0.15,   # Dropped 25%
    'ATR': 0.10,      # Dropped 33%
    'OB': 0.50        # Up 5x! 🔥
}

# Calculate D_JS:
M = {
    'RSI': 0.225,
    'MACD': 0.175,
    'Volume': 0.175,
    'ATR': 0.125,
    'OB': 0.30
}

D_KL(P || M) = Σ P(i) · log(P(i) / M(i))
D_KL(Q || M) = Σ Q(i) · log(Q(i) / M(i))

D_JS ≈ 0.42 > 0.3 🔥 FEATURE DRIFT DETECTED!
Action: Retrain with emphasis on Order Book features
```

---

### 2.3 Incremental Feature Importance

Update feature importance without full recomputation:

**Exponential Moving Average Update:**
```
Importance_new(i) = α · Importance_current(i) + (1-α) · Importance_old(i)
```

**Where:**
- α = learning rate (0.01 - 0.05)
- Importance_current(i) = SHAP value from latest trade
- Importance_old(i) = previous average importance

**Example:**
```python
# Feature: RSI
# Previous avg importance: 0.30

# Trade 1: SHAP_RSI = 0.05
# Importance_new = 0.05 × 0.05 + 0.95 × 0.30 = 0.2875

# Trade 2: SHAP_RSI = 0.08
# Importance_new = 0.05 × 0.08 + 0.95 × 0.2875 = 0.2771

# Trade 100: Average ≈ 0.15 (dropped from 0.30)
# Feature RSI losing importance over time
```

---

## 3. AUTOMATED RETRAINING

### 3.1 Retraining Trigger Logic

**Multi-Criterion Trigger:**
```
Trigger Retraining if ANY:
  1. EWMA decay > 3pp (WR)
  2. CUSUM⁻ > 5.0
  3. SPC out-of-control (8+ consecutive below center)
  4. Feature drift D_JS > 0.3
  5. Scheduled monthly retrain
  6. Manual trigger
```

**Combined Score:**
```
Urgency Score = w1·EWMA_decay + w2·CUSUM + w3·SPC_violations + w4·D_JS

Thresholds:
  Score > 10: 🔥 CRITICAL (retrain immediately)
  Score > 5:  ⚠️  WARNING (retrain within 24h)
  Score > 2:  ℹ️  NOTICE (schedule retrain)
  Score ≤ 2:  ✅ HEALTHY (no action)
```

**Example:**
```python
# Week 8 Metrics:
EWMA_decay = 3.61pp  → w1 = 3.0, contribution = 10.83
CUSUM⁻ = 4.2        → w2 = 1.0, contribution = 4.2
SPC_violations = 2  → w3 = 0.5, contribution = 1.0
D_JS = 0.25         → w4 = 10, contribution = 2.5

Urgency Score = 10.83 + 4.2 + 1.0 + 2.5 = 18.53 > 10 🔥 CRITICAL!

Action: Trigger retraining IMMEDIATELY
```

---

### 3.2 Training Data Window Selection

**Sliding Window Approach:**
```
Data = [Trade_(t-n), Trade_(t-n+1), ..., Trade_(t-1), Trade_t]
```

**Where:**
- n = window size (10,000 trades recommended)
- t = current time

**Window Size Selection:**
```
Optimal Window = arg max E[Performance(model_trained_on_window)]
                   n

Trade-off:
  Small n (1,000):  Fast, adapts quickly, high variance
  Medium n (10,000): Balanced (recommended)
  Large n (50,000):  Slow, stable, may include stale data
```

**Example:**
```python
# Current: November 2025, 15,000 trades total

# Window 1 (Last 10K): October-November data
# Contains: Post-halving volatility, current market regime
# Performance: 61% WR

# Window 2 (Last 5K): November only
# Contains: Very recent data, high variance
# Performance: 59% WR (less stable)

# Window 3 (All 15K): August-November
# Contains: Pre-halving + post-halving mixed
# Performance: 57% WR (stale patterns)

# Optimal: Window 1 (10K trades) ✅
```

---

### 3.3 Retraining Algorithm

**Procedure:**
```
1. Fetch Data:
   - Last 10,000 trades
   - Features: Technical indicators + order book + sentiment
   - Labels: Win/Loss + PnL

2. Feature Engineering:
   - Update feature importance weights
   - Add new features (if drift detected)
   - Remove obsolete features (importance < 5%)

3. Train Model:
   - XGBoost with updated hyperparameters
   - 5-fold cross-validation
   - Early stopping (patience=50)

4. Evaluate:
   - Holdout test set (20% of data)
   - Metrics: WR, Sharpe, Sortino, MDD, Calmar

5. Version Control:
   - Save as model_v{version}
   - Metadata: timestamp, metrics, features used

6. Shadow Test:
   - Deploy as challenger (Module 5)
   - Test 500 trades (0% allocation)
   - Compare to champion

7. Promote (if better):
   - Statistical tests passed
   - Shadow test WR > champion WR
   - Promotion score ≥ 70/100
```

**Hyperparameter Optimization:**
```
# XGBoost Hyperparameters (auto-tuned)

# Learning rate decay:
lr(epoch) = lr_initial × decay_factor^(epoch / decay_steps)

# Optimal range (Bayesian optimization):
params = {
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 10],
    'n_estimators': [100, 1000],
    'subsample': [0.6, 1.0],
    'colsample_bytree': [0.6, 1.0],
    'min_child_weight': [1, 10],
    'gamma': [0, 5]
}

# Objective: Maximize Sharpe Ratio (not just accuracy)
```

---

## 4. ONLINE LEARNING

### 4.1 Stochastic Gradient Descent (SGD) Update

Update model weights with each new trade:

**Formula:**
```
θ_{t+1} = θ_t - η · ∇L(θ_t, x_t, y_t)
```

**Where:**
- θ_t = model weights at time t
- η = learning rate (0.001 - 0.01)
- ∇L = gradient of loss function
- x_t = features from trade t
- y_t = outcome (win/loss)

**Loss Function (Binary Cross-Entropy):**
```
L(θ, x, y) = -[y·log(ŷ) + (1-y)·log(1-ŷ)]

where ŷ = σ(θᵀx) = 1 / (1 + e^(-θᵀx))
```

**Example:**
```python
# Current weights: θ = [0.5, 0.3, 0.2, 0.1]
# Trade features: x = [65, -0.02, 1.2M, 0.15] (RSI, MACD, Vol, OB)
# Prediction: ŷ = σ(θᵀx) = 0.72 (72% confidence LONG)
# Actual: y = 1 (WIN)

# Compute gradient:
∇L = (ŷ - y) · x
   = (0.72 - 1.0) · [65, -0.02, 1.2, 0.15]
   = -0.28 · [65, -0.02, 1.2, 0.15]
   = [-18.2, 0.0056, -0.336, -0.042]

# Update weights:
θ_new = θ - 0.01 · ∇L
      = [0.5, 0.3, 0.2, 0.1] - 0.01 · [-18.2, 0.0056, -0.336, -0.042]
      = [0.682, 0.2999, 0.2034, 0.1004]

# RSI weight increased (good predictor in this case)
```

---

### 4.2 Momentum-Based Online Learning

Add momentum to SGD for faster convergence:

**Formula:**
```
v_{t+1} = β · v_t + (1-β) · ∇L(θ_t, x_t, y_t)
θ_{t+1} = θ_t - η · v_{t+1}
```

**Where:**
- v_t = velocity (momentum term)
- β = momentum coefficient (0.9 recommended)

**Advantages:**
- Faster convergence
- Smooths out noisy gradients
- Escapes local minima

**Example:**
```python
# Trade 1:
∇L_1 = [-18.2, 0.0056, -0.336, -0.042]
v_1 = 0.9 × [0, 0, 0, 0] + 0.1 × ∇L_1 = [-1.82, 0.00056, -0.0336, -0.0042]
θ_1 = θ_0 - 0.01 × v_1 = [0.5182, 0.29999, 0.20034, 0.10004]

# Trade 2:
∇L_2 = [-15.0, 0.002, -0.28, -0.035]
v_2 = 0.9 × v_1 + 0.1 × ∇L_2 = [-3.138, 0.00070, -0.0582, -0.00728]
θ_2 = θ_1 - 0.01 × v_2 = [0.54958, 0.29999, 0.20092, 0.10011]

# Momentum accelerates learning in consistent direction (RSI)
```

---

### 4.3 Adaptive Learning Rate (Adam)

Adjust learning rate per parameter:

**Formula:**
```
m_t = β1 · m_{t-1} + (1-β1) · ∇L
v_t = β2 · v_{t-1} + (1-β2) · (∇L)²

m̂_t = m_t / (1 - β1^t)
v̂_t = v_t / (1 - β2^t)

θ_{t+1} = θ_t - η · m̂_t / (√v̂_t + ε)
```

**Where:**
- m_t = first moment (mean)
- v_t = second moment (variance)
- β1 = 0.9, β2 = 0.999 (recommended)
- ε = 10^-8 (numerical stability)

**Advantages:**
- Parameter-specific learning rates
- Works well with sparse gradients
- Converges faster than SGD

---

### 4.4 Online Learning Safety

**Constraints:**
```
1. Maximum Weight Change per Update:
   |θ_new - θ_old| < max_delta (0.1 recommended)

2. Regularization:
   L_total = L_prediction + λ · ||θ||²
   
   where λ = 0.01 (L2 regularization)

3. Validation Check:
   if Performance_after_update < Performance_before - threshold:
       rollback_weights()

4. Update Frequency:
   Update every N trades (10-100 recommended)
   Not every single trade (too noisy)
```

---

## 5. MODEL VERSIONING

### 5.1 Version Control Schema

**Git-like versioning:**
```
model_v1.0.0  → Initial deployment
model_v1.1.0  → First retrain (minor update)
model_v1.1.1  → Online learning checkpoint
model_v1.2.0  → Second retrain (feature drift fix)
model_v2.0.0  → Major architecture change
```

**Semantic Versioning:**
```
MAJOR.MINOR.PATCH

MAJOR: Breaking changes (new architecture)
MINOR: Retrain with new data
PATCH: Online learning updates
```

---

### 5.2 Model Metadata

**Stored with each version:**
```json
{
  "version": "1.2.3",
  "timestamp": "2025-11-26T04:00:00Z",
  "training_data": {
    "start_date": "2025-10-01",
    "end_date": "2025-11-25",
    "n_trades": 10000,
    "symbols": ["BTCUSDT", "ETHUSDT", ...]
  },
  "metrics": {
    "train": {"wr": 0.62, "sharpe": 2.1},
    "test": {"wr": 0.59, "sharpe": 1.95},
    "production": {"wr": 0.58, "sharpe": 1.85}
  },
  "features": {
    "RSI": 0.15,
    "MACD": 0.10,
    "Volume": 0.15,
    "OrderBook": 0.50,
    "ATR": 0.10
  },
  "hyperparameters": {
    "learning_rate": 0.05,
    "max_depth": 6,
    "n_estimators": 500
  },
  "parent_version": "1.2.2",
  "retrain_reason": "Feature drift (D_JS=0.42)"
}
```

---

### 5.3 Rollback Strategy

**Three-level rollback:**
```
Level 1: Instant Rollback (<30s)
  - Keep champion always in memory
  - Archive last 3 versions
  - Swap pointer instantly

Level 2: Recent Rollback (1-5 min)
  - Load from disk
  - Last 10 versions stored
  - Verify checksum

Level 3: Historical Rollback (5-15 min)
  - Load from S3/archive
  - All versions since v1.0
  - Rebuild if needed
```

---

## 6. PERFORMANCE BENCHMARKS

### Target Latencies:
- Performance monitoring: <10ms per trade
- SHAP computation: <50ms per prediction
- Online update: <100ms per trade
- Retraining: 30-60 minutes (full retrain)
- Shadow testing: 2-3 days (500 trades)
- Model deployment: <30 seconds

### Resource Requirements:
- Memory: 2-4 GB (model + data)
- CPU: 4 cores (online learning)
- GPU: 1x for retraining (optional)
- Storage: 50-100 GB (model versions + data)

---

## 7. INTEGRATION WITH PREVIOUS MODULES

### Module 1: Memory States
```python
# Retraining includes all memory states
data = fetch_trades(n=10000, include_memory_states=True)

# Online learning updates state-action values
if state == 'volatile_bullish':
    update_weights(state_features, outcome)
```

### Module 2: Reinforcement Signals
```python
# Feature importance tracks reward signals
feature_importance['reward_sharpe'] = 0.25
feature_importance['reward_win_rate'] = 0.20

# Retraining optimizes for reward function
loss = -reward_function(predictions, outcomes)
```

### Module 3: Drift Detection
```python
# Drift detector triggers retraining
if drift_detector.check_drift():
    retraining_engine.trigger(reason='drift_detected')
```

### Module 4: Covariate Shift
```python
# Feature drift detection = covariate shift detection
D_JS_features = compute_js_divergence(P_old, P_new)
if D_JS_features > 0.3:
    retrain(emphasize_new_covariates=True)
```

### Module 5: Shadow Models
```python
# Every retrained model = new challenger
new_model = retrain(data)
shadow_manager.register_model(
    model=new_model,
    role=ModelRole.CHALLENGER
)

# Promote only if proven better
if shadow_test_passed(new_model):
    shadow_manager.promote_challenger(new_model.name)
```

---

**Module 6 Section 2: Technical Framework - COMPLETE ✅**

Next: Implementation (Python code)
