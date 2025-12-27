# COVARIATE SHIFT HANDLING: TECHNICAL FRAMEWORK

**Module 4: Covariate Shift Handling - Section 2**

## Overview

**Covariate shift** occurs when the distribution of input features P(X) changes between training and deployment, while the conditional distribution P(Y|X) remains stable. Unlike concept drift (where P(Y|X) changes), covariate shift means the model's learned patterns are still valid, but it encounters feature values outside its training distribution.

**Mathematical Definition:**

Training distribution: P_train(X, Y) = P_train(X) · P(Y|X)
Test distribution: P_test(X, Y) = P_test(X) · P(Y|X)

Covariate shift: P_train(X) ≠ P_test(X), but P_train(Y|X) = P_test(Y|X)

**Key Challenge:** Model performs poorly on P_test(X) despite valid patterns because:
1. Out-of-distribution (OOD) predictions have high uncertainty
2. Training samples with features similar to P_test(X) were under-represented
3. Model confidence calibration was based on P_train(X)

**Solution Approach:** Adapt the model to P_test(X) without full retraining by:
1. Re-weighting training samples to match P_test(X)
2. Applying domain adaptation transforms
3. Calibrating confidence for OOD predictions

---

## 1. DISTRIBUTION DIVERGENCE DETECTION

### 1.1 Maximum Mean Discrepancy (MMD)

MMD measures the distance between two distributions in a reproducing kernel Hilbert space (RKHS).

**Formula:**

```
MMD²(P_train, P_test) = ||μ_train - μ_test||²_H
```

Where:
- μ_train = E_{x~P_train}[φ(x)] (mean embedding in RKHS)
- μ_test = E_{x~P_test}[φ(x)]
- φ(x) = kernel feature map

**Empirical Estimator:**

```python
MMD²(X_train, X_test) = (1/n²) Σ_i,j k(x_i, x_j) 
                         + (1/m²) Σ_i,j k(x'_i, x'_j)
                         - (2/nm) Σ_i,j k(x_i, x'_j)
```

Where:
- X_train = {x_1, ..., x_n} (training samples)
- X_test = {x'_1, ..., x'_m} (test samples)
- k(·, ·) = kernel function (e.g., RBF: k(x, y) = exp(-||x-y||²/2σ²))

**Threshold:**
- MMD² < 0.01: No significant shift
- 0.01 ≤ MMD² < 0.05: Moderate shift
- MMD² ≥ 0.05: Severe shift

### 1.2 Kullback-Leibler Divergence (KL Divergence)

For continuous distributions, KL divergence quantifies information loss when using P_train to approximate P_test.

**Formula:**

```
D_KL(P_test || P_train) = ∫ P_test(x) log(P_test(x) / P_train(x)) dx
```

**Monte Carlo Estimator:**

```python
D_KL ≈ (1/m) Σ_i log(p_test(x'_i) / p_train(x'_i))
```

Where:
- p_test(x), p_train(x) estimated via kernel density estimation (KDE)

**Properties:**
- D_KL ≥ 0 (always non-negative)
- D_KL = 0 iff P_test = P_train (identical distributions)
- Asymmetric: D_KL(P||Q) ≠ D_KL(Q||P)

**Threshold:**
- D_KL < 0.1: No significant shift
- 0.1 ≤ D_KL < 0.5: Moderate shift
- D_KL ≥ 0.5: Severe shift

### 1.3 Kolmogorov-Smirnov Test (Per Feature)

For univariate feature distributions, KS-test detects shifts.

**Formula:**

```
D_n,m = sup_x |F_train(x) - F_test(x)|
```

Where:
- F_train(x), F_test(x) = empirical CDFs

**Statistical Test:**
- H_0: P_train = P_test (no shift)
- H_1: P_train ≠ P_test (shift exists)
- Reject H_0 if p-value < α (typically α = 0.01)

**Advantage:** Detects shifts in individual features (interpretable)

---

## 2. IMPORTANCE WEIGHTING METHODS

### 2.1 Kernel Mean Matching (KMM)

KMM finds weights β_i for training samples such that the weighted training distribution matches the test distribution in RKHS.

**Objective:**

```
minimize_β ||Σ_i β_i φ(x_i) - (1/m) Σ_j φ(x'_j)||²_H

subject to:
  β_i ∈ [0, B]  (bounded weights)
  |Σ_i β_i - n| ≤ nε  (sum constraint)
```

Where:
- β_i = importance weight for training sample x_i
- B = upper bound (prevents extreme weights, typically B = 1000)
- ε = tolerance (typically ε = 0.1)

**Solution (Quadratic Programming):**

Reformulate as:
```
minimize_β  (1/2) β^T K β - κ^T β

subject to:
  0 ≤ β_i ≤ B
  (1-ε)n ≤ 1^T β ≤ (1+ε)n
```

Where:
- K = kernel Gram matrix, K_ij = k(x_i, x_j) (n × n)
- κ = vector, κ_i = (n/m) Σ_j k(x_i, x'_j) (n × 1)

**Algorithm:**

```python
def kernel_mean_matching(X_train, X_test, kernel='rbf', B=1000, eps=0.1):
    """
    Compute importance weights via KMM
    
    Returns:
        beta: Array of shape (n_train,) with importance weights
    """
    n = X_train.shape[0]
    m = X_test.shape[0]
    
    # Compute kernel matrices
    K = compute_kernel_matrix(X_train, X_train, kernel)  # n × n
    kappa = (n / m) * compute_kernel_matrix(X_train, X_test, kernel).sum(axis=1)  # n × 1
    
    # Quadratic programming
    P = K  # Quadratic term
    q = -kappa  # Linear term
    
    # Constraints: 0 ≤ β ≤ B, (1-ε)n ≤ Σβ ≤ (1+ε)n
    G = np.vstack([np.eye(n), -np.eye(n), np.ones(n), -np.ones(n)])
    h = np.hstack([B * np.ones(n), np.zeros(n), (1+eps)*n, -(1-eps)*n])
    
    # Solve QP
    beta = solve_qp(P, q, G, h)
    
    return beta
```

**Properties:**
- Weights satisfy E_{P_train}[β(x)] ≈ 1 (unbiased)
- Bounded weights (0 ≤ β_i ≤ B) prevent instability
- Computationally efficient (O(n³) for QP solver)

### 2.2 Kullback-Leibler Importance Estimation Procedure (KLIEP)

KLIEP directly estimates the density ratio w(x) = P_test(x) / P_train(x) by minimizing KL divergence.

**Objective:**

```
minimize_α  D_KL(P_test || w_α · P_train)
          = - ∫ P_test(x) log w_α(x) dx

subject to:
  ∫ w_α(x) P_train(x) dx = 1  (normalization)
  w_α(x) ≥ 0  (non-negativity)
```

**Parametric Model:**

```
w_α(x) = Σ_j α_j k(x, x'_j)
```

Where:
- α_j = learnable coefficients
- k(·, ·) = kernel function
- {x'_j} = test samples (or subset as "centers")

**Solution (Lagrangian):**

```
maximize_α  (1/m) Σ_i log(Σ_j α_j k(x'_i, x'_j))

subject to:
  (1/n) Σ_i Σ_j α_j k(x_i, x'_j) = 1
  α_j ≥ 0
```

**Algorithm:**

```python
def kliep(X_train, X_test, kernel='rbf', n_centers=100):
    """
    Kullback-Leibler Importance Estimation Procedure
    
    Returns:
        alpha: Coefficients for density ratio model
        centers: Kernel centers (subset of X_test)
    """
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    
    # Select kernel centers (subset of test samples)
    centers = X_test[np.random.choice(n_test, n_centers, replace=False)]
    
    # Compute kernel matrices
    K_train = compute_kernel_matrix(X_train, centers, kernel)  # n_train × n_centers
    K_test = compute_kernel_matrix(X_test, centers, kernel)    # n_test × n_centers
    
    # Solve optimization (gradient ascent)
    alpha = np.ones(n_centers) / n_centers  # Initialize
    
    for iteration in range(100):
        # Compute density ratio
        w_test = K_test @ alpha  # n_test × 1
        w_test = np.maximum(w_test, 1e-8)  # Avoid log(0)
        
        # Gradient
        grad = K_test.T @ (1 / w_test) / n_test  # Maximize log-likelihood
        
        # Constraint gradient (normalization)
        constraint_grad = K_train.T @ np.ones(n_train) / n_train
        
        # Lagrangian step
        alpha += 0.01 * (grad - constraint_grad * (K_train.T @ np.ones(n_train) @ alpha / n_train - 1))
        alpha = np.maximum(alpha, 0)  # Non-negativity
        
        # Normalize
        alpha /= (K_train.T @ np.ones(n_train) @ alpha / n_train)
    
    return alpha, centers

def compute_density_ratio(X, alpha, centers, kernel='rbf'):
    """Compute w(x) = P_test(x) / P_train(x)"""
    K = compute_kernel_matrix(X, centers, kernel)
    w = K @ alpha
    return np.maximum(w, 0)  # Non-negative
```

**Properties:**
- Directly estimates density ratio (no intermediate density estimation)
- KL divergence minimization (principled objective)
- Unbounded weights (may have high variance)

### 2.3 Logistic Regression Discriminator

Train a binary classifier to distinguish training vs test samples. The probability ratio gives importance weights.

**Approach:**

1. Create labeled dataset: {(x_i, y_i = 0)} for x_i ∈ X_train, {(x'_j, y_j = 1)} for x'_j ∈ X_test
2. Train logistic regression: P(y=1|x) = σ(θ^T x)
3. Compute density ratio: w(x) = P(y=1|x) / P(y=0|x) = P(y=1|x) / (1 - P(y=1|x))

**Formula:**

```
w(x) = P_test(x) / P_train(x) 
     ≈ [P(y=1|x) / P(y=1)] / [P(y=0|x) / P(y=0)]
     = [P(y=1|x) / (m/(n+m))] / [(1 - P(y=1|x)) / (n/(n+m))]
     = (n/m) · P(y=1|x) / (1 - P(y=1|x))
```

**Algorithm:**

```python
from sklearn.linear_model import LogisticRegression

def discriminator_importance_weights(X_train, X_test):
    """
    Estimate importance weights via logistic regression discriminator
    
    Returns:
        weights_train: Importance weights for training samples
    """
    n = X_train.shape[0]
    m = X_test.shape[0]
    
    # Create binary classification dataset
    X = np.vstack([X_train, X_test])
    y = np.hstack([np.zeros(n), np.ones(m)])
    
    # Train logistic regression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    
    # Predict probabilities
    p_y1 = clf.predict_proba(X_train)[:, 1]  # P(y=1|x) for training samples
    
    # Compute density ratio
    weights = (n / m) * p_y1 / (1 - p_y1 + 1e-8)
    
    # Clip extreme weights
    weights = np.clip(weights, 0.1, 10)
    
    return weights
```

**Properties:**
- Simple and fast (logistic regression is efficient)
- Interpretable (feature coefficients show which features differ)
- May underperform for high-dimensional data (linear decision boundary)

---

## 3. DOMAIN ADAPTATION TRANSFORMS

### 3.1 Feature Standardization

Normalize features to have zero mean and unit variance based on test distribution.

**Formula:**

```
x_adapted = (x - μ_test) / σ_test
```

Where:
- μ_test = E_{x~P_test}[x] (test mean)
- σ_test = sqrt(Var_{x~P_test}[x]) (test std)

**Algorithm:**

```python
def standardize_to_test_distribution(X_train, X_test):
    """
    Standardize training data to match test distribution
    
    Returns:
        X_train_adapted: Standardized training features
    """
    mu_test = X_test.mean(axis=0)
    sigma_test = X_test.std(axis=0)
    
    X_train_adapted = (X_train - mu_test) / (sigma_test + 1e-8)
    
    return X_train_adapted
```

**Use Case:** When feature scales differ (e.g., volatility increased 3x)

### 3.2 Quantile Transformation

Map training feature quantiles to match test feature quantiles.

**Algorithm:**

```python
from sklearn.preprocessing import QuantileTransformer

def quantile_transform_to_test(X_train, X_test):
    """
    Transform training data quantiles to match test quantiles
    
    Returns:
        X_train_adapted: Quantile-transformed training features
    """
    # Fit on test distribution
    qt = QuantileTransformer(output_distribution='normal')
    qt.fit(X_test)
    
    # Transform training data
    X_train_adapted = qt.transform(X_train)
    
    return X_train_adapted
```

**Use Case:** When feature distributions have different shapes (e.g., skewness changes)

### 3.3 CORAL (Correlation Alignment)

Align second-order statistics (covariance) between training and test distributions.

**Objective:**

```
minimize_A  ||Cov(A · X_train) - Cov(X_test)||²_F
```

Where:
- A = linear transformation matrix
- ||·||_F = Frobenius norm

**Solution:**

```
A = Σ_test^(1/2) Σ_train^(-1/2)
```

Where:
- Σ_train = Cov(X_train)
- Σ_test = Cov(X_test)

**Algorithm:**

```python
def coral_transform(X_train, X_test):
    """
    Correlation Alignment (CORAL)
    
    Returns:
        X_train_adapted: CORAL-transformed training features
        A: Transformation matrix
    """
    # Compute covariance matrices
    Sigma_train = np.cov(X_train.T)
    Sigma_test = np.cov(X_test.T)
    
    # Compute transformation matrix
    # A = Σ_test^(1/2) Σ_train^(-1/2)
    U_train, S_train, Vt_train = np.linalg.svd(Sigma_train)
    Sigma_train_inv_sqrt = U_train @ np.diag(1 / np.sqrt(S_train + 1e-8)) @ Vt_train
    
    U_test, S_test, Vt_test = np.linalg.svd(Sigma_test)
    Sigma_test_sqrt = U_test @ np.diag(np.sqrt(S_test)) @ Vt_test
    
    A = Sigma_test_sqrt @ Sigma_train_inv_sqrt
    
    # Transform training data
    X_train_adapted = X_train @ A.T
    
    return X_train_adapted, A
```

**Properties:**
- Aligns feature correlations (second-order statistics)
- Linear transformation (fast, interpretable)
- Effective when correlations change (e.g., volatility co-movement shifts)

---

## 4. CONFIDENCE CALIBRATION FOR OOD PREDICTIONS

### 4.1 Uncertainty Quantification

Reduce confidence for out-of-distribution (OOD) samples.

**Approach:**

1. Compute distance to training distribution (e.g., Mahalanobis distance)
2. Scale confidence inversely with distance

**Mahalanobis Distance:**

```
D_M(x) = sqrt((x - μ)^T Σ^(-1) (x - μ))
```

Where:
- μ = mean of training distribution
- Σ = covariance of training distribution

**Confidence Adjustment:**

```
confidence_adjusted = confidence_original · exp(-λ · D_M(x))
```

Where:
- λ = decay rate (hyperparameter, typically λ = 0.1)

**Algorithm:**

```python
def calibrate_ood_confidence(X, predictions, confidences, X_train):
    """
    Calibrate confidence for OOD predictions
    
    Returns:
        confidences_adjusted: Calibrated confidence scores
        ood_scores: OOD scores (0 = in-distribution, 1 = OOD)
    """
    # Compute Mahalanobis distance
    mu = X_train.mean(axis=0)
    Sigma = np.cov(X_train.T)
    Sigma_inv = np.linalg.inv(Sigma + np.eye(Sigma.shape[0]) * 1e-6)
    
    D_M = np.array([
        np.sqrt((x - mu).T @ Sigma_inv @ (x - mu))
        for x in X
    ])
    
    # Normalize to [0, 1]
    D_M_norm = (D_M - D_M.min()) / (D_M.max() - D_M.min() + 1e-8)
    
    # Adjust confidence
    lambda_decay = 0.1
    confidences_adjusted = confidences * np.exp(-lambda_decay * D_M_norm)
    
    return confidences_adjusted, D_M_norm
```

### 4.2 Ensemble Disagreement

Use ensemble variance as OOD indicator. High variance → uncertain → reduce confidence.

**Algorithm:**

```python
def calibrate_via_ensemble_disagreement(predictions_ensemble, confidences):
    """
    Calibrate confidence based on ensemble disagreement
    
    Args:
        predictions_ensemble: Array of shape (n_models, n_samples)
        confidences: Array of shape (n_samples,)
    
    Returns:
        confidences_adjusted: Calibrated confidence scores
    """
    # Compute ensemble variance (disagreement)
    ensemble_variance = predictions_ensemble.var(axis=0)
    
    # Normalize to [0, 1]
    variance_norm = (ensemble_variance - ensemble_variance.min()) / (
        ensemble_variance.max() - ensemble_variance.min() + 1e-8
    )
    
    # Adjust confidence (higher variance → lower confidence)
    confidences_adjusted = confidences * (1 - variance_norm)
    
    return confidences_adjusted
```

---

## 5. COVARIATE SHIFT HANDLER ARCHITECTURE

### 5.1 System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                  Covariate Shift Handler                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Distribution Divergence Detector                            │
│     ├─ MMD Calculator                                           │
│     ├─ KL Divergence Estimator                                  │
│     └─ Per-Feature KS Tests                                     │
│                                                                 │
│  2. Importance Weight Estimator                                 │
│     ├─ Kernel Mean Matching (KMM)                               │
│     ├─ KLIEP                                                    │
│     └─ Discriminator (Logistic Regression)                      │
│                                                                 │
│  3. Domain Adaptation Module                                    │
│     ├─ Feature Standardization                                  │
│     ├─ Quantile Transformation                                  │
│     └─ CORAL Transform                                          │
│                                                                 │
│  4. OOD Confidence Calibrator                                   │
│     ├─ Mahalanobis Distance                                     │
│     └─ Ensemble Disagreement                                    │
│                                                                 │
│  5. Adaptive Strategy Selector                                  │
│     └─ Choose adaptation method based on shift severity         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Adaptation Strategy Selection

**Decision Tree:**

```
IF MMD² < 0.01:
    → No adaptation needed
    
ELIF 0.01 ≤ MMD² < 0.05:
    → Moderate shift
    → Apply importance weighting (KMM or KLIEP)
    → Calibrate OOD confidence (Mahalanobis)
    
ELIF MMD² ≥ 0.05:
    → Severe shift
    → Check if P(Y|X) changed:
        IF performance drop > 5pp:
            → Concept drift detected (trigger retraining via Module 3)
        ELSE:
            → Pure covariate shift
            → Apply domain adaptation (CORAL or Quantile Transform)
            → Apply importance weighting
            → Calibrate OOD confidence
```

### 5.3 Integration with Existing Modules

**Module 1 (Memory States):**
- Covariate shift detection per regime (TRENDING/RANGING/VOLATILE)
- Importance weights adjusted for regime-specific distributions

**Module 2 (Reinforcement Learning):**
- RL reward signal includes covariate shift severity (penalty for OOD predictions)
- RL ensemble weights reduced for models with high OOD scores

**Module 3 (Drift Detection):**
- Covariate shift handler activates BEFORE drift detection triggers retraining
- If adaptation fails (performance still drops), escalate to retraining
- Complementary: Covariate shift (fast adaptation) + Drift detection (full retraining)

---

## 6. PERFORMANCE CONSIDERATIONS

### 6.1 Computational Complexity

| Method | Time Complexity | Space Complexity | Speed |
|--------|----------------|------------------|-------|
| MMD | O(n² + m²) | O(n²) (kernel matrix) | Slow |
| KL Divergence (KDE) | O(n·m) | O(n+m) | Medium |
| KS Test | O(n log n) | O(n) | Fast |
| KMM | O(n³) (QP solver) | O(n²) | Slow |
| KLIEP | O(k·n) (k=centers) | O(k·n) | Medium |
| Discriminator | O(n·d) | O(d) | Fast |
| CORAL | O(d³) (SVD) | O(d²) | Fast |

**Recommendations:**
- For n > 1000: Use KLIEP or Discriminator (avoid KMM)
- For d > 50: Use Discriminator (avoid CORAL)
- Real-time: Use KS test + Discriminator (fastest)

### 6.2 Update Frequency

**Continuous Monitoring:**
- Check for covariate shift every 100 trades (same as drift detection)
- Compute MMD incrementally (avoid full recalculation)

**Incremental MMD Update:**

```python
# Maintain running kernel sums
K_train_sum += k(x_new, X_train).sum()
K_test_sum += k(x_new, X_test).sum()

# Update MMD²
MMD_squared = (K_train_sum / n²) + (K_test_sum / m²) - (2 * K_cross_sum / (n*m))
```

---

## 7. EXAMPLE WORKFLOW

**Scenario:** Volatility increases from 3% to 7% (covariate shift)

**Step 1: Detect Shift**
```
MMD²(X_train, X_recent) = 0.042 (MODERATE shift)
KL(P_recent || P_train) = 0.28
Feature KS tests:
  - volatility: p-value = 0.001 (SIGNIFICANT)
  - rsi_14: p-value = 0.15 (not significant)
```

**Step 2: Select Adaptation Strategy**
```
Shift severity: MODERATE
Performance drop: 2pp (not severe)
→ Strategy: Importance weighting (KLIEP) + OOD calibration
```

**Step 3: Compute Importance Weights**
```
alpha, centers = kliep(X_train, X_recent)
weights_train = compute_density_ratio(X_train, alpha, centers)

Mean weight: 1.02 (balanced)
Max weight: 8.4 (some training samples highly relevant)
Min weight: 0.12 (some samples downweighted)
```

**Step 4: Re-weight Training Data**
```
# Apply weights to loss function during prediction
for sample in X_train:
    weighted_loss = weight[i] * loss(y_pred, y_true)
```

**Step 5: Calibrate OOD Confidence**
```
D_M = mahalanobis_distance(X_recent, X_train)
confidence_adjusted = confidence_original * exp(-0.1 * D_M)

Before: confidence = 0.68 (overconfident on OOD data)
After: confidence = 0.54 (calibrated for uncertainty)
```

**Step 6: Monitor Performance**
```
Win rate after adaptation: 57% (maintained, vs 55% without adaptation)
OOD score: 0.42 (moderate, flagged for monitoring)
```

---

## 8. SUCCESS METRICS

### 8.1 Adaptation Effectiveness

**Primary Metrics:**
- **Win rate preservation:** Maintain within 2pp of baseline after shift
- **Adaptation latency:** <6 hours from shift detection to adaptation
- **OOD false positive rate:** <20% (avoid over-flagging)

**Secondary Metrics:**
- **Importance weight stability:** Max/Mean ratio < 20 (avoid extreme weights)
- **MMD reduction:** Post-adaptation MMD² < 0.02 (closer to training distribution)
- **Confidence calibration:** Expected Calibration Error (ECE) < 0.05

### 8.2 Expected Calibration Error (ECE)

Measure how well confidence matches actual accuracy.

**Formula:**

```
ECE = Σ_m (n_m / n) |acc(m) - conf(m)|
```

Where:
- Predictions binned by confidence: m = 1, ..., M
- n_m = number of predictions in bin m
- acc(m) = accuracy in bin m
- conf(m) = average confidence in bin m

**Target:** ECE < 0.05 (well-calibrated)

---

**Technical Framework Complete.** Next: Full Python implementation (covariate_shift_handler.py).
