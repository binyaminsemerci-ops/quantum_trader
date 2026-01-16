# TRAINING PIPELINE OPTIMIZATION ROADMAP

## Current State Analysis

### What Works ✅
- Ensemble loading fixed (syntax error resolved)
- XGBoost + LightGBM trained on real trade data
- Models actively predicting in production
- Feature engineering pipeline functional
- Multi-source data collection ready

### What Needs Improvement ⚠️
- Only 3/82 training samples valid (79 failed)
- Models severely overfitted (R²=1.0 on 3 samples)
- Small dataset → poor generalization
- Need debugging for feature calculation failures

---

## Issue 1: Only 3/82 Samples Valid

### Root Cause Analysis
```
Feature Engineering Process:
1. Calculate RSI (14-period) → needs 14 bars minimum
2. Calculate MACD (12/26/9) → needs 26 bars minimum
3. Rolling volatility (20-period) → needs 20 bars minimum
4. Bollinger Bands (10-period) → needs 10 bars minimum

Current Synthetic Window: 30 bars
Expected: Should support feature calc for all trades
Observed: Only 3 trades generated valid features
```

### Why It Failed
**Hypothesis 1**: Synthetic OHLCV generation created invalid data
- Check: `generate_synthetic_window()` validation
- Fix: Add data quality checks

**Hypothesis 2**: Trade timestamps incompatible with lookback windows
- Check: Trade timestamp distribution
- Fix: Ensure synthetic bars align with trade timing

**Hypothesis 3**: Feature engineering NaN filtering too aggressive
- Check: `engineer_features()` dropna() calls
- Fix: Use forward-fill or linear interpolation

### Solution: Extended Window Approach
```python
# Current: 30-bar window
synthetic_data = generate_synthetic_window(pnl_pct, window=30)

# Improved: 100-bar window
synthetic_data = generate_synthetic_window(pnl_pct, window=100)

# Expected Result: 60+ valid training samples
```

### Expected Outcome
- **Before**: 3 samples → R²=1.0 (overfitting) + R²=-0.0 (LGBM poor)
- **After**: 60+ samples → R²=0.5-0.7 (realistic) + stable LGBM
- **Timeline**: 2 hours to implement + test

---

## Issue 2: Model Overfitting

### Current Model Quality

**XGBoost**:
```
Training R²: 1.0000 (perfect fit on training data)
CV R²: nan (invalid with 3 samples across 3 folds)
Interpretation: Severely overfitted, memorizing noise
```

**LightGBM**:
```
Training R²: -0.0000 (negative = worse than mean baseline)
CV R²: nan
Interpretation: Failed to learn any pattern
```

### Why Overfitting Occurred
1. **Too few samples**: 3 samples vs typical 100+ for tree models
2. **No regularization pressure**: Tree depth unconstrained
3. **High feature dimensionality**: 14 features / 3 samples = bad ratio
4. **No validation feedback**: CV unreliable with tiny datasets

### Solutions

#### Solution A: Expand Training Data (Recommended)
```python
# Method 1: Larger synthetic window (30→100 bars)
# Expected: 20x sample generation, 60+ valid samples

# Method 2: Include real market.tick stream data
# Collect: OHLCV bars from Redis during training
# Benefit: Real data patterns vs synthetic

# Method 3: Multi-symbol training
# Currently: Single symbol (BTCUSDT)
# Expand: Train across all active pairs
# Benefit: Cross-symbol generalization
```

#### Solution B: Model Regularization
```python
# XGBoost tuning
xgb = XGBRegressor(
    max_depth=3,              # Reduce from 6 to prevent overfitting
    learning_rate=0.05,       # Reduce from 0.1
    subsample=0.8,            # Add subsampling
    colsample_bytree=0.8,     # Feature subsampling
    min_child_weight=5,       # Require more samples per leaf
)

# LightGBM tuning
lgbm = LGBMRegressor(
    num_leaves=15,            # Reduce from 31
    max_depth=3,
    min_data_in_leaf=5,       # Minimum samples per leaf
    learning_rate=0.05,
)
```

#### Solution C: Cross-Validation Strategy
```python
# Current: 3-fold CV (invalid with 3 samples)
# Better: Time-series cross-validation
from sklearn.model_selection import TimeSeriesSplit

cv = TimeSeriesSplit(n_splits=3)
# Each fold: train on past samples, test on future
# More realistic for time-series prediction
```

### Expected Improvements
| Metric | Current | Target | Method |
|--------|---------|--------|--------|
| Train Samples | 3 | 60+ | Extended window |
| XGB R² | 1.0000 | 0.5-0.7 | Regularization |
| LGBM R² | -0.0000 | 0.4-0.6 | Tuning + data |
| CV Stability | nan | 0.4-0.5 | Time-series CV |

---

## Issue 3: Data Collection Efficiency

### Current Architecture
```
Redis trade.closed (82 records)
    ↓
Synthetic OHLCV generation (30 bars)
    ↓
14 technical indicators
    ↓
Training
```

### Problem
- Synthetic data ≠ real market patterns
- Limited diversity in training signal
- No real tick-by-tick data

### Solution: Multi-Stream Collection
```
Real-time Streams:
  1. market.tick (real ticks)
  2. trade.closed (real outcomes)
  3. exchange.normalized (cross-exchange)

Potential New Streams:
  4. market.klines (1m OHLCV bars)
  5. orderbook (depth data)
  6. funding_rates (futures data)

Feature Sources:
  7. Binance klines (historical fallback)
  8. Backtest data archive
```

### Implementation Steps
1. **Enable market.tick collection** (1 hour)
   - Currently: Stream exists but maybe not populated
   - Action: Verify Redis stream has data
   - Benefit: Real OHLCV vs synthetic

2. **Add market.klines stream** (2 hours)
   - Currently: Not used
   - Action: Create Redis consumer for 1m bars
   - Benefit: Proper OHLCV sequences

3. **Archive daily trades** (1 hour)
   - Currently: Only 82 lifetime trades
   - Action: Snapshot trade.closed daily
   - Benefit: Long-term dataset growth

4. **Backfill training data** (3 hours)
   - Query Binance historical klines
   - Label with historical trades
   - Create pre-training dataset of 1000+ samples

---

## Improvement Prioritization

### Priority 1 (Do Today - 1 hour)
**Goal**: Get model quality to baseline
```
1. Debug why 79 samples failed feature engineering
2. Extend synthetic window: 30 → 100 bars
3. Retrain and verify 60+ samples
4. Expected R²: XGB 0.5-0.7, LGBM 0.4-0.6
```

### Priority 2 (This Week - 4 hours)
**Goal**: Stabilize model performance
```
1. Implement time-series cross-validation
2. Add regularization to both models
3. Collect real market.tick data for training
4. Retrain on mixed synthetic + real data
5. Expected R²: XGB 0.6-0.8, stable CV
```

### Priority 3 (Next Week - 8 hours)
**Goal**: Scale training infrastructure
```
1. Build market.klines consumer
2. Set up daily trade archiving
3. Create historical data pipeline
4. Backfill 1000+ trade samples
5. Rebuild ensemble on full dataset
6. Expected improvement: 20-30% accuracy gain
```

---

## Quick Win: Extend Window to 100 Bars

### Current Code (30 bars)
```python
def generate_synthetic_window(pnl_pct, window=30):
    """Generate realistic OHLCV matching target PnL"""
    # Creates 30-bar sequence
    ...
    return ohlcv_df
```

### Improved Code (100 bars)
```python
def generate_synthetic_window(pnl_pct, window=100):
    """Generate realistic OHLCV matching target PnL"""
    # Creates 100-bar sequence
    # Longer history → more feature space → less NaN
    ...
    return ohlcv_df
```

### Expected Impact
```
Window Size: 30 → 100 bars (+230%)
Feature Lookback Requirements:
  - RSI: needs 14 bars ✓ (satisfied at 30, still good at 100)
  - MACD: needs 26 bars ✓ (satisfied at 30, more robust at 100)
  - Bollinger: needs 10 bars ✓ (good either way)

NaN Rates:
  - At 30 bars: ~96% NaN (79/82 failed)
  - At 100 bars: ~5-10% NaN (expected, 3-4 failed)
  
Valid Samples:
  - At 30 bars: 3 samples
  - At 100 bars: 60-65 samples

Impact on Models:
  - XGB overfitting reduction: 1.0 → 0.65
  - LGBM stability: -0.0 → 0.5
  - CV reliability: nan → 0.45-0.55
```

---

## Recommended Execution Plan

### Step 1: Deploy Extended Window (30 min)
```bash
# Edit train_ensemble_models_v4.py
# Change: window=30 → window=100

# Execute training
cd /home/qt/quantum_trader
python scripts/train_ensemble_models_v4.py
```

### Step 2: Verify Results (15 min)
```bash
# Expected output
[DATA] Generated 60+ samples, 14 features
[XGB] Train R² = 0.65 ± 0.15
[LGBM] Train R² = 0.50 ± 0.20
[SAVE] Models saved: v4_20260116_<new_timestamp>
```

### Step 3: Deploy to AI Engine (10 min)
```bash
# Models auto-reload from latest in /models/
# Verify in logs
tail -50 /tmp/ai_engine_new.log | grep "Loading ensemble"
```

### Step 4: Monitor Performance (ongoing)
```bash
# Watch for improved predictions
tail -f /tmp/ai_engine_new.log | grep "XGB-Agent\|LGBM-Agent"
```

---

## Success Metrics

### Before (Current)
- ✅ System operational
- ❌ Only 3 training samples
- ❌ Severe overfitting (R²=1.0)
- ⚠️ Unstable cross-validation

### After (Target)
- ✅ System operational
- ✅ 60+ training samples
- ✅ Realistic overfitting (R²=0.5-0.7)
- ✅ Stable cross-validation (CV=0.45-0.55)
- ✅ Real-world prediction reliability

---

**Recommendation**: Execute Priority 1 today to improve model robustness
**Estimated Time**: 1 hour total
**Expected Benefit**: 10-20% improvement in prediction accuracy
