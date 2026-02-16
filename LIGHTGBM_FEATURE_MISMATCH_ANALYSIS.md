# 🚨 LightGBM Feature Mismatch - Complete Analysis
**Date**: February 16, 2026 22:40 UTC  
**Issue**: LightGBM model completely broken due to massive feature mismatch

---

## 📊 THE PROBLEM

### Feature Mismatch Summary

| Component | Expected Features | Actual Features | Gap |
|-----------|------------------|-----------------|-----|
| **LightGBM Model** | **49 features** | 5 provided | **-44 features** |
| **Scaler** | 14 features | 5 provided | -9 features |
| **Feature Stream** | N/A | 17 available | N/A |
| **lgbm_agent.py Code** | 49 needed | 5 extracted | **-44 features** |

---

## 🔍 DETAILED BREAKDOWN

### Current Feature Stream (17 available):
```
✅ price
✅ price_return_1
✅ price_return_5  
✅ price_volatility_10
✅ price_change
✅ ma_10
✅ ma_20
✅ ma_50
✅ ma_cross_10_20
✅ rsi_14
✅ macd
✅ volume
✅ volume_ratio
✅ bb_upper
✅ bb_lower
✅ bb_position
✅ momentum_10
```

### LightGBM Model Requirements (49 features):

**Candlestick Features (10)**:
```
❌ returns
❌ log_returns
❌ price_range
❌ body_size
❌ upper_wick
❌ lower_wick
❌ is_doji
❌ is_hammer
❌ is_engulfing
❌ gap_up / gap_down
```

**Momentum & Oscillators (13)**:
```
✅ rsi (as rsi_14 in stream)
✅ macd
❌ macd_signal
❌ macd_hist
❌ stoch_k
❌ stoch_d
❌ roc
✅ momentum_10
❌ momentum_5
❌ momentum_20
❌ acceleration
❌ relative_spread
```

**Moving Averages (12)**:
```
✅ ma_10 / ma_20 / ma_50 (SMA in stream)
❌ sma_20
❌ ema_9, ema_9_dist
❌ ema_21, ema_21_dist
❌ ema_50, ema_50_dist
❌ ema_200, ema_200_dist
```

**Bollinger Bands (5)**:
```
❌ bb_middle
✅ bb_upper
✅ bb_lower
❌ bb_width
✅ bb_position
```

**Volatility (4)**:
```
✅ price_volatility_10 (as volatility?)
❌ atr
❌ atr_pct
```

**Trend (4)**:
```
❌ adx
❌ plus_di
❌ minus_di
```

**Volume (5)**:
```
✅ volume
✅ volume_ratio
❌ volume_sma
❌ obv
❌ obv_ema
❌ vpt
```

---

## 🎯 ROOT CAUSE

**Model Training vs Production Mismatch**:

1. **Training Phase** (December 2025):
   - Model trained with **49 rich technical features**
   - Includes advanced indicators (ADX, Stochastic, OBV, candlestick patterns)
   - Scaler fitted to 14-feature subset?

2. **Production Deployment** (Now):
   - Feature publisher only generates **17 basic features**
   - lgbm_agent.py code only extracts **5 features**:
     ```python
     feature_names = [
         'price_change',
         'rsi_14',
         'macd',
         'volume_ratio',
         'momentum_10'
     ]
     ```

3. **Result**: 
   - Model expects 49 → Gets 5 → **FAILS EVERY TIME**
   - 6,985 errors/hour (116 errors/minute)
   - System falls back to simple RSI/MACD rules

---

## 📋 SOLUTION OPTIONS

### ⚡ **Option A: Disable LightGBM (RECOMMENDED - 15 minutes)**

**Action**:
1. Comment out LightGBM agent in `ensemble_predictor_service.py`
2. Restart ensemble predictor service
3. System uses XGBoost + fallback only

**Pros**:
- ✅ Immediate fix (15 min)
- ✅ Stops error spam
- ✅ Trading continues normally
- ✅ XGBoost may still work

**Cons**:
- ⚠️ Lose LightGBM predictions (already not working)
- ⚠️ Slightly degraded ensemble quality

**Code Change**:
```python
# In ensemble_predictor_service.py
agents = {
    "xgboost": xgb_agent,
    # "lgbm": lgbm_agent,  # DISABLED until retrained with correct features
}
```

---

### 🔄 **Option B: Retrain LightGBM with 17 Features (2-4 hours)**

**Action**:
1. Collect training data from Redis/database
2. Extract 17 available features per sample
3. Retrain LightGBM model with smaller feature set
4. Generate new scaler.pkl
5. Deploy model + scaler
6. Restart service

**Pros**:
- ✅ Get ML predictions working again
- ✅ Use existing feature infrastructure
- ✅ Modern model with current features

**Cons**:
- ⏳ 2-4 hours of work
- ⏳ Need training data access
- ⚠️ Lower quality than 49-feature model

---

### 🏗️ **Option C: Expand Feature Engineering (4-8 hours)**

**Action**:
1. Update feature publisher to calculate all 49 features
2. Add candlestick pattern detection
3. Add EMAs, ADX, Stochastic, OBV, etc.
4. Test feature generation
5. Deploy updated feature publisher
6. Use existing 49-feature model

**Pros**:
- ✅ Use existing trained model (proven good)
- ✅ Full feature richness
- ✅ Best prediction quality

**Cons**:
- ⏳ 4-8 hours development
- ⏳ Complex feature engineering
- ⏳ Need to test all indicators
- ⚠️ Higher computational cost

---

## 💡 IMMEDIATE RECOMMENDATION

**Do Option A NOW + Option B LATER**

1. **Immediate (15 min)**: Disable LightGBM to stop errors
2. **This week (2-4 hours)**: Retrain with 17 features
3. **Future (optional)**: Expand to 49 features if performance warrants it

**Rationale**:
- System is already trading on fallback signals (working fine)
- XGBoost may still provide value
- Quick fix stops error spam and clarifies logs
- Can retrain properly when time permits

---

## 📝 IMPLEMENTATION PLAN

### Step 1: Disable LightGBM (NOW)

**File**: `ai_engine/services/ensemble_predictor_service.py`

Find the agent initialization section and comment out LGBM:
```python
# Initialize agents
xgb_agent = XGBoostAgent()
# lgbm_agent = LGBMAgent()  # DISABLED: Feature mismatch (needs 49, we have 17)

agents = {
    "xgboost": xgb_agent,
    # "lgbm": lgbm_agent,  # DISABLED
}
```

**Restart**:
```bash
systemctl restart quantum-ensemble-predictor
```

**Verify**:
```bash
journalctl -u quantum-ensemble-predictor -f
# Should see no more "X has 5 features but StandardScaler is expecting..." errors
```

---

### Step 2: Retrain LightGBM (LATER)

**Training Script**:
```python
# train_lgbm_17features.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import pickle

# Features available in production
FEATURES = [
    'price_return_1', 'price_return_5', 'price_volatility_10',
    'price_change', 'ma_10', 'ma_20', 'ma_50', 'ma_cross_10_20',
    'rsi_14', 'macd', 'volume', 'volume_ratio',
    'bb_upper', 'bb_lower', 'bb_position', 'momentum_10'
]

# Load training data
data = load_training_data()  # From Redis/database
X = data[FEATURES]
y = data['target']  # Returns or signal

# Train scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train LightGBM
model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05)
model.fit(X_scaled, y)

# Save
pickle.dump(model, open('lightgbm_v20260216_v2.pkl', 'wb'))
pickle.dump(scaler, open('lightgbm_scaler_v20260216_v2.pkl', 'wb'))
```

---

## ✅ SUCCESS CRITERIA

### After Disabling LightGBM:
```
✅ No more StandardScaler errors in logs
✅ Ensemble predictor producing predictions
✅ Trading continues normally
✅ XGBoost predictions visible in signals
✅ Error rate drops from 116/min to 0
```

### After Retraining:
```
✅ LightGBM agent loads successfully
✅ Predictions generated for all symbols
✅ Model votes show "lgbm" instead of "fallback"
✅ Confidence scores vary (not stuck at 0.72/0.68)
✅ Trading performance improves (measure over 1 week)
```

---

## 📊 CURRENT SYSTEM STATUS

**Trading**: ✅ Working (fallback signals)  
**LightGBM**: ❌ Broken (49vs5 feature mismatch)  
**XGBoost**: ❓ Unknown (not logging errors, may work)  
**Ensemble**: ⚠️ Degraded (using fallback only)  

**Impact**: Low (trading works, just without ML intelligence)  
**Urgency**: Medium (should fix to restore ML capabilities)  
**Complexity**: Low (Option A) to High (Option C)

---

**Report End**
