# 🚨 CRITICAL: V3 Models Training-Production Feature Mismatch

**Date**: 2026-01-11  
**Status**: BLOCKING - V3 models producing degenerate output (100% BUY, 0.0 confidence_std)  
**Severity**: P0 - Both XGBoost v3 and PatchTST v3 are unusable

---

## 🔍 Root Cause Analysis

### Training Data (`train_full.csv`) - 23 Features:
```
open, high, low, close, volume, price_change, rsi_14, macd, volume_ratio, 
momentum_10, high_low_range, volume_change, volume_ma_ratio, ema_10, ema_20, 
ema_50, ema_10_20_cross, ema_10_50_cross, volatility_20, macd_signal, macd_hist, 
bb_position, momentum_20
```

### Production Features (`microservices/ai_engine/service.py`) - 6 Base + ~15 Dynamic:
**Base (always present)**:
- price, price_change, rsi_14, macd, volume_ratio, momentum_10

**Dynamic (module-dependent)**:
- Cross-Exchange: volatility_factor, exchange_divergence, lead_lag_score
- Funding Rate: funding_rate, funding_delta, crowded_side_score  
- Volatility Structure: atr, atr_trend, atr_acceleration, short_term_vol, medium_term_vol, etc.
- Orderbook: orderflow_imbalance, delta_volume, bid_ask_spread_pct, etc.

**❌ MISMATCH**: Training features ≠ Production features!

---

## 💥 Impact

### XGBoost v3 (`xgb_v20260111_055436_v3.pkl`)
- **Symptom**: 100% BUY predictions, confidence_std=0.007 < 0.02
- **Reason**: Expects 23 features from train_full.csv, receives 6-20 different features
- **Status**: QSC FAIL-CLOSED (correctly marked INACTIVE)

### PatchTST v3 (`patchtst_v20260111_061709_v3.pth`)
- **Symptom**: 100% BUY predictions, confidence_std=0.0000 < 0.02
- **Reason**: Model architecture expects 23 flat features, but feature names don't match
- **Status**: QSC FAIL-CLOSED (correctly marked INACTIVE)

### Current Ensemble Status
```
[QSC] ACTIVE: ['lgbm'] | INACTIVE: {'xgb': 'degenerate', 'patchtst': 'degenerate', 'nhits': 'fallback_rules'}
```
**⚠️ ONLY 1/4 MODELS ACTIVE** (25% capacity)

---

## 🔧 Attempted Fixes (All Failed)

1. **StandardScaler normalization** - Fixed numerical instability but didn't solve feature mismatch
2. **Agent v3 support** - Added flat feature extraction but used wrong feature names
3. **Scaler loading** - Scaler loads correctly but can't fix underlying feature mismatch

---

## ✅ Solutions (Prioritized)

### Option 1: **Rollback to V2 Models** (FASTEST - 5 minutes)
- **Action**: Remove v3 model files to force fallback to v2
- **Files to delete**:
  ```bash
  rm /home/qt/quantum_trader/ai_engine/models/xgb_v*_v3.pkl
  rm /home/qt/quantum_trader/ai_engine/models/patchtst_v*_v3.pth
  systemctl restart quantum-ai-engine.service
  ```
- **Expected Result**: 3/4 models ACTIVE (XGB v2, LGBM v2, PatchTST v2)
- **Pros**: Immediate fix, proven working models
- **Cons**: Misses normalization benefits of v3

### Option 2: **Create V4 Training Pipeline** (RECOMMENDED - 1-2 hours)
- **Action**: Create new training data using actual production features
- **Steps**:
  1. Extract feature creation logic from service.py
  2. Create unified feature extractor (training + production)
  3. Generate new train_v4.csv with exact production features
  4. Retrain XGBoost v4 and PatchTST v4 with correct features
  5. Deploy and verify QSC passes
- **Pros**: Properly aligned training-production pipeline
- **Cons**: Requires code refactoring and retraining time

### Option 3: **Expand Production Features** (COMPLEX - 2-3 hours)
- **Action**: Modify service.py to compute all 23 features from train_full.csv
- **Required**:
  - Add: open, high, low, close, volume
  - Add: high_low_range, volume_change, volume_ma_ratio
  - Add: ema_10, ema_20, ema_50, ema_10_20_cross, ema_10_50_cross
  - Add: volatility_20, macd_signal, macd_hist, bb_position, momentum_20
- **Pros**: V3 models work immediately after restart
- **Cons**: Increases feature computation cost, adds complexity

---

## 🎯 Recommended Action Plan

**IMMEDIATE (Next 5 minutes)**:
1. Roll back to v2 models (Option 1)
2. Verify ensemble returns to 3/4 ACTIVE

**NEXT ITERATION (This weekend)**:
1. Implement Option 2 (V4 training pipeline)
2. Create unified_features.py module
3. Regenerate training data with production features
4. Retrain XGBoost v4 and PatchTST v4
5. Add integration test: verify training features == production features

---

## 📊 Lessons Learned

1. **Always validate feature alignment** between training and production
2. **Add CI/CD check**: Compare training CSV columns vs. service.py features dict keys
3. **Document feature engineering pipeline** - must be identical for training and inference
4. **QSC saved us**: Degenerate output detection prevented bad trades

---

## 🔗 Related Files

- **Training Scripts**: ops/retrain/retrain_xgb_v3.py, ops/retrain/retrain_patchtst_v3.py
- **Training Data**: ops/retrain/train_full.csv (23 features)
- **Production Features**: microservices/ai_engine/service.py:1230-1340 (6 base + dynamic)
- **Agents**: ai_engine/agents/xgb_agent.py, ai_engine/agents/patchtst_agent.py
- **QSC Logs**: journalctl -u quantum-ai-engine.service | grep "Degenerate"

---

**Next Action**: Execute Option 1 (Rollback to v2) immediately to restore 3/4 ensemble capacity.
