"""
RETRAINING SCRIPTS - FEATURE SCHEMA UPDATE REQUIRED
====================================================

**Status**: ⚠️ Training scripts use 18-feature schema (v5), need 49-feature update (v6)

**Current State**:
- train_lightgbm_v5.py → 18 features
- train_xgb_v5.py → 18 features
- train_nhits_v5.py → 18 features
- train_patchtst_v5.py → 18 features

**Required Action**:
Create v6 training scripts that:
1. Import from `ai_engine.common_features import FEATURES_V6`
2. Implement feature calculation matching `feature_publisher_service.py` logic
3. Train models with all 49 features for optimal performance

**Why This Matters**:
- Agents NOW use 49-feature schema (Feb 2026 update)
- Old models (18 features) work via scaler bypass (graceful degradation)
- New models (49 features) will have superior prediction quality

**Feature Calculation Source**:
See `ai_engine/services/feature_publisher_service.py` lines 79-450 for complete 
feature engineering logic including:
- Candlestick patterns (doji, hammer, engulfing, gaps)
- Oscillators (RSI, MACD, Stochastic, ROC)
- EMAs with distance metrics
- ADX system (ADX, +DI, -DI)
- Bollinger Bands with position
- Volume indicators (OBV, VPT)
- Momentum and acceleration

**Priority**: Medium (P2)
- Current models work (bypass scaler when mismatch)
- Future retraining should use 49-feature schema
- No immediate impact on trading

**Next Steps**:
1. Create `calculate_features_v6()` function (port from feature_publisher_service)
2. Create train_*_v6.py scripts using FEATURES_V6
3. Schedule retraining to generate 49-feature models
4. Models will automatically be picked up by agents (no code changes needed)

**See Also**:
- `ai_engine/common_features.py` - Canonical feature list
- `ai_engine/services/feature_publisher_service.py` - Feature calculation reference
- Agent implementation examples in `ai_engine/agents/`
