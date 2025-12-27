# COVARIATE SHIFT HANDLING: INTEGRATION GUIDE

**Module 4: Covariate Shift Handling - Section 4**

## Overview

This guide explains how to integrate the **Covariate Shift Handler** with your existing trading system. The handler detects distribution shifts and adapts models without full retraining, using importance weighting, domain adaptation, and OOD confidence calibration.

**Integration Points:**
1. `AITradingEngine` - Covariate shift detection and adaptation
2. `EnsembleManager` - Apply importance weights to ensemble predictions
3. `EventDrivenExecutor` - Feature extraction for shift monitoring
4. `AI-HFOS` - Health monitoring with covariate shift status
5. `routes/ai.py` - API endpoints for shift diagnostics

---

## 1. FILE MODIFICATIONS

### 1.1 `backend/services/ai/ai_trading_engine.py`

**Add imports:**

```python
from services.ai.covariate_shift_handler import (
    CovariateShiftHandler,
    ShiftSeverity,
    AdaptationMethod
)
```

**Initialize handler in `__init__`:**

```python
class AITradingEngine:
    def __init__(self, ...):
        # Existing initialization
        self.memory_system = ...
        self.rl_orchestrator = ...
        self.drift_manager = ...
        
        # NEW: Covariate shift handler
        self.covariate_handler = CovariateShiftHandler(
            mmd_threshold_moderate=float(os.getenv('COVARIATE_MMD_MODERATE', '0.01')),
            mmd_threshold_severe=float(os.getenv('COVARIATE_MMD_SEVERE', '0.05')),
            kl_threshold_moderate=float(os.getenv('COVARIATE_KL_MODERATE', '0.1')),
            kl_threshold_severe=float(os.getenv('COVARIATE_KL_SEVERE', '0.5')),
            importance_weight_upper_bound=float(os.getenv('COVARIATE_WEIGHT_BOUND', '1000')),
            importance_weight_clip=(
                float(os.getenv('COVARIATE_WEIGHT_MIN', '0.1')),
                float(os.getenv('COVARIATE_WEIGHT_MAX', '10'))
            ),
            ood_threshold=float(os.getenv('COVARIATE_OOD_THRESHOLD', '0.7')),
            checkpoint_path='data/covariate_shift_checkpoint.json'
        )
        
        # Store recent feature samples for shift detection
        self.recent_features_buffer = {model: [] for model in self.models}
        self.feature_buffer_size = 200
```

**Add method to check covariate shift (every 100 trades):**

```python
async def _check_covariate_shift(self, model_name: str):
    """
    Check for covariate shift and adapt if needed
    
    Called every 100 trades to detect distribution changes
    """
    if model_name not in self.recent_features_buffer:
        return
    
    recent_features = self.recent_features_buffer[model_name]
    
    if len(recent_features) < 100:
        return  # Need minimum samples
    
    # Get training features (baseline)
    training_features = self._get_training_features(model_name)
    
    if training_features is None or len(training_features) < 500:
        logger.warning(f"[{model_name}] Insufficient training features for covariate shift detection")
        return
    
    # Convert to numpy arrays
    X_train = np.array(training_features)
    X_recent = np.array(recent_features[-100:])  # Last 100 trades
    
    # Get feature names
    feature_names = self._get_feature_names()
    
    # Get current performance (for strategy selection)
    current_performance = {
        'win_rate_drop': self.drift_manager.get_drift_context(model_name).win_rate_delta
        if hasattr(self.drift_manager, 'get_drift_context') else 0.0
    }
    
    # Detect and adapt
    result = self.covariate_handler.adapt_to_covariate_shift(
        model_name=model_name,
        X_train=X_train,
        X_recent=X_recent,
        feature_names=feature_names,
        current_performance=current_performance
    )
    
    # Log result
    logger.info(
        f"[{model_name}] Covariate Shift: "
        f"Severity={result.shift_severity}, "
        f"MMD²={result.distribution_metrics.mmd_squared:.4f}, "
        f"Strategy={result.adaptation_method}"
    )
    
    # If severe shift + performance drop, escalate to drift detection (retraining)
    if result.shift_severity == ShiftSeverity.SEVERE.value and current_performance.get('win_rate_drop', 0) > 0.05:
        logger.warning(f"[{model_name}] Severe covariate shift + performance drop → Escalating to drift detection")
        await self._trigger_drift_detection(model_name)

async def _trigger_drift_detection(self, model_name: str):
    """Escalate to Module 3 drift detection for retraining"""
    # Module 3 handles this
    pass

def _get_training_features(self, model_name: str) -> Optional[List]:
    """Retrieve training feature samples (from database or cache)"""
    # Implementation depends on your data storage
    # Could load from:
    # - Database (last 500 trades used for training)
    # - Cached training data
    # - Model metadata
    pass

def _get_feature_names(self) -> List[str]:
    """Get list of feature names"""
    return [
        'rsi_14', 'rsi_9', 'macd', 'macd_signal', 'macd_histogram',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_position',
        'volume', 'volume_sma_20', 'volatility', 'atr_14',
        'trend_strength', 'regime_encoded'
    ]
```

**Modify `generate_signal` to use importance weights:**

```python
async def generate_signal(self, symbol: str, ...):
    """
    Generate trading signal with covariate shift adaptation
    """
    # Extract features
    features = self._extract_features(market_data)
    
    # Store features for shift detection
    for model_name in self.models:
        if model_name not in self.recent_features_buffer:
            self.recent_features_buffer[model_name] = []
        
        self.recent_features_buffer[model_name].append(features)
        
        # Keep buffer at fixed size
        if len(self.recent_features_buffer[model_name]) > self.feature_buffer_size:
            self.recent_features_buffer[model_name].pop(0)
    
    # Generate predictions with importance weighting
    ensemble_result = await self.ensemble_manager.predict_with_covariate_adaptation(
        symbol=symbol,
        features=features,
        covariate_handler=self.covariate_handler
    )
    
    # ... rest of signal generation logic
    
    # Check covariate shift every 100 trades
    trades_count = self.trades_since_shift_check.get(model_name, 0) + 1
    self.trades_since_shift_check[model_name] = trades_count
    
    if trades_count >= 100:
        await self._check_covariate_shift(model_name)
        self.trades_since_shift_check[model_name] = 0
    
    return signal
```

---

### 1.2 `backend/services/ai/ensemble_manager.py`

**Add method for covariate-adapted predictions:**

```python
async def predict_with_covariate_adaptation(
    self,
    symbol: str,
    features: Dict,
    covariate_handler: CovariateShiftHandler
) -> Dict:
    """
    Generate ensemble prediction with covariate shift adaptation
    
    Applies importance weights to model predictions based on 
    how well each model's training distribution matches current distribution.
    
    Args:
        symbol: Trading symbol
        features: Extracted features
        covariate_handler: Covariate shift handler instance
    
    Returns:
        ensemble_result: Weighted prediction + metadata
    """
    predictions = {}
    confidences = {}
    base_weights = self.model_weights.copy()
    
    # Get predictions from each model
    for model_name, model in self.models.items():
        pred = await model.predict(features)
        predictions[model_name] = pred['prediction']
        confidences[model_name] = pred['confidence']
    
    # Adjust weights based on covariate shift
    adapted_weights = base_weights.copy()
    
    for model_name in self.models:
        # Get importance weight for current sample
        importance_weights = covariate_handler.get_current_weights(model_name)
        
        if importance_weights is not None:
            # Use mean importance weight as model weight adjustment
            # High importance weight = training distribution matches current distribution well
            mean_importance = importance_weights.mean()
            
            # Adjust model weight
            adapted_weights[model_name] *= mean_importance
            
            logger.debug(
                f"[{model_name}] Weight adjusted: "
                f"{base_weights[model_name]:.3f} → {adapted_weights[model_name]:.3f} "
                f"(importance={mean_importance:.3f})"
            )
    
    # Renormalize weights
    total_weight = sum(adapted_weights.values())
    if total_weight > 0:
        adapted_weights = {k: v / total_weight for k, v in adapted_weights.items()}
    else:
        adapted_weights = base_weights  # Fallback
    
    # Weighted ensemble prediction
    ensemble_pred = sum(
        predictions[model] * adapted_weights[model]
        for model in self.models
    )
    
    ensemble_confidence = sum(
        confidences[model] * adapted_weights[model]
        for model in self.models
    )
    
    return {
        'prediction': ensemble_pred,
        'confidence': ensemble_confidence,
        'model_predictions': predictions,
        'model_confidences': confidences,
        'base_weights': base_weights,
        'adapted_weights': adapted_weights,
        'weight_adjustments': {
            model: adapted_weights[model] - base_weights[model]
            for model in self.models
        }
    }
```

---

### 1.3 `backend/services/ai/ai_hedgefund_os.py`

**Add covariate shift diagnostics to health check:**

```python
async def system_health_check(self) -> Dict:
    """
    Comprehensive system health check
    """
    health = {
        'status': 'HEALTHY',
        'timestamp': datetime.utcnow().isoformat(),
        'components': {}
    }
    
    # Existing checks (memory, RL, drift)
    # ...
    
    # NEW: Covariate shift status
    if hasattr(self.ai_engine, 'covariate_handler'):
        covariate_status = await self._check_covariate_shift_health()
        health['components']['covariate_shift'] = covariate_status
        
        if covariate_status['status'] == 'WARNING':
            health['status'] = 'WARNING'
            health['warnings'] = health.get('warnings', [])
            health['warnings'].extend(covariate_status.get('issues', []))
    
    return health

async def _check_covariate_shift_health(self) -> Dict:
    """Check covariate shift handler health"""
    handler = self.ai_engine.covariate_handler
    
    status = {
        'status': 'HEALTHY',
        'models_with_severe_shift': [],
        'models_with_ood_predictions': [],
        'issues': []
    }
    
    # Check each model
    for model_name in self.ai_engine.models:
        history = handler.get_adaptation_history(model_name)
        
        if not history:
            continue
        
        latest = history[-1]
        
        # Check for severe shift
        if latest.shift_severity == ShiftSeverity.SEVERE.value:
            status['models_with_severe_shift'].append(model_name)
            status['status'] = 'WARNING'
            status['issues'].append(
                f"Model {model_name} has severe covariate shift (MMD²={latest.distribution_metrics.mmd_squared:.4f})"
            )
        
        # Check for high OOD predictions
        if latest.ood_calibration:
            ood_ratio = latest.ood_calibration.ood_count / len(latest.ood_calibration.ood_scores)
            if ood_ratio > 0.3:  # >30% OOD
                status['models_with_ood_predictions'].append(model_name)
                status['status'] = 'WARNING'
                status['issues'].append(
                    f"Model {model_name} has {ood_ratio:.1%} OOD predictions"
                )
    
    return status
```

---

### 1.4 `backend/routes/ai.py`

**Add API endpoints:**

```python
from services.ai.covariate_shift_handler import CovariateShiftHandler

@router.get("/covariate/diagnostics")
async def get_covariate_diagnostics():
    """
    Get covariate shift diagnostics for all models
    
    Returns:
        {
            "timestamp": "2024-03-15T10:30:00Z",
            "models": {
                "xgboost": {
                    "shift_severity": "moderate",
                    "mmd_squared": 0.023,
                    "kl_divergence": 0.18,
                    "adaptation_method": "importance_weighting",
                    "significant_features": ["volatility", "volume"],
                    "ood_ratio": 0.12
                },
                ...
            }
        }
    """
    handler: CovariateShiftHandler = ai_engine.covariate_handler
    
    diagnostics = {
        'timestamp': datetime.utcnow().isoformat(),
        'models': {}
    }
    
    for model_name in ai_engine.models:
        history = handler.get_adaptation_history(model_name)
        
        if not history:
            diagnostics['models'][model_name] = {
                'status': 'no_data',
                'message': 'No covariate shift checks performed yet'
            }
            continue
        
        latest = history[-1]
        
        model_diag = {
            'shift_severity': latest.shift_severity,
            'mmd_squared': latest.distribution_metrics.mmd_squared,
            'kl_divergence': latest.distribution_metrics.kl_divergence,
            'adaptation_method': latest.adaptation_method,
            'significant_features': latest.distribution_metrics.significant_features,
            'timestamp': latest.timestamp
        }
        
        if latest.importance_weights:
            model_diag['importance_weights'] = {
                'mean': latest.importance_weights.mean_weight,
                'max': latest.importance_weights.max_weight,
                'stability': latest.importance_weights.stability_score
            }
        
        if latest.ood_calibration:
            total = len(latest.ood_calibration.ood_scores)
            ood_count = latest.ood_calibration.ood_count
            model_diag['ood_ratio'] = ood_count / total if total > 0 else 0
        
        diagnostics['models'][model_name] = model_diag
    
    return diagnostics


@router.get("/covariate/weights/{model_name}")
async def get_importance_weights(model_name: str):
    """
    Get current importance weights for a model
    
    Returns weight distribution statistics
    """
    handler: CovariateShiftHandler = ai_engine.covariate_handler
    
    weights_obj = handler.current_weights.get(model_name)
    
    if not weights_obj:
        raise HTTPException(status_code=404, detail=f"No weights found for model {model_name}")
    
    return {
        'model_name': model_name,
        'method': weights_obj.method,
        'timestamp': weights_obj.timestamp,
        'statistics': {
            'mean': weights_obj.mean_weight,
            'max': weights_obj.max_weight,
            'min': weights_obj.min_weight,
            'variance': weights_obj.weight_variance,
            'stability_score': weights_obj.stability_score
        },
        'weights_sample': weights_obj.weights[:50].tolist()  # First 50 weights
    }


@router.get("/covariate/history/{model_name}")
async def get_adaptation_history(model_name: str, limit: int = 10):
    """
    Get adaptation history for a model
    
    Args:
        model_name: Model identifier
        limit: Max number of results
    
    Returns adaptation timeline
    """
    handler: CovariateShiftHandler = ai_engine.covariate_handler
    
    history = handler.get_adaptation_history(model_name)
    
    if not history:
        return {
            'model_name': model_name,
            'count': 0,
            'history': []
        }
    
    # Latest first
    history = history[-limit:][::-1]
    
    return {
        'model_name': model_name,
        'count': len(history),
        'history': [
            {
                'timestamp': r.timestamp,
                'shift_severity': r.shift_severity,
                'adaptation_method': r.adaptation_method,
                'mmd_squared': r.distribution_metrics.mmd_squared,
                'kl_divergence': r.distribution_metrics.kl_divergence,
                'significant_features': r.distribution_metrics.significant_features
            }
            for r in history
        ]
    }


@router.post("/covariate/force-adaptation/{model_name}")
async def force_covariate_adaptation(model_name: str):
    """
    Manually trigger covariate shift adaptation
    
    Useful for testing or after manual model updates
    """
    if model_name not in ai_engine.models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    # Trigger adaptation
    await ai_engine._check_covariate_shift(model_name)
    
    return {
        'message': f'Covariate shift check triggered for {model_name}',
        'timestamp': datetime.utcnow().isoformat()
    }
```

---

## 2. CONFIGURATION (.env)

Add these environment variables:

```bash
# Covariate Shift Handler Configuration

# MMD Thresholds
COVARIATE_MMD_MODERATE=0.01      # Moderate shift threshold
COVARIATE_MMD_SEVERE=0.05        # Severe shift threshold

# KL Divergence Thresholds
COVARIATE_KL_MODERATE=0.1        # Moderate shift threshold
COVARIATE_KL_SEVERE=0.5          # Severe shift threshold

# KS Test
COVARIATE_KS_P_VALUE=0.01        # Significance level

# Importance Weighting
COVARIATE_WEIGHT_BOUND=1000      # Max weight (KMM parameter B)
COVARIATE_WEIGHT_MIN=0.1         # Min clipped weight
COVARIATE_WEIGHT_MAX=10          # Max clipped weight

# OOD Calibration
COVARIATE_OOD_THRESHOLD=0.7      # OOD threshold (0-1)
COVARIATE_MAHALANOBIS_LAMBDA=0.1 # Confidence decay rate

# Kernel Parameters
COVARIATE_KERNEL=rbf             # Kernel type (rbf, poly, linear)
COVARIATE_KERNEL_GAMMA=0.1       # Kernel bandwidth

# Detection Frequency
COVARIATE_CHECK_INTERVAL=100     # Check every N trades

# Buffer Size
COVARIATE_FEATURE_BUFFER=200     # Recent features buffer size
```

---

## 3. INTEGRATION FLOW

```
┌────────────────────────────────────────────────────────────────┐
│                   Covariate Shift Integration Flow              │
└────────────────────────────────────────────────────────────────┘

1. INITIALIZATION
   ├─ Load CovariateShiftHandler
   ├─ Initialize feature buffers
   └─ Load checkpoint (if exists)

2. FEATURE EXTRACTION (Per Trade)
   ├─ Extract features from market data
   ├─ Store in recent_features_buffer
   └─ Maintain buffer size (200 samples)

3. PERIODIC SHIFT DETECTION (Every 100 Trades)
   ├─ Get last 100 recent features
   ├─ Get training distribution (500 samples)
   ├─ Compute MMD², KL divergence, KS tests
   ├─ Determine shift severity
   └─ Select adaptation strategy

4. ADAPTATION (If Shift Detected)
   ├─ MODERATE Shift:
   │  ├─ Compute importance weights (Discriminator)
   │  └─ Store weights for ensemble weighting
   │
   └─ SEVERE Shift:
      ├─ Check performance drop
      ├─ If WR drop >5pp → Escalate to Module 3 (retraining)
      └─ Else:
         ├─ Apply domain adaptation (CORAL)
         ├─ Compute importance weights
         └─ Calibrate OOD confidence

5. ENSEMBLE PREDICTION (With Adaptation)
   ├─ Get base model predictions
   ├─ Adjust model weights by importance weights
   ├─ Renormalize ensemble weights
   └─ Generate weighted prediction

6. OOD CONFIDENCE CALIBRATION (If Needed)
   ├─ Compute Mahalanobis distance
   ├─ Flag OOD predictions (distance > threshold)
   └─ Reduce confidence: conf_adjusted = conf * exp(-λ · D_M)

7. MONITORING & LOGGING
   ├─ Log shift severity, MMD², KL divergence
   ├─ Track adaptation method applied
   ├─ Monitor OOD ratio
   └─ Store in adaptation history

8. HEALTH CHECK
   ├─ Include covariate shift status
   ├─ Flag models with severe shift
   ├─ Alert on high OOD ratios (>30%)
   └─ Report via API endpoints
```

---

## 4. INTEGRATION TEST SCRIPT

```python
"""
Test covariate shift integration
"""
import asyncio
import numpy as np
from services.ai.ai_trading_engine import AITradingEngine

async def test_covariate_shift_integration():
    """Test complete covariate shift workflow"""
    
    engine = AITradingEngine()
    
    print("=== COVARIATE SHIFT INTEGRATION TEST ===\n")
    
    # Step 1: Check initial state
    print("Step 1: Check initial state...")
    handler = engine.covariate_handler
    assert handler is not None
    print("✓ Handler initialized\n")
    
    # Step 2: Simulate feature extraction for 200 trades
    print("Step 2: Simulating 200 trades with feature extraction...")
    for i in range(200):
        features = {
            'rsi_14': np.random.uniform(40, 60),
            'macd': np.random.normal(0, 1),
            'volatility': np.random.uniform(2, 4),
            'volume': np.random.lognormal(10, 1)
        }
        
        # Store in buffer
        for model_name in engine.models:
            engine.recent_features_buffer[model_name].append(
                [features[k] for k in sorted(features.keys())]
            )
    
    print("✓ 200 features extracted and buffered\n")
    
    # Step 3: Introduce covariate shift (change distribution)
    print("Step 3: Introducing covariate shift (volatility 2-4 → 6-10)...")
    for i in range(100):
        features = {
            'rsi_14': np.random.uniform(40, 60),
            'macd': np.random.normal(0, 1),
            'volatility': np.random.uniform(6, 10),  # SHIFTED
            'volume': np.random.lognormal(11, 1)     # SHIFTED
        }
        
        for model_name in engine.models:
            engine.recent_features_buffer[model_name].append(
                [features[k] for k in sorted(features.keys())]
            )
    
    print("✓ 100 shifted features added\n")
    
    # Step 4: Trigger shift detection
    print("Step 4: Triggering covariate shift detection...")
    model_name = 'xgboost'
    await engine._check_covariate_shift(model_name)
    print("✓ Shift detection completed\n")
    
    # Step 5: Verify adaptation applied
    print("Step 5: Verifying adaptation...")
    history = handler.get_adaptation_history(model_name)
    
    if history:
        latest = history[-1]
        print(f"✓ Adaptation applied:")
        print(f"  - Severity: {latest.shift_severity}")
        print(f"  - MMD²: {latest.distribution_metrics.mmd_squared:.4f}")
        print(f"  - KL Divergence: {latest.distribution_metrics.kl_divergence:.4f}")
        print(f"  - Method: {latest.adaptation_method}")
        print(f"  - Significant Features: {latest.distribution_metrics.significant_features}")
    else:
        print("✗ No adaptation history found")
    
    print("\n✅ Integration test complete!")

if __name__ == "__main__":
    asyncio.run(test_covariate_shift_integration())
```

---

## 5. VERIFICATION CHECKLIST

- [ ] **Handler initialized** in AITradingEngine
- [ ] **Feature buffer** collecting samples (size 200)
- [ ] **Shift detection** runs every 100 trades
- [ ] **MMD² and KL divergence** calculated correctly
- [ ] **KS tests** performed per feature
- [ ] **Shift severity** determined (none/minor/moderate/severe)
- [ ] **Adaptation strategy** selected based on severity
- [ ] **Importance weights** computed (discriminator/KMM/KLIEP)
- [ ] **Domain adaptation** applied (CORAL) for severe shifts
- [ ] **Ensemble weights** adjusted by importance weights
- [ ] **OOD calibration** reduces confidence for OOD predictions
- [ ] **Adaptation history** stored and retrievable
- [ ] **Checkpoint** saved every 5 minutes
- [ ] **Checkpoint** restored on restart
- [ ] **Health check** includes covariate shift status
- [ ] **API endpoints** return diagnostics (4 endpoints functional)
- [ ] **Severe shift + performance drop** escalates to drift detection
- [ ] **Logging** includes MMD², KL, severity, method

---

## 6. TROUBLESHOOTING

### Issue 1: "Insufficient training features for covariate shift detection"

**Cause:** `_get_training_features()` returns None or < 500 samples

**Solution:**
- Ensure training features are stored/cached during model training
- Load last 500 trades from database
- Alternatively, use drift detection baseline (Module 3) as training reference

---

### Issue 2: "Importance weights unstable (max/mean ratio > 100)"

**Cause:** Extreme weights due to outliers or poor kernel bandwidth

**Solution:**
- Increase `COVARIATE_WEIGHT_BOUND` to allow higher weights
- Adjust `COVARIATE_KERNEL_GAMMA` (lower = smoother, higher = sharper)
- Use `discriminator` method instead of `kmm` (more stable)

---

### Issue 3: "High OOD ratio (>50%) after adaptation"

**Cause:** Domain adaptation insufficient or distribution shift too severe

**Solution:**
- Increase `COVARIATE_OOD_THRESHOLD` (less sensitive)
- Escalate to retraining (Module 3) instead of adapting
- Check if concept drift (P(Y|X) changed) rather than covariate shift

---

### Issue 4: "MMD² always near 0 even with obvious shift"

**Cause:** Kernel bandwidth too large (over-smoothing)

**Solution:**
- Decrease `COVARIATE_KERNEL_GAMMA` (0.01 or 0.001)
- Try `linear` kernel instead of `rbf`
- Check feature scaling (standardize features first)

---

### Issue 5: "KL divergence computation fails"

**Cause:** KDE fails for high-dimensional data

**Solution:**
- Rely on MMD² and KS tests only
- Compute KL per feature instead of joint distribution
- Use per-feature KS tests as primary metric

---

## 7. EXPECTED OUTPUT EXAMPLES

### Healthy System (No Shift)

```json
GET /api/v1/ai/covariate/diagnostics

{
  "timestamp": "2024-03-15T10:30:00Z",
  "models": {
    "xgboost": {
      "shift_severity": "none",
      "mmd_squared": 0.0042,
      "kl_divergence": 0.031,
      "adaptation_method": "none",
      "significant_features": [],
      "timestamp": "2024-03-15T10:25:00Z"
    },
    "lightgbm": {
      "shift_severity": "minor",
      "mmd_squared": 0.0087,
      "kl_divergence": 0.065,
      "adaptation_method": "none",
      "significant_features": ["volume"],
      "timestamp": "2024-03-15T10:28:00Z"
    }
  }
}
```

### System with Covariate Shift

```json
GET /api/v1/ai/covariate/diagnostics

{
  "timestamp": "2024-03-15T14:30:00Z",
  "models": {
    "xgboost": {
      "shift_severity": "moderate",
      "mmd_squared": 0.0234,
      "kl_divergence": 0.182,
      "adaptation_method": "importance_weighting",
      "significant_features": ["volatility", "volume", "atr_14"],
      "importance_weights": {
        "mean": 1.08,
        "max": 6.4,
        "stability": 5.9
      },
      "ood_ratio": 0.14,
      "timestamp": "2024-03-15T14:28:00Z"
    },
    "neural_net": {
      "shift_severity": "severe",
      "mmd_squared": 0.0612,
      "kl_divergence": 0.421,
      "adaptation_method": "hybrid",
      "significant_features": ["volatility", "volume", "rsi_14", "macd"],
      "importance_weights": {
        "mean": 1.24,
        "max": 8.7,
        "stability": 7.0
      },
      "ood_ratio": 0.28,
      "timestamp": "2024-03-15T14:30:00Z"
    }
  }
}
```

---

**Module 4 Section 4: Integration Guide - COMPLETE ✅**

Next: Risk Analysis, Test Suite, Benefits Analysis
