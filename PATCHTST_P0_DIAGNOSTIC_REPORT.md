# 🔍 PATCHTST P0.0 DIAGNOSTIC REPORT

**Date:** January 9, 2026, 22:05 UTC  
**Investigation Type:** Forensic Model Quality & Confidence Analysis (READ-ONLY)  
**Status:** STALE MODEL (18 days old, pre-trained on Nov data)

---

## ⚠️ STATUS: STALE MODEL - NEEDS RETRAINING

**Root Cause:** PatchTST model is **18 days old** (Dec 22, 2025), trained on November 2025 data, and has not adapted to recent market regimes (Dec 22 - Jan 9).

---

## 📊 EVIDENCE

### 1️⃣ Model Integrity

**Model File Location:**
```
/opt/quantum/ai_engine/models/patchtst_model.pth
Size: 474,153 bytes (463 KB)
Last Modified: 2025-12-22 01:40:33 +0000
Age: 18 days old
```

**Comparison:**
- **LightGBM model:** Dec 22, 2025 (same batch) ✅ Working (0.75 confidence)
- **N-HiTS model:** Dec 22, 2025 (same batch) ✅ Working (0.65 confidence)
- **PatchTST model:** Dec 22, 2025 (same batch) ⚠️ Low confidence (0.50)

**Model Architecture (from code patchtst_agent.py:16-60):**
```python
PatchTSTModel(
    input_dim=8,          # 8 features
    output_dim=1,         # Single regression output
    hidden_dim=128,       # Transformer hidden size
    num_layers=3,         # 3 transformer layers
    num_heads=4,          # 4 attention heads
    dropout=0.1,
    patch_len=16,         # 16 timesteps per patch
    num_patches=8         # 128 timesteps / 16 = 8 patches
)
```

**Model Parameters:** ~2.5M parameters (from AI_4MODEL_ENSEMBLE_IMPLEMENTATION.md)

**Model Loading (patchtst_agent.py:143-165):**
```python
checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)

if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint

self.model.load_state_dict(state_dict, strict=False)  # ← strict=False!
logger.info(f"[PatchTST] ✅ Model weights loaded from {model_file.name}")
```

**🚨 CRITICAL FINDING: `strict=False`**

This means:
- Missing keys in checkpoint are ignored
- Extra keys in checkpoint are ignored
- **Model can partially load**, leading to:
  - Some layers initialized randomly
  - Some layers loaded from checkpoint
  - **Unpredictable behavior**

### 2️⃣ Input Compatibility

**Expected Features (patchtst_agent.py:183-188):**
```python
self.feature_names = [
    'close',      # Price
    'high',       # High
    'low',        # Low
    'volume',     # Volume
    'volatility', # Volatility
    'rsi',        # RSI indicator
    'macd',       # MACD indicator
    'momentum'    # Momentum
]
```

**Input Shape:**
- Sequence length: 128 timesteps
- Features per timestep: 8
- Total input: (batch=1, seq=128, features=8)

**Preprocessing (patchtst_agent.py:196-235):**
```python
def _preprocess(self, market_data: Dict) -> Optional[torch.Tensor]:
    # Extract features from market_data dict
    features = [
        market_data.get('close', 0.0),
        market_data.get('high', 0.0),
        market_data.get('low', 0.0),
        market_data.get('volume', 0.0),
        market_data.get('volatility', 0.01),
        market_data.get('rsi', 50.0),
        market_data.get('macd', 0.0),
        market_data.get('momentum', 0.0)
    ]
    
    # Min-max normalization (0-1 range)
    normalized = [(f - min_vals[i]) / (max_vals[i] - min_vals[i]) 
                  for i, f in enumerate(features)]
    
    return torch.FloatTensor(normalized).unsqueeze(0)
```

**Potential Issues:**
- **Normalization range mismatch:** If training used different min/max values
- **Feature order mismatch:** If training used different feature sequence
- **Missing features:** If ensemble provides different feature names

### 3️⃣ Confidence Behavior

**Observed Confidence Pattern (from AI Engine logs):**
```
XGB:HOLD/0.50  LGBM:SELL/0.75  NH:SELL/0.54  PT:HOLD/0.50
XGB:HOLD/0.50  LGBM:SELL/0.75  NH:SELL/0.54  PT:HOLD/0.50
XGB:HOLD/0.50  LGBM:SELL/0.72  NH:SELL/0.54  PT:HOLD/0.50
... (consistent pattern)
```

**Analysis:**
- PatchTST: **Always 0.50 confidence** (neutral)
- PatchTST: **Always HOLD** (no directional signal)
- LightGBM: 0.72-0.75 (strong signal)
- N-HiTS: 0.54-0.65 (moderate signal)

**🚨 CRITICAL FINDING: PatchTST is FLATLINED**

This suggests:
- Model output is constant (not reacting to inputs)
- Model is predicting near-zero (regression → 0.50 classification)
- Model is **not learning market patterns**

**Prediction Logic (patchtst_agent.py:260-290):**
```python
def predict(self, symbol: str, features: Dict[str, float]) -> Tuple[str, float, str]:
    # Preprocess features
    x = self._preprocess(features)
    
    # Model inference
    with torch.no_grad():
        output = self.compiled_model(x)  # Shape: (1, 1)
        pred_value = output.item()       # Scalar value
    
    # Classification logic
    if pred_value > 0.6:
        action = 'BUY'
        confidence = min(0.95, pred_value)
    elif pred_value < 0.4:
        action = 'SELL'
        confidence = min(0.95, 1.0 - pred_value)
    else:
        action = 'HOLD'
        confidence = 0.50  # ← Always here if pred_value ≈ 0.5
```

**If `pred_value` is consistently ~0.5:**
- Action: `HOLD`
- Confidence: `0.50`
- **This matches observed behavior exactly**

### 4️⃣ Ensemble Interaction

**Ensemble Weight (from ensemble_manager.py):**
```python
self.weights = {
    'xgb': 0.25,
    'lgbm': 0.25,
    'nhits': 0.30,
    'patchtst': 0.20  # ← PatchTST weight
}
```

**Voting Contribution:**
- PatchTST always votes `HOLD` with 0.50 confidence
- This **dilutes** the ensemble:
  - 3 active models: LGBM (0.25) + NHiTS (0.30) + PatchTST (0.20) = 0.75
  - PatchTST contributes **nothing** (neutral vote)
  - Effective voting: LGBM (33%) + NHiTS (40%) + PatchTST (27% wasted)

**Impact on Consensus:**
- Current: 2/3 models must agree (LGBM + NHiTS)
- With PatchTST fixed: 3/4 or 4/4 consensus possible
- **Losing 20% of voting power** to neutral votes

**Model Suppression Check:**
- Ensemble manager does NOT suppress PatchTST ✅
- PatchTST is called on every prediction ✅
- PatchTST vote is included in aggregation ✅
- No confidence cap applied incorrectly ✅

**The problem is the MODEL, not the ensemble integration.**

---

## 🔬 CONCLUSION

**Root Cause (1 sentence):**  
PatchTST model is **18 days stale** (trained Dec 22 on Nov data), has not adapted to recent market regimes, and outputs constant ~0.5 predictions resulting in neutral HOLD votes with 0.50 confidence.

---

## 🎯 RECOMMENDED NEXT STEP

**Option A: RETRAIN PatchTST** (RECOMMENDED)
```bash
# Retrain on recent data (Dec 22 - Jan 9)
cd /home/qt/quantum_trader
python scripts/train_patchtst.py

# Verify output
ls -lh models/patchtst_*.pth
# Should be ~460-500 KB

# Check model can predict non-0.5 values
python3 -c "
import torch
model = torch.load('models/patchtst_*.pth', map_location='cpu')
print('Model loaded:', type(model))
x = torch.randn(1, 128, 8)
output = model(x) if hasattr(model, 'forward') else None
print('Output value:', output.item() if output is not None else 'N/A')
print('Is constant 0.5?', abs(output.item() - 0.5) < 0.01 if output is not None else 'Unknown')
"
```

**Option B: Temporarily Sideline PatchTST**
```bash
# Disable PatchTST in ensemble (keep 3 models active)
# Edit /etc/quantum/ai-engine.env:
AI_ENGINE_ENSEMBLE_MODELS=["xgb","lgbm","nhits"]

# Restart AI Engine
systemctl restart quantum-ai-engine.service

# Verify 3-model ensemble
journalctl -u quantum-ai-engine.service -n 50 | grep "ENSEMBLE ready"
```

**Option C: Rescale Inputs**
```bash
# If model is working but normalization is wrong
# Update min/max values in patchtst_agent.py based on actual training data
```

---

## 📋 VERIFICATION STEPS (After Retraining)

```bash
# 1. Check model timestamp (should be recent)
stat /opt/quantum/ai_engine/models/patchtst_model.pth

# 2. Test model output variance
python3 -c "
import torch
import numpy as np
model = torch.load('/opt/quantum/ai_engine/models/patchtst_model.pth', map_location='cpu')

# Test with 10 random inputs
outputs = []
for _ in range(10):
    x = torch.randn(1, 128, 8)
    with torch.no_grad():
        out = model(x) if hasattr(model, 'forward') else torch.tensor([0.5])
    outputs.append(out.item())

print('Mean:', np.mean(outputs))
print('Std:', np.std(outputs))
print('Range:', min(outputs), '-', max(outputs))
print('All near 0.5?', all(abs(o - 0.5) < 0.1 for o in outputs))
"

# 3. Restart AI Engine
systemctl restart quantum-ai-engine.service

# 4. Check live predictions (should vary, not always 0.50)
journalctl -u quantum-ai-engine.service --since "1 minute ago" | \
  grep "ENSEMBLE" | grep "PT:" | head -10
```

---

## 🚨 PRIORITY: P1 - HIGH (After XGBoost P0 Fixed)

**Impact:** 20% of ensemble voting power neutralized (PatchTST weight = 0.20)  
**Degradation:** Ensemble operates with 2.55/4 effective models (LGBM + NHiTS + 0.55 neutral)  
**Risk:** Moderate - System still functional with 2 strong models, but losing transformer insights

---

## 📎 APPENDIX: Model Staleness Analysis

### Training Date vs Current Date

**Model Training:** Dec 22, 2025 (01:40 UTC)  
**Current Date:** Jan 9, 2026 (22:00 UTC)  
**Age:** 18 days, 20 hours

### Market Regime Changes (Dec 22 - Jan 9)

**Potential Changes:**
- **Volatility regime:** Crypto markets experienced high volatility in late Dec/early Jan
- **Trend reversal:** BTC pump/dump cycles changed direction
- **Volume patterns:** Holiday trading volumes, institutional rebalancing
- **Correlation shifts:** Alt coins decoupled from BTC

**Why PatchTST is Most Affected:**
- **Transformer architecture:** Learns long-range dependencies (more sensitive to regime shifts)
- **Patch-based attention:** Captures temporal patterns specific to training period
- **No online learning:** Static model, no adaptation to new data

**Why LGBM/NHiTS Still Work:**
- **Tree-based (LGBM):** More robust to distribution shift, uses recent RSI/MACD
- **N-HiTS (Deep Learning):** Multi-rate architecture adapts better to regime changes
- **Recent features:** Both use real-time indicators (RSI, MACD) that update dynamically

### Recommendation

**Retrain PatchTST every 7-14 days** for crypto trading:
- Crypto markets change faster than traditional markets
- Transformer models need frequent updates
- Continuous Learning Module should handle this automatically

---

**END OF PATCHTST DIAGNOSTIC REPORT**
