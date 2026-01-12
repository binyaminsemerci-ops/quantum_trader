# 🚨 CRITICAL: PatchTST v3 Training-Production Feature Mismatch
**Date**: January 11, 2026 21:00 UTC  
**Priority**: P0 - BLOCKS DEPLOYMENT  
**Status**: ⚠️ REQUIRES ACTION

## 🔥 Issue Summary

PatchTST v3 model was successfully trained with **23 features**, but the production PatchTST agent (`ai_engine/agents/patchtst_agent.py`) expects **8 features**. This creates an architecture mismatch that will cause the agent to fail when loading the v3 model.

## 📊 Current State

### Training (ops/retrain/retrain_patchtst_v3.py)
```python
✅ Features: 23
✅ Model: SimplePatchTST(num_features=23, d_model=128, num_heads=4)
✅ File: patchtst_v20260111_205907_v3.pth (1.1M)
✅ Scaler: patchtst_v20260111_205907_v3_scaler.pkl (1.7K)
✅ Best Val Loss: 0.2791
```

### Production Agent (ai_engine/agents/patchtst_agent.py)
```python
❌ Expected Features: 8 (close, high, low, volume, volatility, rsi, macd, momentum)
❌ Model: PatchTSTModel(input_dim=8, hidden_dim=128, num_layers=3)
❌ Search Pattern: "patchtst_v*_v2.pth" (doesn't search for v3)
❌ No Scaler Support: Agent doesn't load or use scaler
```

## 🎯 Root Cause

The agent was designed for v2 models with 8 simple features, but:
1. XGBoost v3 and N-HiTS v2 use 23 features
2. PatchTST v3 training script was created to match the 23-feature standard
3. **Agent was not updated to handle 23-feature v3 models**

## 🔴 Impact

### If v3 Model is Loaded (WILL FAIL):
```python
RuntimeError: Error(s) in loading state_dict for PatchTSTModel:
    size mismatch for patch_embedding.weight: 
    copying a param with shape torch.Size([128, 368]) from checkpoint,
    where the shape is torch.Size([128, 2944]) in current model.
```

**Calculation**:
- v2: `patch_len (16) * input_dim (8) = 128` → Linear(128, 128)
- v3: `patch_len (16) * input_dim (23) = 368` → Linear(368, 128)  
  OR for simplified model: `input_dim (23)` → Linear(23, 128)

### Current Behavior:
- Agent searches for `"patchtst_v*_v2.pth"` → Finds v2 model (19K)
- v3 model exists but is ignored
- Ensemble runs with outdated v2 model

## 📋 Feature Comparison

| Feature Set | v2 (Current) | v3 (Trained) |
|-------------|--------------|--------------|
| **Features** | 8 | 23 |
| **Model Size** | 19K | 1.1M |
| **Architecture** | PatchTSTModel(input_dim=8) | SimplePatchTST(num_features=23) |
| **Scaler** | ❌ No | ✅ Yes |
| **Pattern** | patchtst_v*_v2.pth | patchtst_v*_v3.pth |

### v2 Features (8):
```python
['close', 'high', 'low', 'volume', 'volatility', 'rsi', 'macd', 'momentum']
```

### v3 Features (23):
```python
['open', 'high', 'low', 'close', 'volume', 'price_change', 'rsi_14', 'macd', 
 'volume_ratio', 'momentum_10', 'high_low_range', 'volume_change', 
 'volume_ma_ratio', 'ema_10', 'ema_20', 'ema_50', 'ema_10_20_cross', 
 'ema_10_50_cross', 'volatility_20', 'macd_signal', 'macd_hist', 
 'bb_position', 'momentum_20']
```

## ✅ Solution Options

### Option 1: Update Agent for v3 (RECOMMENDED)
**Pros**: Future-proof, matches XGBoost v3 / N-HiTS v2 standard  
**Cons**: More complex, requires agent rewrite  

**Steps**:
1. Add SimplePatchTST class to agent (or import from shared module)
2. Update model search to look for v3 models first:
   ```python
   latest_model = self._find_latest_model(retraining_dir, "patchtst_v*_v3.pth") or \
                  self._find_latest_model(retraining_dir, "patchtst_v*_v2.pth")
   ```
3. Add scaler loading:
   ```python
   scaler_path = model_path.replace(".pth", "_scaler.pkl")
   if os.path.exists(scaler_path):
       self.scaler = joblib.load(scaler_path)
   ```
4. Detect model version from metadata and use correct architecture
5. Apply scaler before prediction if v3 model

### Option 2: Retrain v3 with 8 Features
**Pros**: Works with existing agent immediately  
**Cons**: Inconsistent with other v3 models, less features = less accuracy  

**Steps**:
1. Modify retrain_patchtst_v3.py to use only 8 features
2. Retrain model
3. Deploy (will work with existing agent)

### Option 3: Hybrid Approach
**Pros**: Gradual migration, both v2 and v3 work  
**Cons**: Requires careful version detection  

**Steps**:
1. Agent detects model version from filename or metadata
2. If v3: Load SimplePatchTST + scaler, use 23 features
3. If v2: Load PatchTSTModel, use 8 features
4. Gradually phase out v2

## 🎯 Recommended Action Plan

### Phase 1: Agent Update (Priority: P0)
```python
# ai_engine/agents/patchtst_agent.py

class SimplePatchTST(nn.Module):
    """v3 architecture for 23-feature models"""
    def __init__(self, num_features: int, d_model=128, num_heads=4, num_layers=2, num_classes=3):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=256, 
            dropout=0.1, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )
    
    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.head(x)

class PatchTSTAgent:
    def __init__(self, ...):
        # Search for v3 first
        latest_model = self._find_latest_model(retraining_dir, "patchtst_v*_v3.pth") or \
                       self._find_latest_model(retraining_dir, "patchtst_v*_v2.pth")
        
        # Detect version
        self.model_version = "v3" if "v3" in str(latest_model) else "v2"
        
        # Load appropriate architecture
        if self.model_version == "v3":
            self.num_features = 23
            self.model = SimplePatchTST(num_features=23)
            # Load scaler
            scaler_path = str(latest_model).replace(".pth", "_scaler.pkl")
            if Path(scaler_path).exists():
                self.scaler = joblib.load(scaler_path)
        else:
            self.num_features = 8
            self.model = PatchTSTModel(input_dim=8, ...)
            self.scaler = None
```

### Phase 2: Feature Extraction Update
```python
# Update predict() to extract 23 features for v3
def predict(self, symbol, features: dict):
    if self.model_version == "v3":
        # Extract all 23 features
        feature_vector = np.array([[
            features.get("open", 0.0),
            features.get("high", 0.0),
            features.get("low", 0.0),
            features.get("close", 0.0),
            # ... all 23 features
        ]])
        if self.scaler:
            feature_vector = self.scaler.transform(feature_vector)
    else:
        # v2: Extract only 8 features
        feature_vector = np.array([[
            features.get("close", 0.0),
            features.get("high", 0.0),
            # ... 8 features
        ]])
```

### Phase 3: Testing & Deployment
1. Update agent code
2. Deploy to test environment
3. Verify v3 model loads correctly
4. Check predictions are varied (not degenerate)
5. Monitor QSC status
6. Deploy to production

## 📁 Files Affected

### Need Updates:
- [ ] `ai_engine/agents/patchtst_agent.py` - Add v3 support
- [ ] `microservices/ai_engine/inference/__init__.py` - May need feature list update

### Already Correct:
- [x] `ops/retrain/retrain_patchtst_v3.py` - Training script (23 features)
- [x] Model files in `ai_engine/models/` - v3 models exist

## ⚠️ Blockers

### Cannot Deploy v3 Model Until:
1. ✅ Training complete (DONE - best_val_loss: 0.2791)
2. ✅ Model files copied to correct location (DONE)
3. ❌ **Agent updated to support v3 architecture** (BLOCKED)
4. ❌ **Agent loads scaler for v3 models** (BLOCKED)
5. ❌ **Feature extraction updated for 23 features** (BLOCKED)

## 🎬 Next Steps

### Immediate (Today):
1. **Update patchtst_agent.py** to support v3 models
2. Test locally or on staging
3. Deploy updated agent to production
4. Restart service and verify v3 model loads
5. Monitor ensemble status for PatchTST activation

### Validation Commands:
```bash
# Check which model is loaded
journalctl -u quantum-ai-engine.service --since "30 seconds ago" | grep "PatchTST.*model"

# Verify v3 model is found
journalctl -u quantum-ai-engine.service --since "30 seconds ago" | grep "Found latest.*patchtst_v.*v3"

# Check scaler loading
journalctl -u quantum-ai-engine.service --since "30 seconds ago" | grep "Scaler"

# Monitor QSC status
journalctl -u quantum-ai-engine.service --since "1 minute ago" | grep "QSC.*ACTIVE"
```

### Expected Logs After Fix:
```
🔍 Found latest PatchTST model: patchtst_v20260111_205907_v3.pth
[PatchTST] Model version: v3 (23 features)
[PatchTST] Loading scaler from: patchtst_v20260111_205907_v3_scaler.pkl
[PatchTST] ✅ Model weights loaded successfully
[QSC] ACTIVE: ['xgb', 'lgbm', 'patchtst', 'nhits']
```

## 📊 Success Criteria

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Model Version | v3 | v2 | ❌ |
| Features | 23 | 8 | ❌ |
| Scaler Loaded | Yes | No | ❌ |
| QSC Status | ACTIVE | INACTIVE (using v2) | ⚠️ |
| Ensemble Weight | 31% | 31% (v2) | ⚠️ |

---
**Status**: 🚨 **BLOCKED - REQUIRES AGENT UPDATE**  
**ETA**: 30-60 minutes to update agent + testing  
**Risk**: Medium (wrong architecture = runtime error)  
**Priority**: P0 (blocks full 4/4 ensemble deployment)
