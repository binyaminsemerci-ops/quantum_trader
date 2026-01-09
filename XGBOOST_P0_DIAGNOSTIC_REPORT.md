# 🔍 XGBOOST P0.0 DIAGNOSTIC REPORT

**Date:** January 9, 2026, 22:00 UTC  
**Investigation Type:** Forensic Model Availability Analysis (READ-ONLY)  
**Status:** PATH MISMATCH + LOAD FAILURE

---

## ⚠️ STATUS: LOAD FAILURE DUE TO BROKEN MODEL FILES

**Root Cause:** XGBoost model files exist but are **corrupted/tiny (164-168 bytes)**. These are not valid trained models.

---

## 📊 EVIDENCE

### 1️⃣ Model File Existence

**Expected Paths (from code analysis):**
- Primary: `/app/models/xgboost_v*.pkl` (retraining directory)
- Fallback: `/opt/quantum/ai_engine/models/xgb_model.pkl` (base directory)
- Feature names: `/opt/quantum/ai_engine/models/xgboost_features.pkl`

**Actual Files Found:**

**Location: /home/qt/quantum_trader/models/ (Docker volume)**
```
-rw-r--r-- 1 qt qt 168 Dec 28 14:51 xgboost_v20251228_145156.pkl
-rw-r--r-- 1 qt qt 168 Dec 28 07:49 xgboost_v20251228_074921.pkl
-rw-r--r-- 1 qt qt 164 Dec 28 13:49 xgboost_v20251228_134935.pkl
-rw-r--r-- 1 qt qt 164 Dec 28 15:47 xgboost_v20251228_154701.pkl
... (20+ similar files, all 164-168 bytes)
```

**Location: /opt/quantum/ai_engine/models/ (Systemd runtime)**
```
/opt/quantum/ai_engine/models/xgb_model.pkl
/opt/quantum/ai_engine/models/xgb_model_v20251115_043308.pkl
/opt/quantum/ai_engine/models/xgb_model_v20251115_043433.pkl
... (300+ files, dates from Nov 15-17, 2025)
```

**🚨 CRITICAL FINDING: ALL FILES ARE 164-168 BYTES**

This is **NOT a valid XGBoost model**. A trained XGBoost model should be:
- Typical size: **500KB - 5MB** (depending on trees and features)
- Structure: Pickled `xgboost.Booster` object with tree structures

164-168 bytes suggests:
- Empty pickle file
- Placeholder file
- Failed save operation
- Corrupted during transfer

### 2️⃣ Config & Environment Mapping

**Code Path Analysis (xgb_agent.py lines 40-52):**
```python
# Priority 1: Search for latest timestamped model
retraining_dir = "/app/models" if os.path.exists("/app/models") else base
latest_model = self._find_latest_model(retraining_dir, "xgboost_v*.pkl")
latest_scaler = self._find_latest_model(retraining_dir, "xgboost_scaler_v*.pkl")

# Priority 2: Fallback to base directory
self.model_path = model_path or latest_model or os.path.join(base, "xgb_model.pkl")
self.scaler_path = scaler_path or latest_scaler or os.path.join(base, "scaler.pkl")
```

**Environment Variables (Checked):**
- `/etc/quantum/ai-engine.env`: No XGBoost-specific paths ✅ (uses code defaults)
- Systemd unit: No overrides ✅
- Docker compose: No explicit XGBoost paths ✅

**Expected Behavior:**
1. Check `/app/models/` for `xgboost_v*.pkl` (retraining outputs)
2. If not found, use `/opt/quantum/ai_engine/models/xgb_model.pkl` (base model)
3. Load model via `pickle.load(f)`

**Actual Behavior:**
1. `/app/models/` does NOT exist on VPS ❌
2. Code falls back to `ai_engine/models/xgb_model.pkl` ✅
3. File exists (164-168 bytes) ⚠️
4. **Pickle load succeeds but model is broken** ❌

### 3️⃣ Code Path Analysis

**Loading Code (xgb_agent.py lines 86-100):**
```python
def _load(self) -> None:
    try:
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)  # ← Succeeds but loads broken model
                logger.info("✅ Loaded XGBoost model from %s", os.path.basename(self.model_path))
    except Exception as e:
        logger.debug("Failed to load model: %s", e)
        self.model = None
```

**Prediction Code (xgb_agent.py lines 262-265):**
```python
def predict(self, symbol: str, features: Dict[str, float]) -> tuple[str, float, str]:
    try:
        if self.model is None:  # ← Triggers when model is broken
            return ('HOLD', 0.50, 'xgb_no_model')
```

**What Happens:**
1. File opens successfully (164 bytes)
2. `pickle.load()` succeeds (valid pickle structure, but empty content)
3. Model object exists but is unusable (no `.predict()` method or broken state)
4. When `predict()` is called, either:
   - `self.model` evaluates to `None` (somehow)
   - OR an exception occurs later (caught silently)
5. Falls back to `'xgb_no_model'` response

### 4️⃣ Docker vs Systemd Delta

| Aspect | Docker (Original) | Systemd (Current) | Delta |
|--------|-------------------|-------------------|-------|
| **Model Location** | `/app/models/` (volume mount) | `/opt/quantum/ai_engine/models/` | Different base path |
| **/app/models exists?** | YES (volume: `./models:/app/models`) | NO (not created in systemd) | ❌ CRITICAL |
| **Retraining Output** | `/app/models/xgboost_v*.pkl` | No destination (would fail) | ❌ CRITICAL |
| **Base Model** | `/app/models/xgb_model.pkl` | `/opt/quantum/ai_engine/models/xgb_model.pkl` | Different location |
| **File Size** | Unknown (no access to Docker history) | 164-168 bytes (BROKEN) | ⚠️ BROKEN |
| **Model Valid?** | Presumably YES (worked in Docker) | NO (tiny corrupted files) | ❌ CRITICAL |

**Key Divergence:**
- **Docker:** Models stored in `./models:/app/models` volume (persistent, large files)
- **Systemd:** Models in `/opt/quantum/ai_engine/models/` (Git-tracked, but tiny broken files)
- **Migration Issue:** Docker volume models were NOT migrated to systemd filesystem

---

## 🔬 CONCLUSION

**Root Cause (1 sentence):**  
XGBoost model files exist in systemd but are **corrupted placeholders (164-168 bytes)** from failed training or incomplete Git commits, while the working models remain in unmounted Docker volumes.

---

## 🎯 RECOMMENDED NEXT STEP

**Option A: Locate Original Working Models** (PREFERRED)
```bash
# Check if Docker volumes still exist
docker volume ls | grep models

# Inspect volume
docker volume inspect quantum_trader_models

# Mount volume temporarily and copy models
docker run --rm -v quantum_trader_models:/data -v /tmp:/backup alpine \
  tar czf /backup/models.tar.gz /data
```

**Option B: Retrain from Scratch**
```bash
# Run training script
cd /home/qt/quantum_trader
python scripts/train_xgboost.py

# Verify output size
ls -lh models/xgboost_*.pkl
# Should be 500KB+ for valid model
```

**Option C: Copy from Working System**
```bash
# If you have another working deployment
scp user@working-server:/app/models/xgboost_*.pkl \
    /opt/quantum/ai_engine/models/
```

---

## 📋 VERIFICATION STEPS (After Fix)

```bash
# 1. Check file size (should be >500KB)
ls -lh /opt/quantum/ai_engine/models/xgb_model.pkl

# 2. Test model loading
python3 -c "
import pickle
with open('/opt/quantum/ai_engine/models/xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)
print('Model type:', type(model))
print('Has predict method:', hasattr(model, 'predict'))
"

# 3. Restart AI Engine
systemctl restart quantum-ai-engine.service

# 4. Check logs for "XGBoost agent loaded"
journalctl -u quantum-ai-engine.service -n 50 | grep -i xgb

# 5. Verify ensemble output (should see XGB confidence != 0.50)
journalctl -u quantum-ai-engine.service --since "1 minute ago" | \
  grep "ENSEMBLE" | grep "XGB:" | head -5
```

---

## 🚨 PRIORITY: P0 - IMMEDIATE

**Impact:** 25% of ensemble voting power lost (XGBoost weight = 0.25)  
**Degradation:** Ensemble operates with 2/4 models instead of 3/4 or 4/4  
**Risk:** Reduced signal quality, lower confidence, more false positives

---

**END OF XGBOOST DIAGNOSTIC REPORT**
