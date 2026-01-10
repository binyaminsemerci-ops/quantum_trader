# 🚨 MODEL DEGRADATION DIAGNOSTIC REPORT
**Date:** 2026-01-10  
**System:** quantum-ai-engine @ 46.224.116.254  
**Severity:** **CRITICAL** - QSC deployment BLOCKED

---

## ✅ EXECUTIVE SUMMARY

**Quality Gate Status:** ❌ **FAIL** (15 violations)  
**Root Cause:** **3 of 4 models NOT functioning correctly**  
**Impact:** Ensemble voting degraded to **1 active model** (only XGBoost working)

---

## 🔍 DETAILED FINDINGS

### 1️⃣ PatchTST: SHADOW MODE ACTIVE ❌
**Evidence:**
```json
"patchtst": {
  "action": "BUY",
  "confidence": 0.6150,
  "model": "patchtst_shadow",
  "shadow": true
}
```

**Code Location:** `ai_engine/agents/patchtst_agent.py:367-381`
```python
shadow_mode = os.getenv('PATCHTST_SHADOW_ONLY', 'false').lower() == 'true'
if shadow_mode:
    return action, confidence, 'patchtst_shadow'
```

**Ensemble Exclusion:** `ai_engine/ensemble_manager.py:430-432`
```python
if model_name == 'patchtst' and 'shadow' in str(prediction[2]):
    shadow_predictions[model_name] = prediction
    # EXCLUDED FROM VOTING
```

**Telemetry Impact:**
- **1418 events** with EXACT confidence = 0.6150 (std = 0.0000)
- 100% BUY predictions
- P10 = P90 = 0.6150 (perfectly constant)

**Ensemble Weight:** Reported as 0.25 (25%), **actual voting weight: 0%**

---

### 2️⃣ LGBM: FALLBACK RULES ENGINE ACTIVE ❌
**Evidence:**
```json
"lgbm": {
  "action": "SELL",
  "confidence": 1.05,
  "model": "lgbm_fallback_rules"
}
```

**Root Cause:** Model file path mismatch
- **Expected path:** `ai_engine/models/lgbm_model.pkl`
- **Actual location:** `/home/qt/quantum_trader/models/lightgbm_v*.pkl`
- **Result:** `self.model = None` → fallback activated

**Code Location:** `ai_engine/agents/lgbm_agent.py:247`
```python
def predict(...):
    if self.model is None:
        return self._fallback_prediction(features)
```

**Fallback Logic:**
```python
# Simple RSI + EMA rules
if rsi < 30:
    action = 'BUY'; confidence = 0.55 + (30 - rsi) / 60
elif rsi > 70:
    action = 'SELL'; confidence = 0.55 + (rsi - 70) / 60
```

**Telemetry Impact:**
- confidence = 1.05 indicates RULES ENGINE active (not model)
- Action distribution: 49.5% BUY, 50.1% SELL (balanced but rule-based)
- std = 0.0567, range = 0.1074 (narrow but better than collapsed models)

**Ensemble Weight:** Reported as 0.25 (25%), **actual: rules-based voting**

---

### 3️⃣ NHiTS: FEATURE DIMENSION MISMATCH ⚠️
**Evidence:**
```
Jan 10 21:46:51 [WARNING] [NHITS] Adjusted feature dimension 14 -> 12 to match scaler/model
```

**Code Location:** `ai_engine/agents/nhits_agent.py:265`
```python
def _match_sequence(self, sequence, target_len):
    if sequence.shape[-1] < target_len:
        pad_width = target_len - sequence.shape[-1]
        pad = np.zeros((sequence.shape[0], pad_width))
        adjusted = np.concatenate([sequence, pad], axis=-1)
        logger.warning("[NHITS] Adjusted feature dimension %d -> %d", ...)
```

**Impact:**
- Feature vector **truncated/padded** every prediction
- Loses 2 features OR pads with zeros
- Results in **near-constant predictions**:
  - 99.6% SELL predictions
  - confidence P10 = P90 = 0.5364 (CONSTANT)
  - std = 0.0086

**Ensemble Weight:** Reported as 0.25 (25%), **actual: corrupted voting**

---

### 4️⃣ XGBoost: ONLY FUNCTIONAL MODEL ✅
**Status:** LOADED and ACTIVE (but low variance)
- Action: 100% BUY
- confidence std = 0.0069 (very low)
- Range: [0.9750 - 0.9959]

**Ensemble Weight:** Reported as 0.25 (25%), **actual: ~100% effective voting power**

---

## 🎯 SYSTEMIC ISSUE: FALSE ENSEMBLE REPORTING

**Reported Weights:**
```python
{'PatchTST': 0.25, 'NHiTS': 0.25, 'XGBoost': 0.25, 'LightGBM': 0.25}
```

**Actual Effective Weights:**
```
PatchTST:  0% (shadow mode - excluded from voting)
NHiTS:     ~5% (corrupted features - garbage vote)
XGBoost:   ~90% (only functional model)
LightGBM:  ~5% (rules engine - not ML model)
```

**Danger:** System APPEARS robust (4 models @ 25% each) but is **single point of failure** (XGBoost only)

---

## 🚨 QUALITY GATE VIOLATIONS

**Total:** 15 violations across 3 models

**PatchTST:** 5 violations
- ❌ Flat predictions (std = 0.0000)
- ❌ Narrow range (P10-P90 = 0.0000)
- ❌ Action collapse (100% BUY)
- ❌ Constant output (0.6150)
- ❌ Dead zone (confidence in [0.4-0.6])

**NHiTS:** 5 violations
- ❌ Flat predictions (std = 0.0086)
- ❌ Narrow range (P10-P90 = 0.0000)
- ❌ Action collapse (99.6% SELL)
- ❌ Constant output (0.5364)
- ❌ Dead zone (confidence in [0.4-0.6])

**XGBoost:** 4 violations
- ❌ Narrow range (P10-P90 = 0.0162)
- ❌ Action collapse (100% BUY)
- ❌ Low variance (std = 0.0069)
- ⚠️ Flat predictions (borderline)

**LGBM:** 1 violation
- ❌ Narrow range (P10-P90 = 0.1074 < 0.12)

---

## 🛠️ REQUIRED FIXES

### Priority 1: DISABLE SHADOW MODE
**File:** `ai_engine/agents/patchtst_agent.py`
```python
# BEFORE:
shadow_mode = os.getenv('PATCHTST_SHADOW_ONLY', 'false').lower() == 'true'

# AFTER: Force disable shadow mode for canary deployment
shadow_mode = False  # QSC MODE: All models must vote
```

OR: Set environment variable:
```bash
export PATCHTST_SHADOW_ONLY=false
```

### Priority 2: FIX LGBM MODEL PATH
**File:** `ai_engine/agents/lgbm_agent.py:38`
```python
# BEFORE:
self.model_path = model_path or str(latest_model) if latest_model else "ai_engine/models/lgbm_model.pkl"

# AFTER: Use correct path
self.model_path = model_path or str(latest_model) if latest_model else "models/lightgbm_v20251228_154858.pkl"
```

OR: Pass correct model_path at initialization

### Priority 3: FIX NHITS FEATURE DIMENSION
**Option A (FAIL-CLOSED):** Raise exception instead of silent padding
```python
def _match_sequence(self, sequence, target_len):
    if sequence.shape[-1] != target_len:
        raise ValueError(f"Feature dimension mismatch: {sequence.shape[-1]} != {target_len}")
```

**Option B (FIX UPSTREAM):** Fix feature engineering to produce correct dimension

**Option C (TEMPORARY):** Document expected dimension and validate at load time

---

## 📊 POST-FIX VALIDATION PLAN

1. **Restart AI Engine:**
   ```bash
   sudo systemctl restart quantum-ai-engine.service
   ```

2. **Capture New Cutover Timestamp:**
   ```bash
   systemctl show quantum-ai-engine.service -p ActiveEnterTimestamp
   ```

3. **Wait for ≥200 events:**
   ```bash
   redis-cli XLEN quantum:stream:trade.intent
   ```

4. **Re-run Quality Gate:**
   ```bash
   cd /home/qt/quantum_trader
   ops/run.sh ai-engine python3 ops/model_safety/quality_gate.py --after <CUTOVER_TS>
   ```

5. **Expected Outcome:**
   - ✅ Exit code 0
   - ✅ All 4 models with proper variance
   - ✅ No SHADOW markers in telemetry
   - ✅ No fallback_rules markers
   - ✅ No feature dimension warnings
   - ✅ <15 violations total
   - ✅ Confidence distributions: std > 0.05, P10-P90 > 0.12

---

## 🔒 FAIL-CLOSED ENFORCEMENT

**QSC MODE BLOCKED until:**
- [ ] PatchTST shadow mode disabled
- [ ] LGBM model loaded (not fallback)
- [ ] NHiTS feature dimension correct
- [ ] Quality gate exits 0
- [ ] ≥200 post-fix events collected

**No manual override permitted.** System correctly prevented catastrophic deployment.

---

## 📚 LESSONS LEARNED

1. **Silent Degradation:** Models degraded to fallback/shadow without alerts
2. **False Reporting:** Ensemble weights reported as balanced when 3/4 models inactive
3. **Feature Pipeline Fragility:** Dimension mismatches silently padded instead of failing
4. **Path Management:** Model file paths not validated at startup
5. **Quality Gates Work:** FAIL-CLOSED design prevented deployment of broken system

**Mental Model Validated:** 
> "Quality Gate = Dommer, QSC Canary = Sikkerhetsbelte, Model = Mistenkt, Telemetry = Bevis, Systemd = Lovens håndhever. Ingen av disse har lov til å være 'snille'."

System performed exactly as designed - it stopped us before disaster.

---

**Generated:** 2026-01-10T22:15:00Z  
**Quality Gate Report:** `reports/safety/quality_gate_20260110_214620_post_cutover.md`  
**Diagnosis Report:** `reports/safety/diagnosis_20260110_214640.md`
