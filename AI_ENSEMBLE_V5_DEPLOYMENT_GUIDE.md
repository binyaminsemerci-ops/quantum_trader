# 🚀 Unified Ensemble v5 - Complete Deployment Guide

## 📋 Current Status

### ✅ Completed:
- **XGBoost v5**: Active in production (18 features, variety confirmed)
- **Unified Agent System**: Deployed and working
- **Feature Pipeline**: v5 features (18) calculated in service.py
- **Training Scripts**: LightGBM, PatchTST, N-HiTS ready

### 🎯 Objective:
Deploy full 4-model ensemble with real data distribution (no synthetic balancing)

---

## 1️⃣ LIGHTGBM V5 TRAINING

### On VPS:
```bash
cd /home/qt/quantum_trader
source /opt/quantum/venvs/ai-engine/bin/activate

# Train LightGBM v5
python3 ops/retrain/train_lightgbm_v5.py
```

### Expected Output:
```
LightGBM v5 Training - Aligned with XGBoost v5 (18 features)
====================================================================
📊 Loading data from datasets/training_data_v5.csv
   Loaded XXXXX rows
📈 Class distribution (REAL DATA - no balancing):
   SELL (0): XXXX (XX.X%)
   HOLD (1): XXXX (XX.X%)
   BUY (2): XXXX (XX.X%)
⚖️ Class weights (natural balancing):
   SELL (0): X.XX
   HOLD (1): X.XX
   BUY (2): X.XX
🚀 Training LightGBM v5...
   Best iteration: XXX
   Accuracy: XX.XX%
💾 Saving model to lightgbm_vYYYYMMDD_HHMMSS_v5.pkl
🎉 LightGBM v5 training complete!
```

### Deploy:
```bash
sudo cp ai_engine/models/lightgbm_v*_v5*.pkl /opt/quantum/ai_engine/models/
sudo chown qt:qt /opt/quantum/ai_engine/models/lightgbm_v*
sudo systemctl restart quantum-ai-engine.service

# Verify
journalctl -u quantum-ai-engine.service --since "15s ago" | grep LGBM-Agent
tail -f /var/log/quantum/lgbm-agent.log
```

---

## 2️⃣ PATCHTST V5 TRAINING

### Requirements Check:
```bash
# Ensure PyTorch is installed
python3 -c "import torch; print(torch.__version__)"
```

### Train PatchTST:
```bash
cd /home/qt/quantum_trader
source /opt/quantum/venvs/ai-engine/bin/activate

# Train PatchTST v5 (may take 10-30 minutes)
python3 ops/retrain/train_patchtst_v5.py
```

### Expected Output:
```
PatchTST v5 Training - Aligned with XGBoost v5 (18 features)
Device: cuda (or cpu)
====================================================================
📊 Loading data...
📈 Class distribution (REAL DATA - no balancing):
🏗️ Building PatchTST model
   Parameters: XXX,XXX
🚀 Training PatchTST v5 (epochs=100, patience=15)
   Epoch  10/100 | Train Loss: X.XXXX | Val Loss: X.XXXX | Val Acc: XX.XX%
   ...
   Early stopping at epoch XX
📊 Final Evaluation:
   Accuracy: XX.XX%
📋 Classification Report:
              precision    recall  f1-score   support
        SELL       X.XX      X.XX      X.XX      XXXX
        HOLD       X.XX      X.XX      X.XX      XXXX
         BUY       X.XX      X.XX      X.XX      XXXX
✅ Variety Check: 3/3 unique predictions
💾 Saving model to patchtst_vYYYYMMDD_HHMMSS_v5.pth
🎉 PatchTST v5 training complete!
```

### Deploy:
```bash
sudo cp ai_engine/models/patchtst_v*_v5* /opt/quantum/ai_engine/models/
sudo chown qt:qt /opt/quantum/ai_engine/models/patchtst_v*
sudo systemctl restart quantum-ai-engine.service

# Verify
journalctl -u quantum-ai-engine.service --since "15s ago" | grep PatchTST-Agent
tail -f /var/log/quantum/patchtst-agent.log
```

---

## 3️⃣ N-HITS V5 TRAINING

### Train N-HiTS:
```bash
cd /home/qt/quantum_trader
source /opt/quantum/venvs/ai-engine/bin/activate

# Train N-HiTS v5
python3 ops/retrain/train_nhits_v5.py
```

### Expected Output:
```
N-HiTS v5 Training - Aligned with XGBoost v5 (18 features)
====================================================================
📊 Loading data...
📈 Class distribution (REAL DATA - no balancing):
🏗️ Building N-HiTS model
   Parameters: XXX,XXX
🚀 Training N-HiTS v5 (epochs=100, patience=15)
   ...
📊 Final Evaluation:
   Accuracy: XX.XX%
✅ Variety Check: 3/3 unique predictions
💾 Saving model to nhits_vYYYYMMDD_HHMMSS_v5.pth
🎉 N-HiTS v5 training complete!
```

### Deploy:
```bash
sudo cp ai_engine/models/nhits_v*_v5* /opt/quantum/ai_engine/models/
sudo chown qt:qt /opt/quantum/ai_engine/models/nhits_v*
sudo systemctl restart quantum-ai-engine.service

# Verify
journalctl -u quantum-ai-engine.service --since "15s ago" | grep NHiTS-Agent
tail -f /var/log/quantum/nhits-agent.log
```

---

## 4️⃣ ENSEMBLE VALIDATION

### Run Validation Script:
```bash
cd /home/qt/quantum_trader
source /opt/quantum/venvs/ai-engine/bin/activate

# Comprehensive ensemble test
python3 ops/validation/ensemble_validate_v5.py
```

### Expected Output:
```
====================================================================
QUANTUM TRADER - ENSEMBLE V5 VALIDATION
====================================================================

🔧 Initializing Ensemble Manager...

📊 AGENT STATUS:
   ✅ XGBoost      ACTIVE (version: xgb_v20260112_040603_v5.pkl)
   ✅ LightGBM     ACTIVE (version: lightgbm_vXXXXXXXX_XXXXXX_v5.pkl)
   ✅ PatchTST     ACTIVE (version: patchtst_vXXXXXXXX_XXXXXX_v5.pth)
   ✅ N-HiTS       ACTIVE (version: nhits_vXXXXXXXX_XXXXXX_v5.pth)

🎯 Active Agents: 4/4
✅ PASS: 4/4 agents active

🔮 ENSEMBLE PREDICTIONS:
----------------------------------------------------------------------

BTCUSDT:
   Action: BUY
   Confidence: 0.723
   Active Models: ['xgb', 'lgbm', 'patchtst', 'nhits']
      [xgb      ] BUY  (conf=0.720)
      [lgbm     ] BUY  (conf=0.680)
      [patchtst ] HOLD (conf=0.750)
      [nhits    ] BUY  (conf=0.740)

ETHUSDT:
   Action: HOLD
   Confidence: 0.685
   Active Models: ['xgb', 'lgbm', 'patchtst', 'nhits']
      [xgb      ] HOLD (conf=0.650)
      [lgbm     ] HOLD (conf=0.720)
      [patchtst ] HOLD (conf=0.680)
      [nhits    ] SELL (conf=0.690)

BNBUSDT:
   Action: HOLD
   Confidence: 0.701
   Active Models: ['xgb', 'lgbm', 'patchtst', 'nhits']
      [xgb      ] HOLD (conf=0.870)
      [lgbm     ] HOLD (conf=0.650)
      [patchtst ] BUY  (conf=0.620)
      [nhits    ] HOLD (conf=0.730)

====================================================================
VALIDATION RESULTS:
====================================================================

1️⃣ Active Agents: 4/4 ✅ PASS
2️⃣ Prediction Variety: 3/3 unique actions ✅ PASS
   Actions: {'BUY', 'HOLD', 'SELL'}
3️⃣ Confidence Distribution: mean=0.703, std=0.019 ✅ PASS
4️⃣ Error Check: 0 errors ✅ PASS

====================================================================
🎉 VALIDATION PASSED: Ensemble v5 is operational!
====================================================================
```

---

## 5️⃣ PRODUCTION MONITORING

### Check Agent Logs:
```bash
# All agents
ls -lh /var/log/quantum/

# Individual agents
tail -f /var/log/quantum/xgb-agent.log
tail -f /var/log/quantum/lgbm-agent.log
tail -f /var/log/quantum/patchtst-agent.log
tail -f /var/log/quantum/nhits-agent.log
```

### Check Service Logs:
```bash
# Real-time monitoring
journalctl -u quantum-ai-engine.service -f | grep -E "Agent.*→|ENSEMBLE|ERROR"

# Recent predictions
journalctl -u quantum-ai-engine.service --since "5 minutes ago" | grep "Agent.*→"
```

### Signal Distribution:
```bash
# Count signals by type (last 5 min)
journalctl -u quantum-ai-engine.service --since "5 minutes ago" | \
  grep -oP "Agent\] \K\w+USDT → (HOLD|BUY|SELL)" | \
  sort | uniq -c
```

---

## 6️⃣ TROUBLESHOOTING

### Issue: LightGBM "X has 5 features, but StandardScaler is expecting 12"
**Cause**: Old LightGBM model trained with different features  
**Fix**: Retrain with v5 script (step 1)

### Issue: PatchTST/N-HiTS not loading
**Cause**: Missing PyTorch or model files  
**Fix**:
```bash
pip install torch
ls -lh /opt/quantum/ai_engine/models/*_v5*
```

### Issue: "Degenerate output detected"
**Cause**: Model predicting 100% same action  
**Fix**: Retrain with natural class balancing (scripts provided)

### Issue: Confidence std < 0.02
**Cause**: All models too confident in same direction  
**Fix**: Check feature variety, retrain if needed

### Issue: Agent logs empty
**Cause**: Permissions or path issues  
**Fix**:
```bash
sudo mkdir -p /var/log/quantum
sudo chown -R qt:qt /var/log/quantum
sudo chmod 755 /var/log/quantum
```

---

## 7️⃣ SUCCESS CRITERIA

### ✅ PASS Conditions:
- [ ] **Active Agents**: ≥3/4 models loaded
- [ ] **Variety**: ≥2 unique actions in predictions
- [ ] **Confidence Std**: >0.02 across ensemble
- [ ] **All v5**: All models report version=v5
- [ ] **Features**: 18 features used, 0 missing
- [ ] **No Degeneracy**: No "100% HOLD/BUY/SELL" warnings

### ❌ FAIL Conditions:
- Less than 3/4 agents active
- Only 1 unique action predicted
- Confidence std < 0.02
- "Degenerate output" errors
- Feature count mismatch

---

## 8️⃣ PRODUCTION METRICS

### Target Performance:
```
Ensemble Configuration:
  - XGBoost:  25% weight (tree-based)
  - LightGBM: 25% weight (fast tree)
  - PatchTST: 30% weight (transformer)
  - N-HiTS:   20% weight (multi-rate)

Expected Accuracy: 75-85%
Signal Variety: BUY/HOLD/SELL all present
Confidence: 0.55-0.90 range
Std: >0.02 (healthy disagreement)
```

### Monitoring Commands:
```bash
# Quick status
systemctl status quantum-ai-engine.service

# Active models
journalctl -u quantum-ai-engine.service --since "1 minute ago" | grep "✅.*ACTIVE"

# Recent signals
journalctl -u quantum-ai-engine.service --since "5 minutes ago" | \
  grep -E "XGB-Agent|LGBM-Agent|PatchTST-Agent|NHiTS-Agent" | \
  grep "→" | tail -20

# Error check
journalctl -u quantum-ai-engine.service --since "10 minutes ago" | grep ERROR
```

---

## 📁 File Locations

### Training Scripts:
```
ops/retrain/train_lightgbm_v5.py   - LightGBM training
ops/retrain/train_patchtst_v5.py   - PatchTST training
ops/retrain/train_nhits_v5.py      - N-HiTS training
ops/retrain/fetch_and_train_xgb_v5.py  - XGBoost training (already done)
```

### Production Models:
```
/opt/quantum/ai_engine/models/xgb_v20260112_040603_v5.pkl
/opt/quantum/ai_engine/models/lightgbm_v*_v5.pkl
/opt/quantum/ai_engine/models/patchtst_v*_v5.pth
/opt/quantum/ai_engine/models/nhits_v*_v5.pth
```

### Agent Code:
```
/home/qt/quantum_trader/ai_engine/agents/unified_agents.py
/home/qt/quantum_trader/ai_engine/ensemble_manager.py
```

### Logs:
```
/var/log/quantum/xgb-agent.log
/var/log/quantum/lgbm-agent.log
/var/log/quantum/patchtst-agent.log
/var/log/quantum/nhits-agent.log
```

---

## ⏭️ Next Steps

1. **Train LightGBM v5** (highest priority - 10-20 minutes)
2. **Train PatchTST v5** (medium priority - 20-40 minutes)
3. **Train N-HiTS v5** (medium priority - 15-30 minutes)
4. **Run Validation** (verify 3-4/4 active)
5. **Monitor Production** (24-48 hours)
6. **Generate Performance Report** (track accuracy, variety, confidence)

---

**Status**: Ready for full v5 ensemble deployment 🚀
