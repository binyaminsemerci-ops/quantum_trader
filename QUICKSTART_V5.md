# Quantum Trader v5 - Quick Start Guide

## 🚀 Complete Training & Deployment (One Command)

### On VPS:
```bash
cd /home/qt/quantum_trader
chmod +x ops/deploy_all_v5_models.sh
./ops/deploy_all_v5_models.sh
```

This script will:
1. ✅ Train XGBoost v5 (if not already done)
2. ✅ Train LightGBM v5
3. ✅ Train PatchTST v5
4. ✅ Train N-HiTS v5
5. ✅ Deploy all models to production
6. ✅ Restart AI engine service
7. ✅ Validate ensemble (all 4 agents)
8. ✅ Show results

**Total time**: ~20-30 minutes (depending on dataset size)

---

## 📋 Step-by-Step (Manual)

### 1. Train Individual Models

```bash
cd /home/qt/quantum_trader
source /opt/quantum/venvs/ai-engine/bin/activate

# Train each model
python3 ops/retrain/train_lightgbm_v5.py
python3 ops/retrain/train_patchtst_v5.py  # ~10-15 min
python3 ops/retrain/train_nhits_v5.py
```

### 2. Deploy to Production

```bash
# Copy models
sudo cp ai_engine/models/*_v5*.pkl /opt/quantum/ai_engine/models/
sudo cp ai_engine/models/*_v5*.pth /opt/quantum/ai_engine/models/
sudo chown -R qt:qt /opt/quantum/ai_engine/models/

# Restart service
sudo systemctl restart quantum-ai-engine.service
```

### 3. Verify Deployment

```bash
# Check model loading
journalctl -u quantum-ai-engine.service --since "15s ago" | grep "Agent.*Loaded"

# Expected output:
# [XGB-Agent] ✅ Loaded xgb_v20260112_040603_v5.pkl (18 features)
# [LGBM-Agent] ✅ Loaded lightgbm_v20260113_XXXXXX_v5.pkl (18 features)
# [PatchTST-Agent] ✅ Loaded patchtst_v20260113_XXXXXX_v5.pth (18 features)
# [NHiTS-Agent] ✅ Loaded nhits_v20260113_XXXXXX_v5.pth (18 features)
```

### 4. Validate Ensemble

```bash
python3 ops/validate_ensemble_v5.py
```

**Expected output:**
```
Active Models: 4/4
Signal Variety: 2-3 unique actions
Confidence Std: >0.02
✅ ENSEMBLE V5 VALIDATION: PASSED
```

---

## 🔍 Monitoring

### Real-time Predictions
```bash
# Watch all agents
tail -f /var/log/quantum/*agent.log

# Watch specific agent
tail -f /var/log/quantum/xgb-agent.log
```

### Check Service Status
```bash
sudo systemctl status quantum-ai-engine.service
```

### View Recent Predictions
```bash
journalctl -u quantum-ai-engine.service --since '5 minutes ago' | grep 'Agent.*→'
```

---

## ✅ Success Criteria

After deployment, verify:

- [ ] **4/4 models active** (XGBoost, LightGBM, PatchTST, N-HiTS)
- [ ] **Signal variety**: BUY, HOLD, SELL mix (not all same)
- [ ] **Confidence std > 0.02** (no degeneracy)
- [ ] **Log files exist**: `/var/log/quantum/*agent.log`
- [ ] **All agents version=v5**
- [ ] **18 features used, 0 missing**

---

## 🐛 Troubleshooting

### Model Not Loading
```bash
# Check if model files exist
ls -lh /opt/quantum/ai_engine/models/*_v5*

# Check permissions
ls -l /opt/quantum/ai_engine/models/ | grep qt
```

### Service Crashes
```bash
# Check for errors
journalctl -u quantum-ai-engine.service --since "1 minute ago" | grep -E "ERROR|Traceback"
```

### Low Accuracy (<70%)
- Check training data quality
- Verify feature engineering in service.py
- Review class distribution (should be natural, not synthetic)

### Degeneracy (all same predictions)
- Check scaler is loading correctly
- Verify 18 features calculated in service.py
- Ensure no data leakage or overfitting

---

## 📊 Current Status

**Deployed:**
- ✅ XGBoost v5 (18 features, 80.64% accuracy)
- ✅ Unified Agents system
- ✅ Feature pipeline (service.py)
- ✅ Dual logging (journald + files)

**Ready for Training:**
- ⚙️ LightGBM v5
- ⚙️ PatchTST v5
- ⚙️ N-HiTS v5

**Target:**
- 🎯 4/4 active ensemble
- 🎯 Varied predictions (BUY/HOLD/SELL)
- 🎯 Confidence std > 0.02

---

## 📁 Key Files

**Training Scripts:**
- `ops/retrain/fetch_and_train_xgb_v5.py`
- `ops/retrain/train_lightgbm_v5.py`
- `ops/retrain/train_patchtst_v5.py`
- `ops/retrain/train_nhits_v5.py`

**Validation:**
- `ops/validate_ensemble_v5.py`

**Deployment:**
- `ops/deploy_all_v5_models.sh` (automated)

**Production:**
- `/opt/quantum/ai_engine/models/*_v5*` (model files)
- `/var/log/quantum/*agent.log` (prediction logs)
- `/home/qt/quantum_trader/ai_engine/agents/unified_agents.py`

---

**Questions? Check the logs first!** 📋
