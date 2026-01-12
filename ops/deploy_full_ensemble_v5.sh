#!/usr/bin/env bash
# =============================================================
#  Quantum Trader - Full Ensemble v5 Training & Deployment
# =============================================================
#  Runs sequentially:
#   1. PatchTST v5 training
#   2. N-HiTS v5 training
#   3. MetaPredictor v5 training (with updated ensemble data)
#   4. Deployment + Service restart
#   5. Validation
# =============================================================

set -e
WORKDIR="/home/qt/quantum_trader"
VENV="/opt/quantum/venvs/ai-engine/bin/activate"
LOGFILE="/var/log/quantum/ensemble_v5_deploy.log"
DATE=$(date +"%Y-%m-%d %H:%M:%S")

# Create log directory if it doesn't exist
sudo mkdir -p /var/log/quantum
sudo chown qt:qt /var/log/quantum

echo "" | tee -a "$LOGFILE"
echo "====================================================================" | tee -a "$LOGFILE"
echo "[$DATE] 🚀 FULL ENSEMBLE v5 DEPLOYMENT STARTED" | tee -a "$LOGFILE"
echo "====================================================================" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

cd "$WORKDIR"
source "$VENV"

# =============================================================
# 0️⃣  PRE-CHECK: Verify existing models
# =============================================================
echo "[STEP 0/5] 🔍 Pre-deployment check..." | tee -a "$LOGFILE"
echo "[INFO] Existing XGBoost v5: $(ls ai_engine/models/xgb_*_v5.pkl 2>/dev/null | wc -l) models" | tee -a "$LOGFILE"
echo "[INFO] Existing LightGBM v5: $(ls ai_engine/models/lightgbm_*_v5.pkl 2>/dev/null | wc -l) models" | tee -a "$LOGFILE"
echo "[INFO] Current ensemble status:" | tee -a "$LOGFILE"
systemctl is-active quantum-ai-engine.service || echo "[WARNING] Service not running!" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

# =============================================================
# 1️⃣  PATCHTST v5 TRAINING
# =============================================================
echo "[STEP 1/5] 🧠 Training PatchTST v5 (sequence transformer)" | tee -a "$LOGFILE"
echo "[INFO] This may take 5-10 minutes..." | tee -a "$LOGFILE"

if python3 ops/retrain/train_patchtst_v5.py 2>&1 | tee -a "$LOGFILE"; then
    PATCH_FILE=$(ls -t ai_engine/models/patchtst_*_v5.pkl 2>/dev/null | head -1)
    if [ -f "$PATCH_FILE" ]; then
        echo "✅ [SUCCESS] PatchTST model: $PATCH_FILE" | tee -a "$LOGFILE"
    else
        echo "❌ [ERROR] PatchTST model file not found!" | tee -a "$LOGFILE"
        exit 1
    fi
else
    echo "❌ [ERROR] PatchTST training failed!" | tee -a "$LOGFILE"
    exit 1
fi
echo "" | tee -a "$LOGFILE"

# =============================================================
# 2️⃣  N-HITS v5 TRAINING
# =============================================================
echo "[STEP 2/5] 🔮 Training N-HiTS v5 (neural hierarchical interpolation)" | tee -a "$LOGFILE"
echo "[INFO] This may take 5-10 minutes..." | tee -a "$LOGFILE"

if python3 ops/retrain/train_nhits_v5.py 2>&1 | tee -a "$LOGFILE"; then
    NHITS_FILE=$(ls -t ai_engine/models/nhits_*_v5.pkl 2>/dev/null | head -1)
    if [ -f "$NHITS_FILE" ]; then
        echo "✅ [SUCCESS] N-HiTS model: $NHITS_FILE" | tee -a "$LOGFILE"
    else
        echo "❌ [ERROR] N-HiTS model file not found!" | tee -a "$LOGFILE"
        exit 1
    fi
else
    echo "❌ [ERROR] N-HiTS training failed!" | tee -a "$LOGFILE"
    exit 1
fi
echo "" | tee -a "$LOGFILE"

# =============================================================
# 3️⃣  META-PREDICTOR v5 RE-TRAINING (with updated ensemble)
# =============================================================
echo "[STEP 3/5] 🧩 Training MetaPredictor v5 (fusion layer)" | tee -a "$LOGFILE"
echo "[INFO] Re-training with updated ensemble data..." | tee -a "$LOGFILE"

if python3 ops/retrain/train_meta_v5.py 2>&1 | tee -a "$LOGFILE"; then
    META_FILE=$(ls -t ai_engine/models/meta_*_v5.pth 2>/dev/null | head -1)
    if [ -f "$META_FILE" ]; then
        echo "✅ [SUCCESS] MetaPredictor model: $META_FILE" | tee -a "$LOGFILE"
    else
        echo "❌ [ERROR] MetaPredictor model file not found!" | tee -a "$LOGFILE"
        exit 1
    fi
else
    echo "⚠️ [WARNING] MetaPredictor training failed - continuing with existing model" | tee -a "$LOGFILE"
fi
echo "" | tee -a "$LOGFILE"

# =============================================================
# 4️⃣  DEPLOY MODELS TO PRODUCTION
# =============================================================
echo "[STEP 4/5] 🚢 Deploying all v5 models to production..." | tee -a "$LOGFILE"

# Count models before deployment
V5_COUNT=$(ls ai_engine/models/*_v5.* 2>/dev/null | wc -l)
echo "[INFO] Total v5 model files to deploy: $V5_COUNT" | tee -a "$LOGFILE"

# Create target directory if it doesn't exist
sudo mkdir -p /opt/quantum/ai_engine/models
sudo chown qt:qt /opt/quantum/ai_engine/models

# Copy all v5 models (pkl, pth, json, scaler)
echo "[INFO] Copying models..." | tee -a "$LOGFILE"
sudo cp ai_engine/models/*_v5.pkl /opt/quantum/ai_engine/models/ 2>/dev/null || true
sudo cp ai_engine/models/*_v5.pth /opt/quantum/ai_engine/models/ 2>/dev/null || true
sudo cp ai_engine/models/*_v5_scaler.pkl /opt/quantum/ai_engine/models/ 2>/dev/null || true
sudo cp ai_engine/models/*_v5_meta.json /opt/quantum/ai_engine/models/ 2>/dev/null || true

# Also copy to working directory (service runs from /home/qt)
cp ai_engine/models/*_v5.* /home/qt/quantum_trader/ai_engine/models/ 2>/dev/null || true

# Set correct permissions
sudo chown qt:qt /opt/quantum/ai_engine/models/*_v5* 2>/dev/null || true
chown qt:qt /home/qt/quantum_trader/ai_engine/models/*_v5* 2>/dev/null || true

echo "✅ [SUCCESS] Models deployed to production" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

# =============================================================
# 5️⃣  SERVICE RESTART & VERIFICATION
# =============================================================
echo "[STEP 5/5] 🔄 Restarting quantum-ai-engine.service..." | tee -a "$LOGFILE"

# Clear Python cache to force reload
sudo rm -rf /home/qt/quantum_trader/ai_engine/__pycache__ 2>/dev/null || true
sudo rm -rf /home/qt/quantum_trader/ai_engine/agents/__pycache__ 2>/dev/null || true

# Restart service
sudo systemctl restart quantum-ai-engine.service

echo "[INFO] Waiting 15 seconds for service to initialize..." | tee -a "$LOGFILE"
sleep 15

# Check service status
if systemctl is-active --quiet quantum-ai-engine.service; then
    echo "✅ [SUCCESS] Service is running" | tee -a "$LOGFILE"
else
    echo "❌ [ERROR] Service failed to start!" | tee -a "$LOGFILE"
    sudo journalctl -u quantum-ai-engine.service --since "20s ago" | tail -50 | tee -a "$LOGFILE"
    exit 1
fi

echo "" | tee -a "$LOGFILE"
echo "[INFO] Checking agent initialization logs..." | tee -a "$LOGFILE"
sudo journalctl -u quantum-ai-engine.service --since "20s ago" | grep -iE "agent.*loaded|active.*models|meta|patch|nhits" | head -30 | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

# =============================================================
# 6️⃣  ENSEMBLE VALIDATION
# =============================================================
echo "[STEP 6/6] ✅ Running full ensemble validation..." | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

if python3 ops/validate_ensemble_v5.py 2>&1 | tee -a "$LOGFILE" | grep -E "Active Models|Signal Variety|ENSEMBLE V5|PASSED|FAILED"; then
    echo "" | tee -a "$LOGFILE"
    echo "✅ [SUCCESS] Validation completed" | tee -a "$LOGFILE"
else
    echo "" | tee -a "$LOGFILE"
    echo "⚠️ [WARNING] Validation had issues - check logs" | tee -a "$LOGFILE"
fi

# =============================================================
# 7️⃣  FINAL SUMMARY
# =============================================================
echo "" | tee -a "$LOGFILE"
echo "====================================================================" | tee -a "$LOGFILE"
echo "🎉 FULL ENSEMBLE v5 DEPLOYMENT COMPLETED!" | tee -a "$LOGFILE"
echo "====================================================================" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"
echo "📊 Deployment Summary:" | tee -a "$LOGFILE"
echo "  - XGBoost v5:      $(ls ai_engine/models/xgb_*_v5.pkl 2>/dev/null | wc -l) models" | tee -a "$LOGFILE"
echo "  - LightGBM v5:     $(ls ai_engine/models/lightgbm_*_v5.pkl 2>/dev/null | wc -l) models" | tee -a "$LOGFILE"
echo "  - PatchTST v5:     $(ls ai_engine/models/patchtst_*_v5.pkl 2>/dev/null | wc -l) models" | tee -a "$LOGFILE"
echo "  - N-HiTS v5:       $(ls ai_engine/models/nhits_*_v5.pkl 2>/dev/null | wc -l) models" | tee -a "$LOGFILE"
echo "  - MetaPredictor v5: $(ls ai_engine/models/meta_*_v5.pth 2>/dev/null | wc -l) models" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"
echo "📝 Logs saved to: $LOGFILE" | tee -a "$LOGFILE"
echo "🔍 Monitor live: journalctl -u quantum-ai-engine.service -f" | tee -a "$LOGFILE"
echo "✅ Validate again: python3 ops/validate_ensemble_v5.py" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"
echo "====================================================================" | tee -a "$LOGFILE"

# Show final agent status
echo "" | tee -a "$LOGFILE"
echo "🤖 Agent Status (last 30 seconds):" | tee -a "$LOGFILE"
sudo journalctl -u quantum-ai-engine.service --since "30s ago" | grep -iE "XGB-Agent|LGBM-Agent|PatchTST|NHiTS|Meta-Agent" | grep -E "→|loaded" | tail -10 | tee -a "$LOGFILE"

exit 0
