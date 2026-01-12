#!/bin/bash
# Quantum Trader v5 - Complete Training & Deployment Pipeline
# Trains all 4 models (XGBoost, LightGBM, PatchTST, N-HiTS) and validates ensemble

set -e  # Exit on error

echo "===================================================================="
echo "QUANTUM TRADER V5 - COMPLETE TRAINING PIPELINE"
echo "===================================================================="

# Check if running on VPS
if [ ! -f "/opt/quantum/venvs/ai-engine/bin/activate" ]; then
    echo "❌ ERROR: Not on VPS or venv not found"
    echo "   Expected: /opt/quantum/venvs/ai-engine/bin/activate"
    exit 1
fi

# Activate venv
echo ""
echo "🔧 Activating virtual environment..."
source /opt/quantum/venvs/ai-engine/bin/activate

# Navigate to project
cd /home/qt/quantum_trader

echo ""
echo "===================================================================="
echo "STEP 1: TRAIN XGBOOST V5 (if not already done)"
echo "===================================================================="
if [ -f "ai_engine/models/xgb_v*_v5.pkl" ]; then
    echo "✅ XGBoost v5 model already exists, skipping..."
else
    echo "🏋️ Training XGBoost v5..."
    python3 ops/retrain/fetch_and_train_xgb_v5.py
fi

echo ""
echo "===================================================================="
echo "STEP 2: TRAIN LIGHTGBM V5"
echo "===================================================================="
echo "🏋️ Training LightGBM v5..."
python3 ops/retrain/train_lightgbm_v5.py

echo ""
echo "===================================================================="
echo "STEP 3: TRAIN PATCHTST V5"
echo "===================================================================="
echo "🏋️ Training PatchTST v5 (this may take 10-15 minutes)..."
python3 ops/retrain/train_patchtst_v5.py

echo ""
echo "===================================================================="
echo "STEP 4: TRAIN N-HITS V5"
echo "===================================================================="
echo "🏋️ Training N-HiTS v5..."
python3 ops/retrain/train_nhits_v5.py

echo ""
echo "===================================================================="
echo "STEP 5: DEPLOY ALL MODELS TO PRODUCTION"
echo "===================================================================="
echo "📦 Copying models to /opt/quantum/ai_engine/models/..."
sudo cp ai_engine/models/*_v5*.pkl /opt/quantum/ai_engine/models/ 2>/dev/null || true
sudo cp ai_engine/models/*_v5*.pth /opt/quantum/ai_engine/models/ 2>/dev/null || true
sudo chown -R qt:qt /opt/quantum/ai_engine/models/

echo "✅ Models deployed"

echo ""
echo "===================================================================="
echo "STEP 6: RESTART AI ENGINE SERVICE"
echo "===================================================================="
echo "🔄 Restarting quantum-ai-engine.service..."
sudo systemctl restart quantum-ai-engine.service

echo "⏳ Waiting for service to start..."
sleep 5

echo ""
echo "===================================================================="
echo "STEP 7: VERIFY MODEL LOADING"
echo "===================================================================="
echo "📋 Checking journalctl for agent initialization..."
journalctl -u quantum-ai-engine.service --since "10 seconds ago" | grep -E "Agent.*Loaded|XGB-Agent|LGBM-Agent|PatchTST-Agent|NHiTS-Agent" | tail -10

echo ""
echo "===================================================================="
echo "STEP 8: VALIDATE ENSEMBLE"
echo "===================================================================="
echo "🧪 Running ensemble validation..."
python3 ops/validate_ensemble_v5.py

echo ""
echo "===================================================================="
echo "STEP 9: CHECK LOG FILES"
echo "===================================================================="
echo "📁 Agent log files:"
ls -lh /var/log/quantum/*agent.log 2>/dev/null || echo "No log files found yet"

echo ""
echo "===================================================================="
echo "✅ COMPLETE! ALL STEPS FINISHED"
echo "===================================================================="
echo ""
echo "📊 To monitor predictions in real-time:"
echo "   tail -f /var/log/quantum/xgb-agent.log"
echo "   tail -f /var/log/quantum/lgbm-agent.log"
echo "   tail -f /var/log/quantum/patchtst-agent.log"
echo "   tail -f /var/log/quantum/nhits-agent.log"
echo ""
echo "🔍 To check service status:"
echo "   sudo systemctl status quantum-ai-engine.service"
echo ""
echo "📈 To view recent predictions:"
echo "   journalctl -u quantum-ai-engine.service --since '5 minutes ago' | grep 'Agent.*→'"
echo ""
