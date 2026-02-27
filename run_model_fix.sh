#!/bin/bash
SSH="ssh -i ~/.ssh/hetzner_fresh -o StrictHostKeyChecking=no root@46.224.116.254"
SCP="scp -i ~/.ssh/hetzner_fresh -o StrictHostKeyChecking=no"

echo "================================================================="
echo "STEP 1: Upload and run agent patches (XGB symlinks + LGBM path fix)"
echo "================================================================="
$SCP /mnt/c/quantum_trader/patch_agents.py root@46.224.116.254:/tmp/patch_agents.py
$SSH 'python3 /tmp/patch_agents.py'

echo ""
echo "================================================================="
echo "STEP 2: Retrain LGBM v6 with 49 features (live Binance data)"
echo "Note: This takes ~60-120 seconds"
echo "================================================================="
$SSH '
cd /opt/quantum
echo "Starting LightGBM v6 training..."
python3 ops/retrain/train_lightgbm_v6.py 2>&1
echo "LGBM v6 training done, exit=$?"
'

echo ""
echo "================================================================="
echo "STEP 3: Verify new LGBM and XGB model files"
echo "================================================================="
$SSH '
echo "--- New LGBM files (sorted newest first) ---"
ls -lht /opt/quantum/ai_engine/models/lightgbm_v*_v2.pkl 2>/dev/null | head -5
ls -lht /opt/quantum/ai_engine/models/lightgbm_scaler_v*_v2.pkl 2>/dev/null | head -5

echo ""
echo "--- XGB symlinks ---"
ls -la /opt/quantum/ai_engine/models/xgb_model.pkl \
        /opt/quantum/ai_engine/models/xgboost_v_prod.pkl \
        /opt/quantum/ai_engine/models/scaler.pkl \
        /opt/quantum/ai_engine/models/xgboost_v_prod_scaler.pkl

echo ""
echo "--- lgbm_agent retraining_dir patch ---"
grep -A 5 "retraining_dir" /opt/quantum/ai_engine/agents/lgbm_agent.py | head -10
'

echo ""
echo "================================================================="
echo "STEP 4: Commit changes"
echo "================================================================="
$SSH '
cd /opt/quantum
git add ai_engine/agents/lgbm_agent.py ai_engine/models/
git commit -m "fix(models): XGB symlinks to v6 (49 features) + LGBM agent retraining_dir path fix + LGBM v6 retrain" || echo "Nothing to commit"
git log --oneline -3
'

echo ""
echo "================================================================="
echo "STEP 5: Restart and verify"
echo "================================================================="
$SSH '
systemctl restart quantum-ai-engine
echo "Waiting 20s for startup..."
sleep 20

echo ""
echo "--- quantum-ai-engine status ---"
systemctl is-active quantum-ai-engine

echo ""
echo "--- Agent startup logs ---"
journalctl -u quantum-ai-engine --no-pager --since "-25s" | grep -E "XGB|LGBM|✅|Loaded|ERROR|features=" | head -25
'
