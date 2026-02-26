#!/bin/bash
SSH="ssh -i ~/.ssh/hetzner_fresh -o StrictHostKeyChecking=no root@46.224.116.254"

echo "=== STEP 1: Check /opt/quantum/models and what LGBM retrain scripts look like ==="
$SSH '
echo "--- /opt/quantum/models/ ---"
ls -la /opt/quantum/models/ 2>/dev/null | tail -30 || echo "NOT FOUND"

echo ""
echo "--- train_lightgbm_v6.py structure ---"
head -80 /opt/quantum/ops/retrain/train_lightgbm_v6.py 2>/dev/null || echo "NOT FOUND"

echo ""
echo "--- LGBM agent exact scaler_path construction ---"
grep -n "scaler_path\|model_path\|retraining_dir\|_v2\|find_latest" /opt/quantum/ai_engine/agents/lgbm_agent.py | head -25

echo ""
echo "--- Where LGBM agent actually resolves scaler ---"
sed -n "30,58p" /opt/quantum/ai_engine/agents/lgbm_agent.py
'
