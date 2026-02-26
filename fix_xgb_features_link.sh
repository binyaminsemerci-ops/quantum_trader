#!/bin/bash
SSH="ssh -i ~/.ssh/hetzner_fresh -o StrictHostKeyChecking=no root@46.224.116.254"

echo "=== Fix XGB: create xgboost_v_prod_features.pkl symlink ==="
$SSH '
MODEL_DIR=/opt/quantum/ai_engine/models
cd $MODEL_DIR

echo "v6 features file exists:"
ls -la xgb_v6_20260217_003058_features.pkl

echo ""
echo "Creating xgboost_v_prod_features.pkl -> xgb_v6_20260217_003058_features.pkl"
ln -sf xgb_v6_20260217_003058_features.pkl xgboost_v_prod_features.pkl

echo ""
echo "Also creating xgb_model_features.pkl for xgb_model.pkl fallback:"
ln -sf xgb_v6_20260217_003058_features.pkl xgb_model_features.pkl

echo ""
echo "All XGB symlinks now:"
ls -la xgb_model.pkl xgboost_v_prod.pkl scaler.pkl xgboost_v_prod_scaler.pkl \
        xgboost_v_prod_features.pkl xgb_model_features.pkl 2>/dev/null

echo ""
echo "Let us also check what feature_names_path the agent builds (from the predict method):"
grep -n "feature_names_path\|_features\|feature_name" /opt/quantum/ai_engine/agents/xgb_agent.py | head -20
'

echo ""
echo "=== Restart service once more ==="
$SSH '
systemctl restart quantum-ai-engine
sleep 25
echo "--- Agent startup (last 28s) ---"
journalctl -u quantum-ai-engine --no-pager --since "-28s" | \
    grep -E "XGB-Agent|LGBM-Agent|✅ Loaded|ACTIVE|ERROR.*features" | head -20
'
