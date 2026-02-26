#!/bin/bash
SSH="ssh -i ~/.ssh/hetzner_fresh -o StrictHostKeyChecking=no root@46.224.116.254"

echo "=== Critical fix: xgboost_features.pkl (hardcoded lookup) ==="
$SSH '
MODEL_DIR=/opt/quantum/ai_engine/models
cd $MODEL_DIR

echo "Current state:"
ls -la xgboost_features.pkl 2>/dev/null || echo "xgboost_features.pkl: NOT FOUND (this is the root cause)"

echo ""
echo "Creating xgboost_features.pkl -> xgb_v6_20260217_003058_features.pkl"
ln -sf xgb_v6_20260217_003058_features.pkl xgboost_features.pkl

echo ""
echo "Verify:"
ls -la xgboost_features.pkl

echo ""
echo "Content:"
python3 -c "
import pickle
with open(\"$MODEL_DIR/xgboost_features.pkl\", \"rb\") as f:
    features = pickle.load(f)
print(f\"Count: {len(features)}\")
print(f\"Features: {features[:5]}...{features[-3:]}\")
"
'

echo ""
echo "=== Restart and check ==="
$SSH '
systemctl restart quantum-ai-engine
sleep 25
echo "--- XGB agent startup + first prediction ---"
journalctl -u quantum-ai-engine --no-pager --since "-30s" | \
    grep -E "XGB|xgboost|LGBM|ACTIVE|ERROR.*features|XGB-Agent|Loaded" | head -20
'
