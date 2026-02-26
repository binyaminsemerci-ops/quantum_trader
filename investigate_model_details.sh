#!/bin/bash
SSH="ssh -i ~/.ssh/hetzner_fresh -o StrictHostKeyChecking=no root@46.224.116.254"

$SSH '
echo "=== /app/models ==="
ls -la /app/models/ 2>/dev/null || echo "/app/models: NOT FOUND"

echo ""
echo "=== XGB agent _find_latest_model implementation ==="
sed -n "90,115p" /opt/quantum/ai_engine/agents/xgb_agent.py

echo ""
echo "=== XGB agent full __init__ model path logic ==="
sed -n "37,110p" /opt/quantum/ai_engine/agents/xgb_agent.py

echo ""
echo "=== XGB agent _load method ==="
sed -n "106,165p" /opt/quantum/ai_engine/agents/xgb_agent.py

echo ""
echo "=== Current model symlinks / files in ai_engine/models ==="
ls -la /opt/quantum/ai_engine/models/*.pkl | grep -E "scaler|xgb|xgboost|lgbm|light"

echo ""
echo "=== v6 model and scaler details ==="
python3 -c "
import pickle
for p in [
    \"/opt/quantum/ai_engine/models/xgb_v6_20260217_003058.pkl\",
    \"/opt/quantum/ai_engine/models/xgb_v6_20260217_003058_scaler.pkl\",
]:
    try:
        with open(p, \"rb\") as f:
            obj = pickle.load(f)
        import os
        print(f\"{os.path.basename(p)}: type={type(obj).__name__}\", end=\"\")
        if hasattr(obj, \"n_features_in_\"): print(f\" features={obj.n_features_in_}\", end=\"\")
        if hasattr(obj, \"feature_names_in_\"): print(f\" names={list(obj.feature_names_in_)}\", end=\"\")
        print()
    except Exception as e:
        print(f\"{p}: ERROR {e}\")
"

echo ""
echo "=== LGBM agent _load_model and scaler loading ==="
sed -n "55,140p" /opt/quantum/ai_engine/agents/lgbm_agent.py

echo ""
echo "=== All LGBM related files in models dir ==="
ls -la /opt/quantum/ai_engine/models/ | grep -iE "lgbm|lightgbm"
'
