#!/bin/bash
SSH="ssh -i ~/.ssh/hetzner_fresh -o StrictHostKeyChecking=no root@46.224.116.254"

echo "=== Check retraining infrastructure ==="
$SSH '
echo "--- Retraining scripts ---"
ls -la /opt/quantum/scripts/ | grep -i train
ls -la /opt/quantum/ | grep -i train
find /opt/quantum -name "*retrain*" -o -name "*training*" | grep -v __pycache__ | grep -v ".pyc" | head -20

echo ""
echo "--- /app/models contents ---"
ls -la /app/models/ 2>/dev/null || echo "/app/models: NOT FOUND"

echo ""
echo "--- ai_engine/models contents ---"
ls -la /opt/quantum/ai_engine/models/

echo ""
echo "--- XGB model exact file + features ---"
python3 - << PYEOF
import sys; sys.path.insert(0, "/opt/quantum")
import pickle, os
# Try to inspect XGB model
for path in ["/opt/quantum/ai_engine/models/xgb_model.pkl",
             "/opt/quantum/ai_engine/models/xgboost_v_prod.pkl"]:
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        print(f"Path: {path}")
        print(f"Type: {type(data)}")
        if hasattr(data, "n_features_in_"): print(f"features: {data.n_features_in_}")
        if hasattr(data, "feature_names_in_"): print(f"names: {list(data.feature_names_in_)}")
        break
PYEOF

echo ""
echo "--- LGBM model exact file + features ---"
python3 - << PYEOF
import sys; sys.path.insert(0, "/opt/quantum")
import pickle, os
for path in ["/opt/quantum/ai_engine/models/lgbm_model.pkl"]:
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        print(f"Path: {path}")
        print(f"Type: {type(data)}")
        if hasattr(data, "n_features_"): print(f"features: {data.n_features_}")
        if hasattr(data, "feature_name_"): print(f"names: {list(data.feature_name_)}")
        break
PYEOF

echo ""
echo "--- XGB agent predict method (how features are built) ---"
grep -n "FEATURE_NAMES\|feature_names\|_build_features\|predict\|def " /opt/quantum/ai_engine/agents/xgb_agent.py | head -40

echo ""
echo "--- Live feature publisher - what 17 features arrive ---"
grep -n "FEATURE_NAMES\|feature_names\|features =" /opt/quantum/ai_engine/services/feature_publisher_service.py | head -30

echo ""
echo "--- Which retrain service handles XGB/LGBM ---"
find /opt/quantum -name "*.py" | xargs grep -l "xgboost\|lightgbm\|XGBClassifier\|LGBMClassifier" 2>/dev/null | grep -v __pycache__ | grep -v ".pyc" | grep -v "agent"
'
