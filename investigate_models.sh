#!/bin/bash
SSH="ssh -i ~/.ssh/hetzner_fresh -o StrictHostKeyChecking=no root@46.224.116.254"

echo "=== XGB: feature mismatch investigation ==="
$SSH '
echo "--- XGB agent feature handling ---"
sed -n "1,80p" /opt/quantum/ai_engine/agents/xgb_agent.py

echo ""
echo "--- XGB scaler info via python ---"
python3 -c "
import pickle
with open(\"/opt/quantum/ai_engine/models/scaler.pkl\", \"rb\") as f:
    s = pickle.load(f)
print(\"Type:\", type(s))
print(\"n_features_in_:\", getattr(s, \"n_features_in_\", \"NOT_FOUND\"))
print(\"feature_names_in_:\", list(getattr(s, \"feature_names_in_\", [])))
"

echo ""
echo "--- XGB model feature count ---"
python3 -c "
import pickle
with open(\"/opt/quantum/ai_engine/models/xgb_model.pkl\", \"rb\") as f:
    m = pickle.load(f)
print(\"Type:\", type(m))
print(\"n_features_in_:\", getattr(m, \"n_features_in_\", \"NOT_FOUND\"))
print(\"feature_names_in_:\", list(getattr(m, \"feature_names_in_\", [])))
"

echo ""
echo "--- XGBoost features .pkl ---"
python3 -c "
import pickle
with open(\"/opt/quantum/ai_engine/models/xgboost_features.pkl\", \"rb\") as f:
    feats = pickle.load(f)
print(\"Type:\", type(feats))
print(\"Count:\", len(feats) if hasattr(feats, \"__len__\") else \"N/A\")
print(\"Features:\", feats)
"

echo ""
echo "=== LGBM: scaler=None investigation ==="
echo "--- LGBM agent ---"
sed -n "1,80p" /opt/quantum/ai_engine/agents/lgbm_agent.py

echo ""
echo "--- LGBM scaler file ---"
python3 -c "
import pickle, os
path = \"/opt/quantum/ai_engine/models/lgbm_scaler.pkl\"
exists = os.path.exists(path)
size = os.path.getsize(path) if exists else 0
print(f\"exists={exists} size={size}\")
if exists and size > 0:
    with open(path, \"rb\") as f:
        s = pickle.load(f)
    print(\"Type:\", type(s))
    print(\"n_features_in_:\", getattr(s, \"n_features_in_\", \"NOT_FOUND\"))
    print(\"feature_names_in_:\", list(getattr(s, \"feature_names_in_\", [])))
"

echo ""
echo "--- live ensemble log for which features arrive ---"
journalctl -u quantum-ai-engine --no-pager --since "-5m" | grep -E "Dropping extras|features|17 feat|49 feat" | head -20
'
