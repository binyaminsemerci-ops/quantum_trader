#!/bin/bash
SSH="ssh -i ~/.ssh/hetzner_fresh -o StrictHostKeyChecking=no root@46.224.116.254"

echo "=== Waiting 30s for prediction cycle ==="
sleep 30

echo ""
echo "=== XGB and LGBM prediction results (last 35s) ==="
$SSH '
journalctl -u quantum-ai-engine --no-pager --since "-35s" | \
    grep -iE "xgb|lgbm|xgboost|lightgbm|prediction failed|features.*scaler|scaler.*features|FAIL-CLOSED" | \
    head -25

echo ""
echo "=== ENSEMBLE log (shows active agents) ==="
journalctl -u quantum-ai-engine --no-pager --since "-35s" | \
    grep -E "ENSEMBLE|QSC|EFFECTIVE_WEIGHTS|ACTIVE.*inactive|xgb.*lgbm" | \
    head -10

echo ""
echo "=== xgboost_v_prod_meta.json current state ==="
cat /opt/quantum/ai_engine/models/xgboost_v_prod_meta.json

echo ""
echo "=== XGB v6 features (what model expects) ==="
python3 -c "
import pickle
with open(\"/opt/quantum/ai_engine/models/xgb_v6_20260217_003058_features.pkl\", \"rb\") as f:
    data = pickle.load(f)
print(f\"Type: {type(data)}\")
if hasattr(data, \"__len__\"): print(f\"Count: {len(data)}\")
print(f\"Features: {data}\")
"
'
