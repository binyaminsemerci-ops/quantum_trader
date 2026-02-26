#!/bin/bash
SSH="ssh -i ~/.ssh/hetzner_fresh -o StrictHostKeyChecking=no root@46.224.116.254"

echo "=== Fix LGBM scaler naming mismatch ==="
$SSH '
MODEL_DIR=/opt/quantum/ai_engine/models

echo "--- Existing files ---"
ls -la $MODEL_DIR/lightgbm_v20260224_v2*.pkl $MODEL_DIR/lightgbm_scaler_v20260224*.pkl 2>/dev/null

echo ""
echo "--- Creating scaler symlink (naming convention fix) ---"
# Agent looks for: lightgbm_v20260224_v2_scaler.pkl
# Actual file is:  lightgbm_scaler_v20260224_v2.pkl
ln -sf lightgbm_scaler_v20260224_v2.pkl $MODEL_DIR/lightgbm_v20260224_v2_scaler.pkl
echo "Created: lightgbm_v20260224_v2_scaler.pkl -> lightgbm_scaler_v20260224_v2.pkl"

echo ""
echo "--- Verify ---"
ls -la $MODEL_DIR/lightgbm_v20260224_v2*.pkl $MODEL_DIR/lightgbm_scaler_v20260224*.pkl 2>/dev/null
'

echo ""
echo "=== Check XGB v6 features file ==="
$SSH '
MODEL_DIR=/opt/quantum/ai_engine/models
python3 -c "
import pickle, json, os

# Check v6 features
p = \"$MODEL_DIR/xgb_v6_20260217_003058_features.pkl\"
with open(p, \"rb\") as f:
    features = pickle.load(f)
print(f\"v6 features: type={type(features)} len={len(features) if hasattr(features, .__len__) else features}\")
if isinstance(features, (list, tuple)):
    print(f\"Features: {features}\")

# Check v6 metadata
meta_p = \"$MODEL_DIR/xgb_v6_20260217_003058_metadata.json\"
if os.path.exists(meta_p):
    with open(meta_p) as f:
        meta = json.load(f)
    print(f\"Metadata: {meta}\")
" 2>&1 || echo "ERROR in python check"

echo ""
echo "--- xgboost_v_prod_meta.json ---"
cat $MODEL_DIR/xgboost_v_prod_meta.json 2>/dev/null || echo "NOT FOUND"
'

echo ""
echo "=== Restart and check all 5 agents load cleanly ==="
$SSH '
systemctl restart quantum-ai-engine
sleep 20
journalctl -u quantum-ai-engine --no-pager --since "-25s" | \
    grep -E "XGB-Agent|LGBM-Agent|NHiTS|PatchTST|TFT|ERROR|Loaded|WARN|features=|scaler" | \
    head -30
'
