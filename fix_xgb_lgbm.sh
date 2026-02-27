#!/bin/bash
# Fix XGB symlinks and retrain LGBM v6 with live Binance data
SSH="ssh -i ~/.ssh/hetzner_fresh -o StrictHostKeyChecking=no root@46.224.116.254"

echo "================================================================="
echo "FIX A: XGB — Update symlinks to use v6 (49 features)"
echo "================================================================="
$SSH '
cd /opt/quantum/ai_engine/models

echo "Before:"
ls -la xgb_model.pkl xgboost_v_prod.pkl scaler.pkl xgboost_v_prod_scaler.pkl

echo ""
echo "Updating symlinks to v6 (49 features)..."

# Point all XGB model aliases to v6
ln -sf xgb_v6_20260217_003058.pkl xgb_model.pkl
ln -sf xgb_v6_20260217_003058.pkl xgboost_v_prod.pkl

# Point all XGB scaler aliases to v6 scaler (not lgbm_scaler!)
ln -sf xgb_v6_20260217_003058_scaler.pkl scaler.pkl
ln -sf xgb_v6_20260217_003058_scaler.pkl xgboost_v_prod_scaler.pkl

echo ""
echo "After:"
ls -la xgb_model.pkl xgboost_v_prod.pkl scaler.pkl xgboost_v_prod_scaler.pkl

echo ""
echo "FIX A: XGB symlinks updated"
'

echo ""
echo "================================================================="
echo "FIX B: LGBM — Place correct scaler in expected fallback path"
echo "================================================================="
# The LGBM agent defaults to "models/lightgbm_scaler_v20251230_223627.pkl"
# resolved from /opt/quantum -> /opt/quantum/models/lightgbm_scaler_v20251230_223627.pkl
# It doesn't exist there. Copy the ai_engine/models one there.
# But that scaler was trained on 49 features vs the 14-feature Feb24 model.
# Better: run train_lightgbm_v6 to create a PROPER matched 49-feature model+scaler.

$SSH '
echo "Checking if /opt/quantum/models/lightgbm_scaler_v20251230_223627.pkl exists..."
ls -la /opt/quantum/models/lightgbm* 2>/dev/null || echo "No lightgbm files in /opt/quantum/models/"

echo ""
echo "Before retrain: show lgbm files in ai_engine/models/"
ls -la /opt/quantum/ai_engine/models/lgbm* 2>/dev/null
ls -la /opt/quantum/ai_engine/models/lightgbm* 2>/dev/null
'

echo ""
echo "================================================================="
echo "FIX B: Running train_lightgbm_v6.py (fetches live Binance data)"
echo "================================================================="
$SSH '
cd /opt/quantum
echo "Starting LightGBM v6 training..."
# Run in the service venv
python3 ops/retrain/train_lightgbm_v6.py
echo "LGBM training exit code: $?"
'

echo ""
echo "================================================================="
echo "FIX B: Check new LGBM model files"
echo "================================================================="
$SSH '
echo "New LGBM model files:"
ls -lh /opt/quantum/ai_engine/models/lightgbm* /opt/quantum/ai_engine/models/lgbm* 2>/dev/null | sort -k6,7

echo ""
echo "Latest LGBM v6 model:"
ls -lt /opt/quantum/ai_engine/models/lightgbm_v*_v2.pkl 2>/dev/null | head -3

echo ""
echo "Latest LGBM v6 scaler:"
ls -lt /opt/quantum/ai_engine/models/lightgbm_scaler_v*_v2.pkl 2>/dev/null | head -3
'
