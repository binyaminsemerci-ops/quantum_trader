#!/bin/bash
SSH="ssh -i ~/.ssh/hetzner_fresh -o StrictHostKeyChecking=no root@46.224.116.254"

$SSH 'bash -s' << '"EOF"'
cd /opt/quantum/ai_engine/models

echo "=== Key XGB symlinks ==="
for f in xgb_model.pkl xgboost_v_prod.pkl scaler.pkl xgboost_features.pkl xgb_v6_20260217_003058.pkl xgb_v6_20260217_003058_scaler.pkl xgb_v6_20260217_003058_features.pkl; do
  if [ -L "$f" ]; then
    echo "SYMLINK: $f -> $(readlink $f)"
  elif [ -f "$f" ]; then
    echo "FILE:    $f ($(stat -c%s $f) bytes)"
  else
    echo "MISSING: $f"
  fi
done

echo ""
echo "=== scaler.pkl: what is it? ==="
/opt/quantum/.venv/bin/python3 -c "
import pickle
with open('scaler.pkl','rb') as f:
    obj = pickle.load(f)
print(f'type={type(obj).__name__}, n_features_in_={getattr(obj,\"n_features_in_\",\"N/A\")}')
" 2>&1

echo ""
echo "=== xgboost_features.pkl: how many features? ==="
/opt/quantum/.venv/bin/python3 -c "
import pickle
with open('xgboost_features.pkl','rb') as f:
    obj = pickle.load(f)
print(f'type={type(obj).__name__}, len={len(obj) if hasattr(obj,\"__len__\") else \"N/A\"}')
if isinstance(obj, list):
    print(f'first5={obj[:5]}')
" 2>&1

echo ""
echo "=== xgb_v6_20260217_003058.pkl: model check ==="
/opt/quantum/.venv/bin/python3 -c "
import pickle
with open('xgb_v6_20260217_003058.pkl','rb') as f:
    obj = pickle.load(f)
print(f'type={type(obj).__name__}')
if hasattr(obj,'n_features_in_'):
    print(f'n_features_in_={obj.n_features_in_}')
" 2>&1
EOF
