#!/bin/bash
# verify_vps_models.sh — run via: wsl bash /mnt/c/quantum_trader/verify_vps_models.sh
SSH="ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254"

echo "=== Checking files exist ==="
$SSH "ls -lh /home/qt/quantum_trader/models/lightgbm_v20260224_v2.pkl /home/qt/quantum_trader/models/lightgbm_scaler_v20260224_v2.pkl"

echo ""
echo "=== Glob pattern test ==="
$SSH "cd /home/qt/quantum_trader && python3 -c \"
from pathlib import Path
scalers=sorted(Path('models').glob('lightgbm_scaler_v*_v2.pkl'))
models=sorted(Path('models').glob('lightgbm_v*_v2.pkl'))
print('Scaler glob results:', scalers)
print('Model glob results:', models)
\""

echo ""
echo "=== Verify scaler n_features (ai-engine venv) ==="
$SSH "/opt/quantum/venvs/ai-engine/bin/python -c \"
import pickle
with open('/home/qt/quantum_trader/models/lightgbm_scaler_v20260224_v2.pkl','rb') as f:
    s = pickle.load(f)
print('scaler n_features_in_:', s.n_features_in_)
\""

echo ""
echo "=== Restart ensemble-predictor and tail startup logs ==="
$SSH "systemctl restart quantum-ensemble-predictor && sleep 7 && journalctl -u quantum-ensemble-predictor -n 20 --no-pager"

echo ""
echo "=== Service status ==="
$SSH "systemctl is-active quantum-ensemble-predictor"
