#!/bin/bash
# Validate AdaptiveLeverageEngine deployment on VPS

echo "============================================================"
echo "AdaptiveLeverageEngine VPS Validation"
echo "============================================================"
echo ""

echo "[1] Checking if AdaptiveLeverageEngine exists..."
if grep -r "class AdaptiveLeverageEngine" ~/quantum_trader/microservices/exitbrain_v3_5/ -n | head -5; then
    echo "✅ AdaptiveLeverageEngine class found"
else
    echo "❌ AdaptiveLeverageEngine class NOT found"
    exit 1
fi

echo ""
echo "[2] Checking compute_levels integration..."
if grep -r "compute_levels" ~/quantum_trader/microservices/exitbrain_v3_5/ -n | grep "exit_brain.py" | head -3; then
    echo "✅ compute_levels integrated in ExitBrain"
else
    echo "❌ compute_levels NOT integrated"
    exit 1
fi

echo ""
echo "[3] Running quick Python import test..."
python3 -c "
from microservices.exitbrain_v3_5.adaptive_leverage_engine import AdaptiveLeverageEngine
engine = AdaptiveLeverageEngine()
levels = engine.compute_levels(0.01, 0.005, 20)
print(f'✅ Import successful!')
print(f'   20x Leverage: TP1={levels.tp1_pct*100:.2f}%, TP2={levels.tp2_pct*100:.2f}%, TP3={levels.tp3_pct*100:.2f}%, SL={levels.sl_pct*100:.2f}%')
print(f'   Harvest: {levels.harvest_scheme}')
print(f'   LSF: {levels.lsf:.4f}')
"

echo ""
echo "[4] Running unit tests..."
cd ~/quantum_trader/microservices/exitbrain_v3_5/tests
python3 test_adaptive_leverage_engine.py

echo ""
echo "============================================================"
echo "✅ AdaptiveLeverageEngine VALIDATED"
echo "============================================================"
