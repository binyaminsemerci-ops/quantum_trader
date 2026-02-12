#!/bin/bash
set -e

echo "🎯 RUNNING CALIBRATION WORKFLOW (PATH 2.4A)"
echo "============================================="
echo ""

cd /home/qt/quantum_trader

# Activate venv
source venv/bin/activate

echo "📊 Current data status:"
echo "----------------------"
redis-cli XLEN quantum:stream:signal.score | awk '{print "Signals: " $1}'
redis-cli XLEN quantum:stream:apply.result | awk '{print "Apply results: " $1}'
redis-cli XLEN quantum:stream:trade.closed | awk '{print "Closed trades: " $1}'

echo ""
echo "🔄 Running calibration workflow..."
echo ""

# Run calibration with 1 day of recent signals
python ai_engine/calibration/run_calibration_workflow.py --days 1 --min-samples 50

echo ""
echo "✅ Calibration workflow complete!"
echo ""
echo "📁 Checking calibrator artifact..."
ls -lh ai_engine/calibration/calibrator_v*.pkl 2>/dev/null || echo "⚠️ No calibrator found"

echo ""
echo "📈 Next steps:"
echo "  1. Verify calibration quality (ECE < 0.10)"
echo "  2. Restart ensemble service to load calibrator"
echo "  3. Monitor calibrated confidence values"
