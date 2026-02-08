#!/bin/bash
cd /root/quantum_trader
echo "🚀 Starting calibration process..."
echo "Timestamp: $(date -Iseconds)"
./run_calibration_safe.sh
echo ""
echo "✅ Calibration execution complete"
