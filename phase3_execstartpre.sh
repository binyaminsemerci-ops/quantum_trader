#!/bin/bash
set -euo pipefail

PROOF_DIR=$(cat /tmp/current_proof_path.txt)
LOG="/var/log/quantum/stream_recover.log"

echo "=== PHASE 3: EXECSTARTPRE VERIFICATION ==="
echo ""

echo "Restarting quantum-execution.service..."
systemctl restart quantum-execution.service
sleep 3

echo "Capturing output..."
echo ""

{
    echo "=== Recovery Log (last 100 lines) ==="
    tail -100 "$LOG" 2>/dev/null || echo "No log"
    echo ""
    echo "=== Execution Service Journal (last 50 lines) ==="
    journalctl -u quantum-execution.service -n 50 --no-pager || true
    echo ""
    echo "=== Service Status ==="
    systemctl status quantum-execution.service --no-pager -l | head -30 || true
} | tee "$PROOF_DIR/restart.txt"

echo ""
echo "✅ PHASE 3 COMPLETE"
