#!/bin/bash
set -euo pipefail

PROOF_DIR=$(cat /tmp/current_proof_path.txt)
LOG="/var/log/quantum/stream_recover.log"

echo "=== PHASE 1: FORCE RECOVERY VIA SYSTEMD ==="
echo ""

echo "Running: systemctl start quantum-stream-recover.service"
systemctl start quantum-stream-recover.service
sleep 3

echo ""
echo "Service execution complete. Capturing output..."
echo ""

{
    echo "=== Service Status ==="
    systemctl status quantum-stream-recover.service --no-pager -l || true
    echo ""
    echo "=== Journal Output (last 100 lines) ==="
    journalctl -u quantum-stream-recover.service -n 100 --no-pager || true
    echo ""
    echo "=== Recovery Log (last 100 lines) ==="
    tail -100 "$LOG" 2>/dev/null || echo "No log yet"
} | tee "$PROOF_DIR/run_now.txt"

echo ""
echo "✅ PHASE 1 COMPLETE"
