#!/bin/bash
set -euo pipefail

PROOF_DIR=$(cat /tmp/current_proof_path.txt)

echo "=== PHASE 2: TIMER VERIFICATION ==="
echo ""

{
    echo "=== Timer Status ==="
    systemctl status quantum-stream-recover.timer --no-pager
    echo ""
    echo "=== Timer List ==="
    systemctl list-timers | grep -i quantum-stream-recover
    echo ""
    echo "=== Timer Details ==="
    systemctl cat quantum-stream-recover.timer
} | tee "$PROOF_DIR/timer.txt"

echo ""
echo "✅ PHASE 2 COMPLETE"
