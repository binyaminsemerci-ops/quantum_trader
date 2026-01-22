#!/bin/bash
set -euo pipefail

PROOF_DIR=$(cat /tmp/current_proof_path.txt)
STREAM="quantum:stream:trade.intent"
GROUP="quantum:group:execution:trade.intent"

echo "=== PHASE 4: CAPTURE AFTER STATE ==="
echo ""

sleep 2

{
    echo "=== Current Consumers ==="
    redis-cli XINFO CONSUMERS "$STREAM" "$GROUP" 2>/dev/null || echo "No consumers"
    echo ""
    echo "=== Pending Messages (first 20) ==="
    redis-cli XPENDING "$STREAM" "$GROUP" - + 20 2>/dev/null || echo "No pending"
    echo ""
    echo "=== Consumer Group Status ==="
    redis-cli XINFO GROUPS "$STREAM" 2>/dev/null || echo "No group"
} | tee "$PROOF_DIR/after.txt"

echo ""
echo "✅ PHASE 4 COMPLETE"
