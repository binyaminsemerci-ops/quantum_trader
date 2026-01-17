#!/bin/bash
set -euo pipefail

echo "=== COLLECTING AFTER EVIDENCE ==="
PROOF_DIR=$(cat /tmp/current_proof_dir.txt)
echo "Proof dir: $PROOF_DIR"
echo ""

STREAM="quantum:stream:trade.intent"
GROUP="quantum:group:execution:trade.intent"

echo "--- XINFO GROUPS (after) ---" | tee "$PROOF_DIR/after.txt"
redis-cli XINFO GROUPS "$STREAM" | tee -a "$PROOF_DIR/after.txt"
echo "" | tee -a "$PROOF_DIR/after.txt"

echo "--- XINFO CONSUMERS (after) ---" | tee -a "$PROOF_DIR/after.txt"
redis-cli XINFO CONSUMERS "$STREAM" "$GROUP" | tee -a "$PROOF_DIR/after.txt"
echo "" | tee -a "$PROOF_DIR/after.txt"

echo "--- XPENDING (after, first 20) ---" | tee -a "$PROOF_DIR/after.txt"
redis-cli XPENDING "$STREAM" "$GROUP" - + 20 | tee -a "$PROOF_DIR/after.txt"
echo "" | tee -a "$PROOF_DIR/after.txt"

echo "--- RECOVERY LOG (last 50 lines) ---" | tee -a "$PROOF_DIR/after.txt"
tail -50 /var/log/quantum/stream_recover.log | tee -a "$PROOF_DIR/after.txt"
echo ""

# Timer status
echo "--- TIMER STATUS ---" | tee -a "$PROOF_DIR/after.txt"
systemctl status quantum-stream-recover.timer --no-pager | tee -a "$PROOF_DIR/after.txt"
echo ""

echo "✅ After evidence collected to: $PROOF_DIR/after.txt"
