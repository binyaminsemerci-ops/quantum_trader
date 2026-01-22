#!/bin/bash
set -euo pipefail

# PHASE 0: BASELINE + SETUP
TS=$(date +%Y%m%d_%H%M%S)
PROOF_DIR="/tmp/zombie_auto_proof_$TS"
mkdir -p "$PROOF_DIR"
echo "$PROOF_DIR" > /tmp/current_proof_path.txt

# Define streams
STREAM="quantum:stream:trade.intent"
GROUP="quantum:group:execution:trade.intent"
LOG="/var/log/quantum/stream_recover.log"

# MODE CHECK
if grep -qiE "BINANCE_TESTNET=true|USE_BINANCE_TESTNET=true" /etc/quantum/testnet.env 2>/dev/null; then
    MODE="TESTNET"
else
    MODE="LIVE"
fi

if [[ "$MODE" == "LIVE" ]]; then
    echo "❌ LIVE detected - read-only mode only"
    echo "MODE=$MODE" > "$PROOF_DIR/mode.txt"
    exit 1
fi

echo "✅ TESTNET mode confirmed"
echo "MODE=$MODE" > "$PROOF_DIR/mode.txt"
echo "PROOF_DIR=$PROOF_DIR" > "$PROOF_DIR/vars.txt"
echo "STREAM=$STREAM" >> "$PROOF_DIR/vars.txt"
echo "GROUP=$GROUP" >> "$PROOF_DIR/vars.txt"
echo "LOG=$LOG" >> "$PROOF_DIR/vars.txt"

# PHASE 0: Capture BEFORE state
echo ""
echo "=== PHASE 0: CAPTURING BEFORE STATE ==="
{
    echo "=== Timer Status ==="
    systemctl status quantum-stream-recover.timer --no-pager || true
    echo ""
    echo "=== Timer List ==="
    systemctl list-timers | grep -i quantum-stream-recover || true
    echo ""
    echo "=== ExecStartPre Config ==="
    systemctl cat quantum-execution.service | grep ExecStartPre || true
    echo ""
    echo "=== Current Consumers ==="
    redis-cli XINFO CONSUMERS "$STREAM" "$GROUP" 2>/dev/null || echo "No consumers"
    echo ""
    echo "=== Pending Messages (first 20) ==="
    redis-cli XPENDING "$STREAM" "$GROUP" - + 20 2>/dev/null || echo "No pending"
    echo ""
    echo "=== Consumer Group Status ==="
    redis-cli XINFO GROUPS "$STREAM" 2>/dev/null || echo "No group"
} | tee "$PROOF_DIR/before.txt"

echo ""
echo "✅ PHASE 0 COMPLETE"
echo "Proof dir: $PROOF_DIR"
