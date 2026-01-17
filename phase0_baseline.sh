#!/bin/bash
set -euo pipefail

echo "=== PHASE 0: BASELINE + BACKUP ==="
TS=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/tmp/zombiefix_backup_$TS"
PROOF_DIR="/tmp/zombiefix_proof_$TS"
mkdir -p "$BACKUP_DIR" "$PROOF_DIR"

echo "Timestamp: $TS"
echo "Backup: $BACKUP_DIR"
echo "Proof: $PROOF_DIR"
echo "$BACKUP_DIR" > /tmp/current_backup_dir.txt
echo "$PROOF_DIR" > /tmp/current_proof_dir.txt
echo ""

# Mode detection
echo "=== MODE DETECTION ==="
if grep -RqiE "BINANCE_TESTNET=true|USE_BINANCE_TESTNET=true" /etc/quantum/*.env 2>/dev/null; then
    MODE="TESTNET"
    echo "✅ MODE: TESTNET"
else
    MODE="LIVE"
    echo "❌ MODE: LIVE - ABORTING"
    exit 1
fi
echo "$MODE" > "$PROOF_DIR/mode.txt"
echo ""

# Define streams
STREAM="quantum:stream:trade.intent"
GROUP="quantum:group:execution:trade.intent"

# Collect baseline evidence
echo "=== BASELINE EVIDENCE ==="
echo "Stream: $STREAM"
echo "Group: $GROUP"
echo ""

echo "--- XINFO GROUPS ---" | tee "$PROOF_DIR/before.txt"
redis-cli XINFO GROUPS "$STREAM" | tee -a "$PROOF_DIR/before.txt"
echo "" | tee -a "$PROOF_DIR/before.txt"

echo "--- XINFO CONSUMERS ---" | tee -a "$PROOF_DIR/before.txt"
redis-cli XINFO CONSUMERS "$STREAM" "$GROUP" | tee -a "$PROOF_DIR/before.txt"
echo "" | tee -a "$PROOF_DIR/before.txt"

echo "--- XPENDING (first 20) ---" | tee -a "$PROOF_DIR/before.txt"
redis-cli XPENDING "$STREAM" "$GROUP" - + 20 | tee -a "$PROOF_DIR/before.txt"
echo ""

# Backup existing files
echo "=== CREATING BACKUPS ==="
for file in \
    /usr/local/bin/quantum_stream_recover.sh \
    /etc/systemd/system/quantum-stream-recover.service \
    /etc/systemd/system/quantum-stream-recover.timer \
    /etc/systemd/system/quantum-execution.service.d/10-zombiefix.conf \
    /etc/systemd/system/quantum-execution.service.d/10-stream-recover.conf
do
    if [ -f "$file" ]; then
        cp -v "$file" "$BACKUP_DIR/"
        echo "✅ Backed up: $file"
    else
        echo "⚠️  Not found: $file (will create new)"
    fi
done
echo ""

echo "✅ PHASE 0 COMPLETE"
echo "Backup dir: $BACKUP_DIR"
echo "Proof dir: $PROOF_DIR"
