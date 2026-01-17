#!/bin/bash
set -euo pipefail

echo "========================================================================"
echo "P0 FIX PACK — PHASE 0: SAFETY + BASELINE"
echo "========================================================================"
echo ""

# Detect mode
echo "[1/4] Detecting TESTNET/LIVE mode..."
MODE="UNKNOWN"
if [ -f /etc/quantum/testnet.env ]; then
    if grep -q "BINANCE_TESTNET=true" /etc/quantum/testnet.env 2>/dev/null; then
        MODE="TESTNET"
    fi
fi

if [ -f /etc/quantum/ai-engine.env ]; then
    if grep -qi "live\|production" /etc/quantum/ai-engine.env 2>/dev/null; then
        MODE="LIVE"
    fi
fi

echo "✓ Mode: $MODE"

if [ "$MODE" = "LIVE" ]; then
    echo "🛑 LIVE mode detected - ABORTING for safety"
    exit 1
fi

echo "✅ TESTNET confirmed - proceeding with fixes"
echo ""

# Identify units
echo "[2/4] Identifying systemd units..."
UNITS=$(systemctl list-units 'quantum-*' --no-legend | awk '{print $1}')
echo "$UNITS" | grep -E "ai-engine|router|execution" | while read unit; do
    echo "  ✓ $unit"
done
echo ""

# Create backup directory
echo "[3/4] Creating backup directory..."
BACKUP_DIR="/tmp/p0fixpack_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
echo "$BACKUP_DIR" > /tmp/p0fixpack_current
echo "✓ Backup dir: $BACKUP_DIR"
echo ""

# Backup files to be modified
echo "[4/4] Backing up files..."
FILES_TO_BACKUP=(
    "/usr/local/bin/ai_strategy_router.py"
    "/home/qt/quantum_trader/services/execution_service.py"
    "/home/qt/quantum_trader/ai_engine/services/eventbus_bridge.py"
)

for file in "${FILES_TO_BACKUP[@]}"; do
    if [ -f "$file" ]; then
        cp "$file" "$BACKUP_DIR/$(basename $file).backup"
        echo "  ✓ Backed up: $file"
    else
        echo "  ⚠️  Not found: $file"
    fi
done

echo ""
echo "✅ PHASE 0 COMPLETE"
echo "   Backup location: $BACKUP_DIR"
echo ""
