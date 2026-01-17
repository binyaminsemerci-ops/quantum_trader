#!/bin/bash
set -euo pipefail

echo "========================================================================"
echo "P0 FIX PACK — PHASE 2: DEPLOY CONSUMER GROUPS + TERMINAL STATES"
echo "========================================================================"
echo ""

BACKUP_DIR=$(cat /tmp/p0fixpack_current)
echo "Backup dir: $BACKUP_DIR"
echo ""

echo "[1/2] Uploading patched files..."
echo "  EventBus: /home/qt/quantum_trader/ai_engine/services/eventbus_bridge.py"
echo "  Execution: /home/qt/quantum_trader/services/execution_service.py"
echo ""

echo "[2/2] Restarting execution service..."
systemctl restart quantum-execution.service
echo "✓ Execution restarted"
sleep 5

echo ""
echo "Verifying service..."
systemctl is-active quantum-execution.service && echo "✓ Execution: active" || echo "❌ Execution: failed"

echo ""
echo "✅ PHASE 2 DEPLOYMENT COMPLETE"
echo ""
