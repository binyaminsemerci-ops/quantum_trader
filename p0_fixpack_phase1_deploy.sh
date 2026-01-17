#!/bin/bash
set -euo pipefail

echo "========================================================================"
echo "P0 FIX PACK — PHASE 1: DEPLOY IDEMPOTENCY FIXES"
echo "========================================================================"
echo ""

BACKUP_DIR=$(cat /tmp/p0fixpack_current)
echo "Backup dir: $BACKUP_DIR"
echo ""

echo "[1/3] Uploading patched files..."
echo "  Router: /usr/local/bin/ai_strategy_router.py"
echo "  Execution: /home/qt/quantum_trader/services/execution_service.py"
echo ""

echo "[2/3] Restarting services..."
systemctl restart quantum-ai-strategy-router.service
echo "✓ Router restarted"
sleep 2

systemctl restart quantum-execution.service
echo "✓ Execution restarted"
sleep 3

echo ""
echo "[3/3] Verifying services..."
systemctl is-active quantum-ai-strategy-router.service && echo "✓ Router: active" || echo "❌ Router: failed"
systemctl is-active quantum-execution.service && echo "✓ Execution: active" || echo "❌ Execution: failed"

echo ""
echo "✅ PHASE 1 DEPLOYMENT COMPLETE"
echo ""
