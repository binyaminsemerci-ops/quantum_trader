#!/bin/bash

###############################################################################
# PHASE D ROLLBACK SCRIPT
# Reverts all PHASE D permanent fix changes
# Date: January 17, 2026
# Author: AI Trader System
###############################################################################

set -e

REPO_ROOT="/home/qt/quantum_trader"
LOG_FILE="/var/log/quantum/rollback_phase_d_$(date +%s).log"

echo "=========================================="
echo "PHASE D ROLLBACK SCRIPT"
echo "=========================================="
echo "Date: $(date -u)"
echo "Log: $LOG_FILE"
echo ""

# Create log file
mkdir -p "$(dirname "$LOG_FILE")"
exec 1> >(tee -a "$LOG_FILE")
exec 2>&1

###############################################################################
# PRE-FLIGHT CHECKS
###############################################################################

echo "[1/6] PRE-FLIGHT CHECKS"
echo "========================================"

if ! git -C "$REPO_ROOT" status > /dev/null 2>&1; then
    echo "ERROR: Not in a git repository at $REPO_ROOT"
    exit 1
fi

CURRENT_BRANCH=$(git -C "$REPO_ROOT" branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo "WARNING: Not on main branch (current: $CURRENT_BRANCH)"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "✓ Pre-flight checks passed"
echo ""

###############################################################################
# STOP SERVICES
###############################################################################

echo "[2/6] STOPPING AFFECTED SERVICES"
echo "========================================"

systemctl stop quantum-router.service 2>/dev/null || echo "Router service not running (OK)"
systemctl stop quantum-execution.service 2>/dev/null || echo "Execution service not running (OK)"
systemctl stop quantum-ai-engine.service 2>/dev/null || echo "AI Engine service not running (OK)"

sleep 2
echo "✓ Services stopped"
echo ""

###############################################################################
# RESTORE CODE FILES
###############################################################################

echo "[3/6] RESTORING ORIGINAL CODE FILES"
echo "========================================"

echo "Restoring ai_strategy_router.py..."
git -C "$REPO_ROOT" checkout HEAD -- ai_strategy_router.py

echo "Restoring governer_agent.py..."
git -C "$REPO_ROOT" checkout HEAD -- governer_agent.py

echo "Restoring ai_engine/services/eventbus_bridge.py..."
git -C "$REPO_ROOT" checkout HEAD -- ai_engine/services/eventbus_bridge.py

echo "✓ Code files restored to main branch version"
echo ""

###############################################################################
# RESTORE SYSTEMD UNITS
###############################################################################

echo "[4/6] RESTORING SYSTEMD UNIT FILES"
echo "========================================"

echo "Restoring quantum-router.service..."
git -C "$REPO_ROOT" checkout HEAD -- quantum-router.service
cp -v "$REPO_ROOT/quantum-router.service" /etc/systemd/system/quantum-router.service

echo "Restoring quantum-execution.service..."
git -C "$REPO_ROOT" checkout HEAD -- quantum-execution.service
cp -v "$REPO_ROOT/quantum-execution.service" /etc/systemd/system/quantum-execution.service

systemctl daemon-reload
echo "✓ Systemd units restored and daemon reloaded"
echo ""

###############################################################################
# RESTORE DIRECTORY PERMISSIONS
###############################################################################

echo "[5/6] RESTORING DIRECTORY PERMISSIONS"
echo "========================================"

echo "Restoring /etc/quantum permissions to mode 700 (root only)..."
chmod 700 /etc/quantum
chmod 644 /etc/quantum/*.env

echo "Current permissions:"
ls -ld /etc/quantum
echo "✓ Directory permissions restored"
echo ""

###############################################################################
# RESTART SERVICES
###############################################################################

echo "[6/6] RESTARTING SERVICES"
echo "========================================"

echo "Starting quantum-router.service..."
systemctl start quantum-router.service || echo "WARNING: Router service failed to start"
sleep 2

echo "Starting quantum-execution.service..."
systemctl start quantum-execution.service || echo "WARNING: Execution service failed to start"
sleep 2

echo "Starting quantum-ai-engine.service..."
systemctl start quantum-ai-engine.service || echo "WARNING: AI Engine service failed to start"
sleep 2

echo ""
echo "Service status after rollback:"
echo "========================================"
systemctl status quantum-router.service --no-pager | head -10
echo ""
systemctl status quantum-execution.service --no-pager | head -10
echo ""
systemctl status quantum-ai-engine.service --no-pager | head -10
echo ""

###############################################################################
# VERIFICATION
###############################################################################

echo "ROLLBACK VERIFICATION"
echo "========================================"

# Check stream state
echo ""
echo "Checking Redis streams..."
TRADE_INTENT_XLEN=$(redis-cli XLEN quantum:stream:trade.intent 2>/dev/null || echo "ERROR")
EXECUTION_RESULT_XLEN=$(redis-cli XLEN quantum:stream:execution.result 2>/dev/null || echo "ERROR")

echo "trade.intent XLEN: $TRADE_INTENT_XLEN"
echo "execution.result XLEN: $EXECUTION_RESULT_XLEN"

# Check code files are reverted
echo ""
echo "Checking code file revisions..."
if grep -q "composite.*dedup.*key" "$REPO_ROOT/ai_strategy_router.py" 2>/dev/null; then
    echo "WARNING: ai_strategy_router.py still contains PHASE D dedup key logic"
else
    echo "✓ ai_strategy_router.py reverted"
fi

if grep -q "redis_client.*incr" "$REPO_ROOT/governer_agent.py" 2>/dev/null; then
    echo "WARNING: governer_agent.py still contains Redis persistence logic"
else
    echo "✓ governer_agent.py reverted"
fi

if grep -q "EXECUTION_RESULT_STREAM.*getenv" "$REPO_ROOT/ai_engine/services/eventbus_bridge.py" 2>/dev/null; then
    echo "WARNING: eventbus_bridge.py still contains env-driven stream logic"
else
    echo "✓ eventbus_bridge.py reverted"
fi

echo ""
echo "=========================================="
echo "ROLLBACK COMPLETED SUCCESSFULLY"
echo "=========================================="
echo "Date: $(date -u)"
echo "Log saved to: $LOG_FILE"
echo ""
echo "Next steps:"
echo "1. Verify services are running: systemctl status quantum-*"
echo "2. Check streams are healthy: redis-cli XINFO STREAM quantum:stream:*"
echo "3. Review rollback log: tail -50 $LOG_FILE"
echo ""
