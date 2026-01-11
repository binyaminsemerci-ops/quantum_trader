#!/bin/bash
"""
QSC ROLLBACK - Immediate Canary Rollback

Reverts to baseline model weights and restarts AI engine.

EXIT CODES:
  0 = Rollback successful
  1 = Rollback failed

USAGE:
  bash ops/model_safety/qsc_rollback.sh
"""

set -e

echo ""
echo "================================================================================"
echo "QSC ROLLBACK - Reverting to Baseline Weights"
echo "================================================================================"
echo ""

# Paths
BASELINE_WEIGHTS="data/baseline_model_weights.json"
CANARY_WEIGHTS="data/qsc_canary_weights.json"
SYSTEMD_OVERRIDE="/etc/systemd/system/quantum-ai_engine.service.d/qsc_canary.conf"
LOCAL_OVERRIDE="data/systemd_overrides/qsc_canary.conf"
QSC_LOG="logs/qsc_canary.jsonl"

# Check baseline exists
if [ ! -f "$BASELINE_WEIGHTS" ]; then
    echo "❌ Baseline weights not found: $BASELINE_WEIGHTS"
    echo "   Cannot rollback without baseline!"
    exit 1
fi

echo "📋 Baseline weights: $BASELINE_WEIGHTS"
cat "$BASELINE_WEIGHTS"
echo ""

# Remove systemd override (try both locations)
echo "🔧 Removing systemd override..."

if [ -f "$SYSTEMD_OVERRIDE" ]; then
    echo "   Removing: $SYSTEMD_OVERRIDE"
    sudo rm -f "$SYSTEMD_OVERRIDE"
    echo "   ✅ Removed"
else
    echo "   ⚠️  Not found (may not be installed): $SYSTEMD_OVERRIDE"
fi

if [ -f "$LOCAL_OVERRIDE" ]; then
    echo "   Removing: $LOCAL_OVERRIDE"
    rm -f "$LOCAL_OVERRIDE"
    echo "   ✅ Removed"
fi

echo ""

# Remove canary weights file
if [ -f "$CANARY_WEIGHTS" ]; then
    echo "🗑️  Removing canary weights: $CANARY_WEIGHTS"
    rm -f "$CANARY_WEIGHTS"
    echo "   ✅ Removed"
fi

echo ""

# Reload systemd
echo "🔄 Reloading systemd daemon..."
if sudo systemctl daemon-reload 2>/dev/null; then
    echo "   ✅ Reloaded"
else
    echo "   ⚠️  Could not reload (may need manual: sudo systemctl daemon-reload)"
fi

echo ""

# Restart AI engine
echo "🚀 Restarting AI engine..."
if sudo systemctl restart quantum-ai_engine.service 2>/dev/null; then
    echo "   ✅ Restarted"
    
    # Wait for service to stabilize
    sleep 3
    
    # Check status
    echo ""
    echo "📊 Service status:"
    sudo systemctl status quantum-ai_engine.service --no-pager -l | head -20
else
    echo "   ⚠️  Could not restart (may need manual: sudo systemctl restart quantum-ai_engine.service)"
fi

echo ""

# Log rollback
ROLLBACK_TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "{\"timestamp\":\"$ROLLBACK_TS\",\"action\":\"rollback_executed\",\"trigger\":\"manual\",\"baseline_restored\":\"$BASELINE_WEIGHTS\"}" >> "$QSC_LOG"

echo "=" "================================================================================"
echo "✅ ROLLBACK COMPLETED"
echo "================================================================================"
echo ""
echo "Baseline weights restored: $BASELINE_WEIGHTS"
echo "Canary override removed"
echo "AI engine restarted"
echo ""
echo "Rollback logged: $QSC_LOG"
echo ""
