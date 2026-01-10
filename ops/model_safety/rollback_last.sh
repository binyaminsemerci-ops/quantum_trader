#!/bin/bash
# Rollback Last Canary Activation

set -e

BACKUP_DIR="/opt/quantum/backups/model_activations"

if [[ ! -d "$BACKUP_DIR" ]]; then
    echo "❌ No backup directory found: $BACKUP_DIR"
    exit 1
fi

# Find most recent backup
LATEST_BACKUP=$(ls -t "$BACKUP_DIR"/ai-engine.env.* 2>/dev/null | head -n 1)

if [[ -z "$LATEST_BACKUP" ]]; then
    echo "❌ No backup files found in $BACKUP_DIR"
    exit 1
fi

echo "======================================================================"
echo "ROLLBACK TO PREVIOUS CONFIG"
echo "======================================================================"
echo ""
echo "Latest backup: $LATEST_BACKUP"
echo ""
read -p "Restore this backup? (y/N): " -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Rollback cancelled"
    exit 0
fi

# Restore backup
echo "[1/3] Restoring config..."
cp "$LATEST_BACKUP" /etc/quantum/ai-engine.env
echo "✅ Config restored"
echo ""

# Restart service
echo "[2/3] Restarting quantum-ai-engine..."
systemctl restart quantum-ai-engine
sleep 5

if ! systemctl is-active --quiet quantum-ai-engine; then
    echo "❌ SERVICE FAILED TO START AFTER ROLLBACK"
    exit 2
fi

echo "✅ Service restarted"
echo ""

# Journal proof
echo "[3/3] Journal proof (last 15 lines):"
echo "----------------------------------------------------------------------"
journalctl -u quantum-ai-engine --no-pager -n 15

echo ""
echo "======================================================================"
echo "✅ ROLLBACK COMPLETE"
echo "======================================================================"
echo ""
echo "Restored from: $LATEST_BACKUP"
echo ""
echo "NEXT STEPS:"
echo "  1. Verify ensemble is working: make scoreboard"
echo "  2. Check journal for errors: journalctl -u quantum-ai-engine -f"
