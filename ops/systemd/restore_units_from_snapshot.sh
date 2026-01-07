#!/bin/bash
# Restore Quantum Trader systemd units from snapshot
# Usage: ./restore_units_from_snapshot.sh [snapshot_dir]

set -e

SNAPSHOT_DIR="${1:-/root/unit_backups_clean_20260107_final}"
BACKUP_DIR="/root/unit_backups_$(date +%Y%m%d_%H%M%S)"

echo "========================================="
echo "RESTORE SYSTEMD UNITS FROM SNAPSHOT"
echo "========================================="
echo "Snapshot: $SNAPSHOT_DIR"
echo "Backup current state to: $BACKUP_DIR"
echo

# Verify snapshot exists
if [ ! -d "$SNAPSHOT_DIR" ]; then
    echo "❌ Error: Snapshot directory not found: $SNAPSHOT_DIR"
    exit 1
fi

# Count files in snapshot
SNAPSHOT_COUNT=$(ls -1 "$SNAPSHOT_DIR" | wc -l)
echo "Snapshot contains $SNAPSHOT_COUNT files"
echo

# Confirm with user
read -p "Continue with restore? (yes/no): " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
    echo "Aborted."
    exit 0
fi

# Step 1: Stop services
echo "Step 1: Stopping quantum-trader.target..."
systemctl stop quantum-trader.target
echo "✅ Services stopped"
echo

# Step 2: Backup current state
echo "Step 2: Backing up current state to $BACKUP_DIR..."
mkdir -p "$BACKUP_DIR"
cp /etc/systemd/system/quantum-*.service \
   /etc/systemd/system/quantum-trader.target \
   "$BACKUP_DIR/" 2>/dev/null || true
BACKUP_COUNT=$(ls -1 "$BACKUP_DIR" | wc -l)
echo "✅ Backed up $BACKUP_COUNT files"
echo

# Step 3: Restore from snapshot
echo "Step 3: Restoring units from snapshot..."
cp "$SNAPSHOT_DIR"/* /etc/systemd/system/
echo "✅ Units restored"
echo

# Step 4: Reload systemd
echo "Step 4: Reloading systemd..."
systemctl daemon-reload
echo "✅ Systemd reloaded"
echo

# Step 5: Start services
echo "Step 5: Starting quantum-trader.target..."
systemctl start quantum-trader.target
echo "✅ Services started"
echo

# Step 6: Wait for services to stabilize
echo "Step 6: Waiting for services to stabilize (10 seconds)..."
sleep 10
echo

# Step 7: Verify
echo "Step 7: Verifying health..."
echo
RUNNING_COUNT=$(systemctl list-units "quantum*.service" --state=running --no-legend | wc -l)
FAILED_COUNT=$(systemctl --failed --no-legend | wc -l)

echo "Running services: $RUNNING_COUNT"
echo "Failed units: $FAILED_COUNT"
echo

if [ "$RUNNING_COUNT" -ge 10 ] && [ "$FAILED_COUNT" -eq 0 ]; then
    echo "✅ Restore successful!"
    echo
    echo "Backup of previous state: $BACKUP_DIR"
else
    echo "⚠️  Restore completed but health check shows issues"
    echo "Check: systemctl status quantum-trader.target"
    echo "Rollback: cp $BACKUP_DIR/* /etc/systemd/system/ && systemctl daemon-reload"
fi

echo
echo "========================================="
