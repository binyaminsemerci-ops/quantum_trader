#!/bin/bash
set -euo pipefail

# ops/rollback_reconcile_secret.sh
# Rollback to previous RECONCILE_CLOSE secret if rotation failed

SCRIPT_NAME="rollback_reconcile_secret.sh"
LOG_FILE="/var/log/quantum/secret_rotation.log"
BACKUP_DIR="/etc/quantum/backups"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

error() {
    log "ERROR: $*"
    exit 1
}

# Ensure running as root
if [[ $EUID -ne 0 ]]; then
    error "This script must be run as root"
fi

# Require backup timestamp argument
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <backup_timestamp>"
    echo ""
    echo "Available backups:"
    ls -1 "$BACKUP_DIR"/apply-layer.env.* | sed 's/.*apply-layer.env./  /' || echo "  (none)"
    exit 1
fi

TIMESTAMP=$1
APPLY_BACKUP="$BACKUP_DIR/apply-layer.env.$TIMESTAMP"
P34_BACKUP="$BACKUP_DIR/reconcile-engine.env.$TIMESTAMP"

# Verify backup files exist
if [[ ! -f "$APPLY_BACKUP" ]]; then
    error "Backup not found: $APPLY_BACKUP"
fi

if [[ ! -f "$P34_BACKUP" ]]; then
    error "Backup not found: $P34_BACKUP"
fi

log "Starting rollback to backup: $TIMESTAMP"

# Extract old secret from backup
OLD_SECRET=$(grep "^RECONCILE_CLOSE_SECRET=" "$APPLY_BACKUP" | cut -d'=' -f2 || true)

if [[ -z "$OLD_SECRET" ]]; then
    error "No RECONCILE_CLOSE_SECRET found in backup file"
fi

log "Found secret in backup: ${OLD_SECRET:0:16}... (truncated)"

# Restore backup files
cp "$APPLY_BACKUP" /etc/quantum/apply-layer.env
cp "$P34_BACKUP" /etc/quantum/reconcile-engine.env
log "Restored environment files from backup"

# Verify secrets match
APPLY_SECRET=$(grep "^RECONCILE_CLOSE_SECRET=" /etc/quantum/apply-layer.env | cut -d'=' -f2)
P34_SECRET=$(grep "^RECONCILE_CLOSE_SECRET=" /etc/quantum/reconcile-engine.env | cut -d'=' -f2)

if [[ "$APPLY_SECRET" != "$P34_SECRET" ]]; then
    error "Secret mismatch after rollback! This should not happen"
fi

log "Verified secrets match after rollback"

# Restart services
log "Restarting quantum-reconcile-engine..."
systemctl restart quantum-reconcile-engine
sleep 2

log "Restarting quantum-apply-layer..."
systemctl restart quantum-apply-layer
sleep 5

# Verify services running
if ! systemctl is-active --quiet quantum-reconcile-engine; then
    error "quantum-reconcile-engine failed to start after rollback!"
fi

if ! systemctl is-active --quiet quantum-apply-layer; then
    error "quantum-apply-layer failed to start after rollback!"
fi

log "Both services running after rollback"

# Check for HMAC failures
sleep 10
HMAC_FAILURES=$(journalctl -u quantum-apply-layer --since "20 seconds ago" | grep -c "HMAC signature invalid" || true)

if [[ $HMAC_FAILURES -gt 0 ]]; then
    log "WARNING: Still seeing $HMAC_FAILURES HMAC failures after rollback"
    log "This indicates a deeper issue - investigate manually"
else
    log "No HMAC failures after rollback - system stable"
fi

log "Rollback complete!"
log "Restored secret from backup: $TIMESTAMP"
log "Investigate rotation failure before attempting again"
exit 0
