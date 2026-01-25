#!/bin/bash
set -euo pipefail

# ops/rotate_reconcile_secret.sh
# Monthly RECONCILE_CLOSE HMAC secret rotation script
# Run as root on VPS

SCRIPT_NAME="rotate_reconcile_secret.sh"
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

log "Starting RECONCILE_CLOSE secret rotation"

# Create backup directory if not exists
mkdir -p "$BACKUP_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

# Backup current environment files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
cp /etc/quantum/apply-layer.env "$BACKUP_DIR/apply-layer.env.$TIMESTAMP"
cp /etc/quantum/reconcile-engine.env "$BACKUP_DIR/reconcile-engine.env.$TIMESTAMP"
log "Backed up environment files to $BACKUP_DIR/*.$TIMESTAMP"

# Generate new secret (64-character hex = 32 bytes)
NEW_SECRET=$(openssl rand -hex 32)
log "Generated new secret: ${NEW_SECRET:0:16}... (truncated for security)"

# Update Apply Layer environment file
if grep -q "^RECONCILE_CLOSE_SECRET=" /etc/quantum/apply-layer.env; then
    sed -i "s/^RECONCILE_CLOSE_SECRET=.*/RECONCILE_CLOSE_SECRET=$NEW_SECRET/" /etc/quantum/apply-layer.env
    log "Updated RECONCILE_CLOSE_SECRET in apply-layer.env"
else
    echo "RECONCILE_CLOSE_SECRET=$NEW_SECRET" >> /etc/quantum/apply-layer.env
    log "Added RECONCILE_CLOSE_SECRET to apply-layer.env"
fi

# Update P3.4 environment file
if grep -q "^RECONCILE_CLOSE_SECRET=" /etc/quantum/reconcile-engine.env; then
    sed -i "s/^RECONCILE_CLOSE_SECRET=.*/RECONCILE_CLOSE_SECRET=$NEW_SECRET/" /etc/quantum/reconcile-engine.env
    log "Updated RECONCILE_CLOSE_SECRET in reconcile-engine.env"
else
    echo "RECONCILE_CLOSE_SECRET=$NEW_SECRET" >> /etc/quantum/reconcile-engine.env
    log "Added RECONCILE_CLOSE_SECRET to reconcile-engine.env"
fi

# Verify secrets match in both files
APPLY_SECRET=$(grep "^RECONCILE_CLOSE_SECRET=" /etc/quantum/apply-layer.env | cut -d'=' -f2)
P34_SECRET=$(grep "^RECONCILE_CLOSE_SECRET=" /etc/quantum/reconcile-engine.env | cut -d'=' -f2)

if [[ "$APPLY_SECRET" != "$P34_SECRET" ]]; then
    error "Secret mismatch after update! Apply: ${APPLY_SECRET:0:8}... P3.4: ${P34_SECRET:0:8}..."
fi

log "Verified secrets match in both files"

# Restart services (P3.4 first, then Apply Layer)
log "Restarting quantum-reconcile-engine..."
systemctl restart quantum-reconcile-engine
sleep 2

log "Restarting quantum-apply-layer..."
systemctl restart quantum-apply-layer
sleep 2

# Wait for services to stabilize
log "Waiting 10 seconds for services to stabilize..."
sleep 10

# Verify services are running
if ! systemctl is-active --quiet quantum-reconcile-engine; then
    error "quantum-reconcile-engine failed to start after rotation!"
fi

if ! systemctl is-active --quiet quantum-apply-layer; then
    error "quantum-apply-layer failed to start after rotation!"
fi

log "Both services running"

# Check for HMAC failures in recent logs
HMAC_FAILURES=$(journalctl -u quantum-apply-layer --since "30 seconds ago" | grep -c "HMAC signature invalid" || true)

if [[ $HMAC_FAILURES -gt 0 ]]; then
    log "WARNING: Detected $HMAC_FAILURES HMAC failures after rotation"
    log "This may indicate old plans in stream or service sync issue"
    log "Monitoring for 30 more seconds..."
    sleep 30
    
    HMAC_FAILURES_AFTER=$(journalctl -u quantum-apply-layer --since "60 seconds ago" | grep -c "HMAC signature invalid" || true)
    
    if [[ $HMAC_FAILURES_AFTER -gt $HMAC_FAILURES ]]; then
        log "ERROR: HMAC failures increasing! Rollback recommended"
        log "Run: ops/rollback_reconcile_secret.sh $TIMESTAMP"
        exit 1
    else
        log "HMAC failures stopped, likely old plans in stream"
    fi
else
    log "No HMAC failures detected"
fi

# Check metrics endpoint
if curl -s http://localhost:8043/metrics | grep -q "reconcile_close"; then
    log "Metrics endpoint responding correctly"
else
    log "WARNING: Metrics endpoint not responding or no reconcile_close metrics"
fi

# Success summary
log "Secret rotation complete!"
log "New secret deployed to both services"
log "Backup available at: $BACKUP_DIR/*.$TIMESTAMP"
log "To rollback: ops/rollback_reconcile_secret.sh $TIMESTAMP"

# Cleanup old backups (keep last 10)
BACKUP_COUNT=$(ls -1 "$BACKUP_DIR"/apply-layer.env.* | wc -l)
if [[ $BACKUP_COUNT -gt 10 ]]; then
    log "Cleaning up old backups (keeping last 10)"
    ls -1t "$BACKUP_DIR"/apply-layer.env.* | tail -n +11 | xargs rm -f
    ls -1t "$BACKUP_DIR"/reconcile-engine.env.* | tail -n +11 | xargs rm -f
    log "Old backups cleaned"
fi

log "Rotation complete - monitoring recommended for next 24h"
exit 0
