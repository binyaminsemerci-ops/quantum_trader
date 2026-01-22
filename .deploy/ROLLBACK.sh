#!/bin/bash
#
# CORE KERNEL ROLLBACK SCRIPT
# ============================
# Restores system to pre-kernel state
#
set -euo pipefail

DIR=/tmp/core_kernel_20260117_104640
LOG=$DIR/logs/rollback.log

log() { echo "$(date -Iseconds) | $*" | tee -a $LOG; }

log "================================"
log "CORE KERNEL ROLLBACK START"
log "================================"

# Restore original files
log "Restoring original files..."
cp -a $DIR/backup/execution_service.py /home/qt/quantum_trader/services/ || log "No execution_service backup"

# Remove systemd drop-ins
log "Removing systemd drop-ins..."
rm -f /etc/systemd/system/quantum-execution.service.d/20-core-kernel.conf
rm -f /etc/systemd/system/quantum-ai-strategy-router.service.d/20-core-kernel.conf

# Stop and disable timer
log "Stopping recovery timer..."
systemctl stop quantum-stream-recover.timer || true
systemctl disable quantum-stream-recover.timer || true

# Reload and restart services
log "Reloading systemd..."
systemctl daemon-reload

log "Restarting services..."
systemctl restart quantum-execution
systemctl restart quantum-ai-strategy-router

log "================================"
log "ROLLBACK COMPLETE"
log "================================"
log ""
log "System restored to pre-kernel state"
log "Evidence preserved in: $DIR"
