#!/bin/bash
set -e

VENVS=("ai-engine" "ai-client-base" "strategy-ops" "rl-sizer")
VENV_BASE="/opt/quantum/venvs"
LOG_FILE="/var/log/quantum/pytorch-migration-$(date +%Y%m%d-%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "=== PyTorch CPU-Only Migration Starting ==="
log "Log file: $LOG_FILE"

# Pre-flight checks
log "=== Pre-Flight Checks ==="
if lspci | grep -i nvidia &>/dev/null; then
    log "ERROR: NVIDIA GPU detected. Aborting."
    exit 1
fi
log "✅ No GPU detected"

# Backup
log "Creating backup (this will take ~5 minutes)..."
cd /mnt/HC_Volume_104287969
BACKUP_FILE="quantum-venvs-backup-$(date +%Y%m%d-%H%M%S).tar.gz"
tar -czf "$BACKUP_FILE" quantum-venvs/ || {
    log "ERROR: Backup failed"
    exit 1
}
BACKUP_SIZE=$(du -sh "$BACKUP_FILE" | awk '{print $1}')
log "✅ Backup created: $BACKUP_FILE ($BACKUP_SIZE)"

# Migrate each venv
TOTAL_SAVED=0
for venv in "${VENVS[@]}"; do
    log ""
    log "=== Migrating $venv ==="
    
    log "Stopping all quantum services..."
    systemctl stop quantum-*.service
    sleep 2
    
    BEFORE_SIZE=$(du -sm "$VENV_BASE/$venv" | awk '{print $1}')
    log "Before: ${BEFORE_SIZE}MB"
    
    source "$VENV_BASE/$venv/bin/activate"
    
    log "Uninstalling CUDA PyTorch..."
    pip uninstall -y torch torchvision torchaudio >> "$LOG_FILE" 2>&1
    
    log "Installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu >> "$LOG_FILE" 2>&1
    
    log "Verifying installation..."
    python -c "import torch; print(f'PyTorch: {torch.__version__}'); assert not torch.cuda.is_available(), 'CUDA still available!'" | tee -a "$LOG_FILE"
    
    deactivate
    
    AFTER_SIZE=$(du -sm "$VENV_BASE/$venv" | awk '{print $1}')
    SAVED=$((BEFORE_SIZE - AFTER_SIZE))
    TOTAL_SAVED=$((TOTAL_SAVED + SAVED))
    log "After: ${AFTER_SIZE}MB (Saved: ${SAVED}MB)"
    
    log "Restarting quantum services..."
    systemctl start quantum-*.service
    sleep 10
    
    log "Verifying critical services..."
    if systemctl is-active quantum-ai-engine quantum-ai-strategy-router quantum-execution &>/dev/null; then
        log "✅ Critical services are active"
    else
        log "❌ WARNING: Some critical services failed to start"
        systemctl status quantum-ai-engine quantum-ai-strategy-router quantum-execution --no-pager | tee -a "$LOG_FILE"
    fi
    
    log "✅ $venv migration complete"
done

# Final validation
log ""
log "=== Final Validation ==="
log "Total disk space saved: ${TOTAL_SAVED}MB (~$((TOTAL_SAVED / 1024))GB)"

log "Testing AI Engine health..."
if curl -s http://localhost:8001/health | grep -q "OK"; then
    log "✅ AI Engine is healthy"
else
    log "❌ WARNING: AI Engine health check failed"
fi

log "Final disk usage:"
df -h / | tail -1 | tee -a "$LOG_FILE"
df -h /mnt/HC_Volume_104287969 | tail -1 | tee -a "$LOG_FILE"

log ""
log "=== Migration Complete ==="
log "Backup location: /mnt/HC_Volume_104287969/$BACKUP_FILE"
log "Full log: $LOG_FILE"
