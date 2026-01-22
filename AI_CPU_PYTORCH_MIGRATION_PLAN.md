# CPU-Only PyTorch Migration Plan
## Spare 17GB Disk Space

**Created:** 2026-01-19  
**Status:** Planning Phase  
**Estimated Savings:** 17.2GB (4.3GB CUDA × 4 venvs)

---

## 🎯 Executive Summary

System har **NO GPU** men 17.2GB NVIDIA CUDA libraries installert i 4 virtual environments. Migrasjon til CPU-only PyTorch vil:
- ✅ Spare 17GB disk space
- ✅ Redusere venv størrelse fra 7.7GB → 3.4GB per venv
- ✅ Ingen performance impact (allerede kjører på CPU)
- ⚠️ Krever koordinert restart av alle AI services

---

## 📊 Current State Analysis

### Venvs Affected (4 total)

| Venv | Size | CUDA | PyTorch | Location |
|------|------|------|---------|----------|
| **ai-engine** | 7.7GB | 4.3GB | 1.7GB | /mnt/HC_Volume_104287969/quantum-venvs/ |
| **ai-client-base** | 7.7GB | 4.3GB | 1.7GB | /mnt/HC_Volume_104287969/quantum-venvs/ |
| **strategy-ops** | 7.6GB | 4.3GB | 1.7GB | /mnt/HC_Volume_104287969/quantum-venvs/ |
| **rl-sizer** | 7.6GB | 4.3GB | 1.7GB | /mnt/HC_Volume_104287969/quantum-venvs/ |
| **TOTAL** | **30.6GB** | **17.2GB** | **6.8GB** | - |

### Services Using These Venvs

**ai-engine venv (19 services):**
- quantum-ai-engine.service (kritisk)
- quantum-ai-strategy-router.service (kritisk)
- quantum-exit-monitor.service (kritisk)
- quantum-ceo-brain.service
- quantum-confidence-tracker.service
- quantum-position-tracker.service
- quantum-pnl-tracker.service
- quantum-risk-telemetry.service
- quantum-strategy-validator.service
- 10+ andre services

**ai-client-base venv (8 services):**
- quantum-ai-orchestrator.service
- quantum-binance-pnl-tracker.service
- quantum-circuit-breaker.service
- quantum-cross-exchange-aggregator.service
- quantum-exchange-stream-bridge.service
- 3+ andre services

**strategy-ops venv (5 services):**
- quantum-strategy-ops.service
- quantum-strategy-analytics.service
- 3+ andre services

**rl-sizer venv (3 services):**
- quantum-rl-sizer.service
- quantum-rl-training.service
- quantum-rl-monitor.service

---

## 🔧 Migration Strategy

### Phase 1: Pre-Migration Validation (15 min)

```bash
# 1. Verify system has no GPU
lspci | grep -i nvidia  # Should return nothing
nvidia-smi  # Should fail

# 2. Backup current venv state
cd /mnt/HC_Volume_104287969
tar -czf quantum-venvs-backup-$(date +%Y%m%d).tar.gz quantum-venvs/
# Expected size: ~8GB compressed

# 3. Document current PyTorch version
for venv in ai-engine ai-client-base strategy-ops rl-sizer; do
    echo "$venv:"
    /opt/quantum/venvs/$venv/bin/python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
    /opt/quantum/venvs/$venv/bin/python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"
done
```

**Expected Output:**
```
ai-engine:
  PyTorch: 2.1.0+cu121
  CUDA available: False
```

### Phase 2: Migration Execution (30 min per venv)

**For each venv (ai-engine, ai-client-base, strategy-ops, rl-sizer):**

```bash
#!/bin/bash
VENV_NAME="ai-engine"  # Change for each venv

echo "=== Migrating $VENV_NAME to CPU-only PyTorch ==="

# 1. Stop all services using this venv
systemctl stop quantum-*.service  # Safest: stop all
# OR: Stop only services using this venv (requires manual list)

# 2. Activate venv
source /opt/quantum/venvs/$VENV_NAME/bin/activate

# 3. Uninstall CUDA PyTorch
pip uninstall -y torch torchvision torchaudio

# 4. Install CPU-only PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 5. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 6. Check venv size reduction
du -sh /opt/quantum/venvs/$VENV_NAME

# 7. Deactivate
deactivate

# 8. Restart services
systemctl start quantum-*.service

# 9. Verify services are healthy
sleep 5
systemctl is-active quantum-ai-engine quantum-ai-strategy-router quantum-execution
```

**Expected Before/After:**

| Venv | Before | After | Saved |
|------|--------|-------|-------|
| ai-engine | 7.7GB | 3.4GB | 4.3GB |
| ai-client-base | 7.7GB | 3.4GB | 4.3GB |
| strategy-ops | 7.6GB | 3.3GB | 4.3GB |
| rl-sizer | 7.6GB | 3.3GB | 4.3GB |
| **TOTAL** | **30.6GB** | **13.4GB** | **17.2GB** |

### Phase 3: Validation (15 min)

```bash
# 1. Verify all services are running
systemctl list-units "quantum-*.service" --state=active | wc -l

# 2. Test AI Engine
curl http://localhost:8001/health

# 3. Test model inference
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BTCUSDT","features":[1.0,2.0,3.0]}'

# 4. Check logs for errors
journalctl -u quantum-ai-engine -n 50 | grep -i "error\|cuda\|gpu"

# 5. Verify no CUDA references
for venv in ai-engine ai-client-base strategy-ops rl-sizer; do
    /opt/quantum/venvs/$venv/bin/python -c "import torch; assert not torch.cuda.is_available(), 'CUDA still available!'"
done

# 6. Final disk usage check
df -h /
df -h /mnt/HC_Volume_104287969
```

---

## 📋 Detailed Migration Script

**File:** `/opt/quantum/scripts/migrate_to_cpu_pytorch.sh`

```bash
#!/bin/bash
set -e

# Configuration
VENVS=("ai-engine" "ai-client-base" "strategy-ops" "rl-sizer")
VENV_BASE="/opt/quantum/venvs"
LOG_FILE="/var/log/quantum/pytorch-migration-$(date +%Y%m%d-%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Pre-flight checks
log "=== Pre-Flight Checks ==="

# Check for GPU
if lspci | grep -i nvidia &>/dev/null; then
    log "ERROR: NVIDIA GPU detected. Migration may not be appropriate."
    exit 1
fi

# Backup
log "Creating backup..."
cd /mnt/HC_Volume_104287969
tar -czf quantum-venvs-backup-$(date +%Y%m%d).tar.gz quantum-venvs/ || {
    log "ERROR: Backup failed"
    exit 1
}
log "Backup created: quantum-venvs-backup-$(date +%Y%m%d).tar.gz"

# Migrate each venv
for venv in "${VENVS[@]}"; do
    log ""
    log "=== Migrating $venv ==="
    
    # Stop services (safe approach: stop all)
    log "Stopping all quantum services..."
    systemctl stop quantum-*.service
    
    # Check current size
    BEFORE_SIZE=$(du -sm "$VENV_BASE/$venv" | awk '{print $1}')
    log "Before: ${BEFORE_SIZE}MB"
    
    # Activate and migrate
    source "$VENV_BASE/$venv/bin/activate"
    
    log "Uninstalling CUDA PyTorch..."
    pip uninstall -y torch torchvision torchaudio
    
    log "Installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    
    # Verify
    python -c "import torch; print(f'PyTorch: {torch.__version__}'); assert not torch.cuda.is_available(), 'CUDA still available!'"
    
    deactivate
    
    # Check new size
    AFTER_SIZE=$(du -sm "$VENV_BASE/$venv" | awk '{print $1}')
    SAVED=$((BEFORE_SIZE - AFTER_SIZE))
    log "After: ${AFTER_SIZE}MB (Saved: ${SAVED}MB)"
    
    # Restart services
    log "Restarting all quantum services..."
    systemctl start quantum-*.service
    
    # Wait for services to start
    sleep 10
    
    # Verify critical services
    log "Verifying services..."
    if systemctl is-active quantum-ai-engine quantum-ai-strategy-router quantum-execution &>/dev/null; then
        log "✅ Critical services are active"
    else
        log "❌ WARNING: Some critical services failed to start"
    fi
    
    log "$venv migration complete"
done

# Final validation
log ""
log "=== Final Validation ==="

# Test AI Engine
log "Testing AI Engine..."
if curl -s http://localhost:8001/health | grep -q "OK"; then
    log "✅ AI Engine is healthy"
else
    log "❌ WARNING: AI Engine health check failed"
fi

# Check disk space
log "Disk usage:"
df -h / | tail -1 | tee -a "$LOG_FILE"
df -h /mnt/HC_Volume_104287969 | tail -1 | tee -a "$LOG_FILE"

log ""
log "=== Migration Complete ==="
log "Log saved to: $LOG_FILE"
```

---

## ⚠️ Risks & Mitigation

### Risk 1: Service Downtime (30-120 minutes)
**Impact:** All AI services offline during migration  
**Mitigation:**
- Schedule during low-volume period
- Migrate one venv at a time
- Keep backup for quick rollback

### Risk 2: Import Errors
**Impact:** Code may reference torch.cuda explicitly  
**Mitigation:**
- Test imports after each venv migration
- Check logs for CUDA references
- Have rollback plan ready

### Risk 3: Performance Degradation
**Impact:** CPU-only PyTorch *might* be slower (unlikely)  
**Mitigation:**
- Already running on CPU (no GPU present)
- Monitor inference latency before/after
- Rollback if >10% slowdown

### Risk 4: Backup Failure
**Impact:** Cannot rollback if migration fails  
**Mitigation:**
- Test backup creation first
- Verify tar archive integrity
- Keep backup for 7 days minimum

---

## 🔄 Rollback Plan

If migration fails:

```bash
#!/bin/bash
# Rollback to CUDA PyTorch

# 1. Stop all services
systemctl stop quantum-*.service

# 2. Restore backup
cd /mnt/HC_Volume_104287969
rm -rf quantum-venvs
tar -xzf quantum-venvs-backup-YYYYMMDD.tar.gz

# 3. Verify restoration
ls -lh quantum-venvs/

# 4. Restart services
systemctl start quantum-*.service

# 5. Verify
systemctl is-active quantum-ai-engine quantum-ai-strategy-router
curl http://localhost:8001/health
```

**Time to Rollback:** ~15 minutes

---

## 📈 Expected Outcomes

### Disk Space

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Root Disk (/) | 73% (105GB) | 73% (105GB) | No change (venvs on volume) |
| Volume (/mnt) | 34% (35GB) | 20% (18GB) | **-17GB** |
| Total venvs | 30.6GB | 13.4GB | **-17.2GB** |

### Services

| Service | Downtime | Impact |
|---------|----------|--------|
| AI Engine | 30-60 min | No inference during migration |
| Strategy Router | 30-60 min | No signal generation |
| Exit Monitor | 30-60 min | No position monitoring |
| **Total** | **30-60 min** | **Trading paused** |

### Performance

- ✅ **No expected change** (already running on CPU)
- ✅ Faster venv creation/cloning in future
- ✅ Smaller disk footprint for snapshots

---

## 🎬 Execution Timeline

**Total Time:** 2-3 hours (including validation)

| Time | Phase | Duration |
|------|-------|----------|
| T+0 | Pre-flight checks & backup | 15 min |
| T+15 | Migrate ai-engine | 30 min |
| T+45 | Validate ai-engine | 15 min |
| T+60 | Migrate ai-client-base | 30 min |
| T+90 | Validate ai-client-base | 15 min |
| T+105 | Migrate strategy-ops | 30 min |
| T+135 | Migrate rl-sizer | 30 min |
| T+165 | Final validation | 15 min |
| **T+180** | **Complete** | **3 hours** |

---

## ✅ Go/No-Go Criteria

**Prerequisites for Migration:**

- [ ] System confirmed to have no GPU (`lspci | grep nvidia` = empty)
- [ ] Backup space available (8GB for compressed backup)
- [ ] Low trading volume period identified
- [ ] All services currently healthy
- [ ] Rollback plan tested and documented
- [ ] Team available for monitoring

**Abort Criteria:**

- ❌ Any critical service fails health check before migration
- ❌ Backup creation fails
- ❌ Disk space < 50GB free on volume
- ❌ High trading volume period

---

## 📞 Post-Migration Monitoring

**First 24 Hours:**

```bash
# Check every hour
watch -n 3600 '
    echo "=== Services ==="
    systemctl is-active quantum-ai-engine quantum-ai-strategy-router quantum-execution
    echo ""
    echo "=== Health ==="
    curl -s http://localhost:8001/health | grep status
    echo ""
    echo "=== Disk ==="
    df -h / | tail -1
    df -h /mnt/HC_Volume_104287969 | tail -1
'
```

**Monitor for:**
- ✅ All services remain active
- ✅ No CUDA-related errors in logs
- ✅ Inference latency unchanged
- ✅ Disk space stable at new level

---

## 📝 Post-Migration Cleanup

**After 7 days of stable operation:**

```bash
# Remove backup
rm /mnt/HC_Volume_104287969/quantum-venvs-backup-*.tar.gz

# Document new baseline
echo "Venv sizes after CPU-only migration:" > /opt/quantum/docs/venv-sizes.txt
du -sh /opt/quantum/venvs/* >> /opt/quantum/docs/venv-sizes.txt
```

---

## 🎯 Success Metrics

**Migration considered successful when:**

1. ✅ All 30+ services active and healthy
2. ✅ AI Engine responding to health checks
3. ✅ No CUDA-related errors in logs
4. ✅ Inference latency within 5% of baseline
5. ✅ Disk space reduced by 17GB
6. ✅ System stable for 24 hours

---

## 🔗 Related Documents

- [AI_DISK_UPGRADE_102GB_SUCCESS.md](AI_DISK_UPGRADE_102GB_SUCCESS.md) - Previous disk space work
- [deploy/disk_space_monitor.sh](deploy/disk_space_monitor.sh) - Monitoring script
- `/var/log/quantum/disk-monitor.log` - Current disk usage logs

---

**Status:** Ready for execution when approved  
**Recommendation:** Execute during weekend low-volume period  
**Risk Level:** 🟡 MEDIUM (reversible with backup)
