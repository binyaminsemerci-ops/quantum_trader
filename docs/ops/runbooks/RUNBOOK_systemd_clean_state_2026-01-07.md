# RUNBOOK: Systemd Clean State + Portfolio Intelligence Redis Fix

**Date:** 2026-01-07  
**Scope:** Production VPS systemd cleanup, service stability fixes  
**Status:** ✅ Complete - System in clean stable state

---

## Executive Summary

Achieved clean systemd state for Quantum Trader production deployment:
- **0 failed units** (was 1 with stale ai-engine reference)
- **0 activating/restart loops** (was 5 failing services)
- **12/32 services running stable**
- **OS-level fixes only** (no code changes, "INGEN SLETTING" requirement met)

---

## Current System Health

### Service Status
```bash
systemctl --failed --no-pager
# → 0 loaded units listed ✅

systemctl list-units "quantum*.service" --state=running --no-legend | wc -l
# → 12 services running ✅
```

### Running Services (12)
1. quantum-binance-pnl-tracker
2. quantum-ceo-brain
3. quantum-execution
4. quantum-market-publisher
5. quantum-portfolio-intelligence
6. quantum-position-monitor
7. quantum-risk-brain
8. quantum-rl-feedback-v2
9. quantum-rl-monitor
10. quantum-rl-sizer
11. quantum-strategy-brain
12. quantum-strategy-ops

### Not-Found Units (13 - Safe/Expected)
These services are referenced in quantum-trader.target but have no unit files yet:
- quantum-cross-exchange, quantum-exposure-balancer, quantum-meta-regime
- quantum-model-federation, quantum-model-supervisor, quantum-pil
- quantum-portfolio-governance, quantum-retraining-worker, quantum-rl-dashboard
- quantum-strategic-evolution, quantum-strategic-memory
- quantum-trade-intent-consumer, quantum-universe-os

**Status:** inactive dead (not causing issues, safe to ignore)

---

## Golden Snapshot (Rollback Reference)

**Location:** /root/unit_backups_clean_20260107_final/  
**Contents:** 20 files (84K total)
- All quantum-*.service unit files (19 services)
- quantum-trader.target (master target)

### Restore Procedure
```bash
# Backup current state first
mkdir -p /root/unit_backups_$(date +%Y%m%d_%H%M%S)
cp /etc/systemd/system/quantum-*.service /etc/systemd/system/quantum-trader.target /root/unit_backups_$(date +%Y%m%d_%H%M%S)/

# Restore from golden snapshot
cp /root/unit_backups_clean_20260107_final/* /etc/systemd/system/
systemctl daemon-reload
systemctl restart quantum-trader.target

# Verify
systemctl status quantum-trader.target
systemctl --failed
```

---

## Fix 1: Portfolio Intelligence Redis Hostname Resolution

### Symptom
Service stuck in activating/restart loop with error:
```
redis.exceptions.ConnectionError: Error -3 connecting to quantum_redis:6379
socket.gaierror: Temporary failure in name resolution
```

### Root Cause
- **Code uses Docker hostname:** quantum_redis (from Docker Compose environment)
- **Systemd has no Docker DNS:** Hostname not resolvable under systemd deployment
- **Environment file sets IP:** /etc/quantum/portfolio-intelligence.env has REDIS_HOST=127.0.0.1
- **BUT:** Python code also uses hardcoded quantum_redis hostname in some connection paths

### Solution (OS-Level Fix)
Added hostname alias to /etc/hosts:
```bash
echo "127.0.0.1 quantum_redis" >> /etc/hosts
```

**Line added:** Line 16 in /etc/hosts

### Verification
```bash
# Check hostname resolution
getent hosts quantum_redis
# → 127.0.0.1 quantum_redis ✅

# Check service status
systemctl is-active quantum-portfolio-intelligence.service
# → active ✅

systemctl status quantum-portfolio-intelligence.service --no-pager -l | head -20
# → Active: active (running) since... ✅
# → Logs show successful operation (syncing positions every 30s)

# Check process environment
PID=$(systemctl show -p MainPID --value quantum-portfolio-intelligence.service)
tr "\0" "\n" < /proc/$PID/environ | grep -E "^REDIS_"
# → REDIS_HOST=127.0.0.1
# → REDIS_PORT=6379
# → REDIS_DB=0
```

### Important Notes
⚠️ **Persistence Warning:** /etc/hosts changes are NOT persistent across:
- Cloud instance reprovisioning
- Image rebuilds
- Cloud-init resets

**Recommendation:** Add to cloud-init user-data or instance startup script:
```yaml
# cloud-init snippet
write_files:
  - path: /etc/hosts
    append: true
    content: |
      127.0.0.1 quantum_redis
```

---

## Fix 2: Systemd Stale Reference Cleanup

### Issue
systemctl --failed showed:
```
quantum-ai-engine.service    not-found failed
```

### Root Cause
- Unit file /etc/systemd/system/quantum-ai-engine.service was removed/never existed
- References still existed in:
  - quantum-trader.target → Wants=quantum-ai-engine.service
  - quantum-clm.service → After=quantum-ai-engine.service

### Solution
```bash
# Remove from target
sed -i "/^Wants=quantum-ai-engine\.service$/d" /etc/systemd/system/quantum-trader.target

# Remove from clm service
sed -i "s/ quantum-ai-engine\.service//" /etc/systemd/system/quantum-clm.service

# Reload and clear failed state
systemctl daemon-reload
systemctl reset-failed

# Verify no references remain
grep -RIn "quantum-ai-engine\.service" /etc/systemd/system
# → (no output) ✅
```

### Generic Procedure (Future Reference)
To remove any stale service reference:
```bash
# 1. Find all references
grep -RIn "quantum-<service-name>\.service" /etc/systemd/system

# 2. Remove from Wants= lines
sed -i "/^Wants=quantum-<service-name>\.service$/d" /etc/systemd/system/quantum-trader.target

# 3. Remove from After=/Before= lines
sed -i "s/ quantum-<service-name>\.service//" /etc/systemd/system/quantum-*.service

# 4. Reload and clear
systemctl daemon-reload
systemctl reset-failed

# 5. Verify
systemctl --failed --no-pager
```

---

## Fix 3: Previous Fixes (Reference Only)

### quantum-execution.service
**Issue:** EventBus DiskBuffer write permission denied  
**Fix:** Added ReadWritePaths=/home/qt/quantum_trader/runtime

### quantum-rl-monitor.service
**Issue:** CSV log write permission denied  
**Fix:** 
- Changed WorkingDirectory to /home/qt/quantum_trader
- Added ReadWritePaths=/var/log

### quantum-clm.service (DISABLED)
**Issue:** Tries to write to /app/models (read-only filesystem)  
**Status:** Disabled until code change to use /data/quantum/models

### quantum-risk-safety.service (DISABLED)
**Issue:** ModuleNotFoundError: No module named microservices  
**Fix Needed:** Add Environment="PYTHONPATH=/home/qt/quantum_trader" to unit file

---

## Standard Verification Checklist

Run these commands to verify system health:

```bash
# 1. Check for failed units
systemctl --failed --no-pager
# Expected: 0 loaded units listed

# 2. Count running quantum services
systemctl list-units "quantum*.service" --state=running --no-legend | wc -l
# Expected: 12

# 3. Check for bad states (activating/failed/not-found issues)
systemctl list-units "quantum*.service" --all --no-pager | grep -E "activating|failed"
# Expected: no output (or only "not-found inactive dead" which is safe)

# 4. Verify quantum_redis hostname resolution
getent hosts quantum_redis
# Expected: 127.0.0.1 quantum_redis

# 5. Check portfolio-intelligence specifically
systemctl is-active quantum-portfolio-intelligence.service
# Expected: active

systemctl status quantum-portfolio-intelligence.service --no-pager -l | head -20
# Expected: Active: active (running), logs show regular position syncing

# 6. Check all quantum services status
systemctl list-units "quantum*.service" --all --no-pager

# 7. Verify no restart loops (check uptime)
systemctl show quantum-portfolio-intelligence.service -p ActiveEnterTimestamp
# Should show stable timestamp (not changing rapidly)

# 8. Check Redis connectivity from portfolio-intelligence process
PID=$(systemctl show -p MainPID --value quantum-portfolio-intelligence.service)
if [ -n "$PID" ] && [ "$PID" -gt 0 ]; then
  echo "Process running with PID: $PID"
  tr "\0" "\n" < /proc/$PID/environ | grep -E "^REDIS_" || echo "No REDIS_ vars"
fi
```

---

## Troubleshooting Guide

### Service Will Not Start (Activating Loop)

1. **Check logs:**
   ```bash
   journalctl -u quantum-<service-name>.service -n 100 --no-pager
   ```

2. **Common issues:**
   - **Write permission denied:** Add ReadWritePaths=/path/to/writable/dir
   - **Module not found:** Check PYTHONPATH or WorkingDirectory
   - **Network/hostname errors:** Check /etc/hosts for hostname aliases
   - **Redis connection:** Verify Redis is running on 127.0.0.1:6379

3. **Test fix:**
   ```bash
   systemctl restart quantum-<service-name>.service
   sleep 5
   systemctl status quantum-<service-name>.service
   ```

### Hostname Resolution Issues

```bash
# Test DNS resolution
getent hosts <hostname>

# Check /etc/hosts
grep <hostname> /etc/hosts

# Test Redis connection directly
redis-cli -h 127.0.0.1 -p 6379 PING
# or
redis-cli -h quantum_redis -p 6379 PING  # should work if alias exists
```

### Rollback to Previous State

```bash
# Stop all services
systemctl stop quantum-trader.target

# Restore from backup
cp /root/unit_backups_clean_20260107_final/* /etc/systemd/system/
systemctl daemon-reload

# Start services
systemctl start quantum-trader.target

# Verify
systemctl status quantum-trader.target --no-pager
systemctl --failed
```

---

## Next Steps (Optional)

### Low Priority Maintenance
1. **Clean not-found units from target** (optional)
   - Edit /etc/systemd/system/quantum-trader.target
   - Remove Wants= entries for services without code
   - Only do this if clutter is causing confusion

2. **Code refactor** (future consideration)
   - Replace hardcoded quantum_redis hostname in Python code
   - Use REDIS_HOST environment variable consistently
   - Eliminates need for /etc/hosts alias

3. **Cloud-init snippet** (production best practice)
   - Add /etc/hosts alias to cloud-init configuration
   - Ensures persistence across instance rebuilds

4. **Re-enable disabled services**
   - quantum-clm: Needs code change (use /data/quantum/models instead of /app/models)
   - quantum-risk-safety: Add PYTHONPATH=/home/qt/quantum_trader to unit file

5. **Documentation**
   - Create service dependency diagram
   - Document standard troubleshooting workflows
   - Add monitoring/alerting for service state

---

## References

### Source Documentation
- /root/OPS_STATE_2026-01-07_systemd_clean.md - Master state document
- /root/OPS_FIX_2026-01-07_portfolio-intelligence_hosts-alias.md - Specific fix documentation

### Related Files
- **Golden snapshot:** /root/unit_backups_clean_20260107_final/ (20 files, 84K)
- **Unit files:** /etc/systemd/system/quantum-*.service
- **Target:** /etc/systemd/system/quantum-trader.target
- **Env files:** /etc/quantum/*.env
- **Repo:** /home/qt/quantum_trader

### System Info
- **VPS:** 46.224.116.254
- **OS:** Ubuntu 24.04 Noble
- **Systemd:** v255
- **Python:** 3.12.3
- **Redis:** 127.0.0.1:6379 (system service)
- **User:** qt:qt (repo/services)
- **Venvs:** /opt/quantum/venvs/ai-client-base (7.9GB shared)

---

**Last Updated:** 2026-01-07  
**Verified By:** VPS systemd health check  
**Status:** Production stable ✅
