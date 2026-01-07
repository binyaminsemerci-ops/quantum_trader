# Quantum Trader Systemd Operations Runbook

**Last Updated:** 2026-01-07  
**Golden Snapshot:** `/root/unit_backups_clean_20260107_final/`  
**Status:** ✅ Production Stable

---

## Known-Good State

**Current Health (2026-01-07):**
- **Failed units:** 0
- **Running services:** 12/32
- **Activating loops:** 0
- **Restart issues:** 0

**Running Services:**
```
quantum-binance-pnl-tracker    quantum-ceo-brain
quantum-execution              quantum-market-publisher
quantum-portfolio-intelligence quantum-position-monitor
quantum-risk-brain             quantum-rl-feedback-v2
quantum-rl-monitor             quantum-rl-sizer
quantum-strategy-brain         quantum-strategy-ops
```

**Not-Found Units (Expected):** 13 services without code/entrypoints yet:
- quantum-cross-exchange, quantum-exposure-balancer, quantum-meta-regime
- quantum-model-federation, quantum-model-supervisor, quantum-pil
- quantum-portfolio-governance, quantum-retraining-worker, quantum-rl-dashboard
- quantum-strategic-evolution, quantum-strategic-memory
- quantum-trade-intent-consumer, quantum-universe-os

Status: `inactive dead` (safe, not attempting to start)

---

## Quick Health Check

```bash
# Run automated health check
cd /home/qt/quantum_trader
./ops/systemd/healthcheck.sh

# Manual checks
systemctl --failed --no-pager
systemctl list-units "quantum*.service" --state=running --no-legend | wc -l
systemctl list-units "quantum*.service" --all --no-pager | grep -E "activating|failed"
```

**Expected Results:**
- 0 failed units
- 12 running services
- No activating/failed states (except safe "not-found inactive dead")

---

## Service Control

### Start All Services
```bash
systemctl start quantum-trader.target
systemctl status quantum-trader.target --no-pager
```

### Stop All Services
```bash
systemctl stop quantum-trader.target
systemctl status quantum-trader.target --no-pager
```

### Restart All Services
```bash
systemctl restart quantum-trader.target
# Wait for services to stabilize
sleep 10
./ops/systemd/healthcheck.sh
```

### Individual Service Control
```bash
# Restart specific service
systemctl restart quantum-<service-name>.service

# Check logs
journalctl -u quantum-<service-name>.service -n 50 --no-pager

# Check status
systemctl status quantum-<service-name>.service --no-pager
```

---

## Restore from Golden Snapshot

**Golden snapshot location:** `/root/unit_backups_clean_20260107_final/`

### Automated Restore
```bash
cd /home/qt/quantum_trader
./ops/systemd/restore_units_from_snapshot.sh
```

### Manual Restore
```bash
# 1. Stop services
systemctl stop quantum-trader.target

# 2. Backup current state
mkdir -p /root/unit_backups_$(date +%Y%m%d_%H%M%S)
cp /etc/systemd/system/quantum-*.service \
   /etc/systemd/system/quantum-trader.target \
   /root/unit_backups_$(date +%Y%m%d_%H%M%S)/

# 3. Restore from snapshot
cp /root/unit_backups_clean_20260107_final/* /etc/systemd/system/

# 4. Reload systemd
systemctl daemon-reload

# 5. Start services
systemctl start quantum-trader.target

# 6. Verify
./ops/systemd/healthcheck.sh
```

---

## Critical Configuration: /etc/hosts Alias

**Issue:** Python code uses Docker hostname `quantum_redis`  
**Environment:** Systemd deployment has no Docker DNS  
**Solution:** OS-level hostname alias

**Current Configuration:**
```bash
# /etc/hosts line 16
127.0.0.1 quantum_redis
```

**Verification:**
```bash
getent hosts quantum_redis
# Expected: 127.0.0.1 quantum_redis
```

**⚠️ Persistence Warning:**
- `/etc/hosts` changes are NOT persistent across:
  - Cloud instance reprovisioning
  - Image rebuilds
  - Cloud-init resets

**Action Required:** Add to cloud-init user-data:
```yaml
write_files:
  - path: /etc/hosts
    append: true
    content: |
      127.0.0.1 quantum_redis
```

---

## Troubleshooting Guide

| **Symptom** | **Check Command** | **Fix** |
|-------------|-------------------|---------|
| Service won't start (activating loop) | `journalctl -u quantum-<service>.service -n 100` | Check for permission errors, add ReadWritePaths |
| "No module named microservices" | `grep PYTHONPATH /etc/systemd/system/quantum-<service>.service` | Add `Environment="PYTHONPATH=/home/qt/quantum_trader"` |
| "Permission denied" writing files | `journalctl -u quantum-<service>.service \| grep "Permission denied"` | Add `ReadWritePaths=/path/to/dir` to unit file |
| "quantum_redis: Name or service not known" | `getent hosts quantum_redis` | Add `127.0.0.1 quantum_redis` to /etc/hosts |
| Redis connection failed | `redis-cli -h 127.0.0.1 -p 6379 PING` | Check redis-server.service status |
| Unit file not found | `systemctl cat quantum-<service>.service` | Check if unit file exists in /etc/systemd/system |
| Stale failed units | `systemctl --failed` | Run `systemctl reset-failed` |

### Common Permission Fix Pattern
```bash
# 1. Identify permission error
journalctl -u quantum-<service>.service -n 50 | grep -i permission

# 2. Add ReadWritePaths to unit file
sudo nano /etc/systemd/system/quantum-<service>.service
# Add: ReadWritePaths=/path/to/writable/directory

# 3. Reload and restart
systemctl daemon-reload
systemctl restart quantum-<service>.service
systemctl status quantum-<service>.service
```

---

## Export Current State

Export current live unit files to repo:
```bash
cd /home/qt/quantum_trader
./ops/systemd/export_live_units.sh
```

This creates: `ops/systemd/live_units/YYYYMMDD_HHMMSS/` with all unit files.

---

## Key Files Reference

**Systemd Unit Files:**
- `/etc/systemd/system/quantum-*.service` (19 services)
- `/etc/systemd/system/quantum-trader.target` (master target)

**Environment Files:**
- `/etc/quantum/*.env` (per-service configuration)

**Logs:**
- `journalctl -u quantum-<service>.service`
- `/var/log/quantum/*.log` (if configured)

**Backups:**
- `/root/unit_backups_clean_20260107_final/` (golden snapshot)
- `/root/unit_backups_*/` (timestamped backups)

**Repository:**
- `/home/qt/quantum_trader/ops/systemd/` (runbooks + scripts)

---

## Recent Fixes Applied (2026-01-07)

### Fix 1: Removed Stale AI-Engine References
- Removed `quantum-ai-engine.service` from quantum-trader.target Wants=
- Removed from quantum-clm.service After=
- Cleared failed state with `systemctl reset-failed`

### Fix 2: Portfolio-Intelligence Redis Hostname
- Added `/etc/hosts` alias: `127.0.0.1 quantum_redis`
- No code changes required
- Service now resolves Docker hostname under systemd

### Fix 3: EventBus/RL Monitor Write Permissions
- quantum-execution: Added `ReadWritePaths=/home/qt/quantum_trader/runtime`
- quantum-rl-monitor: Added `ReadWritePaths=/var/log`, fixed WorkingDirectory
- quantum-portfolio-intelligence: Added `ReadWritePaths=/home/qt/quantum_trader/logs /home/qt/quantum_trader/runtime`

---

## Emergency Contacts

**Production VPS:** 46.224.116.254  
**System User:** qt:qt (services run as qt)  
**Python Venvs:** /opt/quantum/venvs/ai-client-base  
**Redis:** 127.0.0.1:6379 (system service)

**Escalation:**
1. Check `./ops/systemd/healthcheck.sh`
2. Review `/root/OPS_STATE_2026-01-07_systemd_clean.md`
3. Restore from `/root/unit_backups_clean_20260107_final/`

---

**Version:** 1.0  
**Author:** Principal Ops Engineer  
**Review Date:** 2026-01-07
