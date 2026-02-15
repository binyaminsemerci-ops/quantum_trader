# Emergency Exit System - Operational Finalization

**Date**: 2026-02-14  
**Status**: Complete  
**Author**: AI Agent

---

## Executive Summary

This document describes the operational finalization of the Emergency Exit System, implementing the institutional-grade specification for fail-closed emergency position closure.

**Key deliverables:**
1. ✅ Redis Streams Schema (exact field definitions)
2. ✅ Emergency Exit Worker schema compliance
3. ✅ Systemd units with WatchdogSec
4. ✅ Exit Brain sd_notify integration
5. ✅ Chaos-test runbook

---

## Redis Streams Schema

### Streams Overview

| Stream | Purpose | Producer |
|--------|---------|----------|
| `system:panic_close` | Global emergency channel | Risk Kernel, Watchdog, Ops |
| `system:panic_close:completed` | EEW execution report | Emergency Exit Worker |
| `exit_brain:heartbeat` | Health monitoring | Exit Brain |
| `system:state:trading` | Trading halt state | Emergency Exit Worker |

### Schema Documentation

Full schema documented in:
- [services/REDIS_STREAMS_SCHEMA.md](services/REDIS_STREAMS_SCHEMA.md)

---

## Components Updated

### 1. Emergency Exit Worker

**File**: `services/emergency_exit_worker/emergency_exit_worker.py`

**Changes:**
- Stream names updated to `system:panic_close`, `system:panic_close:completed`
- Field names aligned with schema:
  - `event_id` (uuid) - tracks event through system
  - `ts`, `ts_started`, `ts_completed` (int64 epoch ms)
  - `positions_total` instead of `positions_found`
  - `issued_by` instead of `source` for incoming events
- Added `"watchdog"` to AUTHORIZED_SOURCES
- Added idempotency key: `system:panic_close:processed`

### 2. Exit Brain Watchdog

**File**: `services/exit_brain/exit_brain_watchdog.py`

**Changes:**
- Stream names updated to `exit_brain:heartbeat`, `system:panic_close`
- DEGRADED_THRESHOLD changed from 10s → 5s (per spec)
- Heartbeat parsing updated to use `ts` (int64 ms) and `active_positions`
- Panic close publishing uses new schema fields:
  - `event_id`, `reason`, `severity`, `issued_by`, `ts`

### 3. Exit Brain with sd_notify

**File**: `services/exit_brain/main_with_watchdog.py` (NEW)

**Features:**
- `sd_notify("READY=1")` at startup
- `sd_notify("WATCHDOG=1")` every 1 second
- Redis heartbeat publishing to `exit_brain:heartbeat`
- Health status tracking (ALIVE/DEGRADED)

### 4. Systemd Unit Files

**quantum-exit-brain.service** (NEW):
```ini
Type=notify
WatchdogSec=3
NotifyAccess=all
```

If Exit Brain fails to notify systemd within 3 seconds, systemd will kill and restart it.

**Existing units updated:**
- quantum-exit-brain-watchdog.service (stream binding confirmed)
- quantum-emergency-exit-worker.service (no restart policy - failures need inspection)

### 5. Test Tools

**tools/test_panic_close.py**

Updated to use new schema:
- Generates `event_id` (uuid4)
- Uses `ts` (epoch ms)
- Uses `issued_by` field
- Waits for completion with matching `event_id`

---

## Deployment Files

| File | Purpose |
|------|---------|
| `ops/systemd/quantum-exit-brain.service` | Exit Brain with WatchdogSec |
| `ops/systemd/quantum-exit-brain-watchdog.service` | Watchdog service |
| `ops/systemd/quantum-emergency-exit-worker.service` | EEW service |
| `services/deploy_emergency_exit_system.sh` | Deployment script |

---

## Watchdog Thresholds

| Condition | Threshold | Action |
|-----------|-----------|--------|
| Heartbeat missing | > 5 seconds | TRIGGER panic_close |
| Status = DEGRADED | > 5 seconds | TRIGGER panic_close |
| Positions unguarded | > 3 seconds | TRIGGER panic_close |
| Decisions stagnant | > 30 seconds | TRIGGER panic_close |

**CRITICAL**: The "positions unguarded" threshold (3s) is the most important - if there are open positions and heartbeat is missing, panic_close triggers faster.

---

## Chaos Testing

Full runbook: [CHAOS_TEST_RUNBOOK.md](services/emergency_exit_worker/CHAOS_TEST_RUNBOOK.md)

### Quick Test Procedure

```bash
# 1. Verify services
systemctl status quantum-exit-brain.service
systemctl status quantum-emergency-exit-worker.service

# 2. Kill Exit Brain
kill -9 $(pgrep -f "main_with_watchdog.py")

# 3. Watch (should complete within 15s)
journalctl -u quantum-exit-brain-watchdog -f
journalctl -u quantum-emergency-exit-worker -f

# 4. Verify
redis-cli XREVRANGE system:panic_close:completed + - COUNT 1
redis-cli HGETALL system:state:trading
```

### Expected Timeline

```
T=0      Exit Brain killed
T+3-5s   Watchdog detects failure
T+3-5s   system:panic_close published
T+5-10s  EEW closes all positions
T+5-10s  system:state:trading.halted = true
```

---

## Safety Guarantees

1. **Fail-closed**: Uncertainty → close positions
2. **No false negatives**: Better to close unnecessarily than miss a failure
3. **AI-independent**: EEW has NO strategy, NO optimization
4. **Testable**: Monthly chaos tests required
5. **Auditable**: All events persisted to Redis streams

---

## Next Steps

1. **VPS Deployment**:
   ```bash
   # Copy files to VPS
   scp -r services/emergency_exit_worker root@46.224.116.254:/home/qt/quantum_trader/services/
   scp ops/systemd/*.service root@46.224.116.254:/etc/systemd/system/
   
   # Install sdnotify
   pip install sdnotify
   
   # Enable services
   systemctl daemon-reload
   systemctl enable quantum-emergency-exit-worker
   systemctl enable quantum-exit-brain-watchdog
   ```

2. **Create Redis consumer groups**:
   ```bash
   redis-cli XGROUP CREATE system:panic_close emergency_exit_worker $ MKSTREAM
   redis-cli XGROUP CREATE system:panic_close audit_logger $ MKSTREAM
   ```

3. **Run chaos test** (after deployment)

---

## File Inventory

### New Files
| Path | Lines | Purpose |
|------|-------|---------|
| `services/REDIS_STREAMS_SCHEMA.md` | ~300 | Complete schema documentation |
| `services/exit_brain/main_with_watchdog.py` | ~180 | sd_notify wrapper |
| `ops/systemd/quantum-exit-brain.service` | ~55 | Systemd unit with WatchdogSec |
| `services/emergency_exit_worker/CHAOS_TEST_RUNBOOK.md` | ~250 | Test procedures |

### Modified Files
| Path | Changes |
|------|---------|
| `services/emergency_exit_worker/emergency_exit_worker.py` | Stream names, schema fields |
| `services/exit_brain/exit_brain_watchdog.py` | Stream names, thresholds, schema |
| `tools/test_panic_close.py` | New schema, event_id tracking |

---

## Conclusion

The Emergency Exit System is now operationally complete with:
- Exact Redis schema compliance
- Systemd watchdog integration (sd_notify)
- Documented chaos testing procedures
- Timeline guarantees (< 15s to safety)

The system is ready for VPS deployment and chaos testing.
