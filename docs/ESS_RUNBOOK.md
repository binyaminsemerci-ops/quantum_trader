# ESS (Emergency Stop System) Runbook

**Purpose:** OS-level emergency stop for Quantum Trader, independent of Python/Redis.

**Critical:** ESS uses a latch mechanism - once activated, services will NOT auto-restart until manually deactivated.

---

## Quick Reference

### Activate ESS (Stop Trading)

```bash
# Method 1: Direct command
bash /home/qt/quantum_trader/ops/ess_controller.sh activate

# Method 2: Systemd
systemctl start quantum-ess.service

# Method 3: Flag file
mkdir -p /var/run/quantum
touch /var/run/quantum/ESS_ON
```

**Result:** All trading services stop within 2 seconds.

---

### Check ESS Status

```bash
bash /home/qt/quantum_trader/ops/ess_controller.sh status
```

**Output:**
- `ESS STATUS: ACTIVE` - Trading stopped
- `ESS STATUS: INACTIVE` - Trading active

---

### Deactivate ESS (Resume Trading)

```bash
bash /home/qt/quantum_trader/ops/ess_controller.sh deactivate
```

**Result:** Services restart in controlled sequence.

---

## Architecture

### Components

1. **ESS Controller:** `/home/qt/quantum_trader/ops/ess_controller.sh`
   - Bash script (no Python dependencies)
   - Manages latch flag and service lifecycle

2. **Latch Flag:** `/var/run/quantum/ESS_ON`
   - Presence = ESS active
   - Absence = ESS inactive

3. **Systemd Units:**
   - `quantum-ess.service` - Manual activation
   - `quantum-ess.path` - Monitors flag file
   - `quantum-ess-trigger.service` - Activated by path monitor

4. **Trading Services Stopped:**
   - `quantum-ai-engine`
   - `quantum-execution`
   - `quantum-apply-layer`
   - `quantum-governor`

5. **Monitor Services (Keep Running):**
   - `grafana-server`
   - `quantum-rl-monitor`
   - `redis-server`
   - `prometheus`

---

## Scenarios

### Scenario 1: Market Crash / Flash Crash

**Situation:** Need to stop all trading immediately.

**Action:**
```bash
systemctl start quantum-ess.service
```

**Verification:**
```bash
# Check ESS status
bash /home/qt/quantum_trader/ops/ess_controller.sh status

# Verify services stopped
systemctl status quantum-ai-engine
systemctl status quantum-execution
systemctl status quantum-governor
```

**Resume Trading:**
```bash
# After market stabilizes and manual review
bash /home/qt/quantum_trader/ops/ess_controller.sh deactivate
```

---

### Scenario 2: System Instability / Python Hang

**Situation:** Python services unresponsive, need emergency stop.

**Action:**
```bash
# ESS works even if Python is hung
touch /var/run/quantum/ESS_ON
```

**Result:** Path monitor detects flag within 10s, triggers systemd stop.

---

### Scenario 3: Maintenance Window

**Situation:** Need to perform maintenance, stop trading temporarily.

**Action:**
```bash
bash /home/qt/quantum_trader/ops/ess_controller.sh activate
```

**Perform Maintenance:**
- Deploy code updates
- Restart services individually
- Test changes

**Resume Trading:**
```bash
bash /home/qt/quantum_trader/ops/ess_controller.sh deactivate
```

---

### Scenario 4: ESS Test / Drill

**Purpose:** Verify ESS works in production.

**Action:**
```bash
cd /home/qt/quantum_trader
bash scripts/proof_ess.sh
```

**Expected Output:**
```
SUMMARY: PASS
ESS is production-ready for emergency stop scenarios
```

**Frequency:** Weekly or after system changes.

---

## Latch Mechanism

**Design:** ESS uses a latch to prevent accidental restart.

**Behavior:**
1. ESS activated → Flag created → Services stopped
2. Services **DO NOT** auto-restart while flag exists
3. Systemd `Restart=always` is overridden by ESS latch
4. Manual deactivation required to resume trading

**Rationale:**
- Prevents flapping (start/stop loops)
- Ensures human review before resuming
- Fail-safe: trading stays stopped until explicit approval

---

## Troubleshooting

### ESS Won't Activate

**Symptom:** Services still running after ESS activation.

**Debug:**
```bash
# Check ESS controller
bash /home/qt/quantum_trader/ops/ess_controller.sh activate

# Check journalctl for ESS logs
journalctl -u quantum-ess -u quantum-ess-trigger --since "5 minutes ago"

# Manually stop services
for svc in quantum-ai-engine quantum-execution quantum-apply-layer quantum-governor; do
    systemctl stop $svc
done
```

---

### ESS Won't Deactivate

**Symptom:** Services won't restart after deactivation.

**Debug:**
```bash
# Check flag removed
ls -la /var/run/quantum/ESS_ON

# Manually restart services
for svc in quantum-ai-engine quantum-execution quantum-apply-layer quantum-governor; do
    systemctl start $svc
    systemctl status $svc
done
```

---

### Flag File Missing After Reboot

**Symptom:** `/var/run/quantum/ESS_ON` disappears after reboot.

**Expected Behavior:** This is correct. `/var/run` is tmpfs, cleared on reboot.

**Action:** ESS automatically deactivates on reboot (fail-safe: trading resumes).

---

## Monitoring

### Logs

**ESS activation/deactivation:**
```bash
journalctl -u quantum-ess -u quantum-ess-trigger -f
```

**Trading services:**
```bash
journalctl -u quantum-ai-engine -u quantum-execution -u quantum-governor -f
```

---

### Metrics

**ESS State:**
```bash
# Check Redis marker (best-effort)
redis-cli GET quantum:ess:active
```

**Service Status:**
```bash
for svc in quantum-ai-engine quantum-execution quantum-apply-layer quantum-governor; do
    echo "$svc: $(systemctl is-active $svc)"
done
```

---

## Production Deployment

### Initial Setup

```bash
# Copy systemd units
cp /home/qt/quantum_trader/deploy/systemd/quantum-ess.* /etc/systemd/system/

# Reload systemd
systemctl daemon-reload

# Enable path monitor
systemctl enable quantum-ess.path
systemctl start quantum-ess.path

# Verify path monitor active
systemctl status quantum-ess.path
```

---

### Verify Installation

```bash
# Run proof script
bash /home/qt/quantum_trader/scripts/proof_ess.sh

# Expected output
SUMMARY: PASS
```

---

## Audit Trail

All ESS activations/deactivations are logged to journalctl:

```bash
journalctl -t quantum-ess --since today
```

**Example Log Entry:**
```
Jan 28 00:15:23 quantumtrader-prod-1 quantum-ess[123456]: [2026-01-28 00:15:23 UTC] ESS: === ESS ACTIVATION INITIATED ===
Jan 28 00:15:23 quantumtrader-prod-1 quantum-ess[123456]: [2026-01-28 00:15:23 UTC] ESS: Stopping quantum-ai-engine.service
Jan 28 00:15:24 quantumtrader-prod-1 quantum-ess[123456]: [2026-01-28 00:15:24 UTC] ESS: === ESS ACTIVATED - TRADING STOPPED ===
```

---

## Security

**Access Control:** Only `root` can activate/deactivate ESS.

**File Permissions:**
```bash
chmod 700 /home/qt/quantum_trader/ops/ess_controller.sh
chown root:root /home/qt/quantum_trader/ops/ess_controller.sh
```

**Systemd Units:** Installed in `/etc/systemd/system/` (root-only).

---

## Summary

**ESS Provides:**
- ✅ OS-level emergency stop (independent of Python)
- ✅ Fast shutdown (<2 seconds)
- ✅ Latch mechanism (prevents auto-restart)
- ✅ Controlled rollback
- ✅ Audit logging
- ✅ Systemd integration

**Use Cases:**
- Market crash / flash crash
- System instability
- Maintenance windows
- Emergency intervention

**Next Steps:**
- Test ESS weekly with `proof_ess.sh`
- Review logs after each activation
- Document any edge cases in this runbook
