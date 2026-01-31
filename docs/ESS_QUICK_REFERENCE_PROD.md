# ESS QUICK REFERENCE — Production Operations

**OS-Level Emergency Stop System** • Systemd-based • Prod-grade • Idempotent

---

## ACTIVATE ESS (Stop All Trading)

```bash
# Method 1: Direct command (recommended)
bash /home/qt/quantum_trader/ops/ess_controller.sh activate

# Method 2: Via systemd (oneshot)
systemctl start quantum-ess.service

# Method 3: Flag file (automatic pickup in 5s via watch timer)
touch /var/run/quantum/ESS_ON
```

**Result:** All trading services stop within **2 seconds**.

Stops:
- `quantum-ai-engine`
- `quantum-execution`
- `quantum-apply-layer`
- `quantum-governor`

Keeps running:
- `grafana-server`
- `quantum-rl-monitor`
- `redis-server`
- `prometheus`

---

## CHECK ESS STATUS

```bash
bash /home/qt/quantum_trader/ops/ess_controller.sh status
```

**Output:**
```
ESS Status: ACTIVE          # Trading stopped
ESS Status: INACTIVE        # Trading active
```

**Check flag file directly:**
```bash
[[ -f /var/run/quantum/ESS_ON ]] && echo "ESS ACTIVE" || echo "ESS INACTIVE"
```

---

## DEACTIVATE ESS (Resume Trading)

```bash
# Method 1: Direct command (recommended)
bash /home/qt/quantum_trader/ops/ess_controller.sh deactivate

# Method 2: Remove flag (automatic pickup in 5s)
rm /var/run/quantum/ESS_ON
```

**Result:** Trading services restart in controlled sequence.

---

## KEY PROPERTIES

### Latch Mechanism ✅
- Once activated, services **WILL NOT** auto-restart
- Must be manually deactivated
- **No flapping** (prevents oscillation)

### Fail-Safe ✅
- Works **even if Python stack hangs**
- Works **even if Redis is down**
- Uses systemd, not Python/Redis
- Writes audit logs to `journalctl`

### Idempotent ✅
- Safe to call multiple times
- No side effects from duplicates
- Controller handles latch internally

### Rapid Stop ✅
- Stops trading within **~2 seconds**
- Uses `systemctl stop` (immediate)
- No graceful shutdown window

---

## VERIFY PROOF

Run automated proof suite:

```bash
bash /home/qt/quantum_trader/scripts/proof_ess.sh
```

This verifies:
1. ✓ Services stop within 2s
2. ✓ ESS flag file mechanism works
3. ✓ Services restart correctly
4. ✓ Audit logs written
5. ✓ Redis marker set/cleared

**Expected output:**
```
============================================================
ESS (EMERGENCY STOP SYSTEM) - E2E PROOF
============================================================
...
[SUMMARY] PASS: All ESS components verified
```

---

## AUDIT LOGS

View ESS events:

```bash
# All ESS events (latest 50)
journalctl -u quantum-ess -u quantum-ess-trigger -n 50

# Real-time ESS events
journalctl -u quantum-ess -u quantum-ess-trigger -f

# Today's ESS events
journalctl -t quantum-ess --since today

# Search for activation
journalctl -t quantum-ess | grep "ACTIVATED"
```

**Example audit log:**
```
Jan 28 00:15:23 quantumtrader quantum-ess[12345]: [ESS AUDIT] ESS ACTIVATION INITIATED
Jan 28 00:15:23 quantumtrader quantum-ess[12345]: [ESS AUDIT] ESS LATCH FLAG SET
Jan 28 00:15:23 quantumtrader quantum-ess[12345]: [ESS AUDIT] ESS STOPPED SERVICE: quantum-ai-engine
Jan 28 00:15:23 quantumtrader quantum-ess[12345]: [ESS AUDIT] ESS REDIS MARKER SET
Jan 28 00:15:23 quantumtrader quantum-ess[12345]: [ESS AUDIT] ESS ACTIVATION COMPLETE
```

---

## DEPLOYMENT

Install ESS systemd units to production:

```bash
# Copy units to systemd
sudo cp /home/qt/quantum_trader/deploy/systemd/quantum-ess*.* \
        /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable path watcher (auto-detects flag file)
sudo systemctl enable quantum-ess.path
sudo systemctl start quantum-ess.path

# Verify installation
sudo systemctl status quantum-ess.path
sudo systemctl list-unit-files | grep quantum-ess
```

---

## SYSTEMD UNITS

| Unit | Purpose | Trigger |
|------|---------|---------|
| `quantum-ess.service` | Manual activation | `systemctl start quantum-ess.service` |
| `quantum-ess.path` | Flag file monitor | Detects `/var/run/quantum/ESS_ON` |
| `quantum-ess-trigger.service` | Oneshot executor | Activated by path unit |
| `quantum-ess-watch.service` | Fallback watcher | Polls flag every 5s |
| `quantum-ess-watch.timer` | Timer for watch service | Every 5s |

---

## TROUBLESHOOTING

### ESS won't activate
```bash
# Verify flag directory exists
sudo mkdir -p /var/run/quantum
sudo chmod 755 /var/run/quantum

# Verify ess_controller.sh is executable
ls -la /home/qt/quantum_trader/ops/ess_controller.sh
# Should have 'x' permission

# Check systemd units
systemctl status quantum-ess.path
systemctl status quantum-ess-watch.timer
```

### Services not stopping
```bash
# Verify services are enabled
systemctl is-enabled quantum-ai-engine.service
systemctl is-enabled quantum-execution.service

# Manually stop if ESS fails
systemctl stop quantum-ai-engine quantum-execution quantum-apply-layer quantum-governor
```

### Check Redis marker
```bash
redis-cli GET quantum:ess:active
# Returns timestamp if ESS active, nil if not
```

---

## INCIDENT PLAYBOOK

### Trading Halt Emergency
```bash
# 1. Activate ESS immediately
bash /home/qt/quantum_trader/ops/ess_controller.sh activate

# 2. Verify all trading stopped
bash /home/qt/quantum_trader/ops/ess_controller.sh status

# 3. Check audit logs
journalctl -t quantum-ess --since "5 minutes ago"

# 4. Assess situation, then deactivate when safe
bash /home/qt/quantum_trader/ops/ess_controller.sh deactivate
```

### Market Anomaly
```bash
# ESS activated automatically if conditions met
# Check Redis for metric thresholds
redis-cli GET quantum:state:portfolio

# Monitor with
tail -f /var/log/quantum/governor.log
journalctl -u quantum-governor -f
```

---

## MONITORING

Add to Grafana dashboard:

```promql
# ESS active?
[[ -f /var/run/quantum/ESS_ON ]] ? 1 : 0

# Services running?
node_systemd_unit_state{name="quantum-ai-engine.service", state="active"} == 1
```

---

## SLA

| Metric | Target | Status |
|--------|--------|--------|
| Activation time | < 2s | ✅ |
| Deactivation time | < 5s | ✅ |
| Audit logging | 100% | ✅ |
| Latch reliability | 100% | ✅ |
| Fail-safe | Works offline | ✅ |

---

**Last Updated:** Jan 31, 2026  
**Status:** PROD-GRADE • VERIFIED • OPERATIONALIZED
