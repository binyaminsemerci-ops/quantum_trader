# Quantum Trader V3 - Audit & Notification Layer

## Overview

The audit and notification layer provides comprehensive logging and daily summaries of all auto-repair actions performed by Quantum Trader V3.

## Components

### 1. Audit Logger (`tools/audit_logger.py`)

Central logging system that records all auto-repair actions with UTC timestamps.

**Features:**
- Timestamped audit logging to `/mnt/status/AUTO_REPAIR_AUDIT.log`
- Daily summary generation
- Discord/Slack webhook notifications
- Email notifications (optional)
- Command-line interface

**Usage:**
```bash
# Log an action (from other scripts)
from audit_logger import log_action
log_action("Auto-repair: Created database 'quantum'")

# Generate daily summary
python3 audit_logger.py daily_summary

# View statistics
python3 audit_logger.py stats

# Test logging
python3 audit_logger.py test
```

### 2. Integration Points

#### Real-Time Monitor (`tools/realtime_monitor.py`)
- Logs when threshold breaches trigger auto-heal
- Logs completion status of heal operations
- Tracks success/failure of repair attempts

#### Weekly Self-Heal (`tools/weekly_self_heal.py`)
- Logs start of validation cycle
- Logs results of each validation phase
- Logs report generation

#### Auto-Repair System (`scripts/auto_repair_system.py`)
- Logs PostgreSQL database creation
- Logs XGBoost model retraining
- Logs Grafana restarts
- Logs Docker cache cleanup

### 3. Scheduled Daily Summary

Two options for scheduling:

#### Option A: Cron Job
```bash
bash scripts/install_audit_cron.sh
```

Creates cron job: `/etc/cron.d/quantum_audit_summary`  
Runs at: 00:00 UTC daily

#### Option B: Systemd Timer (Recommended)
```bash
bash scripts/install_audit_systemd.sh
```

Creates:
- Service: `quantum-audit.service`
- Timer: `quantum-audit.timer`

## Configuration

Set environment variables for notifications:

```bash
# Discord/Slack webhook (required for webhooks)
export QT_ALERT_WEBHOOK="https://discord.com/api/webhooks/..."

# Email notifications (optional)
export QT_AUDIT_EMAIL="admin@example.com"
export QT_SMTP_SERVER="localhost"
export QT_SMTP_PORT="25"
```

Add to `/etc/environment` or container environment for persistence.

## Audit Log Format

```
[2025-12-17T21:00:03Z] Auto-repair: Created PostgreSQL database 'quantum'
[2025-12-17T21:00:04Z] Auto-repair: Retrained XGBoost model to 22 features
[2025-12-17T21:00:06Z] Auto-repair: Restarted Grafana container
[2025-12-17T21:00:08Z] Auto-repair: Cleaned Docker cache
[2025-12-17T21:15:00Z] Weekly self-heal: Started validation cycle
[2025-12-17T21:15:45Z] Weekly self-heal: AI Agent validation - PASSED
[2025-12-17T21:16:12Z] Weekly self-heal: Module integrity check - PASSED
[2025-12-17T21:16:30Z] Weekly self-heal: Completed - ✅ All checks passed
```

## Daily Summary Example

**Discord/Slack Message:**
```
Quantum Trader V3 – Daily Audit Summary (2025-12-18)

📊 Total Actions: 12

Recent Actions:
============================================================
[2025-12-18T00:00:00Z] Auto-repair: PostgreSQL check - OK
[2025-12-18T06:00:00Z] Auto-repair: Triggered by threshold breach
[2025-12-18T06:00:15Z] Auto-repair: XGBoost validation - PASSED
[2025-12-18T12:00:00Z] Weekly self-heal: Started validation cycle
[2025-12-18T12:00:45Z] Weekly self-heal: Completed - ✅ All checks passed
============================================================

✅ 12 auto-repair actions executed
```

## Testing

### Test Audit Logging
```bash
python3 tools/audit_logger.py test
```

### Manual Daily Summary
```bash
python3 tools/audit_logger.py daily_summary
```

### View Audit Statistics
```bash
python3 tools/audit_logger.py stats
```

### Check Systemd Timer Status
```bash
sudo systemctl status quantum-audit.timer
sudo systemctl list-timers quantum-audit.timer
```

### View Logs
```bash
# Audit log
tail -f /mnt/status/AUTO_REPAIR_AUDIT.log

# Cron/systemd execution log
tail -f /var/log/quantum_audit.log

# Systemd journal
sudo journalctl -u quantum-audit.service -f
```

## Integration Checklist

- ✅ `audit_logger.py` created in `tools/`
- ✅ `realtime_monitor.py` integrated with audit logging
- ✅ `weekly_self_heal.py` integrated with audit logging
- ✅ `auto_repair_system.py` integrated with audit logging
- ⏳ Cron job or systemd timer installed (run installer script)
- ⏳ Environment variables configured (optional for notifications)
- ⏳ Test daily summary generation

## Deployment Steps

1. **Upload files to VPS:**
```bash
scp -i ~/.ssh/hetzner_fresh tools/audit_logger.py qt@46.224.116.254:/home/qt/quantum_trader/tools/
scp -i ~/.ssh/hetzner_fresh tools/realtime_monitor.py qt@46.224.116.254:/home/qt/quantum_trader/tools/
scp -i ~/.ssh/hetzner_fresh tools/weekly_self_heal.py qt@46.224.116.254:/home/qt/quantum_trader/tools/
scp -i ~/.ssh/hetzner_fresh scripts/auto_repair_system.py qt@46.224.116.254:/home/qt/quantum_trader/scripts/
scp -i ~/.ssh/hetzner_fresh scripts/install_audit_cron.sh qt@46.224.116.254:/home/qt/quantum_trader/scripts/
scp -i ~/.ssh/hetzner_fresh scripts/install_audit_systemd.sh qt@46.224.116.254:/home/qt/quantum_trader/scripts/
```

2. **Make scripts executable:**
```bash
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 "chmod +x ~/quantum_trader/tools/audit_logger.py ~/quantum_trader/scripts/*.sh"
```

3. **Install daily summary scheduler:**
```bash
# Option A: Cron
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 "bash ~/quantum_trader/scripts/install_audit_cron.sh"

# OR Option B: Systemd (recommended)
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 "bash ~/quantum_trader/scripts/install_audit_systemd.sh"
```

4. **Test the system:**
```bash
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 "python3 ~/quantum_trader/tools/audit_logger.py test"
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 "python3 ~/quantum_trader/tools/audit_logger.py daily_summary"
```

## Monitoring

Monitor audit system health:

```bash
# Check if audit log is growing
ls -lh /mnt/status/AUTO_REPAIR_AUDIT.log

# View recent audit entries
tail -20 /mnt/status/AUTO_REPAIR_AUDIT.log

# Check daily summary execution
sudo tail -f /var/log/quantum_audit.log

# For systemd: check timer schedule
sudo systemctl list-timers quantum-audit.timer
```

## Troubleshooting

### No audit entries appearing
- Check that integrated scripts can import `audit_logger`
- Verify `/mnt/status/` directory exists and is writable
- Check Python path in scripts

### Daily summary not sending
- Verify webhook URL is set: `echo $QT_ALERT_WEBHOOK`
- Check internet connectivity from VPS
- Review `/var/log/quantum_audit.log` for errors
- Test webhook manually: `curl -X POST -H "Content-Type: application/json" -d '{"content":"Test"}' $QT_ALERT_WEBHOOK`

### Cron/timer not running
- Check cron service: `sudo systemctl status cron`
- Check timer: `sudo systemctl status quantum-audit.timer`
- Verify file permissions: `ls -l /etc/cron.d/quantum_audit_summary`
- Check syslog: `sudo tail -f /var/log/syslog | grep quantum`

## Future Enhancements

- [ ] Add severity levels (INFO, WARNING, ERROR)
- [ ] Implement log rotation for audit file
- [ ] Add weekly and monthly summary reports
- [ ] Integrate with Grafana for visualization
- [ ] Add Slack thread replies for detailed reports
- [ ] Implement alert escalation for critical failures
- [ ] Add audit log search/query interface

---

**Version:** 1.0  
**Created:** 2025-12-17  
**Author:** GitHub Copilot  
**Status:** Ready for deployment
