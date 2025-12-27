# 🎯 Quantum Trader V3 - Audit Layer Quick Reference

## 📋 One-Line Commands

### View Audit Log
```bash
tail -f ~/quantum_trader/status/AUTO_REPAIR_AUDIT.log
```

### Generate Daily Summary
```bash
python3 ~/quantum_trader/tools/audit_logger.py daily_summary
```

### View Statistics
```bash
python3 ~/quantum_trader/tools/audit_logger.py stats
```

### Test Logging
```bash
python3 ~/quantum_trader/tools/audit_logger.py test
```

---

## 📍 File Locations

| File | Location |
|------|----------|
| Audit Log | `~/quantum_trader/status/AUTO_REPAIR_AUDIT.log` |
| Audit Logger | `~/quantum_trader/tools/audit_logger.py` |
| Summary Log | `~/quantum_trader/status/audit_summary.log` |
| Cron Job | User crontab (run `crontab -l`) |

---

## 🔧 Configuration

### Enable Webhook Notifications
```bash
export QT_ALERT_WEBHOOK="https://discord.com/api/webhooks/YOUR_ID/YOUR_TOKEN"
```

### Enable Email Notifications
```bash
export QT_AUDIT_EMAIL="admin@example.com"
export QT_SMTP_SERVER="localhost"
export QT_SMTP_PORT="25"
```

---

## 📊 Expected Log Entries

### Auto-Repair Actions
```
[2025-12-17T17:40:37Z] Auto-repair: Created PostgreSQL database quantum
[2025-12-17T17:40:37Z] Auto-repair: Retrained XGBoost model to 22 features
[2025-12-17T17:40:37Z] Auto-repair: Restarted Grafana container
[2025-12-17T17:40:37Z] Auto-repair: Cleaned Docker cache
```

### Weekly Self-Heal
```
[2025-12-17T17:40:37Z] Weekly self-heal: Started validation cycle
[2025-12-17T17:40:38Z] Weekly self-heal: AI Agent validation - PASSED
[2025-12-17T17:40:39Z] Weekly self-heal: Module integrity check - PASSED
[2025-12-17T17:40:40Z] Weekly self-heal: Completed - ✅ All checks passed
```

### Real-Time Monitor
```
[2025-12-17T18:00:00Z] Auto-heal: Triggered by realtime monitor threshold breach
[2025-12-17T18:00:15Z] Auto-heal: Completed successfully (exit code 0)
```

---

## 🕐 Schedule

| Action | Frequency | Time |
|--------|-----------|------|
| Real-time monitoring | Hourly | Every hour |
| Weekly self-heal | Weekly | Sunday 00:00 |
| Daily summary | Daily | 00:00 UTC |

---

## ✅ Quick Health Check

```bash
# 1. Check audit log exists and is growing
ls -lh ~/quantum_trader/status/AUTO_REPAIR_AUDIT.log

# 2. View last 10 entries
tail -10 ~/quantum_trader/status/AUTO_REPAIR_AUDIT.log

# 3. Check today's entry count
python3 ~/quantum_trader/tools/audit_logger.py stats

# 4. Verify cron job
crontab -l | grep audit_logger
```

---

## 🚨 Troubleshooting

### No audit entries appearing
```bash
# Check log file permissions
ls -l ~/quantum_trader/status/AUTO_REPAIR_AUDIT.log

# Test logging manually
python3 ~/quantum_trader/tools/audit_logger.py test

# Check Python path in integrated scripts
grep "sys.path.insert" ~/quantum_trader/scripts/auto_repair_system.py
```

### Daily summary not running
```bash
# Check cron job exists
crontab -l | grep audit_logger

# Check cron execution log
tail -20 ~/quantum_trader/status/audit_summary.log

# Test manually
python3 ~/quantum_trader/tools/audit_logger.py daily_summary
```

### Webhook not sending
```bash
# Verify webhook URL is set
echo $QT_ALERT_WEBHOOK

# Test webhook manually
curl -X POST -H "Content-Type: application/json" \
  -d '{"content":"Test from Quantum Trader"}' \
  $QT_ALERT_WEBHOOK
```

---

**Keep this card:** Save to desktop or pin in documentation folder  
**VPS:** 46.224.116.254  
**Status:** 🟢 Operational  
**Version:** 1.0
