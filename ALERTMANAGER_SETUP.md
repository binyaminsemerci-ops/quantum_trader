# Quantum Trader - Alertmanager Configuration Guide

## Quick Setup

### Option 1: Slack Notifications (Recommended)

1. **Create Slack Webhook:**
   - Go to https://api.slack.com/apps
   - Create new app → "Incoming Webhooks"
   - Activate webhooks → Add New Webhook
   - Select channel (e.g., #quantum-trader-alerts)
   - Copy webhook URL

2. **Configure:**
   ```bash
   # Add to .env
   echo "SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL" >> ~/.env
   
   # Copy Slack config
   cp ~/quantum_trader/monitoring/alertmanager-slack.yml ~/quantum_trader/monitoring/alertmanager.yml
   
   # Restart Alertmanager
   cd ~/quantum_trader
   docker compose -f systemctl.alerting.yml restart alertmanager
   ```

3. **Test:**
   ```bash
   # Stop AI Engine to trigger alert
   docker stop quantum_ai_engine
   # Wait 30 seconds, check Slack
   # Restart
   docker start quantum_ai_engine
   ```

### Option 2: Email Notifications

1. **Configure SMTP (Gmail example):**
   ```bash
   # Add to .env
   cat >> ~/.env << EOF
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USERNAME=your-email@gmail.com
   SMTP_PASSWORD=your-app-specific-password
   ALERT_EMAIL_TO=alerts@yourcompany.com
   EOF
   
   # Copy Email config
   cp ~/quantum_trader/monitoring/alertmanager-email.yml ~/quantum_trader/monitoring/alertmanager.yml
   
   # Restart Alertmanager
   docker compose -f systemctl.alerting.yml restart alertmanager
   ```

2. **Gmail App Password:**
   - Enable 2FA on Gmail
   - Go to: https://myaccount.google.com/apppasswords
   - Generate app password for "Mail"
   - Use this password in SMTP_PASSWORD

### Option 3: Multiple Channels

You can combine both by merging receivers in alertmanager.yml:

```yaml
receivers:
  - name: 'critical-alerts'
    slack_configs:
      - channel: '#alerts'
        # ... slack config
    email_configs:
      - to: 'critical@company.com'
        # ... email config
```

## Alert Rules

Current alerts (from monitoring/alert_rules.yml):
- **ServiceDown**: Any service down for 1 minute
- **AIEngineDown**: AI Engine down for 30 seconds
- **RedisHighMemory**: Redis using >80% memory for 5 minutes
- **HighLatency**: API p95 latency >2s for 5 minutes
- **LowDiskSpace**: <10% disk space for 5 minutes
- **BackupFailed**: No backup in last 8 hours

## Testing Alerts

```bash
# Test ServiceDown alert
docker stop quantum_ai_engine
sleep 35  # Wait for alert to fire
docker start quantum_ai_engine

# View Alertmanager UI
curl http://localhost:9093/api/v2/alerts | python3 -m json.tool

# Check alert status
journalctl -u quantum_alertmanager.service --tail 50
```

## Troubleshooting

1. **Alerts not sending:**
   ```bash
   # Check Alertmanager logs
   journalctl -u quantum_alertmanager.service
   
   # Verify config syntax
   docker exec quantum_alertmanager amtool check-config /etc/alertmanager/alertmanager.yml
   ```

2. **Slack webhook not working:**
   - Verify webhook URL is correct
   - Check channel permissions
   - Test webhook directly:
     ```bash
     curl -X POST -H 'Content-type: application/json' \
       --data '{"text":"Test from Quantum Trader"}' \
       $SLACK_WEBHOOK_URL
     ```

3. **Email not sending:**
   - Verify SMTP credentials
   - Check firewall allows port 587/465
   - Test SMTP connection:
     ```bash
     telnet smtp.gmail.com 587
     ```

## Useful Commands

```bash
# Reload Alertmanager config (without restart)
docker exec quantum_alertmanager kill -HUP 1

# Silence an alert
curl -X POST http://localhost:9093/api/v2/silences \
  -d '{"matchers":[{"name":"alertname","value":"HighLatency"}],"startsAt":"2025-12-16T22:00:00Z","endsAt":"2025-12-16T23:00:00Z","comment":"Maintenance window"}'

# View active alerts
curl http://localhost:9093/api/v2/alerts | jq '.[] | {alertname:.labels.alertname, status:.status.state}'

# Test notification
docker exec quantum_alertmanager amtool alert add --alertmanager.url=http://localhost:9093 \
  test_alert severity=warning service=test
```

