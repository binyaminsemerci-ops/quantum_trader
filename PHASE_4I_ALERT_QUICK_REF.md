# 🚨 PHASE 4I: ALERT SYSTEM - QUICK REFERENCE

---

## 🎛️ CONTAINER MANAGEMENT

### View Alert Logs (Live)
```bash
ssh qt@46.224.116.254 'docker logs quantum_governance_alerts -f'
```

### View Recent Alerts
```bash
ssh qt@46.224.116.254 'docker logs quantum_governance_alerts --tail 50'
```

### Check Container Status
```bash
ssh qt@46.224.116.254 'docker ps --filter name=quantum_governance_alerts'
```

### Restart Alert Service
```bash
ssh qt@46.224.116.254 'docker restart quantum_governance_alerts'
```

---

## 📊 ALERT MONITORING

### View All Alerts in Redis
```bash
ssh qt@46.224.116.254 'docker exec quantum_redis redis-cli LRANGE governance_alerts 0 -1'
```

### Get Alert Count
```bash
ssh qt@46.224.116.254 'docker exec quantum_redis redis-cli LLEN governance_alerts'
```

### View Latest Alert (Formatted)
```bash
ssh qt@46.224.116.254 'docker exec quantum_redis redis-cli LINDEX governance_alerts 0 | python3 -m json.tool'
```

### Clear All Alerts
```bash
ssh qt@46.224.116.254 'docker exec quantum_redis redis-cli DEL governance_alerts'
```

---

## 🧪 TESTING ALERTS

### Test Model Drift Alert
```bash
ssh qt@46.224.116.254 'docker exec quantum_redis redis-cli SET latest_metrics "{\"mape\":0.08,\"sharpe_ratio\":0.5}"'
# Wait 2 minutes, then check logs
```

### Test Missing Weights Alert
```bash
ssh qt@46.224.116.254 'docker exec quantum_redis redis-cli DEL governance_weights'
# Wait 2 minutes, then check logs
```

### Test CPU Alert (High Load)
```bash
ssh qt@46.224.116.254 'yes > /dev/null & sleep 120; pkill yes'
# Check logs during high CPU
```

### View Test Results
```bash
ssh qt@46.224.116.254 'docker logs quantum_governance_alerts --tail 30 | grep ALERT'
```

---

## ⚙️ CONFIGURATION

### Current Thresholds
- **CPU:** > 85%
- **Memory:** > 80%
- **MAPE:** > 0.06
- **Sharpe Ratio:** < 0.8

### Check Current Config
```bash
ssh qt@46.224.116.254 'docker logs quantum_governance_alerts | grep "Thresholds"'
```

### Modify Thresholds
Edit `docker-compose.yml`:
```yaml
environment:
  - CPU_THRESHOLD=90        # Raise CPU threshold
  - MAPE_THRESHOLD=0.04     # Lower MAPE threshold (stricter)
```

Then restart:
```bash
ssh qt@46.224.116.254 'cd ~/quantum_trader && docker compose down governance-alerts && docker compose --profile microservices up -d governance-alerts'
```

---

## 📧 ENABLE EMAIL ALERTS

### 1. Get Gmail App Password
1. Go to Google Account settings
2. Security → 2-Step Verification
3. App passwords → Generate password
4. Copy the 16-character password

### 2. Update docker-compose.yml
```yaml
environment:
  - ALERT_EMAIL=your@email.com
  - EMAIL_USER=your@gmail.com
  - EMAIL_PASS=your_app_password
  - SMTP_SERVER=smtp.gmail.com
  - SMTP_PORT=587
```

### 3. Restart Service
```bash
ssh qt@46.224.116.254 'cd ~/quantum_trader && docker compose down governance-alerts && docker compose --profile microservices up -d governance-alerts'
```

### 4. Verify Email Config
```bash
ssh qt@46.224.116.254 'docker logs quantum_governance_alerts | grep "Email alerts"'
# Should show: "Email alerts: ✅ Enabled"
```

---

## 📱 ENABLE TELEGRAM ALERTS

### 1. Create Telegram Bot
1. Open Telegram, search for @BotFather
2. Send `/newbot` command
3. Follow instructions, name your bot
4. Copy the bot token (format: `123456789:ABCDEF...`)

### 2. Get Your Chat ID
1. Search for @userinfobot in Telegram
2. Send `/start` command
3. Copy your ID (format: `123456789`)

### 3. Update docker-compose.yml
```yaml
environment:
  - TELEGRAM_TOKEN=123456789:ABCDEF123456
  - TELEGRAM_CHAT_ID=123456789
```

### 4. Restart Service
```bash
ssh qt@46.224.116.254 'cd ~/quantum_trader && docker compose down governance-alerts && docker compose --profile microservices up -d governance-alerts'
```

### 5. Verify Telegram Config
```bash
ssh qt@46.224.116.254 'docker logs quantum_governance_alerts | grep "Telegram alerts"'
# Should show: "Telegram alerts: ✅ Enabled"
```

---

## 🔍 TROUBLESHOOTING

### No Alerts Generated
```bash
# Check if monitoring loop is running
ssh qt@46.224.116.254 'docker logs quantum_governance_alerts | grep "Cycle"'

# Verify Redis connection
ssh qt@46.224.116.254 'docker exec quantum_governance_alerts python3 -c "import redis; r=redis.Redis(host=\"quantum_redis\"); print(r.ping())"'
```

### Alerts Not Showing in Dashboard
```bash
# Verify alerts in Redis
ssh qt@46.224.116.254 'docker exec quantum_redis redis-cli LLEN governance_alerts'

# Should return a number > 0
```

### Container Not Healthy
```bash
# Check health status
ssh qt@46.224.116.254 'docker inspect quantum_governance_alerts | grep -A10 Health'

# Check for errors
ssh qt@46.224.116.254 'docker logs quantum_governance_alerts --tail 100 | grep -i error'
```

---

## 📈 MONITORING CHECKLIST

### Daily
- [ ] Check alert count: `LLEN governance_alerts`
- [ ] Review recent logs for critical alerts
- [ ] Verify container is healthy

### Weekly
- [ ] Review all alert types triggered
- [ ] Check alert frequency patterns
- [ ] Verify cooldown system working
- [ ] Test one alert type manually

### Monthly
- [ ] Review and adjust thresholds if needed
- [ ] Clear old alerts from Redis
- [ ] Test email/Telegram notifications
- [ ] Verify integration with all Phase 4 components

---

## 🎯 ALERT TYPES REFERENCE

| Alert Title | Trigger | Threshold | Action |
|------------|---------|-----------|--------|
| High CPU | CPU usage | > 85% | Check resource limits |
| High Memory | Memory usage | > 80% | Scale resources |
| Model Drift | MAPE value | > 0.06 | Phase 4F retraining |
| Low Sharpe | Sharpe ratio | < 0.8 | Review predictions |
| No Model Weights | Missing Redis key | N/A | Check Phase 4E |
| Governance Inactive | Redis flag false | N/A | Restart AI Engine |
| Retrainer Disabled | Redis flag false | N/A | Enable Phase 4F |
| Validation Reject | Log contains REJECT | N/A | Review Phase 4G |

---

## 🚀 QUICK COMMANDS

### One-Line Status Check
```bash
ssh qt@46.224.116.254 'echo "Container:" && docker ps --filter name=quantum_governance_alerts --format "{{.Status}}" && echo "Alerts:" && docker exec quantum_redis redis-cli LLEN governance_alerts && echo "Last Alert:" && docker exec quantum_redis redis-cli LINDEX governance_alerts 0'
```

### Full System Check
```bash
ssh qt@46.224.116.254 'echo "=== Alert Service ===" && docker logs quantum_governance_alerts --tail 20 && echo -e "\n=== Alert Count ===" && docker exec quantum_redis redis-cli LLEN governance_alerts && echo -e "\n=== Latest Alert ===" && docker exec quantum_redis redis-cli LINDEX governance_alerts 0 | python3 -m json.tool'
```

### Restart All Phase 4 Services
```bash
ssh qt@46.224.116.254 'cd ~/quantum_trader && docker restart quantum_ai_engine quantum_governance_dashboard quantum_governance_alerts && sleep 10 && docker ps --filter name=quantum'
```

---

## 📚 DOCUMENTATION

- **Full Guide:** [AI_PHASE_4I_ALERT_SYSTEM_COMPLETE.md](AI_PHASE_4I_ALERT_SYSTEM_COMPLETE.md)
- **Phase 4 Overview:** Check Phase 4D-4I documentation
- **Dashboard Integration:** [PHASE_4H_QUICK_ACCESS.md](PHASE_4H_QUICK_ACCESS.md)

---

**Alert System Status:** ✅ OPERATIONAL  
**Monitoring:** 24/7  
**Last Updated:** 2025-12-20  
