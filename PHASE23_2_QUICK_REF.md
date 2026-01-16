# Phase 23.2 Quick Reference
## QuantumFond Monitoring Stack

---

## 🌐 Access URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| **Grafana** | http://46.224.116.254:3100 | admin / quantumfond2025 |
| **Prometheus** | http://46.224.116.254:9090 | - |
| **Alertmanager** | http://46.224.116.254:9093 | - |
| **CI Metrics** | http://46.224.116.254:9100/metrics | - |
| **Error Metrics** | http://46.224.116.254:9200/metrics | - |

---

## 🔧 Common Commands

### View All Monitoring Containers
```bash
ssh root@46.224.116.254 "systemctl list-units | grep quantum"
```

### Check Service Health
```bash
ssh root@46.224.116.254 "
  curl -s http://localhost:9090/-/healthy && echo ' ✓ Prometheus'
  curl -s http://localhost:3100/api/health && echo ' ✓ Grafana'
  curl -s http://localhost:9093/-/healthy && echo ' ✓ Alertmanager'
  curl -s http://localhost:9100/health && echo ' ✓ CI Exporter'
  curl -s http://localhost:9200/health && echo ' ✓ Error Exporter'
"
```

### View Metrics
```bash
# CI Metrics
curl http://46.224.116.254:9100/metrics | grep ^ci_

# Error Metrics
curl http://46.224.116.254:9200/metrics | grep ^backend_error
```

### View Container Logs
```bash
journalctl -u quantum_prometheus.service
journalctl -u quantum_grafana.service
journalctl -u quantum_alertmanager.service
journalctl -u quantum_ci_exporter.service
journalctl -u quantum_error_exporter.service
```

### Restart Services
```bash
docker restart quantum_prometheus
docker restart quantum_grafana
docker restart quantum_alertmanager
docker restart quantum_ci_exporter
docker restart quantum_error_exporter
```

---

## 📝 Log Formats

### CI Status Log (`/home/qt/quantum_trader/monitoring/logs/ci_status.log`)
```
2025-12-27T00:00:00Z,success,45.2
2025-12-27T01:00:00Z,failure,120.5
```
Format: `timestamp,status,duration_seconds`

### CI Test Failures Log (`/home/qt/quantum_trader/monitoring/logs/ci_tests.log`)
```
2025-12-27T00:00:00Z,unit_tests,0
2025-12-27T01:00:00Z,e2e_tests,3
```
Format: `timestamp,test_suite,failures_count`

### Error Log (`/home/qt/quantum_trader/monitoring/logs/errors.log`)
```
2025-12-27T00:00:00Z ERROR: Connection timeout
2025-12-27T00:01:00Z ERROR: Invalid request
```
Format: `timestamp ERROR: message`

---

## 🧪 Testing Alerts

### Test CI Failure Alert
```bash
echo "$(date -Iseconds),failure,120" | \
  ssh root@46.224.116.254 "cat >> /home/qt/quantum_trader/monitoring/logs/ci_status.log"
```
Wait 2 minutes for alert to fire.

### Test Error Rate Alert
```bash
ssh root@46.224.116.254 "
  for i in {1..10}; do 
    echo \"$(date) ERROR: Test error \$i\" >> /home/qt/quantum_trader/monitoring/logs/errors.log
  done
"
```
Wait 1 minute for alert to fire.

### Check Alertmanager
```bash
curl http://46.224.116.254:9093/api/v2/alerts | jq
```

---

## 🔔 Configure Slack Alerts

1. Edit environment file:
```bash
ssh root@46.224.116.254 "nano /home/qt/quantum_trader/monitoring/.env"
```

2. Add Slack webhook:
```
SLACK_WEBHOOK_TOKEN=T00000000/B00000000/XXXXXXXXXXXXXXXXXXXX
```

3. Restart Alertmanager:
```bash
docker restart quantum_alertmanager
```

4. Test alert:
```bash
curl -X POST http://46.224.116.254:9093/api/v2/alerts -d '[{
  "labels": {"alertname": "test", "severity": "info"},
  "annotations": {"description": "Test alert from CLI"}
}]'
```

---

## 📊 Grafana Dashboard

### Import Dashboard
1. Open http://46.224.116.254:3100
2. Login: admin / quantumfond2025
3. Click **+** → **Import Dashboard**
4. Upload: `/home/qt/quantum_trader/monitoring/grafana/dashboards/ci_errors_dashboard.json`
5. Select **Prometheus** datasource
6. Click **Import**

### Dashboard Panels
- CI Build Status (gauge)
- Build Duration (graph)
- Test Failures (stat)
- Error Rate (graph)
- Memory Usage (graph)
- CPU Usage (graph)
- Service Health (table)
- Active Alerts (list)

---

## 🔍 Prometheus Queries

### CI Metrics
```promql
# Build status (1=success, 0=failure)
ci_last_build_status

# Builds over time
rate(ci_builds_total[5m])

# Average build duration
avg_over_time(ci_last_build_duration_seconds[1h])
```

### Error Metrics
```promql
# Error rate
rate(backend_error_total[5m]) * 60

# Errors per day
increase(backend_error_total[24h])

# P95 error rate
quantile_over_time(0.95, backend_error_rate[1h])
```

---

## 🛠️ Troubleshooting

### Exporter Not Collecting Metrics
```bash
# Check log file exists
ls -lh /home/qt/quantum_trader/monitoring/logs/

# Check exporter logs
journalctl -u quantum_ci_exporter.service
journalctl -u quantum_error_exporter.service

# Verify file permissions
chmod 644 /home/qt/quantum_trader/monitoring/logs/*.log
```

### Prometheus Not Scraping
```bash
# Check targets
curl http://46.224.116.254:9090/api/v1/targets

# Check Prometheus config
docker exec quantum_prometheus cat /etc/prometheus/prometheus.yml

# Reload config
curl -X POST http://46.224.116.254:9090/-/reload
```

### Grafana Dashboard Empty
```bash
# Check datasource
curl http://46.224.116.254:3100/api/datasources

# Test Prometheus connection
docker exec quantum_grafana wget -O- http://quantum_prometheus:9090/-/healthy
```

---

## 📦 File Structure

```
/home/qt/quantum_trader/monitoring/
├── systemctl.monitoring.yml
├── .env
├── logs/
│   ├── ci_status.log
│   ├── ci_tests.log
│   └── errors.log
├── prometheus/
│   ├── prometheus.yml
│   └── alert_rules.yml
├── alertmanager/
│   └── alertmanager.yml
├── grafana/
│   ├── datasources.yml
│   └── dashboards/
│       └── ci_errors_dashboard.json
├── exporters/
│   ├── ci_metrics_exporter.js
│   ├── error_exporter.py
│   ├── Dockerfile.ci
│   └── Dockerfile.error
├── deploy_monitoring.sh
└── test_alerts.sh
```

---

## 🎯 Success Indicators

✅ All 5 containers running  
✅ Prometheus scraping 2 exporters  
✅ Grafana showing dashboards  
✅ Alertmanager configured  
✅ Test metrics flowing  
✅ Alert rules loaded (14 rules)  
✅ Health checks passing  

---

**Phase 23.2 Status:** ✅ **OPERATIONAL**  
**Deployment Date:** 2025-12-27  
**Version:** 1.0.0

