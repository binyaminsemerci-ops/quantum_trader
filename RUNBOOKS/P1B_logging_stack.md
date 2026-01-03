# P1-B RUNBOOK: Logging Stack

**Severity**: P1 - HIGH  
**Category**: Observability Infrastructure  
**Owner**: DevOps / On-Call Engineer

---

## 📋 OVERVIEW

P1-B Logging Stack Components:
- **Loki**: Log aggregation engine (port 3100)
- **Promtail**: Log collector (scrapes Docker logs)
- **Grafana**: Log visualization (Loki datasource)
- **JSON Logging**: Structured logs with correlation_id

---

## 🚨 COMMON ISSUES

### Issue 1: Loki Down

**Symptom**: Alert `LokiDown` firing, Grafana can't query logs

**Checks**:
```bash
# Check container
docker ps | grep quantum_loki

# Check logs
docker logs quantum_loki --tail 50

# Check health endpoint
curl http://localhost:3100/ready
# Expected: ready

# Check disk space (Loki stores logs)
df -h /home/qt/quantum_trader/data
# Should have >10GB free
```

**Fix**:
```bash
# Restart Loki
docker compose -f docker-compose.logging.yml restart loki

# If disk full, clean old logs
docker exec quantum_loki rm -rf /loki/chunks/*
docker compose -f docker-compose.logging.yml restart loki

# Verify
curl http://localhost:3100/ready
```

---

### Issue 2: Promtail Down

**Symptom**: Alert `PromtailDown` firing, logs not appearing in Grafana

**Checks**:
```bash
# Check container
docker ps | grep quantum_promtail

# Check logs
docker logs quantum_promtail --tail 50

# Check if it can reach Loki
docker exec quantum_promtail curl -s http://loki:3100/ready
```

**Fix**:
```bash
# Restart Promtail
docker compose -f docker-compose.logging.yml restart promtail

# Check if positions file is corrupt
docker exec quantum_promtail cat /tmp/positions/positions.yaml

# If corrupt, reset positions (will re-scrape all logs)
docker exec quantum_promtail rm /tmp/positions/positions.yaml
docker compose -f docker-compose.logging.yml restart promtail
```

---

### Issue 3: No Logs in Grafana

**Symptom**: Loki and Promtail UP, but Grafana shows no logs

**Checks**:
```bash
# 1. Check if Loki has data
curl http://localhost:3100/loki/api/v1/label/container/values
# Should return: ["quantum_auto_executor", "quantum_ai_engine", ...]

# 2. Check Grafana datasource
# Go to: Grafana → Configuration → Data Sources → Loki
# Click "Save & Test" - should see "Data source is working"

# 3. Check if services are logging JSON
docker logs quantum_auto_executor --tail 10
# Should see JSON like: {"ts":"2026-01-03T...","level":"INFO",...}
```

**Fix**:
```bash
# If not JSON, services need to load logging_config.py
# Check if shared/logging_config.py is mounted
docker exec quantum_auto_executor ls -la /app/shared/logging_config.py

# If missing, rebuild containers
docker compose build auto-executor ai-engine
docker compose up -d auto-executor ai-engine
```

---

### Issue 4: correlation_id Not Tracking

**Symptom**: Can't trace orders across services using correlation_id

**Checks**:
```bash
# 1. Check if logs have correlation_id field
docker logs quantum_auto_executor --tail 50 | grep correlation_id

# 2. Check a specific order flow
# Get an ORDER_SUBMIT event
docker logs quantum_auto_executor | grep ORDER_SUBMIT | tail -1
# Extract correlation_id from JSON

# 3. Search all logs for that correlation_id
CORR_ID="<paste_correlation_id_here>"
docker logs quantum_auto_executor 2>&1 | grep "$CORR_ID"
docker logs quantum_ai_engine 2>&1 | grep "$CORR_ID"
```

**Expected Flow**:
1. `INTENT_RECEIVED` (with correlation_id)
2. `ORDER_SUBMIT` (same correlation_id)
3. `ORDER_RESPONSE` (same correlation_id)

**Fix if correlation_id missing**:
```bash
# Check if EventBus is passing correlation_id
# File: backend/core/eventbus/redis_stream_bus.py
# Should have: "correlation_id": correlation_id in publish()

# If not, pull latest code and restart
cd /home/qt/quantum_trader
git pull origin main
docker compose build auto-executor ai-engine
docker compose up -d auto-executor ai-engine
```

---

## 🔍 DIAGNOSTIC QUERIES

### Query 1: Errors in Last Hour
```bash
# Via curl (Loki API)
curl -G -s 'http://localhost:3100/loki/api/v1/query_range' \
  --data-urlencode 'query={container=~"quantum_.*"} |= "ERROR"' \
  --data-urlencode 'start=1h' | jq
```

### Query 2: Order Flow for Symbol
```bash
# Via LogQL (in Grafana Explore)
{container="quantum_auto_executor"} 
  |= "ORDER_" 
  | json 
  | symbol="BTCUSDT"
```

### Query 3: High Error Rate Services
```bash
# Via LogQL
sum by (service) (
  count_over_time(
    {container=~"quantum_.*"} |= "ERROR" [5m]
  )
)
```

---

## 🛠️ MAINTENANCE

### Daily: Check Disk Usage
```bash
# Loki data size
docker exec quantum_loki du -sh /loki

# If >50GB, consider reducing retention in loki-config.yml
# retention_period: 720h → 168h (7 days)
```

### Weekly: Verify Logging Health
```bash
# Run log_status.sh
bash scripts/log_status.sh

# Should show:
# ✅ Loki: UP
# ✅ Promtail: UP
# ✅ Grafana Loki Datasource: OK
# ✅ JSON Logging: Active
# ✅ correlation_id: Tracking
```

### Monthly: Review Alerts
```bash
# Check if any alerts fired in last 30 days
# Grafana → Alerting → Alert Rules → p1b_*

# Review:
# - False positives? Adjust thresholds
# - Missed incidents? Add new alerts
# - Too noisy? Increase repeat_interval
```

---

## 🔄 ROLLBACK

### Disable Logging Stack (Emergency)
```bash
# Stop Loki + Promtail (keeps logs in Docker)
docker compose -f docker-compose.logging.yml stop

# Services continue logging to Docker logs
# Access via: docker logs <container>
```

### Re-enable Logging Stack
```bash
# Start Loki + Promtail
docker compose -f docker-compose.logging.yml up -d

# Verify
curl http://localhost:3100/ready
docker ps | grep promtail
```

---

## 📊 METRICS

Key metrics to monitor:
- **Loki Ingestion Rate**: `rate(loki_ingester_bytes_received_total[1m])`
- **Promtail Sent Bytes**: `rate(promtail_sent_bytes_total[1m])`
- **Error Log Rate**: `sum(rate(container_log_entries_total{level="ERROR"}[5m]))`
- **correlation_id Coverage**: `count({container=~"quantum_.*"} | json | correlation_id!="")`

---

## 🔗 RELATED

- `RUNBOOKS/P0_execution_stuck.md` - Order execution issues
- `RUNBOOKS/alerts.md` - Alert definitions and thresholds
- `observability/loki/loki-config.yml` - Loki configuration
- `observability/promtail/promtail-config.yml` - Promtail configuration
- `shared/logging_config.py` - JSON logging module
