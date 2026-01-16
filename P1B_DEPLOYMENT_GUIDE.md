# P1-B DEPLOYMENT GUIDE

**Status**: Ready for deployment  
**Date**: 2026-01-03  
**Commit**: 125465cc

---

## 📋 WHAT WAS IMPLEMENTED

### 1. JSON Logging with correlation_id
- **File**: `shared/logging_config.py`
- **Features**:
  - Structured JSON logs: `{ts, level, service, event, correlation_id, msg, ...}`
  - Context-aware correlation_id tracking (thread-safe)
  - Convenience functions: `log_intent_received`, `log_order_submit`, `log_order_response`
- **Integrated in**:
  - `backend/microservices/auto_executor/executor_service.py`
  - `microservices/ai_engine/service.py`
- **EventBus enhanced**:
  - `backend/core/event_bus.py` - passes correlation_id
  - `backend/core/eventbus/redis_stream_bus.py` - includes correlation_id in messages

### 2. Log Aggregation (Loki + Promtail)
- **Compose**: `systemctl.logging.yml`
- **Loki Config**: `observability/loki/loki-config.yml`
  - 30 days retention
  - 50MB/s ingestion rate
  - Auto-compaction
- **Promtail Config**: `observability/promtail/promtail-config.yml`
  - Scrapes all `quantum_*` containers
  - Parses JSON logs
  - Extracts labels: level, service, event
  - Preserves correlation_id
- **Grafana Datasource**: `observability/grafana/provisioning/datasources/datasource.yml`
  - Loki added with derived fields (correlation_id, order_id)
- **Dashboard**: `observability/grafana/dashboards/p1b_log_aggregation.json`
  - Errors by level (15min)
  - Order flow logs (filterable by correlation_id)
  - Errors by service
  - Order events timeline

### 3. Alerts + Routing
- **Alert Rules**: `observability/prometheus/rules/p1b_alerts.yml`
  - `HighErrorRate`: >10 errors/sec for 2min (CRITICAL)
  - `OrderSubmitWithoutResponse`: >5 unresponded orders in 5min (CRITICAL)
  - `ContainerRestartLoop`: frequent restarts (WARNING)
  - `ExecutionLatencyHigh`: P95 >5sec (WARNING)
  - `NoOrdersSubmitted`: 0 orders for 30min (WARNING)
  - `RedisConnectionLoss`, `LokiDown`, `PromtailDown` (infrastructure)
- **Alertmanager**: `observability/alertmanager/alertmanager.yml`
  - Critical route: webhook + email
  - Warning route: webhook only
  - Inhibition rules (suppress dependent alerts)

### 4. Runbooks
- `RUNBOOKS/P0_execution_stuck.md` - Order execution issues
- `RUNBOOKS/P1B_logging_stack.md` - Logging infrastructure
- `RUNBOOKS/alerts.md` - Alert catalog + tuning guide

### 5. Verification
- `scripts/log_status.sh` - 10 automated checks

---

## 🚀 DEPLOYMENT STEPS

### Prerequisites
- SSH access to VPS: `ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254`
- Code already pushed to GitHub (commit 125465cc)

### Step 1: Pull Latest Code on VPS

```bash
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254

cd /home/qt/quantum_trader
git pull origin main

# Verify commit
git log --oneline -1
# Should show: 125465cc P1-B: Complete Ops Hardening...
```

### Step 2: Build Updated Services

```bash
# Rebuild services that use new logging module
docker compose build auto-executor ai-engine

# Verify image built
docker images | grep quantum | grep latest
```

### Step 3: Deploy Logging Stack

```bash
# Start Loki + Promtail
docker compose -f systemctl.logging.yml up -d

# Wait for startup (30 seconds)
sleep 30

# Verify containers
systemctl list-units | grep loki
systemctl list-units | grep promtail

# Check Loki health
curl http://localhost:3100/ready
# Should return: ready
```

### Step 4: Restart Trading Services (Apply JSON Logging)

```bash
# Restart services to load new logging module
docker compose restart auto-executor ai-engine

# Wait for startup
sleep 15

# Verify JSON logging active
journalctl -u quantum_auto_executor.service --tail 10

# Should see JSON like:
# {"ts":"2026-01-03T...","level":"INFO","service":"auto_executor",...}
```

### Step 5: Reload Prometheus (New Alert Rules)

```bash
# Reload Prometheus config
curl -X POST http://localhost:9090/-/reload

# Verify rules loaded
curl -s http://localhost:9090/api/v1/rules | grep p1b_

# Or check via UI: http://46.224.116.254:9090/rules
```

### Step 6: Restart Grafana (New Datasource + Dashboard)

```bash
# Restart to load Loki datasource
docker compose restart grafana

# Wait for startup
sleep 10

# Verify Grafana
curl http://localhost:3001/api/health
```

### Step 7: Run Verification Script

```bash
# Make script executable
chmod +x scripts/log_status.sh

# Run verification
bash scripts/log_status.sh

# Expected output:
# ✅ PASS: Loki is UP and ready
# ✅ PASS: Promtail is running
# ✅ PASS: Promtail → Loki connectivity OK
# ✅ PASS: Grafana Loki datasource configured
# ✅ PASS: Auto-executor is logging in JSON format
# ✅ PASS: correlation_id present in logs
# ✅ PASS: Loki is ingesting logs from Quantum containers
# ✅ PASS: P1-B alert rules are loaded in Prometheus
# ✅ PASS: Alertmanager has critical routing configured
# ✅ PASS: All required runbooks exist
# 
# ✅ ALL CHECKS PASSED
```

---

## ✅ ACCEPTANCE CRITERIA VERIFICATION

### Criterion 1: correlation_id Tracks 3+ Services

```bash
# Wait for some trading activity or trigger intent manually
# Then check logs:

# Get a correlation_id from auto_executor
CORR_ID=$(journalctl -u quantum_auto_executor.service --tail 100 2>&1 | grep -o '"correlation_id":"[^"]*"' | head -1 | cut -d'"' -f4)

echo "Testing correlation_id: $CORR_ID"

# Search across services
echo "=== Auto Executor ==="
journalctl -u quantum_auto_executor.service 2>&1 | grep "$CORR_ID" | head -5

echo "=== AI Engine ==="
journalctl -u quantum_ai_engine.service 2>&1 | grep "$CORR_ID" | head -5

echo "=== Redis Event Stream ==="
redis-cli XREAD STREAMS quantum:stream:intent quantum:stream:execution 0 0 | grep "$CORR_ID"
```

**Expected**: Same correlation_id appears in:
1. AI Engine (intent generation)
2. Auto Executor (order submission)
3. Redis streams (event propagation)

### Criterion 2: Loki + Promtail Operational

```bash
# Check via log_status.sh (already run)
# Or manually:

# Loki ready
curl http://localhost:3100/ready

# Promtail scraping
journalctl -u quantum_promtail.service --tail 20

# Grafana can query logs
# Go to: http://46.224.116.254:3001/explore
# Select Loki datasource
# Query: {container=~"quantum_.*"}
# Should see logs flowing
```

### Criterion 3: 2+ Alerts Active and Tested

```bash
# View alerts in Prometheus
# Go to: http://46.224.116.254:9090/rules
# Look for: p1b_logging_alerts, p1b_execution_alerts

# Test HighErrorRate alert (simulate)
# Trigger 20 errors in auto_executor:
docker exec quantum_auto_executor python3 -c "
import logging
import sys
sys.path.insert(0, '/app')
from shared.logging_config import setup_json_logging
logger = setup_json_logging('test')
for i in range(20):
    logger.error(f'Test error {i}', extra={'event': 'TEST_ERROR'})
"

# Wait 2 minutes, then check alert fired:
curl -s http://localhost:9090/api/v1/alerts | grep HighErrorRate

# Or check Alertmanager:
curl -s http://localhost:9093/api/v1/alerts | grep HighErrorRate
```

### Criterion 4: Runbooks Written

```bash
# Verify files exist
ls -la RUNBOOKS/
# Should show:
# P0_execution_stuck.md
# P1B_logging_stack.md
# alerts.md
```

### Criterion 5: Verification Script Works

```bash
# Already tested in Step 7
bash scripts/log_status.sh
# Should return exit code 0 (success)
echo $?
```

---

## 🧪 PROOF SNIPPETS

### Proof 1: JSON Logging with correlation_id

```bash
# Get 5 JSON log entries from auto_executor
journalctl -u quantum_auto_executor.service --tail 5

# Example output:
# {"ts":"2026-01-03T10:15:23.456789Z","level":"INFO","service":"auto_executor","event":"INTENT_RECEIVED","correlation_id":"a1b2c3d4-e5f6-7890-abcd-ef1234567890","msg":"Intent received: BTCUSDT","symbol":"BTCUSDT","intent_id":"intent_123","confidence":0.82}
```

### Proof 2: Loki Query Returns Data

```bash
# Query Loki API
curl -G -s 'http://localhost:3100/loki/api/v1/query_range' \
  --data-urlencode 'query={container=~"quantum_.*"} |= "ORDER_"' \
  --data-urlencode 'limit=10' | jq '.data.result[0]'

# Should return log entries with ORDER_ events
```

### Proof 3: Alert Fired

```bash
# After simulating high error rate:
curl -s http://localhost:9093/api/v1/alerts | jq '.data[] | select(.labels.alertname=="HighErrorRate")'

# Should show alert in firing state
```

---

## 🔙 ROLLBACK (If Issues)

### Rollback Logging Stack

```bash
# Stop Loki + Promtail
docker compose -f systemctl.logging.yml down

# Services continue logging to Docker logs
# Access via: docker logs <container>
```

### Rollback JSON Logging (Revert to Old Format)

```bash
# Revert commit
cd /home/qt/quantum_trader
git revert 125465cc

# Rebuild services
docker compose build auto-executor ai-engine

# Restart
docker compose restart auto-executor ai-engine
```

### Rollback Alert Rules

```bash
# Remove P1-B rules
rm observability/prometheus/rules/p1b_alerts.yml

# Reload Prometheus
curl -X POST http://localhost:9090/-/reload
```

---

## 📊 MONITORING POST-DEPLOYMENT

### Day 1: Verify No Issues
- Check log_status.sh every 2 hours
- Monitor Grafana P1-B dashboard for errors
- Check no false positive alerts

### Week 1: Tune Alerts
- Review alert firing history
- Adjust thresholds if too noisy
- Add new alerts if missed incidents

### Month 1: Review Retention
- Check Loki disk usage
- Adjust retention_period if needed (30d → 14d if disk constrained)

---

## 🔗 RELATED DOCS

- `P2_ROADMAP.md` - Next phase (Performance & Alpha)
- `GO_LIVE_CHECKLIST.md` - Go-live procedures
- `RUNBOOKS/*` - Operational runbooks

---

## ✅ DEPLOYMENT COMPLETE

Once all steps pass:
1. ✅ Logging stack operational
2. ✅ JSON logging with correlation_id active
3. ✅ Alerts firing correctly
4. ✅ Runbooks in place
5. ✅ Verification script passing

**Status**: P1-B COMPLETE - System is production-ready from ops perspective

**Next**: Choose path for Phase B (Shadow Mode) or Phase C (Live Small) from P2_ROADMAP.md

