# RUNBOOK: Alerts Reference

**Category**: Observability  
**Owner**: On-Call Engineer

---

## 📋 ALERT CATALOG

All P1-B production alerts with thresholds, runbooks, and mitigation steps.

---

## 🚨 CRITICAL ALERTS

### HighErrorRate
**Query**: `sum(rate(container_log_entries_total{level="ERROR"}[5m])) > 10`  
**Threshold**: >10 errors/sec for 2min  
**Severity**: CRITICAL

**Symptom**: Service is logging excessive errors

**Quick Check**:
```bash
# Find which service is erroring
docker logs quantum_auto_executor --tail 100 | grep ERROR
docker logs quantum_ai_engine --tail 100 | grep ERROR
```

**Runbook**: `RUNBOOKS/P1B_logging_stack.md#high-error-rate`

**Mitigation**:
1. Identify error source (check logs)
2. If known transient issue (e.g., Binance API timeout): wait
3. If service crash loop: restart container
4. If logic bug: rollback to last known good version

---

### OrderSubmitWithoutResponse
**Query**: `(sum(increase(orders_submitted_total[5m])) - sum(increase(orders_responded_total[5m]))) > 5`  
**Threshold**: >5 unresponded orders in 5min  
**Severity**: CRITICAL

**Symptom**: Orders submitted to Binance but no response received

**Quick Check**:
```bash
# Check for stuck orders
docker logs quantum_auto_executor | grep ORDER_SUBMIT | tail -20
docker logs quantum_auto_executor | grep ORDER_RESPONSE | tail -20

# Compare counts
```

**Runbook**: `RUNBOOKS/P0_execution_stuck.md`

**Mitigation**:
1. Check Binance API connectivity: `curl https://fapi.binance.com/fapi/v1/time`
2. Check executor logs for errors
3. If stuck >5min: consider cancelling orders via Binance UI
4. If persistent: run `scripts/go_live_abort.sh`

---

### RedisConnectionLoss
**Query**: `redis_connected_clients{job="redis"} == 0`  
**Threshold**: 0 clients for 1min  
**Severity**: CRITICAL

**Symptom**: All services disconnected from Redis

**Quick Check**:
```bash
# Check Redis
docker ps | grep quantum_redis
docker exec quantum_redis redis-cli PING
```

**Runbook**: `RUNBOOKS/P1B_logging_stack.md#redis-issues`

**Mitigation**:
1. Restart Redis: `docker compose restart redis`
2. Check if services reconnect automatically
3. If not: restart trading services

---

## ⚠️ WARNING ALERTS

### ContainerRestartLoop
**Query**: `rate(container_last_seen{container=~"quantum_.*"}[5m]) > 0.01`  
**Threshold**: Restarting >0.01/sec for 5min  
**Severity**: WARNING

**Symptom**: Container crash looping

**Quick Check**:
```bash
# Check which container
docker ps -a | grep quantum_

# Check logs for crash reason
docker logs <container_name> --tail 100
```

**Mitigation**:
1. Check logs for error: OOM, Python exception, config issue
2. If OOM: increase container memory limit
3. If config: fix and restart
4. If bug: rollback code

---

### ExecutionLatencyHigh
**Query**: `histogram_quantile(0.95, rate(order_execution_latency_seconds_bucket[5m])) > 5`  
**Threshold**: P95 latency >5sec for 3min  
**Severity**: WARNING

**Symptom**: Orders taking >5sec to execute

**Quick Check**:
```bash
# Check Binance API latency
time curl -s https://fapi.binance.com/fapi/v1/time

# Check executor CPU/memory
docker stats quantum_auto_executor --no-stream
```

**Mitigation**:
1. If Binance API slow: wait (usually resolves)
2. If executor overloaded: check for stuck processes
3. If persistent: increase executor CPU limit

---

### NoOrdersSubmitted
**Query**: `increase(orders_submitted_total[30m]) == 0 and up{job="auto_executor"} == 1`  
**Threshold**: 0 orders for 30min  
**Severity**: WARNING

**Symptom**: Executor alive but not submitting orders

**Possible Causes**:
1. No signals from AI engine
2. Execution policy blocking all orders (GATE_BLOCKED)
3. Risk limits maxed out
4. Bug in executor logic

**Quick Check**:
```bash
# Check for GATE_BLOCKED
docker logs quantum_auto_executor --tail 200 | grep GATE_BLOCKED

# Check if AI engine is publishing intents
docker exec quantum_redis redis-cli XLEN quantum:stream:intent

# Check executor is consuming
docker logs quantum_auto_executor | grep INTENT_RECEIVED | tail -10
```

**Mitigation**:
1. If no intents: check AI engine
2. If GATE_BLOCKED: check execution policy config
3. If logic bug: review code + restart

---

### LokiDown / PromtailDown
**Query**: `up{job="loki"} == 0` / `up{job="promtail"} == 0`  
**Threshold**: Down for 2min  
**Severity**: WARNING

**Symptom**: Log aggregation broken

**Quick Check**:
```bash
docker ps | grep loki
docker ps | grep promtail
```

**Runbook**: `RUNBOOKS/P1B_logging_stack.md#loki-down`

**Mitigation**:
1. Restart: `docker compose -f docker-compose.logging.yml restart loki promtail`
2. Logs still in Docker: `docker logs <container>`
3. Not urgent unless debugging active incident

---

## 🎯 ALERT RESPONSE SLA

| Severity | Response Time | Action Required |
|----------|---------------|-----------------|
| **CRITICAL** | 5 minutes | Immediate investigation, page on-call |
| **WARNING** | 30 minutes | Investigate during business hours |
| **INFO** | N/A | Dashboard only, no action |

---

## 📊 ALERT TUNING

### When to Increase Threshold
- False positives (alerts firing but no real issue)
- Alert fatigue (too many non-actionable alerts)

**Example**: If `HighErrorRate` fires during normal Binance API timeouts, increase threshold from 10 to 20 errors/sec

### When to Decrease Threshold
- Missed incidents (issue occurred but no alert)
- Late detection (alert fired too late)

**Example**: If orders stuck for 10min before `OrderSubmitWithoutResponse` fires, decrease from 5min to 2min

### How to Modify
1. Edit: `observability/prometheus/rules/p1b_alerts.yml`
2. Change `expr` threshold or `for` duration
3. Reload Prometheus: `curl -X POST http://localhost:9090/-/reload`
4. Test: trigger condition and verify alert fires

---

## 🔕 SILENCE ALERTS (Maintenance)

### Temporary Silence (Grafana UI)
1. Go to: Grafana → Alerting → Silences
2. Click "New Silence"
3. Match labels: `alertname=HighErrorRate`
4. Duration: 1h
5. Comment: "Deploying new version"
6. Create

### Permanent Disable (edit rules)
```yaml
# In p1b_alerts.yml, comment out rule:
# - alert: HighErrorRate
#   expr: ...
```

---

## 🔗 RELATED

- `RUNBOOKS/P0_execution_stuck.md`
- `RUNBOOKS/P1B_logging_stack.md`
- `observability/prometheus/rules/p1b_alerts.yml`
- `observability/alertmanager/alertmanager.yml`
