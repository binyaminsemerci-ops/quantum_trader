# 🎯 RL Shadow Metrics - Grafana Integration

**Status**: ✅ LIVE  
**Dashboard**: https://app.quantumfond.com/grafana → Search "RL Shadow"  
**Deployed**: 2026-01-15T23:12:00Z

---

## Quick Access

### Grafana Dashboard
1. **URL**: https://app.quantumfond.com/grafana
2. **Search**: "RL Shadow System - Performance Monitoring"
3. **Auto-refresh**: 30 seconds
4. **Time range**: Last 6 hours (adjustable)

### 8 Real-Time Panels
- **Gate Pass Rate** (12.8% baseline) - Time series by symbol
- **Cooldown Blocking Rate** (19.3% baseline) - Trend over time
- **Average Pass Rate** - Gauge with thresholds
- **Eligible Rate** - Pass + cooldown active
- **Policy Age** - Freshness check (max 600s)
- **Ensemble Confidence** - Pass vs Fail comparison
- **Would Flip Rate** - RL disagreement metric
- **Total Intents** - Cumulative counter

---

## Architecture

```
AI Engine → Redis Stream (quantum:stream:trade.intent)
              ↓
RL Shadow Metrics Exporter (port 9092)
  - Analyzes 500 recent intents every 60s
  - Exports 17 Prometheus metrics
              ↓
Prometheus (port 9091)
  - Scrapes exporter every 60s
  - Stores time-series data
              ↓
Grafana (https://app.quantumfond.com/grafana)
  - Visualizes metrics
  - Auto-refreshes every 30s
```

---

## Prometheus Metrics (17 total)

### Gate Performance
- `quantum_rl_gate_pass_rate{symbol="..."}` - Pass rate per symbol (0.0-1.0)
- `quantum_rl_cooldown_blocking_rate{symbol="..."}` - Cooldown impact (0.0-1.0)
- `quantum_rl_eligible_rate{symbol="..."}` - Pass + cooldown (0.0-1.0)

### RL System Health
- `quantum_rl_policy_age_seconds{symbol="..."}` - Policy freshness (0-600s)
- `quantum_rl_confidence_avg{symbol="..."}` - RL confidence (0.0-1.0)

### Ensemble Comparison
- `quantum_rl_ensemble_confidence_pass{symbol="..."}` - When gate passes
- `quantum_rl_ensemble_confidence_fail{symbol="..."}` - When gate fails

### RL Effects
- `quantum_rl_would_flip_rate{symbol="..."}` - Disagrees with ensemble
- `quantum_rl_reinforce_rate{symbol="..."}` - Agrees with ensemble

### Counters
- `quantum_rl_intents_analyzed_total` - Total intents processed
- `quantum_rl_gate_passes_total{symbol="..."}` - Cumulative passes
- `quantum_rl_gate_failures_total{symbol="..."}` - Cumulative failures

---

## Service Management

### Check Status
```bash
# Metrics exporter
systemctl status quantum-rl-shadow-metrics-exporter.service

# Prometheus
systemctl status prometheus

# Grafana
systemctl status grafana-server
```

### View Logs
```bash
# Exporter logs (see data collection)
journalctl -u quantum-rl-shadow-metrics-exporter.service -f

# Should show: "📊 Updated metrics: 500 intents, 51 symbols"
```

### Restart Services
```bash
# If metrics exporter stops working
systemctl restart quantum-rl-shadow-metrics-exporter.service

# If Prometheus not scraping
systemctl reload prometheus

# If Grafana dashboard not updating
systemctl restart grafana-server
```

---

## Manual Queries

### Query Metrics Directly
```bash
# Check exporter endpoint
curl http://127.0.0.1:9092/metrics | grep quantum_rl | head -20

# Query Prometheus
curl -sS "http://127.0.0.1:9091/api/v1/query?query=quantum_rl_gate_pass_rate{symbol=\"BTCUSDT\"}"

# Get all RL metrics
curl -sS "http://127.0.0.1:9091/api/v1/label/__name__/values" | grep quantum_rl
```

### PromQL Examples (use in Grafana)
```promql
# Average pass rate across all symbols
avg(quantum_rl_gate_pass_rate)

# Top 5 symbols by pass rate
topk(5, quantum_rl_gate_pass_rate)

# Symbols with high cooldown blocking
quantum_rl_cooldown_blocking_rate > 0.3

# Policy age over time
quantum_rl_policy_age_seconds{symbol=~"BTCUSDT|ETHUSDT|SOLUSDT"}

# Ensemble confidence when RL passes vs fails
quantum_rl_ensemble_confidence_pass - quantum_rl_ensemble_confidence_fail
```

---

## Troubleshooting

### Dashboard Shows "No Data"
1. **Check metrics exporter is running**:
   ```bash
   systemctl status quantum-rl-shadow-metrics-exporter.service
   ```
   - Should be `active (running)`
   - If not: `systemctl restart quantum-rl-shadow-metrics-exporter.service`

2. **Verify metrics endpoint responds**:
   ```bash
   curl http://127.0.0.1:9092/metrics | grep quantum_rl_gate_pass_rate
   ```
   - Should return metrics with values
   - If "connection refused": Check service logs

3. **Check Prometheus is scraping**:
   ```bash
   curl -sS "http://127.0.0.1:9091/api/v1/query?query=up{job=\"quantum_rl_shadow\"}"
   ```
   - Should return `"value": [timestamp, "1"]`
   - If `0`: Check Prometheus config `/etc/prometheus/prometheus.yml`

4. **Reload Prometheus**:
   ```bash
   systemctl reload prometheus
   ```

5. **Restart Grafana**:
   ```bash
   systemctl restart grafana-server
   ```

### Metrics Are Stale
```bash
# Check exporter is updating
journalctl -u quantum-rl-shadow-metrics-exporter.service --since "5 minutes ago" | grep "Updated metrics"

# If no updates, check Redis stream
redis-cli XLEN quantum:stream:trade.intent
# Should return > 5000

# Restart exporter
systemctl restart quantum-rl-shadow-metrics-exporter.service
```

### Dashboard Panels Empty After Time Range Change
- Some panels filter by symbol (`BTCUSDT`, `ETHUSDT`, etc.)
- If those symbols haven't traded recently, panels will be empty
- Try: Change time range to "Last 24 hours" or remove symbol filter

---

## Configuration Files

### Metrics Exporter
- **Script**: `/home/qt/quantum_trader/microservices/ai_engine/rl_shadow_metrics_exporter.py`
- **Config**: `/etc/quantum/rl-shadow-metrics-exporter.env`
- **Service**: `/etc/systemd/system/quantum-rl-shadow-metrics-exporter.service`
- **Port**: 9092

### Prometheus
- **Config**: `/etc/prometheus/prometheus.yml`
- **Scrape job**: `quantum_rl_shadow`
- **Target**: `localhost:9092`
- **Interval**: 60s

### Grafana
- **Dashboard**: `/home/qt/quantum_trader/observability/grafana/dashboards/rl_shadow_performance.json`
- **UID**: `rl-shadow-performance`
- **URL**: https://app.quantumfond.com/grafana

---

## 24-48 Hour Monitoring

### What to Watch
1. **Pass Rate Stability**: Should stay > 12%
2. **Cooldown Impact**: Should stay < 25%
3. **Policy Freshness**: Should stay < 300s
4. **Confidence Patterns**: Look for anomalies
5. **Would Flip Rate**: Should stay < 25%

### Alert Conditions
- 🚨 Pass rate < 8% for > 15 minutes
- 🚨 Cooldown blocking > 30% sustained
- 🚨 Policy age > 500s (stale policies)
- 🚨 Would flip rate > 30% (too much disagreement)

### Decision Point (48h)
**IF stable** → Increase RL_INFLUENCE_WEIGHT to 0.10 (10% real influence)  
**IF unstable** → Investigate gate failures, optimize cooldown  
**IF high disagreement** → Analyze RL vs ensemble patterns

---

## Related Documentation
- **RL_SHADOW_OBSERVABILITY_COMPLETE.md** - Full deployment guide
- **RL_SHADOW_MONITORING_GUIDE.md** - Daily monitoring procedures
- **configs/grafana_rl_shadow_dashboard.json** - Dashboard JSON (local copy)

---

**Status**: 🎉 **MONITORING ACTIVE**  
**Next Review**: 2026-01-17T23:00:00Z
