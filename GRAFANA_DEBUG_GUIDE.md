# Grafana Dashboard Debug Guide

## ✅ New Working Dashboard Created
**URL:** https://quantumfond.com/grafana/d/d31466b8-8cd2-44f9-a02b-82a08de7c9d8/quantum-metrics-test-working

**Panels (12):**
1. RL Shadow - Pass Rate % (avg across all symbols)
2. RL Shadow - Total Intents Analyzed
3. RL Shadow - Eligible Rate %
4. Safety - Safe Mode Status (0=OFF green, 1=ON red)
5. Safety - TTL Seconds
6. Redis - Clients Connected
7. Redis - Memory Used (MB)
8. Harvest - Kill Score by Symbol (Top 10)
9. Safety - Faults Last Hour
10. Safety - Trade Intent Rate (per min)
11. Node Exporter - CPU Usage %
12. Quantum Services - Active Count

---

## 🔧 Debug Existing Panel Errors

### Common Issues:

**1) "An unexpected error happened"**
- **Cause:** Query syntax error or metric doesn't exist
- **Fix:** 
  ```
  1. Click panel title → Edit
  2. Check Query tab
  3. Verify metric name exists in Prometheus
  4. Test in Explore: curl http://localhost:9091/api/v1/query?query=<metric_name>
  ```

**2) "No data"**
- **Cause:** Time range too short, or metric has no values yet
- **Fix:**
  ```
  1. Expand time range to Last 24 hours
  2. Check if metric exists: curl http://localhost:9091/api/v1/label/__name__/values | grep <metric>
  3. Query returns empty result? Service may not be emitting that metric
  ```

**3) Panel shows 0 or -1**
- **Cause:** Metric exists but value is actually 0 (normal in shadow mode)
- **Fix:** This is expected behavior, not an error

---

## 📊 Verified Working Metrics

### RL Shadow (port 9092):
```promql
quantum_rl_gate_pass_rate              # Per-symbol pass rate (0-1)
quantum_rl_intents_analyzed_total      # Total intents processed
quantum_rl_eligible_rate               # Eligible rate (0-1)
quantum_rl_would_flip_rate             # Would flip rate (0-1)
quantum_rl_cooldown_blocking_rate      # Cooldown blocking rate
```

### Safety Telemetry (port 9105):
```promql
quantum_safety_safe_mode               # 0=OFF, 1=ON
quantum_safety_safe_mode_ttl_seconds   # TTL in seconds (-1 = disabled)
quantum_safety_faults_last_1h          # Faults in last hour
quantum_safety_fault_stream_length     # Current fault stream length
quantum_trade_intent_rate_per_min      # Intents per minute
quantum_trade_intent_stream_length     # Stream length
```

### Redis (port 9121):
```promql
redis_connected_clients                # Connected clients
redis_memory_used_bytes                # Memory usage in bytes
redis_uptime_in_seconds                # Redis uptime
```

### Harvest Exporter (port 8042):
```promql
quantum_harvest_kill_score{symbol="X"} # Kill score per symbol
quantum_harvest_action{symbol="X"}     # Action per symbol
quantum_harvest_k_regime_flip          # Regime flip K factor
```

### Node Exporter (port 9100):
```promql
node_cpu_seconds_total                 # CPU time
node_memory_MemTotal_bytes             # Total memory
node_systemd_unit_state{name=~"quantum.*"} # Systemd unit states
```

---

## 🛠️ Quick Fixes for Existing Dashboards

### Fix "Redis Connectivity" Panel:
**Old query (broken):**
```promql
redis_up{job="redis_exporter"}
```
**New query (working):**
```promql
redis_uptime_in_seconds > 0
```

### Fix "SAFE MODE Status" Panel:
**Old query (broken):**
```promql
quantum_safety_mode_status
```
**New query (working):**
```promql
quantum_safety_safe_mode
```

### Fix "RL Shadow Pass Rate" Panel:
**If showing "No data":**
```promql
# Old (per-symbol):
quantum_rl_gate_pass_rate{symbol="BTCUSDT"}

# New (avg all symbols):
avg(quantum_rl_gate_pass_rate) * 100
```

---

## 🔍 Testing Queries

**Via SSH:**
```bash
# Test single metric
curl -s "http://localhost:9091/api/v1/query?query=quantum_safety_safe_mode"

# List all quantum metrics
curl -s http://localhost:9091/api/v1/label/__name__/values | grep quantum

# Check specific target status
curl -s "http://localhost:9091/api/v1/query?query=up{job=\"quantum_rl_shadow\"}"
```

**Via Grafana Explore:**
1. Go to Explore tab
2. Select Prometheus datasource
3. Enter query: `quantum_safety_safe_mode`
4. Run query
5. See instant result

---

## 📈 Dashboard Best Practices

1. **Use rate() for counters:**
   ```promql
   rate(quantum_rl_intents_analyzed_total[5m])
   ```

2. **Multiply by 100 for percentages:**
   ```promql
   avg(quantum_rl_gate_pass_rate) * 100
   ```

3. **Filter by label:**
   ```promql
   quantum_harvest_kill_score{symbol=~"BTC.*|ETH.*"}
   ```

4. **Top N values:**
   ```promql
   topk(10, quantum_harvest_kill_score)
   ```

5. **Service health check:**
   ```promql
   count(node_systemd_unit_state{name=~"quantum.*", state="active"})
   ```

---

## 🎯 Next Steps

1. Open new test dashboard: https://quantumfond.com/grafana/d/d31466b8-8cd2-44f9-a02b-82a08de7c9d8
2. Verify all panels show data (even if 0)
3. Use Explore to test queries before adding to dashboards
4. Fix existing dashboards by comparing working queries from test dashboard
5. Set time range to "Last 24 hours" for better data visibility

---

## ✅ Verification Commands

```bash
# Quick health check
/home/qt/quantum_trader/deploy/verify_observability.sh

# Check Prometheus targets
curl -s http://localhost:9091/api/v1/targets | jq '.data.activeTargets[] | {job: .labels.job, health: .health}'

# List all available metrics
curl -s http://localhost:9091/api/v1/label/__name__/values | jq '.data | sort'
```
