# P3 Apply Layer - Prometheus Integration

## Metrics Endpoint

The apply layer exposes Prometheus metrics on port **8043**.

**URL**: `http://localhost:8043/metrics`

## Available Metrics

### 1. quantum_apply_plan_total
**Type**: Counter  
**Labels**: symbol, decision  
**Description**: Total apply plans created

Example:
```
quantum_apply_plan_total{symbol="BTCUSDT",decision="EXECUTE"} 42
quantum_apply_plan_total{symbol="BTCUSDT",decision="SKIP"} 15
quantum_apply_plan_total{symbol="ETHUSDT",decision="SKIP"} 58
quantum_apply_plan_total{symbol="BTCUSDT",decision="BLOCKED"} 3
```

**Use cases**:
- Track how many plans created per symbol
- Monitor decision distribution (EXECUTE vs SKIP vs BLOCKED)
- Alert on excessive BLOCKED decisions

---

### 2. quantum_apply_execute_total
**Type**: Counter  
**Labels**: symbol, step, status  
**Description**: Total execution attempts per step

Example:
```
quantum_apply_execute_total{symbol="BTCUSDT",step="CLOSE_PARTIAL_75",status="success"} 12
quantum_apply_execute_total{symbol="BTCUSDT",step="CLOSE_FULL",status="success"} 5
quantum_apply_execute_total{symbol="BTCUSDT",step="UPDATE_SL",status="failed"} 2
```

**Use cases**:
- Track execution success rate
- Monitor which steps are most common
- Alert on failed executions

---

### 3. quantum_apply_dedupe_hits_total
**Type**: Counter  
**Description**: Total duplicate plan detections (idempotency working)

Example:
```
quantum_apply_dedupe_hits_total 87
```

**Use cases**:
- Verify idempotency mechanism active
- High value indicates proposals unchanged (normal in stable conditions)
- Low value indicates proposals frequently changing

---

### 4. quantum_apply_last_success_epoch
**Type**: Gauge  
**Labels**: symbol  
**Description**: Timestamp of last successful execution per symbol

Example:
```
quantum_apply_last_success_epoch{symbol="BTCUSDT"} 1769130500
```

**Use cases**:
- Monitor execution freshness
- Alert if no executions for extended period
- Track last activity per symbol

---

## Prometheus Scrape Configuration

Add this job to your Prometheus configuration:

### Option A: Append to existing prometheus.yml

```yaml
  - job_name: 'quantum-apply-layer'
    static_configs:
      - targets: ['localhost:8043']
        labels:
          service: 'apply-layer'
          component: 'quantum-trader'
          environment: 'production'
    scrape_interval: 10s
    scrape_timeout: 5s
```

### Option B: Use file_sd_configs (recommended for production)

Create `/etc/prometheus/targets/quantum-apply-layer.json`:
```json
[
  {
    "targets": ["localhost:8043"],
    "labels": {
      "service": "apply-layer",
      "component": "quantum-trader",
      "environment": "production"
    }
  }
]
```

Then in prometheus.yml:
```yaml
  - job_name: 'quantum-apply-layer'
    file_sd_configs:
      - files:
          - '/etc/prometheus/targets/quantum-apply-layer.json'
        refresh_interval: 30s
    scrape_interval: 10s
```

### Reload Prometheus

```bash
# Check config valid
promtool check config /etc/prometheus/prometheus.yml

# Reload (no restart needed)
curl -X POST http://localhost:9091/-/reload

# Or restart if reload not enabled
sudo systemctl restart prometheus
```

---

## Verification

### 1. Check metrics endpoint directly
```bash
curl http://localhost:8043/metrics | grep quantum_apply
```

### 2. Check Prometheus targets
Open Prometheus UI: `http://localhost:9091/targets`

Look for `quantum-apply-layer` job - should show **UP** status.

### 3. Query metrics in Prometheus
```promql
# Total plans by symbol
sum by (symbol) (quantum_apply_plan_total)

# Execution success rate
rate(quantum_apply_execute_total{status="success"}[5m])

# Time since last execution
time() - quantum_apply_last_success_epoch

# Dedupe hit rate
rate(quantum_apply_dedupe_hits_total[5m])
```

---

## Grafana Dashboard (Optional)

### Panel 1: Plans Created (Time Series)
**Query**:
```promql
sum by (symbol, decision) (rate(quantum_apply_plan_total[5m]))
```

**Legend**: `{{symbol}} - {{decision}}`

---

### Panel 2: Execution Success Rate (Gauge)
**Query**:
```promql
sum(rate(quantum_apply_execute_total{status="success"}[5m])) /
sum(rate(quantum_apply_execute_total[5m])) * 100
```

**Unit**: Percent (0-100)

---

### Panel 3: Last Execution Timestamp (Table)
**Query**:
```promql
quantum_apply_last_success_epoch * 1000
```

**Format**: Table, time column format

---

### Panel 4: Dedupe Hits (Counter)
**Query**:
```promql
quantum_apply_dedupe_hits_total
```

---

## Alert Rules (Recommended)

Create `/etc/prometheus/rules/quantum-apply-layer.yml`:

```yaml
groups:
  - name: quantum_apply_layer
    interval: 30s
    rules:
      # Alert if no executions for 1 hour (testnet mode only)
      - alert: QuantumApplyNoExecutions
        expr: |
          (time() - quantum_apply_last_success_epoch > 3600)
          and on() (quantum_apply_plan_total{decision="EXECUTE"} > 0)
        for: 5m
        labels:
          severity: warning
          component: apply-layer
        annotations:
          summary: "No apply layer executions for {{ $labels.symbol }} in 1 hour"
          description: "Last execution: {{ $value | humanizeDuration }} ago"
      
      # Alert on high execution failure rate
      - alert: QuantumApplyHighFailureRate
        expr: |
          (
            sum by (symbol) (rate(quantum_apply_execute_total{status="failed"}[5m]))
            /
            sum by (symbol) (rate(quantum_apply_execute_total[5m]))
          ) > 0.1
        for: 5m
        labels:
          severity: critical
          component: apply-layer
        annotations:
          summary: "High execution failure rate for {{ $labels.symbol }}"
          description: "{{ $value | humanizePercentage }} of executions failing"
      
      # Alert if service metrics endpoint down
      - alert: QuantumApplyMetricsDown
        expr: up{job="quantum-apply-layer"} == 0
        for: 2m
        labels:
          severity: critical
          component: apply-layer
        annotations:
          summary: "Apply layer metrics endpoint unreachable"
          description: "Check if quantum-apply-layer service is running"
      
      # Alert on excessive BLOCKED decisions
      - alert: QuantumApplyExcessiveBlocks
        expr: |
          (
            sum by (symbol) (increase(quantum_apply_plan_total{decision="BLOCKED"}[10m]))
          ) > 10
        for: 5m
        labels:
          severity: warning
          component: apply-layer
        annotations:
          summary: "Excessive BLOCKED decisions for {{ $labels.symbol }}"
          description: "{{ $value }} plans blocked in 10 minutes (check kill_score or kill_switch)"
```

Load rules:
```bash
# Add to prometheus.yml
rule_files:
  - '/etc/prometheus/rules/quantum-apply-layer.yml'

# Reload
curl -X POST http://localhost:9091/-/reload
```

---

## Troubleshooting

### Metrics endpoint not accessible
```bash
# Check service running
systemctl status quantum-apply-layer

# Check port listening
netstat -tlnp | grep 8043

# Check prometheus_client installed
python3 -c "import prometheus_client; print('OK')"
```

### Prometheus not scraping
```bash
# Check targets in Prometheus UI
# Look for quantum-apply-layer job

# Check Prometheus logs
journalctl -u prometheus | grep apply-layer

# Test scrape manually
curl http://localhost:8043/metrics
```

### Metrics not updating
```bash
# Check service logs
journalctl -u quantum-apply-layer -f

# Verify plans being created
redis-cli XLEN quantum:stream:apply.plan

# Check for errors
journalctl -u quantum-apply-layer | grep ERROR
```
