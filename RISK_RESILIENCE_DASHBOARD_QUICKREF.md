# Risk & Resilience Dashboard - Quick Reference

**EPIC-STRESS-DASH-001** | Dashboard Integration for Stress Testing & Risk Enforcement

---

## 📊 4 New Metrics

| Metric | What It Shows | PromQL Example |
|--------|---------------|----------------|
| `risk_gate_decisions_total` | RiskGate allow/block/scale decisions | `sum by (decision) (risk_gate_decisions_total)` |
| `ess_triggers_total` | ESS activation events (critical safety) | `sum by (source) (rate(ess_triggers_total[5m]))` |
| `exchange_failover_events_total` | Exchange failover when primary unhealthy | `sum by (primary, selected) (exchange_failover_events_total)` |
| `stress_scenario_runs_total` | Stress test execution success/failure | `sum by (scenario, success) (stress_scenario_runs_total)` |

---

## 🎯 Use Cases

### Monitor Risk Enforcement
```promql
# Top 5 block reasons
topk(5, sum by (reason) (risk_gate_decisions_total{decision="block"}))

# Block rate by exchange
100 * sum by (exchange) (risk_gate_decisions_total{decision="block"}) 
  / sum by (exchange) (risk_gate_decisions_total)
```

### Detect ESS Triggers (Critical Safety)
```promql
# ESS triggers in last hour
increase(ess_triggers_total[1h])

# ESS trigger rate spike (alert threshold)
rate(ess_triggers_total[5m]) > 0.05  # > 3 per hour
```

### Track Exchange Reliability
```promql
# Failover frequency by primary exchange
sum by (primary) (rate(exchange_failover_events_total[5m]))

# Most common failover paths
topk(3, sum by (primary, selected) (exchange_failover_events_total))
```

### Validate Stress Testing
```promql
# Stress scenario success rate
100 * sum by (scenario) (stress_scenario_runs_total{success="true"})
  / sum by (scenario) (stress_scenario_runs_total)

# Failed scenarios (good = catching issues)
sum by (scenario) (stress_scenario_runs_total{success="false"})
```

---

## 📍 Where Metrics Are Recorded

```python
# RiskGate v3 (backend/risk/risk_gate_v3.py)
def _record_decision_metric(result, account, exchange, strategy):
    risk_gate_decisions_total.labels(
        decision=result.decision,  # allow | block | scale_down
        reason=result.reason,      # ess_halt | leverage_exceeded | ...
        account=account,
        exchange=exchange,
        strategy=strategy,
    ).inc()

# ESS (backend/services/risk/emergency_stop_system.py)
async def activate(reason):
    source = classify_source(reason)  # global_risk | manual | exchange_outage | drawdown
    ess_triggers_total.labels(source=source).inc()

# Exchange Failover (backend/policies/exchange_failover_policy.py)
async def choose_exchange_with_failover(primary, default):
    if selected != primary:
        exchange_failover_events_total.labels(
            primary=primary,
            selected=selected,
        ).inc()

# Stress Scenarios (backend/tests_runtime/stress_scenarios/runner.py)
async def run_scenario(name):
    result = await scenario_fn()
    stress_scenario_runs_total.labels(
        scenario=result.name,
        success=str(result.success).lower(),  # "true" | "false"
    ).inc()
```

---

## 🚀 Quick Start

### 1. Import Dashboard to Grafana
```bash
# Upload this file to Grafana UI:
deploy/k8s/observability/grafana_risk_resilience_dashboard.json
```

### 2. Generate Metrics (Run Stress Scenarios)
```bash
# Via Python
python -c "
import asyncio
from backend.tests_runtime.stress_scenarios.runner import run_all_scenarios, print_results_summary

results = asyncio.run(run_all_scenarios())
print_results_summary(results)
"
```

### 3. Query Metrics Endpoint
```bash
curl http://localhost:8000/metrics | grep -E "risk_gate|ess_triggers|exchange_failover|stress_scenario"
```

---

## 🎨 Grafana Dashboard Panels

1. **RiskGate Decisions (Timeseries)** - `sum by (decision) (risk_gate_decisions_total)`
2. **Top Block Reasons (Bar Gauge)** - `topk(5, sum by (reason) (risk_gate_decisions_total))`
3. **ESS Triggers Rate (Timeseries)** - `sum by (source) (rate(ess_triggers_total[5m]))`
4. **ESS Total Count (Stat)** - `sum(ess_triggers_total)`
5. **Exchange Failovers (Table)** - `sum by (primary, selected) (exchange_failover_events_total)`
6. **Failover Rate (Timeseries)** - `sum by (primary) (rate(exchange_failover_events_total[5m]))`
7. **Stress Scenarios (Timeseries)** - `sum by (scenario, success) (stress_scenario_runs_total)`
8. **Success Rate % (Bar Gauge)** - `100 * sum by (scenario) (...{success="true"}) / sum by (scenario) (...)`

---

## 🔥 Alert Examples (Prometheus)

```yaml
# High ESS trigger rate (critical safety issue)
- alert: ESSTriggersHigh
  expr: rate(ess_triggers_total[5m]) > 0.05  # > 3 per hour
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "ESS triggering frequently ({{ $value }} triggers/min)"

# Exchange failover spike (infrastructure degradation)
- alert: ExchangeFailoverSpike
  expr: rate(exchange_failover_events_total[5m]) > 0.1  # > 6 per hour
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "Exchange failovers spiking ({{ $value }} failovers/min)"

# High RiskGate block rate (too conservative or systemic risk)
- alert: RiskGateBlockingMost
  expr: |
    100 * sum(risk_gate_decisions_total{decision="block"}) 
      / sum(risk_gate_decisions_total) > 70
  for: 15m
  labels:
    severity: warning
  annotations:
    summary: "RiskGate blocking {{ $value }}% of orders"

# Stress scenario regression (resilience degrading)
- alert: StressScenarioFailureHigh
  expr: |
    100 * sum by (scenario) (stress_scenario_runs_total{success="false"})
      / sum by (scenario) (stress_scenario_runs_total) > 30
  for: 1h
  labels:
    severity: warning
  annotations:
    summary: "Stress scenario {{ $labels.scenario }} failing {{ $value }}%"
```

---

## 📚 Related Docs

- **Full Summary**: `EPIC_STRESS_DASH_001_SUMMARY.md`
- **Observability Guide**: `docs/OBSERVABILITY_README.md`
- **Metrics Catalog**: `infra/metrics/metrics.py`
- **RiskGate Integration**: `EPIC_RISK3_EXEC_001_SUMMARY.md`
- **Stress Scenarios**: `backend/tests_runtime/stress_scenarios/`

---

**Updated**: December 4, 2025 | EPIC-STRESS-DASH-001 Complete ✅
