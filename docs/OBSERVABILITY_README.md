# Observability v2 — Quantum Trader Microservices

**EPIC-OBS-001** | Production-ready observability for all microservices

## Overview

The observability layer provides **structured logging**, **Prometheus metrics**, and **distributed tracing** for Quantum Trader v2.0 microservices. All services must implement standardized health probes and metrics endpoints for Kubernetes deployment and Grafana monitoring. This ensures consistent operational visibility across the distributed trading system.

**✅ ALL MICROSERVICES NOW HAVE TRACING ENABLED** (as of 2026-01-11)

### Services with Tracing

All 9 microservices now have distributed tracing enabled:

1. ✅ **ai-engine** (port 8001) - AI model inference & signal generation
2. ✅ **execution** (port 8002) - Order execution & position monitoring
3. ✅ **risk-safety** (port 8003) - Emergency Stop System & risk limits
4. ✅ **portfolio-intelligence** (port 8004) - Portfolio state & PnL tracking
5. ✅ **rl-training** (port 8005) - RL training & continuous learning
6. ✅ **trading-bot** (port 8006) - Autonomous trading signal generation
7. ✅ **clm** (standalone) - Continuous Learning Manager
8. ✅ **eventbus-bridge** (standalone) - Event bus to Redis bridge
9. ✅ **position-monitor** (standalone) - Position monitoring daemon

---

## Required Contract for Every Microservice

Every service **MUST** implement:

- **MUST expose**: `GET /health/live` (liveness probe, always 200 if process alive)
- **MUST expose**: `GET /health/ready` (readiness probe, 200 if deps healthy / 503 if not)
- **MUST expose**: `GET /metrics` (Prometheus text format)
- **MUST call**: `init_observability(service_name)` at startup
- **MUST use**: Service-aware logging (`service_name`, `environment`, `version` in all logs)
- **SHOULD support**: Distributed tracing (graceful fallback if OpenTelemetry not installed)

---

## Quickstart for New Microservices

**3-step integration** (copy-paste ready):

```python
# 1. Import observability module
from backend.infra.observability import (
    init_observability,
    get_logger,
    ObservableService,
)
from prometheus_client import REGISTRY, generate_latest, CONTENT_TYPE_LATEST
from fastapi import FastAPI, Response

# 2. Initialize observability at module level
init_observability(service_name="my-service", log_level="INFO")
logger = get_logger(__name__)

```python
from backend.infra.observability import init_observability, get_logger, ObservableService
from prometheus_client import REGISTRY, generate_latest, CONTENT_TYPE_LATEST
from fastapi import FastAPI, Response

# Step 1: Initialize observability
init_observability(service_name="my-service", log_level="INFO")
logger = get_logger(__name__)

# Step 2: Create and instrument FastAPI app
app = FastAPI(title="My Service")
ObservableService.instrument(app, service_name="my-service")

# Step 3: Add required endpoints
@app.get("/health/live")
async def live(): return {"status": "ok"}

@app.get("/health/ready")
async def ready(): return {"status": "ready"}  # Add dependency checks here

@app.get("/metrics")
async def metrics(): return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
```

**Done!** Service now has JSON logs, Prometheus metrics, tracing, and K8s health probes.ter | symbol, side, status | Trades executed |
---

## Metrics Provided

All metrics auto-include `service`, `environment`, `version` labels.

| Metric | Type | Description |
|--------|------|-------------|
| `http_requests_total` | Counter | Requests by method/endpoint/status |
| `http_request_duration_seconds` | Histogram | P50/P95/P99 latency |
| `http_requests_in_progress` | Gauge | Concurrent requests |
| `signals_generated_total` | Counter | AI Engine signal volume |
| `trades_executed_total` | Counter | Orders submitted |
| `positions_open` | Gauge | Open positions by symbol |
| `errors_total` | Counter | Application exceptions |
| `emergency_stop_triggered_total` | Counter | ESS activations |

**EPIC-STRESS-DASH-001: Risk & Resilience Dashboard Metrics**

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `risk_gate_decisions_total` | Counter | decision, reason, account, exchange, strategy | RiskGate v3 decisions (allow/block/scale) |
| `ess_triggers_total` | Counter | source | ESS activation by trigger source (global_risk, manual, exchange_outage, drawdown) |
| `exchange_failover_events_total` | Counter | primary, selected | Exchange failover events when primary unhealthy |
| `stress_scenario_runs_total` | Counter | scenario, success | Stress test scenario executions (true/false) |

**Purpose**: Monitor when RiskGate blocks/scales trades, see when ESS triggers (critical safety), track which exchanges fail over most frequently, and track stress scenario execution health. These metrics power the Risk & Resilience Dashboard for real-time visibility into system hardening and failure scenarios.

**Dashboard**: `deploy/k8s/observability/grafana_risk_resilience_dashboard.json`

Full catalog: `infra/metrics/metrics.py`
         │                          │
    [/metrics]                [JSON Logs]
---

## Logging Format

**JSON-based** structured logs compatible with Loki ingestion.

Every log entry includes:
- `timestamp` — ISO 8601 format
- `levelname` — INFO / WARNING / ERROR
- `message` — Log message
- `service_name` — Service identifier
- `service_version` — Deployment version
- `environment` — prod / staging / dev
- `correlation_id` — Request trace ID (when available)

---

## Tracing (Optional)

Distributed tracing enabled when OpenTelemetry is installed:
- `init_tracing()` integrated into `init_observability()`
- Exports traces via OTLP to Jaeger/Tempo
- Gracefully disables if `OTLP_ENDPOINT` not configured
- No crashes if OpenTelemetry packages missing

---

## Architecture

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `LOG_LEVEL` | Logging level (DEBUG / INFO / WARNING / ERROR) |
| `SERVICE_NAME` | Service identifier for metrics/logs |
| `SERVICE_VERSION` | Deployment version (defaults to 1.0.0) |
| `ENVIRONMENT` | Deployment env (dev / staging / prod) |
| `OTLP_ENDPOINT` | Optional tracing endpoint (systemd: `http://localhost:4317`, docker: `http://jaeger:4317`) |
| `ENABLE_TRACING` | Enable distributed tracing (default: true) |
---

## How to Validate

Test all required endpoints:

```bash
# Liveness probe (should always return 200)
curl http://localhost:8001/health/live

# Readiness probe (200 if deps healthy, 503 if not)
curl http://localhost:8001/health/ready

# Prometheus metrics (text/plain format)
curl http://localhost:8001/metrics

# Check logs are JSON formatted
# Look for: {"service_name": "...", "timestamp": "...", "message": "..."}
```

---

## Extending Observability

**DO:**
- Add custom metrics specific to your service domain
- Use correct label keys (`service`, `environment`, `version`)
- Keep readiness checks under 2s timeout
- Log structured data with `extra={}` parameter

**DON'T:**
- Override the global Prometheus registry
- Rename default service labels
- Block readiness probe with heavy checks (>2s)
- Mix logging frameworks (stick to observability module)

---

## Compliance Checklist

Before deploying a new service, verify:

- [ ] `init_observability(service_name, log_level)` called at startup
- [ ] `GET /health/live` endpoint implemented
- [ ] `GET /health/ready` endpoint implemented (with real dependency checks)
- [ ] `GET /metrics` endpoint implemented
- [ ] `ObservableService.instrument(app, service_name)` called
- [ ] Service name matches across all config (SERVICE_NAME env var)
- [ ] JSON logs include `service_name`, `timestamp`, `levelname`, `message`
- [ ] Metrics endpoint returns Prometheus text format
- [ ] Service starts without errors if OpenTelemetry not installed

---

## Related Documentation

- **Metrics Catalog**: `backend/infra/metrics/metrics.py`
- **Observability Contract**: `backend/infra/observability/contract.py`
- **K8s Integration**: `deploy/k8s/observability/README.md`
- **Test Suite**: `tests/observability/`

---

**Updated**: December 4, 2025 | **EPIC-OBS-001 Phase 7** | Quantum Trader v2.0