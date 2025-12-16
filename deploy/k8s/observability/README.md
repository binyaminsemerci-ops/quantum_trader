# Observability Infrastructure — Kubernetes Integration

**EPIC-OBS-001 Phase 5** | Monitoring stack integration for Quantum Trader v2.0 microservices

This directory contains **infrastructure stubs** showing how Prometheus, Grafana, Loki, and Alertmanager integrate with Quantum Trader microservices.

---

## 📋 Overview

All Quantum Trader microservices follow the **Observability Contract** (see `docs/OBSERVABILITY_README.md`):

| Requirement | Endpoint | Purpose |
|-------------|----------|---------|
| **Metrics** | `GET /metrics` | Prometheus scraping |
| **Liveness** | `GET /health/live` | K8s liveness probe |
| **Readiness** | `GET /health/ready` | K8s readiness probe |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      QUANTUM TRADER MICROSERVICES                   │
│  ┌──────────┐  ┌───────────┐  ┌──────┐  ┌───────────┐  ┌────────┐ │
│  │AI Engine │  │ Execution │  │ Risk │  │ Portfolio │  │RL Train│ │
│  │  :8001   │  │   :8002   │  │ :8003│  │   :8004   │  │ :8005  │ │
│  └────┬─────┘  └─────┬─────┘  └───┬──┘  └─────┬─────┘  └────┬───┘ │
└───────┼──────────────┼─────────────┼───────────┼──────────────┼─────┘
        │              │             │           │              │
        │ /metrics     │ /metrics    │ /metrics  │ /metrics     │ /metrics
        │ (15s)        │ (15s)       │ (15s)     │ (15s)        │ (15s)
        │              │             │           │              │
        └──────────────┴─────────────┴───────────┴──────────────┘
                                     │
                                     ▼
                        ┌────────────────────────┐
                        │   PROMETHEUS           │
                        │   (Metrics Storage)    │
                        │   - 15s scrape         │
                        │   - 15d retention      │
                        └───┬──────────────┬─────┘
                            │              │
                            │ alerts       │ query
                            ▼              │
                    ┌──────────────┐       │
                    │Alertmanager  │       │
                    │- PagerDuty   │       │
                    │- Slack       │       │
                    │- Email       │       │
                    └──────────────┘       │
                                           │
        ┌──────────────────────────────────┴─────────────────┐
        │                                                      │
        │              JSON Logs (stdout)                     │
        │   ┌──────┬────────┬────────┬─────────┬─────────┐   │
        │   │      │        │        │         │         │   │
        │   ▼      ▼        ▼        ▼         ▼         ▼   │
        │  ┌─────────────────────────────────────────────┐   │
        │  │         LOKI (Log Aggregation)              │   │
        │  │         - JSON parsing                      │   │
        │  │         - 30d retention (prod)              │   │
        │  └─────────────────────────────────────────────┘   │
        │                          │                          │
        │                          │ LogQL queries            │
        │                          ▼                          │
        │              ┌─────────────────────┐                │
        │              │     GRAFANA         │                │
        │              │  - Dashboards       │                │
        │              │  - Alerts           │                │
        │              │  - Explore          │                │
        │              └─────────────────────┘                │
        └──────────────────────────────────────────────────────┘

Optional (EPIC-TRACE-001):
┌─────────────────────────────────────────────────────────┐
│         OpenTelemetry Traces (OTLP export)              │
│  Microservices → OTLP Collector → Jaeger/Tempo         │
│                                   ↓                     │
│                                Grafana                  │
└─────────────────────────────────────────────────────────┘
```

**Key Components:**

| Component | Purpose | Protocol | Port |
|-----------|---------|----------|------|
| **Prometheus** | Metrics storage + querying | HTTP (pull) | 9090 |
| **Loki** | Log aggregation | HTTP (push) | 3100 |
| **Grafana** | Visualization + dashboards | HTTP | 3000 |
| **Alertmanager** | Alert routing | HTTP | 9093 |
| **Jaeger** (optional) | Distributed tracing | OTLP | 14268 |

**Data Flow:**

1. **Metrics Path**: Service `/metrics` → Prometheus (scrape 15s) → Grafana (query) + Alertmanager (alerts)
2. **Logs Path**: Service JSON logs → stdout → Promtail → Loki → Grafana (LogQL)
3. **Traces Path** (optional): Service OTLP export → Collector → Jaeger → Grafana (TraceQL)

---

## 🎯 Prometheus Integration

### ServiceMonitor Pattern

Each microservice is scraped via a **ServiceMonitor** CRD (Prometheus Operator):

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: ai-engine
  namespace: quantum-trader
spec:
  selector:
    matchLabels:
      app: ai-engine
  endpoints:
  - port: http
    path: /metrics
    interval: 15s
```

**Examples in this directory:**
- `ai_engine_servicemonitor.yaml`
- `execution_servicemonitor.yaml`

### Required Labels

All metrics **MUST** include these labels (automatically added by observability module):

| Label | Example | Purpose |
|-------|---------|---------|
| `service` | `ai-engine` | Service identifier |
| `environment` | `production` | Deployment environment |
| `version` | `1.0.0` | Service version |
| `pod` | `ai-engine-7d4f...` | Pod name (auto-added by Prometheus) |
| `namespace` | `quantum-trader` | K8s namespace (auto-added) |

### Scraping Configuration

**Interval**: 15 seconds (configurable per service)  
**Timeout**: 10 seconds  
**Path**: `/metrics` (standard)  
**Port**: Service HTTP port (8001-8005)

---

## 🚨 Alertmanager Integration

Alert rules are defined in `alert_rules.yaml` and loaded into Prometheus.

### Critical Alerts

| Alert | Condition | Severity |
|-------|-----------|----------|
| **HighErrorRate** | HTTP 5xx errors > 5% for 5 minutes | Critical |
| **NoTrades** | Zero trades for 30 minutes (market hours) | Warning |
| **HighLatency** | P95 latency > 2s for 5 minutes | Critical |
| **ServiceDown** | No metrics received for 5 minutes | Critical |

**Template**: `alert_rules.yaml`

### Routing

Alerts are routed to:
- **Slack**: All warnings
- **PagerDuty**: Critical alerts
- **Email**: Non-urgent notifications

---

## 📊 Grafana Dashboards

### Pre-built Dashboards

1. **Service Overview** (`grafana_dashboard_template.json`)
   - HTTP request rate
   - Error rate (by service)
   - Latency percentiles (P50, P95, P99)
   - Signals processed

2. **Trading Activity** (to be created)
   - Trades executed (by symbol)
   - Positions open
   - PnL tracking
   - Order flow

3. **System Health** (to be created)
   - All services status
   - Dependency health (Redis, DB, EventBus)
   - ESS activations

### Dashboard Variables

- `$service`: Service name filter
- `$environment`: Environment filter (prod/staging/dev)
- `$time_range`: Time range selector

---

## 📝 Loki Logging Integration

### Log Labels

All microservices emit **JSON logs** with these labels (automatically added by observability module):

| Label | Example | Purpose |
|-------|---------|---------|
| `service_name` | `ai-engine` | Service identifier |
| `service_version` | `1.0.0` | Deployment version |
| `environment` | `production` | Environment (prod/staging/dev) |
| `correlation_id` | `abc-123-def-456` | Request trace ID |
| `levelname` | `INFO` / `ERROR` / `WARNING` | Log level |

**Example log entry:**

```json
{
  "service_name": "ai-engine",
  "service_version": "1.0.0",
  "environment": "production",
  "timestamp": "2025-12-04T12:34:56.789Z",
  "levelname": "INFO",
  "message": "Signal generated",
  "correlation_id": "abc-123-def-456",
  "signal_type": "LONG",
  "symbol": "BTCUSDT",
  "confidence": 0.87
}
```

### Promtail Configuration (Minimal Example)

**Required configuration** for scraping logs from Quantum Trader pods:

```yaml
# promtail-config.yaml (minimal stub)
scrape_configs:
- job_name: quantum-trader
  kubernetes_sd_configs:
  - role: pod
    namespaces:
      names: [quantum-trader]
  
  # Parse JSON logs
  pipeline_stages:
  - json:
      expressions:
        service: service_name
        level: levelname
        msg: message
        correlation_id: correlation_id
        environment: environment
  
  # Add Kubernetes metadata as labels
  relabel_configs:
  - source_labels: [__meta_kubernetes_pod_label_app]
    target_label: service
  - source_labels: [__meta_kubernetes_namespace]
    target_label: namespace
  - source_labels: [__meta_kubernetes_pod_name]
    target_label: pod
```

**Note**: This is a stub configuration. Full Promtail deployment with DaemonSet is outside Phase 5 scope.

### LogQL Query Examples

**Essential queries** for Grafana Explore:

```promql
# All logs from AI Engine (last hour)
{service="ai-engine"} | json

# All errors across services
{service=~".*"} |= "ERROR" | json | levelname="ERROR"

# Trades executed for specific symbol
{service="execution"} | json | message=~".*executed.*" | symbol="BTCUSDT"

# High-confidence signals only
{service="ai-engine"} | json | confidence > 0.8

# Trace all logs for a specific request
{correlation_id="abc-123-def-456"} | json

# Error rate by service (last 5m)
sum by (service) (rate({service=~".*"} |= "ERROR" [5m]))
```

### Log Retention

| Environment | Retention | Reason |
|-------------|-----------|--------|
| **Production** | 30 days | Compliance + debugging |
| **Staging** | 7 days | Testing validation |
| **Development** | 3 days | Local troubleshooting |

### Integration with Grafana

**Data source setup:**

1. Add Loki data source in Grafana
2. URL: `http://loki:3100`
3. Enable "Derived fields" for correlation IDs:
   - **Regex**: `correlation_id":"([^"]+)`
   - **URL**: `/explore?queries=[{"datasource":"tempo","query":"${__value.raw}"}]`
   - Links logs → traces automatically

---

## 🔧 Deployment Instructions

### Prerequisites

1. **Prometheus Operator** installed in cluster
2. **Grafana** deployed with Prometheus data source
3. **Loki** stack deployed (optional)
4. **Alertmanager** configured with routing

### Deploy ServiceMonitors

```bash
# Apply ServiceMonitors for each service
kubectl apply -f ai_engine_servicemonitor.yaml
kubectl apply -f execution_servicemonitor.yaml

# Verify Prometheus targets
kubectl port-forward -n monitoring svc/prometheus 9090:9090
# Open http://localhost:9090/targets
```

### Load Alert Rules

```bash
# Create PrometheusRule resource
kubectl apply -f alert_rules.yaml

# Verify rules loaded
kubectl port-forward -n monitoring svc/prometheus 9090:9090
# Open http://localhost:9090/alerts
```

### Import Grafana Dashboards

```bash
# Import dashboard JSON via Grafana UI
# Or use Grafana provisioning:
kubectl create configmap grafana-dashboard-quantum-trader \
  --from-file=grafana_dashboard_template.json \
  -n monitoring
```

---

## 🧪 Testing

### Verify Metrics Scraping

```bash
# Check if Prometheus is scraping metrics
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.labels.service=="ai-engine")'

# Query metrics
curl 'http://localhost:9090/api/v1/query?query=http_requests_total{service="ai-engine"}'
```

### Verify Logs in Loki

```bash
# Query logs via LogCLI
logcli query '{service="ai-engine"}' --limit=10

# Or via Grafana Explore:
# Data source: Loki
# Query: {service="ai-engine"} | json
```

### Test Alerts

```bash
# Trigger alert (generate errors)
ab -n 1000 -c 10 http://ai-engine:8001/nonexistent

# Check Alertmanager
curl http://localhost:9093/api/v2/alerts
```

---

## 📚 Related Documentation

- **Observability Contract**: `docs/OBSERVABILITY_README.md`
- **Service Deployment**: `deploy/k8s/services/`
- **Metrics Catalog**: `infra/metrics/metrics.py`
- **Infrastructure Setup**: This directory

---

## 🔄 Next Steps

1. **Customize alert thresholds** in `alert_rules.yaml` based on production traffic
2. **Add more dashboards** for trading-specific metrics
3. **Configure Alertmanager routing** to Slack/PagerDuty
4. **Enable Loki** for centralized logging
5. **Set up tracing** with Jaeger/Tempo (optional)

---

**Updated**: December 4, 2025 | **EPIC-OBS-001 Phase 5** | Quantum Trader v2.0
