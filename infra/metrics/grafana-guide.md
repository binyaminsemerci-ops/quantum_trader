# Grafana Dashboard Integration Guide
# SPRINT 3 - Module F: Metrics + Grafana

**Status**: Skeleton / Configuration Template  
**Priority**: P2 (implement after Sprint 3 core features)

---

## 🎯 OVERVIEW

Grafana provides real-time visualization of Prometheus metrics from all microservices.

---

## 📊 METRICS COLLECTED

### **1. HTTP Metrics**
- `http_requests_total` - Total requests by method, endpoint, status
- `http_request_duration_seconds` - Request latency histogram
- `http_requests_in_progress` - Concurrent requests gauge

### **2. EventBus Metrics**
- `eventbus_messages_published_total` - Messages published by event type
- `eventbus_messages_consumed_total` - Messages consumed by event type
- `eventbus_message_processing_duration_seconds` - Processing time histogram
- `eventbus_queue_lag` - Queue backlog gauge

### **3. Trading Metrics**
- `trades_executed_total` - Trades by symbol, side, status
- `trade_execution_duration_seconds` - Trade latency
- `positions_open` - Open positions by symbol
- `portfolio_value_usd` - Total portfolio value
- `signals_generated_total` - Signals by type and model

### **4. Risk Metrics**
- `risk_checks_total` - Risk checks by type and result
- `emergency_stop_triggered_total` - ESS activations
- `policy_violations_total` - Policy violations by severity

### **5. Infrastructure Metrics**
- `redis_operations_total` - Redis ops by operation and status
- `redis_operation_duration_seconds` - Redis latency
- `database_operations_total` - Database ops by operation and status
- `service_health_status` - Service health by component
- `dependency_health_status` - Dependency health

---

## 🏗️ ARCHITECTURE

```
┌───────────────┐
│ Microservices │ → /metrics endpoint (Prometheus format)
└───────┬───────┘
        │
┌───────▼────────┐
│  Prometheus    │ ← Scrapes metrics every 15s
│  (port 9090)   │
└───────┬────────┘
        │
┌───────▼────────┐
│   Grafana      │ ← Visualizes metrics
│  (port 3000)   │
└────────────────┘
```

---

## 📁 FILES

### **Prometheus Configuration**

```yaml
# infra/metrics/prometheus.yml

global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'ai-engine'
    static_configs:
      - targets: ['ai-engine-service:8001']
    metrics_path: /metrics
  
  - job_name: 'execution'
    static_configs:
      - targets: ['execution-service:8002']
    metrics_path: /metrics
  
  - job_name: 'risk-safety'
    static_configs:
      - targets: ['risk-safety-service:8003']
    metrics_path: /metrics
  
  - job_name: 'portfolio-intelligence'
    static_configs:
      - targets: ['portfolio-intelligence-service:8004']
    metrics_path: /metrics
  
  - job_name: 'monitoring-health'
    static_configs:
      - targets: ['monitoring-health-service:8080']
    metrics_path: /metrics
```

### **Docker Compose**

```yaml
# infra/metrics/docker-compose-monitoring.yml

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: quantum_prometheus
    restart: always
    ports:
      - "9090:9090"
    volumes:
      - ./infra/metrics/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=7d'
    networks:
      - quantum_trader
  
  grafana:
    image: grafana/grafana:latest
    container_name: quantum_grafana
    restart: always
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./infra/metrics/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./infra/metrics/grafana/datasources:/etc/grafana/provisioning/datasources:ro
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=quantum_trader_2025
      - GF_USERS_ALLOW_SIGN_UP=false
    networks:
      - quantum_trader
    depends_on:
      - prometheus

volumes:
  prometheus_data:
  grafana_data:

networks:
  quantum_trader:
    external: true
```

---

## 📈 GRAFANA DASHBOARDS

### **Dashboard 1: Service Health Overview**

**Panels**:
- Service health status (gauge)
- HTTP request rate (graph)
- HTTP latency P50/P95/P99 (graph)
- Error rate (graph)
- Dependency health (gauge)

**PromQL Queries**:
```promql
# Request rate
rate(http_requests_total[5m])

# Latency P95
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Error rate
rate(http_requests_total{status=~"5.."}[5m])
```

### **Dashboard 2: Trading Performance**

**Panels**:
- Trades executed (counter)
- Win rate (percentage)
- Portfolio value (gauge)
- Open positions (table)
- Trade execution latency (heatmap)

**PromQL Queries**:
```promql
# Total trades
sum(trades_executed_total)

# Open positions
sum(positions_open) by (symbol)

# Portfolio value
portfolio_value_usd
```

### **Dashboard 3: EventBus Monitoring**

**Panels**:
- Message throughput (graph)
- Queue lag (gauge)
- Processing duration P95 (graph)
- Message distribution by type (pie chart)

**PromQL Queries**:
```promql
# Message rate
rate(eventbus_messages_published_total[5m])

# Queue lag
eventbus_queue_lag

# Processing duration P95
histogram_quantile(0.95, rate(eventbus_message_processing_duration_seconds_bucket[5m]))
```

### **Dashboard 4: Infrastructure Metrics**

**Panels**:
- Redis latency (graph)
- Redis connection pool (gauge)
- Database latency (graph)
- Database connection pool (gauge)
- Emergency Stop activations (counter)

**PromQL Queries**:
```promql
# Redis latency
histogram_quantile(0.95, rate(redis_operation_duration_seconds_bucket[5m]))

# ESS activations
increase(emergency_stop_triggered_total[1h])
```

---

## 🚀 DEPLOYMENT

### **Step 1: Start Monitoring Stack**

```bash
# Start Prometheus and Grafana
docker-compose -f infra/metrics/docker-compose-monitoring.yml up -d

# Verify Prometheus
curl http://localhost:9090/targets

# Verify Grafana
open http://localhost:3000  # Login: admin / quantum_trader_2025
```

### **Step 2: Add /metrics to All Services**

```python
# backend/services/ai_engine/app.py

from fastapi import FastAPI
from infra.metrics.metrics import metrics_endpoint

app = FastAPI()

@app.get("/metrics")
async def metrics():
    return metrics_endpoint()
```

### **Step 3: Import Dashboards**

1. Login to Grafana: http://localhost:3000
2. Go to Dashboards → Import
3. Upload JSON files from `infra/metrics/grafana/dashboards/`

---

## ⚠️ PRODUCTION CONSIDERATIONS

1. **Security**:
   - Change default Grafana password
   - Enable authentication on Prometheus
   - Use HTTPS for external access

2. **Retention**:
   - Prometheus default: 7 days (configurable)
   - Use remote storage (Thanos, Cortex) for long-term retention

3. **Alerting**:
   - Configure Prometheus AlertManager
   - Define alert rules for critical metrics
   - Integrate with PagerDuty, Slack, email

4. **High Availability**:
   - Deploy multiple Prometheus instances
   - Use Thanos for global view

---

## 📋 TODO (Sprint 4+)

- [ ] Create Grafana dashboard JSON templates
- [ ] Configure Prometheus alert rules
- [ ] Setup AlertManager
- [ ] Add custom business metrics (profit/loss, win rate)
- [ ] Integrate with external monitoring (Datadog, New Relic)
- [ ] Setup long-term metric storage (Thanos)

---

**Next Steps**: Complete Sprint 3 infrastructure skeleton, then implement Grafana dashboards in Sprint 4
