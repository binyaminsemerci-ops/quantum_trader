# Monitoring Health Service

Dedicated microservice for collecting, aggregating, and exposing health status from all Quantum Trader services and infrastructure components.

## Features

- 🔍 **Service Health Monitoring** - HTTP health checks for all microservices
- 🏗️ **Infrastructure Monitoring** - Redis, Postgres, Binance API reachability
- 📊 **Health Aggregation** - System-wide status (OK/DEGRADED/CRITICAL)
- 🚨 **Alert Management** - Automatic alerts for failures
- 🔄 **Event-Driven** - Subscribes to `ess.tripped`, publishes `health.snapshot_updated`
- 🌐 **REST API** - Exposes health status for dashboards

## Quick Start

### Prerequisites

- Python 3.11+
- Redis (for EventBus)
- Running Quantum Trader services

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Set environment variables:

```bash
export REDIS_URL=redis://localhost:6379
export LOG_LEVEL=INFO
export HEALTH_CHECK_INTERVAL=60  # seconds
export BASE_SERVICE_URL=http://localhost:8000
```

### Run Standalone

```bash
python -m backend.services.monitoring_health_service.main
```

### Run with Docker

```bash
docker build -t quantum-monitoring-health .
docker run -p 8080:8080 \
  -e REDIS_URL=redis://redis:6379 \
  quantum-monitoring-health
```

## API Endpoints

### Health Checks

- `GET /health` - Service self-check
- `GET /health/system` - Aggregated system health
- `GET /health/metrics` - Key metrics (Grafana-ready)

### Alerts

- `GET /alerts/active` - Active alerts
- `GET /alerts/history` - Alert history
- `POST /alerts/{id}/clear` - Clear alert

## Architecture

```
┌─────────────────────────────────────────────┐
│         Monitoring Health Service           │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────┐      ┌──────────────┐        │
│  │Collectors│─────▶│  Aggregators │        │
│  └──────────┘      └──────────────┘        │
│       │                    │                │
│       │                    ▼                │
│       │            ┌──────────────┐        │
│       │            │AlertManager  │        │
│       │            └──────────────┘        │
│       │                    │                │
│       └────────────┬───────┘                │
│                    │                        │
│            ┌───────▼────────┐               │
│            │   EventBus     │               │
│            └────────────────┘               │
│                                             │
└─────────────────────────────────────────────┘
         │                   │
         ▼                   ▼
   HTTP /health        Redis Events
    endpoints       (ess.tripped, etc.)
```

## Health Status Logic

| Condition | Status |
|-----------|--------|
| All OK, latency < 1s | **OK** |
| Service degraded | **DEGRADED** |
| Critical service down | **CRITICAL** |
| Infrastructure down | **CRITICAL** |

## Adding New Services

Edit `main.py` → `_get_service_targets()`:

```python
ServiceTarget(
    name="new_service",
    url="http://localhost:9000/health",
    critical=True,  # or False
)
```

## Events

### Subscribed
- `ess.tripped` - Emergency stop system alerts

### Published
- `health.snapshot_updated` - Every 60s (configurable)
- `health.alert_raised` - On failures

## Testing

```bash
pytest tests/unit/test_monitoring_health_service_sprint2_service6.py -v
```

**Test Coverage**: 16 tests, 100% pass rate

## Future Enhancements

- Grafana dashboards
- Historical health logging
- Slack/PagerDuty notifications
- Self-healing triggers
- Predictive alerts

## License

Quantum Trader - Internal Use Only

---

**Author**: Quantum Trader AI Team  
**Date**: December 4, 2025  
**Sprint**: 2 - Service #6
