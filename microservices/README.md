# Quantum Trader - Microservices Architecture

**Sprint 2:** Service-oriented architecture with event-driven communication

## Services

| Service | Port | Responsibility |
|---------|------|----------------|
| **risk-safety** | 8003 | ESS, PolicyStore, risk enforcement |
| **ai-engine** | 8001 | ML decisions, RL sizing, meta-strategy |
| **execution** | 8002 | Order placement, trade lifecycle, rate limiting |
| **portfolio-intelligence** | 8004 | PBA, analytics, performance tracking |
| **rl-training** | 8006 | Offline RL training, model versioning |
| **monitoring-health** | 8005 | Health checks, metrics, alerting |
| **marketdata** | 8007 | Market data aggregation (optional) |

## Architecture

```
API Gateway (8000)
    ├── ai-engine (8001)
    ├── execution (8002)
    ├── risk-safety (8003) ✅ IMPLEMENTED
    ├── portfolio-intelligence (8004)
    ├── monitoring-health (8005)
    ├── rl-training (8006)
    └── marketdata (8007)
         │
    EventBus (Redis Streams)
         │
    Data Layer (Redis + SQLite + Postgres)
```

## Status

- ✅ **risk-safety-service:** COMPLETE (boilerplate + full implementation)
- 🔶 **ai-engine-service:** Boilerplate pending
- 🔶 **execution-service:** Boilerplate pending
- 🔶 **portfolio-intelligence-service:** Boilerplate pending
- 🔶 **rl-training-service:** Boilerplate pending
- 🔶 **monitoring-health-service:** Boilerplate pending
- 🔶 **marketdata-service:** Boilerplate pending

## Running Services

### Individual Service
```bash
cd microservices/<service_name>
python -m pip install -r requirements.txt
python main.py
```

### All Services (Docker Compose)
```bash
docker-compose up -d
```

## Development Guidelines

1. **Health Check:** All services must implement `GET /health`
2. **EventBus:** Subscribe to relevant events on startup
3. **PolicyStore:** Load snapshot on startup, refresh on `policy.updated` event
4. **Logging:** Use service name prefix in all logs
5. **Graceful Shutdown:** Handle SIGTERM/SIGINT signals
6. **Testing:** Minimum 80% coverage for critical paths

## Event Schema

See `docs/EVENT_SCHEMA.md` for complete event definitions.

## Next Steps

1. Implement boilerplate for services #2-7
2. Migrate modules according to mapping table
3. Test inter-service communication
4. Deploy to production with Docker Compose
