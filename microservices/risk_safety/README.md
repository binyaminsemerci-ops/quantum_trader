# Risk & Safety Service

**Port:** 8003  
**Responsibility:** Risk management, ESS, and policy enforcement

## Features

- **Emergency Stop System (ESS):** Autonomous circuit breaker
- **PolicyStore:** Single Source of Truth for configuration
- **Risk Limits:** Enforce per-symbol and portfolio-wide limits
- **Event-Driven:** Publishes state changes and policy updates

## API Endpoints

### ESS
- `GET /api/risk/ess/status` - Current ESS state
- `POST /api/risk/ess/override` - Manual ESS override
- `POST /api/risk/ess/reset` - Reset ESS to NORMAL

### Policy
- `GET /api/policy/{key}` - Get policy value
- `GET /api/policy` - Get all policies
- `POST /api/policy/update` - Update policy value

### Health
- `GET /health` - Service health check

## Events

**Published:**
- `ess.state.changed` - ESS state transition
- `ess.tripped` - ESS entered CRITICAL state
- `policy.updated` - Policy value changed
- `risk.limit.exceeded` - Risk threshold breach

**Consumed:**
- `trade.closed` - For ESS loss tracking
- `order.failed` - For failure rate monitoring

## Running Locally

```bash
cd microservices/risk_safety
python -m pip install -r requirements.txt
python main.py
```

Service will start on http://localhost:8003

## Running with Docker

```bash
docker build -t risk-safety-service .
docker run -p 8003:8003 \
    -e REDIS_HOST=localhost \
    -e REDIS_PORT=6379 \
    risk-safety-service
```

## Testing

```bash
pytest tests/ -v
```

## Configuration

Environment variables (see `config.py`):
- `REDIS_HOST` - Redis hostname (default: localhost)
- `REDIS_PORT` - Redis port (default: 6379)
- `LOG_LEVEL` - Logging level (default: INFO)

## Dependencies

See `requirements.txt`. Requires access to:
- `backend/safety/ess.py`
- `backend/core/policy_store.py`
- `backend/core/event_bus.py`
- `backend/core/disk_buffer.py`
