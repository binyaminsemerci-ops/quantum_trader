# Portfolio Governance Agent

## Overview

Portfolio Governance Agent er en AI-drevet mikrotjeneste som bygger "exposure memory" fra trade-historikk og dynamisk justerer risikoparametere basert på faktisk porteføljeperformance.

## Core Functionality

### 1. Exposure Memory
Lagrer og analyserer:
- PnL per trade
- AI confidence scores  
- Market volatility
- Leverage utilization
- Win rates og performance metrics

### 2. Portfolio Score Calculation

```python
score = (avg_pnl × avg_confidence × win_rate) / max(avg_volatility, 0.01)
```

- Higher score = Better performance
- Drives automatic policy adjustments

### 3. Dynamic Policy Management

| Policy | Max Leverage | Min Confidence | Max Positions | Score Range |
|--------|--------------|----------------|---------------|-------------|
| CONSERVATIVE | 10x | 0.75 | 3 | < 0.3 |
| BALANCED | 20x | 0.65 | 5 | 0.3-0.7 |
| AGGRESSIVE | 30x | 0.55 | 7 | > 0.7 |

## Architecture

```
Trade Events → Exposure Memory → Portfolio Score → Policy Adjuster → Risk Parameters
                                                          ↓
                                        ExitBrain v3.5 + RL Sizing Agent
```

## Components

### exposure_memory.py
Core memory system med rolling window (500 events)

**Key Classes:**
- `ExposureMemory` - Main memory manager

**Key Methods:**
- `record(data)` - Store trade event
- `summarize()` - Generate statistics
- `get_portfolio_score()` - Calculate performance score
- `get_symbol_stats(symbol)` - Symbol-specific analytics
- `get_memory_health()` - Health monitoring

### governance_controller.py
Policy controller og decision maker

**Key Classes:**
- `PortfolioGovernanceAgent` - Main controller

**Key Methods:**
- `adjust_policy()` - Update policy based on score
- `should_allow_trade()` - Trade approval gate
- `get_recommended_position_size()` - Dynamic sizing
- `run()` - Continuous monitoring loop (30s interval)

### portfolio_governance_service.py
Service entry point med graceful shutdown

## Redis Integration

### Streams
```
quantum:stream:portfolio.memory      # Trade events (PnL, confidence, etc.)
quantum:stream:governance.events     # Policy change events
```

### Keys
```
quantum:governance:policy            # Current policy
quantum:governance:score             # Current score
quantum:governance:params            # JSON parameters
quantum:governance:param:*           # Individual params
```

## Environment Variables

```bash
REDIS_URL=redis://redis:6379/0       # Redis connection
LOG_LEVEL=INFO                        # Logging level
POLICY_UPDATE_INTERVAL=30             # Policy check interval (seconds)
MIN_SAMPLES=50                        # Minimum samples for decisions
CONSERVATIVE_THRESHOLD=0.3            # Score threshold for CONSERVATIVE
AGGRESSIVE_THRESHOLD=0.7              # Score threshold for AGGRESSIVE
```

## Docker Deployment

### Build
```bash
docker build -t quantum_trader-portfolio-governance:latest .
```

### Run Standalone
```bash
docker run -d \
  --name quantum_portfolio_governance \
  --network quantum_trader \
  -e REDIS_URL=redis://redis:6379/0 \
  quantum_trader-portfolio-governance:latest
```

### Run with Docker Compose
```bash
docker-compose -f docker-compose.vps.yml up -d portfolio-governance
```

## Health Check

Container health check verifies Redis policy key exists:
```python
import redis
r = redis.from_url('redis://redis:6379/0')
assert r.exists('quantum:governance:policy')
```

## Monitoring

### Check Service Status
```bash
docker logs quantum_portfolio_governance
```

### Query Current State
```bash
# Current policy
docker exec redis redis-cli GET quantum:governance:policy

# Current score
docker exec redis redis-cli GET quantum:governance:score

# Memory samples
docker exec redis redis-cli XLEN quantum:stream:portfolio.memory

# Policy parameters
docker exec redis redis-cli GET quantum:governance:params | python3 -m json.tool
```

## Integration Examples

### ExitBrain v3.5
```python
import redis
r = redis.from_url('redis://redis:6379/0')

policy = r.get('quantum:governance:policy').decode()
if policy == 'CONSERVATIVE':
    # Tighter stops, wider targets
    stop_loss_mult = 1.5
elif policy == 'AGGRESSIVE':
    # Tighter targets, wider stops  
    stop_loss_mult = 0.8
```

### RL Sizing Agent
```python
import redis
import json
r = redis.from_url('redis://redis:6379/0')

params = json.loads(r.get('quantum:governance:params'))
max_leverage = params['max_leverage']
min_confidence = params['min_confidence']

# Use in position sizing calculation
if ai_confidence < min_confidence:
    reject_trade()
```

### Recording Trade Events
```python
import redis
from datetime import datetime
r = redis.from_url('redis://redis:6379/0')

event = {
    'timestamp': datetime.utcnow().isoformat(),
    'symbol': 'BTCUSDT',
    'side': 'LONG',
    'leverage': '20',
    'pnl': '0.32',
    'confidence': '0.72',
    'volatility': '0.14',
    'position_size': '1000',
    'exit_reason': 'dynamic_tp'
}

r.xadd('quantum:stream:portfolio.memory', event)
```

## Troubleshooting

### Service Won't Start
```bash
# Check Redis connection
docker exec redis redis-cli ping

# Check logs
docker logs quantum_portfolio_governance | grep ERROR

# Restart
docker-compose -f docker-compose.vps.yml restart portfolio-governance
```

### Policy Not Updating
```bash
# Check sample count (need minimum 50)
docker exec redis redis-cli XLEN quantum:stream:portfolio.memory

# Check service is running
docker ps | grep portfolio_governance

# Check for errors in logs
docker logs quantum_portfolio_governance | tail -50
```

### Score is 0.0
```bash
# Check if events recorded
docker exec redis redis-cli XLEN quantum:stream:portfolio.memory

# Check last event
docker exec redis redis-cli XREVRANGE quantum:stream:portfolio.memory + - COUNT 1

# Verify PnL values are numeric and non-zero
```

## Performance Tuning

### Adjust Memory Window
```python
# In exposure_memory.py __init__
self.memory = deque(maxlen=1000)  # Increased from 500
```

### Adjust Policy Thresholds
```python
# In governance_controller.py
self.thresholds = {
    "conservative_threshold": 0.25,  # More conservative (was 0.3)
    "aggressive_threshold": 0.8,     # Less aggressive (was 0.7)
}
```

### Adjust Update Frequency
```python
# In portfolio_governance_service.py main()
agent.run(interval=15)  # Update every 15s (was 30s)
```

## Dependencies

- `redis>=5.0.1` - Redis client
- `python-json-logger>=2.0.7` - Structured logging

## Logs

Logs are written to:
- stdout (captured by Docker)
- `/app/logs/` (if mounted)

Log format:
```
2025-12-21 12:00:00 - governance_controller - INFO - Policy changed: BALANCED → AGGRESSIVE (High score (0.82))
```

## Testing

See [AI_PORTFOLIO_GOVERNANCE_VALIDATION.md](../../AI_PORTFOLIO_GOVERNANCE_VALIDATION.md) for:
- Deployment validation
- Test scenarios
- Integration tests
- Performance benchmarks

## Documentation

- **Validation Guide**: [AI_PORTFOLIO_GOVERNANCE_VALIDATION.md](../../AI_PORTFOLIO_GOVERNANCE_VALIDATION.md)
- **Quick Reference**: [AI_PORTFOLIO_GOVERNANCE_QUICKREF.md](../../AI_PORTFOLIO_GOVERNANCE_QUICKREF.md)
- **Phase Summary**: [AI_PHASE_4Q_COMPLETE.md](../../AI_PHASE_4Q_COMPLETE.md)

## Version

**Version:** 1.0.0  
**Phase:** 4Q  
**Status:** Production Ready  
**Date:** 2025-12-21

## License

Part of Quantum Trader System © 2025

## Support

For issues or questions, check:
1. Docker logs
2. Redis key/stream status
3. AI Engine health endpoint
4. Validation documentation

---

**Portfolio Governance Agent** - Adaptive Risk Management through Continuous Learning 🧠
