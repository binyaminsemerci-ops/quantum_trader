# Portfolio Governance - Quick Reference

## 🚀 Deployment One-Liner

```bash
docker-compose -f docker-compose.vps.yml up -d portfolio-governance
```

## 📊 Essential Checks

```bash
# 1. Service Status
docker ps | grep portfolio_governance

# 2. Current Policy
docker exec redis redis-cli GET quantum:governance:policy

# 3. Portfolio Score
docker exec redis redis-cli GET quantum:governance:score

# 4. Memory Samples
docker exec redis redis-cli XLEN quantum:stream:portfolio.memory

# 5. AI Engine Integration
curl -s http://localhost:8001/health | jq '.metrics.portfolio_governance'
```

## 🎯 Policy Decision Logic

| Score Range | Policy | Max Leverage | Min Confidence | Max Positions |
|-------------|--------|--------------|----------------|---------------|
| < 0.3 | **CONSERVATIVE** | 10x | 0.75 | 3 |
| 0.3 - 0.7 | **BALANCED** | 20x | 0.65 | 5 |
| > 0.7 | **AGGRESSIVE** | 30x | 0.55 | 7 |

## 📈 Portfolio Score Formula

```
score = (avg_pnl × avg_confidence × win_rate) / max(avg_volatility, 0.01)
```

## 🔗 Redis Keys

```bash
# Read-Only (queried by other services)
quantum:governance:policy                    # Current policy
quantum:governance:score                     # Current score
quantum:governance:params                    # JSON policy parameters
quantum:governance:param:max_leverage        # Individual param
quantum:governance:param:min_confidence      # Individual param

# Streams
quantum:stream:portfolio.memory              # Exposure memory events
quantum:stream:governance.events             # Policy change events
```

## 🧪 Quick Test

```bash
# Simulate winning trade
docker exec redis redis-cli XADD quantum:stream:portfolio.memory "*" \
  timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  symbol "BTCUSDT" \
  pnl "0.45" \
  confidence "0.75" \
  volatility "0.12" \
  leverage "20"

# Wait 30s, then check policy
sleep 30 && docker exec redis redis-cli GET quantum:governance:policy
```

## 🔥 Emergency Commands

```bash
# Reset to BALANCED
docker exec redis redis-cli SET quantum:governance:policy BALANCED

# Clear memory (DANGER!)
docker exec redis redis-cli DEL quantum:stream:portfolio.memory

# Restart service
docker-compose -f docker-compose.vps.yml restart portfolio-governance
```

## 📡 Integration Points

### ExitBrain v3.5
```python
policy = redis.get("quantum:governance:policy")
# Use for TP/SL adjustment
```

### RL Sizing Agent
```python
params = json.loads(redis.get("quantum:governance:params"))
max_pos = params["max_position_pct"]
```

### Exposure Balancer
```python
max_positions = redis.get("quantum:governance:param:max_concurrent_positions")
```

## 📝 Environment Variables

```bash
PORTFOLIO_GOVERNANCE_ENABLED=true
REDIS_URL=redis://redis:6379/0
POLICY_UPDATE_INTERVAL=30
MIN_SAMPLES=50
CONSERVATIVE_THRESHOLD=0.3
AGGRESSIVE_THRESHOLD=0.7
```

## ✅ Success Criteria

- ✅ Service running (healthy)
- ✅ Policy updates every 30s
- ✅ Memory samples > 50
- ✅ AI Engine integration OK
- ✅ Score calculation working

---

**Status:** ✅ Phase 4Q Complete  
**Version:** 1.0.0  
**Updated:** 2025-12-21
