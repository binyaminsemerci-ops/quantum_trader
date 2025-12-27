# Strategic Memory Sync - Phase 4S

**Continuous Learning & Pattern Recognition System**

## 🎯 Purpose

Strategic Memory Sync collects and synchronizes memory data from all active trading modules, analyzes patterns between strategy decisions and actual results, learns which strategies work best in different market regimes, and provides meta-feedback to the AI Engine.

## 🧠 Architecture

```
┌─────────────────────────┐
│ Portfolio Governance    │
│ (Policy Management)     │
└───────────┬─────────────┘
            │
            ↓
┌─────────────────────────┐
│ Meta-Regime Correlator  │
│ (Regime Detection)      │
└───────────┬─────────────┘
            │
            ↓
┌─────────────────────────┐
│ Strategic Memory Sync   │◄───── Redis Streams
│ (Pattern Learning)      │
└───────────┬─────────────┘
            │
            ↓
┌─────────────────────────┐
│ RL / AI Engine / Exit   │
│ (Action Execution)      │
└─────────────────────────┘
```

## 📦 Components

### 1. MemoryLoader (`memory_loader.py`)
Retrieves strategic data from Redis:
- Portfolio governance policy
- Preferred market regime
- Meta-regime observations (last 50)
- Portfolio PnL stream (last 50)
- Exposure summary
- Leverage settings
- Exit statistics
- Recent trade results (last 30)

### 2. PatternAnalyzer (`pattern_analyzer.py`)
Discovers patterns between strategies and results:
- Analyzes performance by market regime
- Calculates win rates per regime
- Identifies best-performing policies
- Tracks confidence levels
- Computes average PnL by regime

**Metrics Computed:**
- `avg_pnl`: Average profit/loss per regime
- `win_rate`: Wins / (Wins + Losses)
- `confidence`: Weighted average confidence score
- `best_policy`: Policy that performed best in regime

### 3. ReinforcementFeedback (`reinforcement_feedback.py`)
Generates meta-signals for AI Engine:
- Recommends policy based on regime performance
- Calculates confidence boost (0.0 to 1.0)
- Suggests leverage adjustments (0.5x to 2.0x)
- Publishes feedback to Redis and event bus

**Policy Recommendations:**
| Regime | Win Rate | Avg PnL | Recommended Policy |
|--------|----------|---------|-------------------|
| BULL | >60% | >0.2 | AGGRESSIVE |
| BULL | >50% | >0.1 | BALANCED |
| BULL | <50% | <0.1 | CONSERVATIVE |
| BEAR | Any | Any | CONSERVATIVE |
| VOLATILE | Any | Any | CONSERVATIVE |
| RANGE | >55% | Any | BALANCED |
| RANGE | <55% | Any | CONSERVATIVE |

### 4. StrategicMemorySync (`memory_sync_service.py`)
Main orchestration service:
- Runs continuous 60-second analysis loop
- Loads → Analyzes → Generates Feedback
- Handles graceful shutdown (SIGINT/SIGTERM)
- Structured logging with timestamps

## 🔄 Data Flow

```
Redis Streams/Keys
    ↓
[MemoryLoader]
    ↓
Strategic Memory Dict
    ↓
[PatternAnalyzer]
    ↓
Analysis Results
    ↓
[ReinforcementFeedback]
    ↓
quantum:feedback:strategic_memory (Redis)
    ↓
AI Engine / RL Agent / Portfolio Governance
```

## 📊 Redis Keys

### Input Data Sources:
- `quantum:governance:policy` - Current portfolio policy
- `quantum:governance:preferred_regime` - Preferred market regime
- `quantum:governance:regime_stats` - Regime statistics JSON
- `quantum:stream:meta.regime` - Meta-regime observations stream
- `quantum:stream:portfolio.memory` - Portfolio PnL stream
- `quantum:exposure:summary` - Current exposure JSON
- `quantum:risk:leverage_limits` - Leverage settings JSON
- `quantum:exit:statistics` - Exit brain stats JSON
- `quantum:stream:trade.results` - Trade results stream

### Output Feedback:
- `quantum:feedback:strategic_memory` - Strategic feedback JSON
- `quantum:events:strategic_feedback` - Event bus channel

## 🎮 Feedback Structure

```json
{
  "preferred_regime": "BULL",
  "updated_policy": "AGGRESSIVE",
  "confidence_boost": 0.7842,
  "leverage_hint": 1.5,
  "regime_performance": {
    "avg_pnl": 0.3121,
    "win_rate": 0.6667,
    "sample_count": 15
  },
  "timestamp": "2025-12-21T06:45:00.000000Z",
  "version": "1.0.0"
}
```

### Field Descriptions:
- **preferred_regime**: Best performing market regime
- **updated_policy**: Recommended governance policy
- **confidence_boost**: 0.0-1.0 confidence multiplier
- **leverage_hint**: 0.5-2.0 leverage adjustment factor
- **regime_performance**: Performance metrics for best regime

## 🚀 Deployment

### Docker Build:
```bash
docker compose -f docker-compose.vps.yml build strategic-memory
```

### Start Service:
```bash
docker compose -f docker-compose.vps.yml up -d strategic-memory
```

### Check Status:
```bash
docker ps --filter name=quantum_strategic_memory
docker logs --tail 50 quantum_strategic_memory
```

### Health Check:
```bash
curl -s http://localhost:8001/health | jq '.metrics.strategic_memory'
```

## ⚙️ Configuration

### Environment Variables:
- `REDIS_URL`: Redis connection URL (default: `redis://redis:6379/0`)
- `MEMORY_SYNC_INTERVAL`: Analysis interval in seconds (default: `60`)

### Docker Compose:
```yaml
strategic-memory:
  build: ./microservices/strategic_memory
  container_name: quantum_strategic_memory
  restart: always
  environment:
    - REDIS_URL=redis://redis:6379/0
    - MEMORY_SYNC_INTERVAL=60
  depends_on:
    redis:
      condition: service_healthy
  networks:
    - quantum_trader
  deploy:
    resources:
      limits:
        memory: 256M
        cpus: '0.3'
```

## 🔍 Monitoring

### Check Feedback:
```bash
docker exec quantum_redis redis-cli GET quantum:feedback:strategic_memory
```

### Watch Logs:
```bash
docker logs -f quantum_strategic_memory
```

### Sample Log Output:
```json
{"event": "Memory sync iteration complete", "iteration": 5, "samples": 42, "best_regime": "BULL", "policy": "AGGRESSIVE", "confidence": 0.7842, "elapsed_ms": 127.45, "timestamp": "2025-12-21T06:45:32.123456Z", "level": "info"}
```

## 🧪 Testing

### Manual Test Data:
```bash
# Inject test regime observations
docker exec quantum_redis redis-cli XADD quantum:stream:meta.regime '*' \
  regime BULL pnl 0.45 volatility 0.015 trend 0.003 confidence 0.91 timestamp '2025-12-21T06:00:00Z'

# Inject test PnL data
docker exec quantum_redis redis-cli XADD quantum:stream:portfolio.memory '*' \
  regime BULL pnl 0.38 total_pnl 1250.50 timestamp '2025-12-21T06:00:00Z'

# Wait for analysis cycle (60s)
sleep 65

# Check feedback
docker exec quantum_redis redis-cli GET quantum:feedback:strategic_memory | jq
```

### Expected Output:
```json
{
  "preferred_regime": "BULL",
  "updated_policy": "AGGRESSIVE",
  "confidence_boost": 0.7234,
  "leverage_hint": 1.5
}
```

## 📈 Performance Impact

### System Effects:
1. **RL Position Sizing Agent**: Adjusts leverage based on `leverage_hint`
2. **Portfolio Governance**: Switches policy based on `updated_policy`
3. **Exit Brain v3.5**: Adjusts TP/SL aggressiveness based on `confidence_boost`
4. **Meta-Regime Correlator**: Uses feedback to improve regime detection

### Learning Cycle:
```
Trade → Results → Memory → Analysis → Feedback → Adjust Strategy → Trade
```

## 🎓 Machine Learning Aspects

### Reinforcement Learning:
- **Reward Signal**: PnL performance per regime
- **State**: Current market regime
- **Action**: Policy recommendation
- **Exploration**: Tracks multiple policies per regime
- **Exploitation**: Recommends best-performing policy

### Pattern Recognition:
- Time-series analysis of regime transitions
- Performance correlation across regimes
- Win rate optimization per regime
- Confidence-weighted recommendations

### Continuous Improvement:
- Longer memory window = better pattern detection
- More samples = higher confidence
- Adapts to changing market conditions
- Self-correcting based on results

## ⚠️ Important Notes

1. **Minimum Samples**: Requires 3+ samples for feedback generation
2. **Cold Start**: Will show "UNKNOWN" regime until sufficient data collected
3. **Memory Window**: Uses last 50 regime observations, 50 PnL entries, 30 trades
4. **Update Frequency**: 60-second intervals (configurable)
5. **Safe Defaults**: Falls back to CONSERVATIVE policy if uncertain

## 🔗 Integration Points

### Consumes Data From:
- Portfolio Governance Agent (Phase 4Q)
- Meta-Regime Correlator (Phase 4R)
- Exit Brain v3.5
- RL Position Sizing Agent
- Exposure Balancer

### Provides Feedback To:
- AI Engine (via health endpoint)
- Portfolio Governance (policy updates)
- RL Agent (leverage hints)
- Exit Brain (confidence signals)

## 📝 Version History

- **v1.0.0** (2025-12-21): Initial implementation
  - Memory loading from 9 Redis sources
  - Pattern analysis with regime breakdown
  - Reinforcement feedback generation
  - 60-second continuous learning loop

---

**Phase 4S Complete** ✅  
Strategic Memory Sync provides the "brain" that learns from experience and continuously improves trading strategy selection.
