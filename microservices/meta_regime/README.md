# Meta-Regime Correlator - Phase 4R

## 🧭 Purpose

The Meta-Regime Correlator is an AI layer that:

1. **Collects** market and portfolio data (volatility, funding rate, trend, PnL, confidence)
2. **Learns** the relationship between market regimes and system performance
3. **Updates** governance policy and ExitBrain parameters automatically
4. **Feeds back** predictions to RL Sizing Agent and Exposure Balancer

## 🧩 Components

### RegimeDetector
Analyzes market trends and defines regimes:
- **BULL**: Positive trend, moderate volatility
- **BEAR**: Negative trend, moderate volatility
- **RANGE**: Low volatility, no clear trend
- **VOLATILE**: High volatility regardless of trend
- **UNCERTAIN**: Doesn't fit clear patterns

### RegimeMemory
Long-term memory of market regimes:
- Stores up to 1000 regime observations
- Calculates win rates per regime
- Tracks average PnL by regime
- Persistent storage in Redis

### MetaRegimeCorrelator
Links portfolio results to market regimes:
- Identifies best performing regimes
- Suggests policy adjustments (Conservative/Balanced/Aggressive)
- Updates governance automatically based on regime changes
- Provides regime-based recommendations

## 🔧 Configuration

Environment variables:
- `REDIS_URL`: Redis connection URL (default: redis://localhost:6379/0)
- `REGIME_INTERVAL`: Analysis interval in seconds (default: 30)

## 📊 Redis Keys

**Streams:**
- `quantum:stream:meta.regime`: Regime observation stream

**Keys:**
- `quantum:governance:preferred_regime`: Best performing regime
- `quantum:governance:regime_stats`: Full regime statistics (JSON)
- `quantum:governance:policy`: Current governance policy (updated by correlator)

**Events:**
- `quantum:events:policy_change`: Published when policy changes due to regime shift

## 🚀 Usage

### Local Development
```bash
python meta_regime_service.py
```

### Docker
```bash
docker compose -f docker-compose.vps.yml build meta-regime
docker compose -f docker-compose.vps.yml up -d meta-regime
```

### Health Check
```bash
# Check service logs
docker logs quantum_meta_regime

# Verify Redis data
docker exec redis redis-cli XLEN quantum:stream:meta.regime
docker exec redis redis-cli GET quantum:governance:preferred_regime

# Check AI Engine health
curl -s http://localhost:8001/health | jq '.metrics.meta_regime'
```

## 📈 Integration

The Meta-Regime Correlator integrates with:

1. **Portfolio Governance Agent**: Updates policy based on regime
2. **AI Engine**: Exposes regime metrics in health endpoint
3. **Cross-Exchange Feed**: Consumes market data
4. **RL Sizing Agent**: Provides regime context for sizing decisions
5. **Exposure Balancer**: Informs exposure adjustments

## 🧠 Regime → Policy Mapping

| Regime | Default Policy | Adjustable |
|--------|---------------|------------|
| BULL | AGGRESSIVE | Yes, based on historical performance |
| RANGE | BALANCED | Yes |
| BEAR | CONSERVATIVE | Yes |
| VOLATILE | CONSERVATIVE | Yes |
| UNCERTAIN | BALANCED | Yes |

Policy suggestions are refined based on historical performance in each regime.

## 📝 Example Output

```json
{
  "regime": "BULL",
  "volatility": 0.021,
  "trend": 0.0034,
  "confidence": 0.87,
  "pnl": 0.12,
  "best_regime": "BULL",
  "statistics": {
    "count": 145,
    "avg_pnl": 0.085,
    "win_rate": 0.62,
    "avg_volatility": 0.019
  }
}
```

## 🎯 Performance Impact

Expected improvements:
- **15-25%** better risk-adjusted returns through regime-aware policy
- **Reduced drawdowns** by avoiding unfavorable regimes
- **Faster policy adaptation** to market changes
- **Cross-loop feedback** improves RL agent learning
