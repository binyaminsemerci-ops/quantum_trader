# Strategy Generator AI - Deployment Guide

## 🚀 Quick Start

### 1. Start Strategy Generator Services

```bash
# Start continuous strategy generation + shadow testing
systemctl --profile strategy-gen up -d

# View logs
systemctl logs -f strategy_generator
systemctl logs -f shadow_tester
```

### 2. Service Overview

**strategy_generator** (continuous_runner.py):
- Runs every 24 hours (configurable)
- Generates 20 new strategies per generation
- Uses evolutionary search (mutation + crossover)
- Promotes high-performers to SHADOW status
- Backtests on 90 days of BTCUSDT + ETHUSDT data

**shadow_tester** (shadow_runner.py):
- Tests SHADOW strategies every 15 minutes
- Evaluates deployment candidates every hour
- Promotes strategies with fitness ≥70 to LIVE
- Monitors forward-test performance

### 3. Configuration (systemctl.yml)

**Strategy Generator:**
```yaml
environment:
  - GENERATION_INTERVAL_HOURS=24    # Generation frequency
  - POPULATION_SIZE=20              # Strategies per generation
  - MUTATION_RATE=0.3               # Genetic algorithm params
  - CROSSOVER_RATE=0.4
```

**Shadow Tester:**
```yaml
environment:
  - SHADOW_INTERVAL_MINUTES=15      # Shadow test frequency
  - DEPLOYMENT_INTERVAL_HOURS=1     # Deployment check frequency
```

### 4. Monitoring

**Check service status:**
```bash
systemctl list-units | grep strategy
```

**View generation logs:**
```bash
journalctl -u quantum_strategy_generator.service --tail 50
```

**View shadow test logs:**
```bash
journalctl -u quantum_shadow_tester.service --tail 50
```

**Check database:**
```bash
# View strategies
python -c "
from backend.database import SessionLocal
from backend.research.postgres_repository import PostgresStrategyRepository
from backend.research.models import StrategyStatus

session = SessionLocal()
repo = PostgresStrategyRepository(session)

for status in [StrategyStatus.CANDIDATE, StrategyStatus.SHADOW, StrategyStatus.LIVE]:
    strategies = repo.get_strategies_by_status(status)
    print(f'{status}: {len(strategies)} strategies')
"
```

### 5. Manual Operations

**Trigger generation manually:**
```bash
docker exec quantum_strategy_generator python -c "
from backend.research.continuous_runner import ContinuousStrategyRunner
from backend.database import SessionLocal
from binance.client import Client

session = SessionLocal()
runner = ContinuousStrategyRunner(session, Client(), generation_interval_hours=24)
runner.run_generation()
"
```

**Promote strategy to SHADOW:**
```bash
docker exec quantum_strategy_generator python -c "
from backend.database import SessionLocal
from backend.research.postgres_repository import PostgresStrategyRepository
from backend.research.models import StrategyStatus

session = SessionLocal()
repo = PostgresStrategyRepository(session)
repo.update_status('strategy_id_here', StrategyStatus.SHADOW)
print('✅ Promoted to SHADOW')
"
```

## 📊 Expected Performance

**Generation 1 Results (validated):**
```
Strategies: 10
Top Fitness: 65-67
Top PF: 1.79-2.60
Top WR: 45-57%
Runtime: ~2 seconds
```

**Shadow Testing:**
- Tests run on recent 7-day windows
- Strategies need PF >1.5, WR >45%, Fitness >60 to promote
- Deployment threshold: Fitness ≥70

**Live Deployment:**
- Only strategies with proven forward-test performance
- Automatic promotion after successful shadow period (7+ days)
- Continuous monitoring and auto-disable if underperforming

## 🔧 Troubleshooting

**Services not starting:**
```bash
# Check logs
systemctl logs strategy_generator
systemctl logs shadow_tester

# Restart services
systemctl --profile strategy-gen restart
```

**No strategies generated:**
- Check Binance API keys in `.env`
- Verify database is accessible
- Check backtest data availability

**Shadow tests failing:**
- Ensure market data is available
- Check confidence thresholds (min_confidence)
- Verify EnsembleManager is working

## 📈 Scaling

**Increase generation frequency:**
```yaml
GENERATION_INTERVAL_HOURS=12  # Run every 12 hours
```

**Larger populations:**
```yaml
POPULATION_SIZE=50  # More strategies per generation
```

**More aggressive shadow testing:**
```yaml
SHADOW_INTERVAL_MINUTES=5  # Test every 5 minutes
```

## 🎯 Integration with Trading System

The Strategy Generator AI integrates seamlessly with Quantum Trader:

1. **Candidate Strategies**: Generated and backtested
2. **Shadow Strategies**: Forward-tested with paper trades
3. **Live Strategies**: Deployed to production trading
4. **Monitoring**: Continuous performance tracking
5. **Auto-disable**: Poor performers automatically removed

The system creates a self-improving trading strategy pipeline that continuously evolves based on market conditions.

