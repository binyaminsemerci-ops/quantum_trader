# Strategy Generator AI (SG AI)

**Autonomous strategy research engine with genetic algorithms for Quantum Trader.**

## 🎯 Overview

The Strategy Generator AI continuously:
1. **Generates** strategy candidates using genetic algorithms
2. **Backtests** strategies on historical data
3. **Evolves** strategies through crossover and mutation
4. **Shadow tests** promising strategies on live data (paper trading)
5. **Promotes** validated strategies to live trading
6. **Demotes** underperforming strategies

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Strategy Lifecycle                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  CANDIDATE → Backtest → (pass) → SHADOW → Forward Test →    │
│                ↓                            ↓                 │
│             (fail)                       (pass)               │
│                ↓                            ↓                 │
│            DISABLED ←─────────────────── LIVE                │
│                         ↑                                     │
│                    (underperform)                             │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Components

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| **models.py** | Data structures | StrategyConfig, StrategyStats, Status enum, fitness calculation |
| **repositories.py** | Persistence interfaces | Protocol-based for dependency injection |
| **backtest.py** | Historical simulation | Entry/exit logic, TP/SL, commission modeling |
| **search.py** | Genetic algorithms | Crossover, mutation, fitness-based selection |
| **shadow.py** | Forward testing | Paper trading on live data, async monitoring |
| **deployment.py** | Promotion/demotion | Threshold-based lifecycle management |

## 🧬 Genetic Algorithm

### Parameters Evolved

- `min_confidence`: Entry threshold (0.60-0.90)
- `take_profit_pct`: TP percentage (0.01-0.05)
- `stop_loss_pct`: SL percentage (0.005-0.03)
- `risk_per_trade_pct`: Risk per trade (0.005-0.03)
- `leverage`: Leverage multiplier (10-50)
- `trailing_stop_enabled`: Use trailing stops (bool)
- `regime_filter`: Market regime (TRENDING/RANGING/CHOPPY/ANY)
- `entry_type`: Entry logic (ENSEMBLE/MOMENTUM/MEAN_REVERSION)

### Operators

**Crossover (70% of offspring):**
```python
# Combines two parent strategies
child.min_confidence = (parent1.min_confidence + parent2.min_confidence) / 2
child.take_profit_pct = parent1.take_profit_pct  # From parent 1
child.leverage = parent2.leverage  # From parent 2
```

**Mutation (30% of offspring):**
```python
# Adjusts 1-3 parameters randomly
child.min_confidence += random.uniform(-0.05, 0.05)
child.leverage *= random.uniform(0.9, 1.1)
```

### Fitness Function

```
fitness = (
    0.40 * profit_factor_score +
    0.20 * win_rate_score +
    0.20 * drawdown_penalty +
    0.20 * sample_size_score
)
```

**Target:** PF=2.0, WR=60%, DD=0%, Trades=100+

## 🚀 Quick Start

### 1. Generate Initial Population

```python
from backend.research import StrategySearchEngine, StrategyBacktester

search = StrategySearchEngine(backtester, repository)

strategies = search.run_generation(
    population_size=20,
    generation=1,
    symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    start_date=datetime(2024, 1, 1),
    end_date=datetime.utcnow(),
    parent_strategies=None  # First generation
)

# Top strategies by fitness
top_3 = strategies[:3]
```

### 2. Evolutionary Loop

```python
# Select parents (top 5)
parents = strategies[:5]

# Generate offspring
next_gen = search.run_generation(
    population_size=20,
    generation=2,
    symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    start_date=datetime(2024, 1, 1),
    end_date=datetime.utcnow(),
    parent_strategies=parents  # Evolve from parents
)
```

### 3. Shadow Testing

```python
from backend.research import StrategyShadowTester

shadow_tester = StrategyShadowTester(repository, market_data)

# One-off test (for cron)
await shadow_tester.run_once(
    symbols=["BTCUSDT", "ETHUSDT"],
    lookback_days=7
)

# Or continuous monitoring
await shadow_tester.run_shadow_loop(
    symbols=["BTCUSDT", "ETHUSDT"],
    lookback_days=7,
    interval_minutes=15  # Test every 15 min
)
```

### 4. Promotion/Demotion

```python
from backend.research import StrategyDeploymentManager

deployment = StrategyDeploymentManager(
    repository=repository,
    candidate_min_pf=1.5,
    candidate_min_trades=50,
    shadow_min_pf=1.3,
    shadow_min_days=14
)

# Promote qualifying strategies
promoted = deployment.review_and_promote()
# CANDIDATE → SHADOW → LIVE

# Demote underperformers
disabled = deployment.review_and_disable()
# LIVE → DISABLED
```

## 📊 Configuration

### Backtest Parameters

```python
backtester = StrategyBacktester(
    market_data_client=data_client,
    commission_rate=0.0004  # 0.04% per trade
)
```

### Search Parameters

```python
search = StrategySearchEngine(
    backtester=backtester,
    repository=repository,
    min_confidence_range=(0.60, 0.90),
    take_profit_range=(0.010, 0.050),
    stop_loss_range=(0.005, 0.030),
    risk_range=(0.005, 0.030),
    leverage_range=(10, 50)
)
```

### Promotion Thresholds

```python
deployment = StrategyDeploymentManager(
    repository=repository,
    
    # CANDIDATE → SHADOW
    candidate_min_pf=1.5,        # Min profit factor
    candidate_min_trades=50,     # Min trades
    candidate_max_dd=0.20,       # Max 20% drawdown
    
    # SHADOW → LIVE
    shadow_min_pf=1.3,           # Min profit factor
    shadow_min_trades=20,        # Min trades
    shadow_min_days=14,          # Min forward test days
    
    # LIVE → DISABLED
    live_min_pf=1.1,             # Min profit factor
    live_max_dd=0.25,            # Max 25% drawdown
    live_check_days=30           # Check last 30 days
)
```

## 📁 Examples

See `backend/research/examples/` for complete working examples:

| Example | Description |
|---------|-------------|
| `example_1_first_generation.py` | Generate initial population |
| `example_2_evolutionary_loop.py` | Multi-generation evolution |
| `example_3_shadow_testing.py` | Forward testing setup |
| `example_4_full_pipeline.py` | Complete workflow |

**Run examples:**

```bash
cd backend/research/examples
python example_1_first_generation.py
python example_2_evolutionary_loop.py
python example_3_shadow_testing.py
python example_4_full_pipeline.py
```

## 🔌 Integration with Quantum Trader

### 1. Implement Concrete Repository

```python
from backend.research.repositories import StrategyRepository
import psycopg2

class PostgresStrategyRepository(StrategyRepository):
    def __init__(self, connection_string):
        self.conn = psycopg2.connect(connection_string)
    
    def save_strategy(self, config: StrategyConfig):
        # INSERT INTO strategies ...
        pass
    
    def get_strategies_by_status(self, status: StrategyStatus):
        # SELECT * FROM strategies WHERE status = ...
        pass
    
    # ... implement other methods
```

### 2. Connect to Market Data

```python
from backend.research.repositories import MarketDataClient

class BinanceMarketDataClient(MarketDataClient):
    def __init__(self, client):
        self.client = client
    
    def get_history(self, symbol, timeframe, start, end):
        klines = self.client.futures_klines(
            symbol=symbol,
            interval=timeframe,
            startTime=int(start.timestamp() * 1000),
            endTime=int(end.timestamp() * 1000)
        )
        
        return pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', ...
        ])
```

### 3. Wire Up with Ensemble

```python
from backend.research.backtest import StrategyBacktester

class EnsembleBacktester(StrategyBacktester):
    def __init__(self, market_data, ensemble):
        super().__init__(market_data)
        self.ensemble = ensemble
    
    def _check_entry(self, config, df, idx):
        # Use real ensemble predictions
        predictions = self.ensemble.predict(df.iloc[:idx+1])
        signal = predictions[-1]
        
        if config.entry_type == EntryType.ENSEMBLE_CONSENSUS:
            return signal['confidence'] >= config.min_confidence
        
        # ... other entry types
```

### 4. Deploy with Docker

Add to `docker-compose.yml`:

```yaml
services:
  strategy_generator:
    image: quantum_trader:latest
    command: python -m backend.research.examples.continuous_evolution
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/quantum
      - BINANCE_API_KEY=${BINANCE_API_KEY}
      - BINANCE_API_SECRET=${BINANCE_API_SECRET}
    depends_on:
      - db
      - backend
```

### 5. Schedule Shadow Testing

**Option A: Cron**

```bash
*/15 * * * * python -c "import asyncio; from backend.research import StrategyShadowTester; asyncio.run(StrategyShadowTester(...).run_once(...))"
```

**Option B: Async Loop**

```python
# In main application
async def start_shadow_testing():
    shadow_tester = StrategyShadowTester(repo, market_data)
    
    await shadow_tester.run_shadow_loop(
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        lookback_days=7,
        interval_minutes=15
    )

# Run in background
asyncio.create_task(start_shadow_testing())
```

## 📈 Monitoring

### Key Metrics to Track

**Generation Metrics:**
- Best fitness per generation
- Average fitness per generation
- Fitness improvement rate
- Population diversity

**Shadow Metrics:**
- Number of strategies in shadow
- Shadow test win rate
- Shadow test profit factor
- Average shadow test duration

**Live Metrics:**
- Number of live strategies
- Live strategy P&L
- Live strategy drawdown
- Promotion/demotion rate

### Logging

All components use Python's `logging` module:

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Components log key events:
# - Strategy generation
# - Backtest results
# - Shadow test results
# - Promotion/demotion decisions
```

## 🔧 Production Checklist

- [ ] Implement concrete `StrategyRepository` (PostgreSQL/MongoDB)
- [ ] Implement concrete `MarketDataClient` (Binance/exchange API)
- [ ] Integrate with existing ensemble predictions
- [ ] Set production thresholds (stricter than examples)
- [ ] Set up continuous shadow testing (cron or async)
- [ ] Add monitoring/alerting (Prometheus, Grafana)
- [ ] Implement strategy versioning
- [ ] Add rollback capability for underperformers
- [ ] Create analytics dashboard
- [ ] Set up database backups
- [ ] Add circuit breakers for API failures
- [ ] Implement rate limiting
- [ ] Add unit/integration tests

## 🎓 Theory

### Why Genetic Algorithms?

Traditional optimization methods (grid search, Bayesian optimization) struggle with:
- **High dimensionality:** 8+ parameters to optimize
- **Non-convex fitness landscape:** Local optima everywhere
- **Discrete + continuous parameters:** Mix of floats, ints, enums, bools

Genetic algorithms excel at:
- **Exploration:** Mutation introduces randomness
- **Exploitation:** Crossover combines successful traits
- **Parallelization:** Evaluate entire population at once
- **Robustness:** No gradient required, handles noisy fitness

### Fitness Function Design

**Profit Factor (40%):**
```
score = min(pf / 2.0, 1.0)  # Target: PF ≥ 2.0
```

**Win Rate (20%):**
```
score = min(wr / 0.60, 1.0)  # Target: WR ≥ 60%
```

**Drawdown Penalty (20%):**
```
score = max(1.0 - dd * 5.0, 0.0)  # Penalize DD heavily
```

**Sample Size (20%):**
```
score = min(trades / 100.0, 1.0)  # Target: ≥ 100 trades
```

### Shadow Testing Rationale

**Why forward test?**
- Backtest overfitting is common
- Market conditions change
- Live execution differs from simulation

**How long to shadow test?**
- Min 7 days: Detect obvious failures
- Min 14 days: Recommended for promotion
- Max 30 days: Avoid staleness

**Virtual capital allocation:**
- $10,000 per strategy (typical)
- Independent of actual trading capital
- Prevents interference between strategies

## 📚 References

- [Genetic Algorithms in Trading](https://www.quantstart.com/articles/Genetic-Algorithms-in-Trading/)
- [Overfitting in Backtesting](https://www.investopedia.com/articles/trading/12/backtesting-survivorship-bias.asp)
- [Walk-Forward Optimization](https://www.amibroker.com/guide/w_optimization.html)
- [Strategy Evaluation Metrics](https://www.quantifiedstrategies.com/trading-strategy-performance-metrics/)

## 🤝 Contributing

See main Quantum Trader README for contribution guidelines.

## 📄 License

Part of Quantum Trader AI Hedge Fund OS.

---

**Built with ❤️ for autonomous trading**
