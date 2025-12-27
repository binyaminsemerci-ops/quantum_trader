# Strategy Generator AI - Quick Reference

## 🚀 One-Liner Commands

```bash
# Generate first population
python backend/research/examples/example_1_first_generation.py

# Run 5 generations
python backend/research/examples/example_2_evolutionary_loop.py

# Shadow test strategies
python backend/research/examples/example_3_shadow_testing.py

# Full pipeline demo
python backend/research/examples/example_4_full_pipeline.py

# Integration example
python backend/research/examples/example_5_integration.py
```

---

## 📂 File Structure

```
backend/research/
├── __init__.py              # Module exports
├── models.py                # Data structures (140 lines)
├── repositories.py          # Protocol interfaces (50 lines)
├── backtest.py              # Historical simulation (200 lines)
├── search.py                # Genetic algorithms (180 lines)
├── shadow.py                # Forward testing (150 lines)
├── deployment.py            # Promotion/demotion (200 lines)
├── README.md                # Full documentation (500 lines)
├── examples/
│   ├── example_1_first_generation.py    # (140 lines)
│   ├── example_2_evolutionary_loop.py   # (170 lines)
│   ├── example_3_shadow_testing.py      # (140 lines)
│   ├── example_4_full_pipeline.py       # (280 lines)
│   └── example_5_integration.py         # (380 lines)
```

**Total:** ~2,030 lines of production Python code

---

## 🧬 Core Classes

| Class | Purpose | Key Methods |
|-------|---------|-------------|
| `StrategyConfig` | Strategy specification | `.to_dict()`, `.from_dict()` |
| `StrategyStats` | Performance metrics | `.fitness_score` (auto-calculated) |
| `StrategyBacktester` | Historical simulation | `.backtest(config, symbols, start, end)` |
| `StrategySearchEngine` | Genetic algorithms | `.run_generation()`, `.evolve()` |
| `StrategyShadowTester` | Forward testing | `.run_shadow_loop()`, `.run_once()` |
| `StrategyDeploymentManager` | Promotion/demotion | `.review_and_promote()`, `.review_and_disable()` |

---

## 🔧 API Quick Reference

### Generate Strategies

```python
from backend.research import StrategySearchEngine, StrategyBacktester

search = StrategySearchEngine(backtester, repo)

strategies = search.run_generation(
    population_size=20,
    generation=1,
    symbols=["BTCUSDT", "ETHUSDT"],
    start_date=datetime(2024, 1, 1),
    end_date=datetime.utcnow(),
    parent_strategies=None  # First gen
)

# Returns list sorted by fitness (best first)
```

### Backtest Strategy

```python
from backend.research import StrategyBacktester

backtester = StrategyBacktester(market_data_client)

stats = backtester.backtest(
    config=strategy_config,
    symbols=["BTCUSDT", "ETHUSDT"],
    start_date=datetime(2024, 1, 1),
    end_date=datetime.utcnow()
)

print(f"PF: {stats.profit_factor:.2f}")
print(f"WR: {stats.win_rate:.1%}")
print(f"Fitness: {stats.fitness_score:.3f}")
```

### Shadow Test

```python
from backend.research import StrategyShadowTester

shadow_tester = StrategyShadowTester(repo, market_data)

# One-off
await shadow_tester.run_once(
    symbols=["BTCUSDT"],
    lookback_days=7
)

# Continuous
await shadow_tester.run_shadow_loop(
    symbols=["BTCUSDT"],
    lookback_days=7,
    interval_minutes=15
)
```

### Promote/Demote

```python
from backend.research import StrategyDeploymentManager

deployment = StrategyDeploymentManager(repo)

# Promote
promoted_ids = deployment.review_and_promote()
# Returns list of strategy_id strings

# Demote
disabled_ids = deployment.review_and_disable()
# Returns list of strategy_id strings
```

---

## ⚙️ Configuration Templates

### Conservative

```python
deployment = StrategyDeploymentManager(
    repository=repo,
    candidate_min_pf=1.8,     # High PF required
    candidate_min_trades=100,  # Large sample
    candidate_max_dd=0.15,     # Low DD tolerance
    shadow_min_pf=1.5,
    shadow_min_trades=50,
    shadow_min_days=30,        # Long forward test
    live_min_pf=1.3,
    live_max_dd=0.20,
    live_check_days=60
)
```

### Aggressive

```python
deployment = StrategyDeploymentManager(
    repository=repo,
    candidate_min_pf=1.3,     # Lower PF OK
    candidate_min_trades=30,   # Smaller sample
    candidate_max_dd=0.25,     # Higher DD tolerance
    shadow_min_pf=1.2,
    shadow_min_trades=15,
    shadow_min_days=7,         # Short forward test
    live_min_pf=1.0,
    live_max_dd=0.30,
    live_check_days=14
)
```

### Balanced (Default)

```python
deployment = StrategyDeploymentManager(
    repository=repo,
    candidate_min_pf=1.5,
    candidate_min_trades=50,
    candidate_max_dd=0.20,
    shadow_min_pf=1.3,
    shadow_min_trades=20,
    shadow_min_days=14,
    live_min_pf=1.1,
    live_max_dd=0.25,
    live_check_days=30
)
```

---

## 📊 Fitness Calculation

```python
# Automatic in StrategyStats
stats = StrategyStats(
    profit_factor=1.8,
    win_rate=0.62,
    max_drawdown_pct=0.12,
    total_trades=87,
    # ... other fields
)

# Auto-calculated property
print(stats.fitness_score)  # e.g., 0.782
```

**Formula:**
```
fitness = (
    0.40 × min(pf / 2.0, 1.0) +           # PF target: 2.0
    0.20 × min(wr / 0.60, 1.0) +          # WR target: 60%
    0.20 × max(1.0 - dd * 5.0, 0.0) +     # DD target: 0%
    0.20 × min(trades / 100.0, 1.0)       # Sample target: 100
)
```

---

## 🎯 Strategy Status Lifecycle

```
CANDIDATE → SHADOW → LIVE → DISABLED
    ↓
DISABLED (if backtest fails)
```

**Status meanings:**
- `CANDIDATE`: Just generated, backtest complete
- `SHADOW`: Forward testing on live data (paper trading)
- `LIVE`: Active trading with real capital
- `DISABLED`: Underperforming or failed validation

---

## 🔍 Common Queries

### Get all strategies by status

```python
from backend.research.models import StrategyStatus

candidates = repo.get_strategies_by_status(StrategyStatus.CANDIDATE)
shadow = repo.get_strategies_by_status(StrategyStatus.SHADOW)
live = repo.get_strategies_by_status(StrategyStatus.LIVE)
disabled = repo.get_strategies_by_status(StrategyStatus.DISABLED)
```

### Get strategy statistics

```python
# All stats
all_stats = repo.get_stats(strategy_id)

# Backtest only
backtest_stats = repo.get_stats(strategy_id, source="BACKTEST")

# Shadow only
shadow_stats = repo.get_stats(strategy_id, source="SHADOW")

# Last 7 days
recent_stats = repo.get_stats(strategy_id, days=7)
```

### Update strategy status

```python
from backend.research.models import StrategyStatus

repo.update_status(strategy_id, StrategyStatus.LIVE)
```

---

## 🐛 Troubleshooting

### "No strategies qualified for SHADOW"

**Cause:** Backtest performance below thresholds

**Solution:**
- Lower thresholds (relaxed mode)
- Increase population size (more candidates)
- Expand parameter ranges (more diversity)

### "Shadow test returned no stats"

**Cause:** Not enough data or trades

**Solution:**
- Increase `lookback_days` (e.g., 7 → 14)
- Lower `min_confidence` in strategies
- Check market data availability

### "Fitness scores all low (<0.5)"

**Cause:** Poor parameter ranges or overfitting

**Solution:**
- Adjust GA parameter ranges
- Increase sample size (longer backtest)
- Check commission rate (too high?)
- Verify market data quality

### "Strategies immediately disabled after promotion"

**Cause:** Overfitting to backtest period

**Solution:**
- Increase shadow test duration
- Stricter shadow promotion thresholds
- Use walk-forward optimization
- Check regime consistency

---

## 📈 Performance Targets

| Metric | Conservative | Balanced | Aggressive |
|--------|-------------|----------|------------|
| Profit Factor | ≥ 1.8 | ≥ 1.5 | ≥ 1.3 |
| Win Rate | ≥ 60% | ≥ 55% | ≥ 50% |
| Max Drawdown | ≤ 15% | ≤ 20% | ≤ 25% |
| Sample Size | ≥ 100 | ≥ 50 | ≥ 30 |
| Fitness Score | ≥ 0.80 | ≥ 0.65 | ≥ 0.50 |

---

## 🔗 Integration Checklist

### Quantum Trader Services

- [ ] `EnsembleOrchestrator` → Entry signals
- [ ] `TradingMathematician` → Position sizing
- [ ] `HybridTPSL` → TP/SL placement
- [ ] `PositionMonitor` → Live monitoring
- [ ] `RiskManager` → Risk constraints

### Data Sources

- [ ] Binance API → Historical OHLCV
- [ ] PostgreSQL → Strategy storage
- [ ] Redis → Caching (optional)
- [ ] InfluxDB → Time series (optional)

### Infrastructure

- [ ] Docker service
- [ ] Cron jobs (shadow testing)
- [ ] Monitoring (Prometheus/Grafana)
- [ ] Alerting (Slack/Discord)
- [ ] Logging (ELK stack)

---

## 📞 Quick Help

**Examples not working?**
→ Check stub implementations in `example_1_first_generation.py`

**How to add custom parameters?**
→ Extend `StrategyConfig` in `models.py`

**How to change fitness function?**
→ Modify `_calculate_fitness()` in `models.py`

**How to add new entry types?**
→ Update `EntryType` enum and `_check_entry()` in `backtest.py`

**How to optimize performance?**
→ Use multiprocessing for population evaluation

**Need more examples?**
→ See `backend/research/examples/` (5 working examples)

**Full documentation?**
→ See `backend/research/README.md` (500 lines)

---

**Built for Quantum Trader AI Hedge Fund OS** 🚀
