# Stress Testing System (SST)

## Overview

The **Scenario & Stress Testing (SST)** system enables comprehensive robustness validation of Quantum Trader by simulating extreme market conditions, historical crashes, and synthetic stress scenarios. It tests all system components under duress to identify vulnerabilities and validate fail-safes.

## Purpose

SST validates that Quantum Trader can:
- Survive flash crashes and extreme volatility
- Handle liquidity droughts
- Recover from data corruption
- Manage execution failures
- Activate emergency stops appropriately
- Adapt policies under stress
- Maintain model performance under duress

## Architecture

### Core Components

#### 1. **Scenario** (Data Model)
Defines a stress test scenario with:
- Type (flash crash, volatility spike, data corruption, etc.)
- Parameters (drop percentage, multipliers, durations)
- Symbols to test
- Date range (for historical replays)

```python
scenario = Scenario(
    name="BTC Flash Crash",
    type=ScenarioType.FLASH_CRASH,
    parameters={"drop_pct": 0.20, "duration_bars": 5},
    symbols=["BTCUSDT"]
)
```

#### 2. **ScenarioLoader**
Loads or generates market data:
- Historical data from exchange (for replays)
- Synthetic data with realistic properties (GBM + volatility clustering)
- Multi-symbol support
- Data quality validation

#### 3. **ScenarioTransformer**
Applies stress conditions to data:
- **Flash Crash**: Sudden price drops with recovery
- **Volatility Spike**: Multiply ATR by factor
- **Liquidity Drop**: Reduce volume by percentage
- **Spread Explosion**: Widen bid-ask spreads
- **Data Corruption**: Inject NaNs, spikes, duplicates
- **Correlation Breakdown**: Break multi-asset correlations
- **Pump & Dump**: Artificial price manipulation
- **Mixed Custom**: Combine multiple stresses

#### 4. **ExchangeSimulator**
Simulates realistic order execution:
- Slippage based on volatility and liquidity
- Spread costs
- Execution failures
- Partial fills
- Latency spikes
- Adapts to stress conditions

#### 5. **ScenarioExecutor**
Runs full system simulation:
- Steps through market data bar-by-bar
- Feeds strategy runtime engine
- Orchestrator filters signals
- Risk layers validate trades
- Simulates executions
- Tracks PnL and equity
- Monitors emergency stops
- Records policy transitions

#### 6. **StressTestRunner**
Orchestrates batch testing:
- Parallel scenario execution
- Aggregates results
- Generates summary statistics
- Identifies weakest scenarios

#### 7. **ScenarioLibrary**
Pre-defined stress scenarios:
- BTC Flash Crash 2021
- ETH Liquidity Collapse
- SOL Volatility Explosion
- Mixed Regime Chaos
- Model Failure Simulation
- Data Corruption Shock
- Execution Crisis
- Black Swan Combo

## Usage

### Basic Example

```python
from backend.services.stress_testing import (
    Scenario,
    ScenarioType,
    ScenarioExecutor,
    StressTestRunner
)

# Create scenario
scenario = Scenario(
    name="Test Crash",
    type=ScenarioType.FLASH_CRASH,
    parameters={"drop_pct": 0.20}
)

# Setup executor
executor = ScenarioExecutor(
    runtime_engine=my_strategy_engine,
    orchestrator=my_orchestrator,
    risk_guard=my_risk_guard,
    safety_governor=my_safety_governor,
    initial_capital=100000.0
)

# Create runner
runner = StressTestRunner(executor=executor)

# Run test
results = runner.run_batch([scenario])

# Print results
runner.print_summary(results)
```

### Batch Testing

```python
from backend.services.stress_testing import ScenarioLibrary

# Get library scenarios
scenarios = [
    ScenarioLibrary.btc_flash_crash_2021(),
    ScenarioLibrary.eth_liquidity_collapse(),
    ScenarioLibrary.sol_volatility_explosion()
]

# Run batch
results = runner.run_batch(scenarios, parallel=True)

# Analyze
summary = runner.generate_summary(results)
print(f"Avg Max Drawdown: {summary['avg_max_drawdown']:.2f}%")
print(f"Emergency Stops: {summary['total_emergency_stops']}")
```

### Custom Scenario

```python
# Complex multi-stress scenario
scenario = Scenario(
    name="Black Swan",
    type=ScenarioType.MIXED_CUSTOM,
    parameters={
        "flash_crash_drop_pct": 0.30,
        "volatility_multiplier": 8.0,
        "spread_mult": 15.0,
        "liquidity_drop_pct": 0.95
    },
    symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"]
)

results = runner.run_batch([scenario])
```

### Historical Replay

```python
from datetime import datetime

scenario = Scenario(
    name="May 2021 Crash",
    type=ScenarioType.HISTORIC_REPLAY,
    start=datetime(2021, 5, 1),
    end=datetime(2021, 5, 31),
    symbols=["BTCUSDT"]
)

results = runner.run_batch([scenario])
```

## Result Analysis

### ScenarioResult Output

```python
result = results["Test Crash"]

# Performance metrics
print(f"Total Trades: {result.total_trades}")
print(f"Win Rate: {result.winrate:.1f}%")
print(f"Max Drawdown: {result.max_drawdown:.2f}%")
print(f"Profit Factor: {result.profit_factor:.2f}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")

# System behavior
print(f"Emergency Stops: {result.emergency_stops}")
print(f"Execution Failures: {result.execution_failures}")
print(f"Failed Models: {result.failed_models}")
print(f"Policy Transitions: {len(result.policy_transitions)}")

# Curves
import matplotlib.pyplot as plt
plt.plot(result.equity_curve)
plt.title("Equity Curve Under Stress")
plt.show()
```

## Scenario Types

### 1. Historic Replay
Replays real historical data.
```python
ScenarioType.HISTORIC_REPLAY
Parameters: start, end dates
```

### 2. Flash Crash
Sudden price drop with recovery.
```python
ScenarioType.FLASH_CRASH
Parameters:
  - drop_pct: Drop percentage (e.g., 0.20 = 20%)
  - duration_bars: Crash duration
  - recovery_bars: Recovery duration
```

### 3. Volatility Spike
Multiply volatility by factor.
```python
ScenarioType.VOLATILITY_SPIKE
Parameters:
  - multiplier: Volatility multiplier (e.g., 5.0)
  - duration_bars: Spike duration
```

### 4. Liquidity Drop
Reduce trading volume.
```python
ScenarioType.LIQUIDITY_DROP
Parameters:
  - drop_pct: Volume reduction (e.g., 0.90 = 90% drop)
  - duration_bars: Drop duration
```

### 5. Spread Explosion
Widen bid-ask spreads.
```python
ScenarioType.SPREAD_EXPLOSION
Parameters:
  - spread_mult: Spread multiplier (e.g., 10.0)
  - duration_bars: Explosion duration
```

### 6. Data Corruption
Inject data quality issues.
```python
ScenarioType.DATA_CORRUPTION
Parameters:
  - corruption_pct: Percentage to corrupt (e.g., 0.10)
  - corruption_types: ["nan", "spike", "duplicate"]
```

### 7. Mixed Custom
Combine multiple stresses.
```python
ScenarioType.MIXED_CUSTOM
Parameters: Any combination of above
```

## Integration with Quantum Trader

### Required Components

SST integrates with these Quantum Trader components:

```python
executor = ScenarioExecutor(
    runtime_engine=strategy_runtime_engine,    # Generates signals
    orchestrator=orchestrator,                  # Filters signals
    risk_guard=risk_guard,                      # Pre-trade validation
    portfolio_balancer=portfolio_balancer,      # Portfolio constraints
    safety_governor=safety_governor,            # Emergency stops
    msc=meta_strategy_controller,               # Policy controller
    policy_store=policy_store,                  # Global policy
    exchange_simulator=exchange_simulator,      # Order execution
    event_bus=event_bus                         # Event distribution
)
```

### Mock Components for Testing

For standalone testing without full Quantum Trader:

```python
class MockStrategyEngine:
    def generate_signals(self, bar, context):
        # Return mock signals
        return []

class MockOrchestrator:
    def evaluate_signal(self, signal, context):
        # Return trade decision
        return decision

# Similar for other components
```

See `examples.py` for complete mock implementations.

## Configuration

### Exchange Simulator Settings

```python
simulator = ExchangeSimulator(
    base_slippage_pct=0.001,   # 0.1% base slippage
    base_latency_ms=50.0,       # 50ms latency
    failure_rate=0.001          # 0.1% failure rate
)
```

### Executor Settings

```python
executor = ScenarioExecutor(
    initial_capital=100000.0    # Starting capital
)
```

### Runner Settings

```python
runner = StressTestRunner(
    executor=executor,
    max_workers=4               # Parallel scenarios
)
```

## Best Practices

### 1. Start Simple
Begin with single scenarios before batch testing:
```python
# Test one scenario first
results = runner.run_batch([scenario], parallel=False)
```

### 2. Use Deterministic Seeds
For reproducibility:
```python
scenario = Scenario(
    name="Test",
    type=ScenarioType.FLASH_CRASH,
    seed=42  # Fixed seed
)
```

### 3. Validate Data Quality
Check loaded data:
```python
loader = ScenarioLoader()
df = loader.load_data(scenario)
is_valid, issues = loader.validate_data(df)
if not is_valid:
    print(f"Data issues: {issues}")
```

### 4. Monitor Resource Usage
Limit parallel workers for large batches:
```python
runner = StressTestRunner(
    executor=executor,
    max_workers=2  # Reduce for memory constraints
)
```

### 5. Analyze Failures
Review failed scenarios:
```python
for name, result in results.items():
    if not result.success:
        print(f"Failed: {name}")
        print(f"Notes: {result.notes}")
```

## Performance Metrics

### Primary Metrics
- **Win Rate**: Percentage of profitable trades
- **Max Drawdown**: Worst peak-to-trough decline
- **Profit Factor**: Gross profit / gross loss
- **Sharpe Ratio**: Risk-adjusted returns

### System Health Metrics
- **Emergency Stops**: ESS activation count
- **Execution Failures**: Failed order count
- **Failed Models**: Models that crashed
- **Failed Strategies**: Strategies that failed
- **Data Quality Issues**: Corrupted bar count

### Risk Metrics
- **Max DD Duration**: Longest drawdown period
- **Policy Transitions**: MSC policy change count
- **Latency Spikes**: High-latency execution count

## Testing Workflow

### 1. Development Testing
```python
# Quick single scenario
scenario = ScenarioLibrary.btc_flash_crash_2021()
results = runner.run_batch([scenario])
```

### 2. Pre-Production Validation
```python
# Run comprehensive suite
scenarios = ScenarioLibrary.get_all()
results = runner.run_batch(scenarios, parallel=True)
summary = runner.generate_summary(results)

# Check thresholds
if summary['avg_max_drawdown'] > 20.0:
    print("WARNING: High average drawdown")
if summary['total_emergency_stops'] > 10:
    print("WARNING: Frequent ESS activations")
```

### 3. Continuous Monitoring
```python
# Run weekly stress tests
def weekly_stress_test():
    scenarios = ScenarioLibrary.get_by_type(ScenarioType.FLASH_CRASH)
    results = runner.run_batch(scenarios)
    return runner.generate_summary(results)
```

## Examples

Full examples available in `examples.py`:
- Single scenario test
- Batch testing with library
- Custom mixed scenario
- Historical replay
- Comprehensive suite

Run examples:
```bash
python -m backend.services.stress_testing.examples
```

## Testing

Unit tests in `tests/test_stress_testing.py`:
```bash
pytest tests/test_stress_testing.py -v
```

## Limitations

1. **Synthetic Data**: Generated data may not capture all market nuances
2. **Simplified Execution**: Real exchange behavior more complex
3. **No Order Book**: Simplified liquidity model
4. **Single Timeframe**: Currently hourly bars
5. **Linear Simulation**: Bar-by-bar, not tick-by-tick

## Future Enhancements

- [ ] Multi-timeframe support
- [ ] Order book simulation
- [ ] Tick-level replay
- [ ] Real exchange API integration
- [ ] Machine learning stress detection
- [ ] Automated scenario generation
- [ ] Visual dashboard
- [ ] Report generation (PDF/HTML)
- [ ] Database persistence
- [ ] API endpoints

## License

Part of Quantum Trader - AI Hedge Fund OS

---

**Contact**: For questions or issues, refer to main Quantum Trader documentation.
