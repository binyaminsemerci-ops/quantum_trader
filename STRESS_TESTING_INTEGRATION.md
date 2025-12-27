# Stress Testing System Integration Guide

## Overview

The Stress Testing System (SST) is now fully integrated with Quantum Trader, providing comprehensive robustness validation under extreme market conditions.

## Architecture

### Integration Points

```
Quantum Trader Backend
├── API Layer (/api/stress-testing/*)
│   ├── POST /scenarios/run - Run single scenario
│   ├── POST /scenarios/batch - Run batch tests
│   ├── GET /scenarios/library - List scenarios
│   ├── GET /results/{name} - Get results
│   └── GET /status - System status
│
├── Component Adapters (quantum_trader_adapter.py)
│   ├── RuntimeEngineAdapter - Strategy signal generation
│   ├── OrchestratorAdapter - Signal filtering
│   ├── RiskGuardAdapter - Pre-trade validation
│   ├── PortfolioBalancerAdapter - Portfolio constraints
│   ├── SafetyGovernorAdapter - Emergency stops
│   ├── MSCAdapter - Meta strategy control
│   ├── PolicyStoreAdapter - Global policy
│   └── EventBusAdapter - Event distribution
│
└── SST Core Components
    ├── ScenarioLoader - Data loading/generation
    ├── ScenarioTransformer - Stress transformations
    ├── ExchangeSimulator - Execution simulation
    ├── ScenarioExecutor - System simulation
    ├── StressTestRunner - Batch orchestration
    └── ScenarioLibrary - Pre-defined scenarios
```

## Quick Start

### 1. Start Backend

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. List Available Scenarios

```bash
curl http://localhost:8000/api/stress-testing/scenarios/library
```

**Response:**
```json
[
  {
    "name": "BTC Flash Crash 2021",
    "type": "flash_crash",
    "description": "flash_crash scenario: BTC Flash Crash 2021",
    "parameters": {
      "drop_pct": 0.3,
      "duration_bars": 5,
      "recovery_bars": 15
    }
  },
  ...
]
```

### 3. Run Single Scenario

```bash
curl -X POST http://localhost:8000/api/stress-testing/scenarios/run \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Flash Crash",
    "type": "flash_crash",
    "parameters": {
      "drop_pct": 0.20,
      "duration_bars": 5
    },
    "symbols": ["BTCUSDT"]
  }'
```

**Response:**
```json
{
  "status": "submitted",
  "scenario_name": "Test Flash Crash",
  "message": "Stress test 'Test Flash Crash' started. Check /results/Test Flash Crash for results."
}
```

### 4. Get Results

```bash
curl http://localhost:8000/api/stress-testing/results/Test%20Flash%20Crash
```

**Response:**
```json
{
  "scenario_name": "Test Flash Crash",
  "success": true,
  "duration_sec": 2.45,
  "total_pnl": -1250.50,
  "total_pnl_pct": -1.25,
  "max_drawdown": 3.2,
  "sharpe_ratio": -0.45,
  "profit_factor": 0.85,
  "winrate": 45.5,
  "total_trades": 12,
  "winning_trades": 5,
  "losing_trades": 7,
  "emergency_stops": 0,
  "execution_failures": 2,
  "failed_models": [],
  "failed_strategies": [],
  "equity_curve_sample": [100000, 99500, 98800, ...],
  "notes": "Completed successfully"
}
```

### 5. Run Batch Tests

```bash
curl -X POST http://localhost:8000/api/stress-testing/scenarios/batch \
  -H "Content-Type: application/json" \
  -d '{
    "scenario_names": [
      "BTC Flash Crash 2021",
      "ETH Liquidity Collapse",
      "SOL Volatility Explosion"
    ],
    "parallel": true,
    "max_workers": 4
  }'
```

**Response:**
```json
{
  "total_scenarios": 3,
  "successful": 3,
  "failed": 0,
  "total_duration_sec": 8.23,
  "total_trades": 47,
  "avg_winrate": 52.3,
  "avg_max_drawdown": 5.4,
  "worst_drawdown": 8.2,
  "scenarios_with_ess": 0,
  "results": [...]
}
```

## Scenario Types

### 1. Flash Crash
Sudden price drop with recovery.

```json
{
  "type": "flash_crash",
  "parameters": {
    "drop_pct": 0.20,        // 20% drop
    "duration_bars": 5,      // Crash duration
    "recovery_bars": 15      // Recovery time
  }
}
```

### 2. Volatility Spike
Multiply volatility by factor.

```json
{
  "type": "volatility_spike",
  "parameters": {
    "multiplier": 5.0,       // 5x volatility
    "duration_bars": 20      // Spike duration
  }
}
```

### 3. Liquidity Drop
Reduce trading volume.

```json
{
  "type": "liquidity_drop",
  "parameters": {
    "drop_pct": 0.90,        // 90% volume drop
    "duration_bars": 15      // Drop duration
  }
}
```

### 4. Spread Explosion
Widen bid-ask spreads.

```json
{
  "type": "spread_explosion",
  "parameters": {
    "spread_mult": 10.0,     // 10x wider spreads
    "duration_bars": 10      // Duration
  }
}
```

### 5. Data Corruption
Inject data quality issues.

```json
{
  "type": "data_corruption",
  "parameters": {
    "corruption_pct": 0.10,  // 10% corrupted
    "corruption_types": ["nan", "spike", "duplicate"]
  }
}
```

### 6. Mixed Custom
Combine multiple stresses.

```json
{
  "type": "mixed_custom",
  "parameters": {
    "flash_crash_drop_pct": 0.30,
    "volatility_multiplier": 8.0,
    "spread_mult": 15.0,
    "liquidity_drop_pct": 0.95
  }
}
```

## Python Integration

### Using the API

```python
import requests

# List scenarios
response = requests.get("http://localhost:8000/api/stress-testing/scenarios/library")
scenarios = response.json()

# Run scenario
scenario = {
    "name": "My Test",
    "type": "flash_crash",
    "parameters": {"drop_pct": 0.25},
    "symbols": ["BTCUSDT", "ETHUSDT"]
}

response = requests.post(
    "http://localhost:8000/api/stress-testing/scenarios/run",
    json=scenario
)
print(response.json())

# Get results
import time
time.sleep(5)  # Wait for completion

response = requests.get("http://localhost:8000/api/stress-testing/results/My%20Test")
result = response.json()

print(f"PnL: ${result['total_pnl']:.2f}")
print(f"Max DD: {result['max_drawdown']:.2f}%")
print(f"Win Rate: {result['winrate']:.1f}%")
```

### Direct Component Usage

```python
from backend.services.stress_testing import (
    Scenario,
    ScenarioType,
    ScenarioLibrary,
    create_quantum_trader_executor
)
from backend.services.stress_testing import StressTestRunner

# Get app state (in FastAPI context)
from backend.main import app

# Create executor with real components
executor = create_quantum_trader_executor(
    app_state=app.state,
    initial_capital=100000.0
)

# Get scenario from library
scenario = ScenarioLibrary.btc_flash_crash_2021()

# Run test
runner = StressTestRunner(executor=executor)
results = runner.run_batch([scenario])

# Analyze
result = results[scenario.name]
print(f"Emergency Stops: {result.emergency_stops}")
print(f"Failed Models: {result.failed_models}")
print(f"Execution Failures: {result.execution_failures}")
```

## Component Behavior Under Stress

### RuntimeEngine
- Continues generating signals under stress
- May produce lower confidence signals during volatility
- Adapts to changing market conditions

### Orchestrator
- Filters signals based on stress conditions
- Reduces position sizes under high volatility
- Blocks trades during data corruption

### RiskGuard
- Enforces stricter limits under stress
- Blocks trades if max positions reached
- Validates position sizes against equity

### PortfolioBalancer
- Maintains diversification constraints
- Reduces max positions during stress
- Enforces concentration limits

### SafetyGovernor
- Activates emergency stop if DD > 30%
- Halts trading if equity drops too low
- Monitors system health continuously

### MSC (Meta Strategy Controller)
- Adjusts policy based on performance
- Switches to DEFENSIVE mode if win rate < 40%
- Switches to AGGRESSIVE if win rate > 60%

## Output Metrics

### Performance Metrics
- `total_pnl`: Total profit/loss ($)
- `total_pnl_pct`: P&L as percentage (%)
- `max_drawdown`: Worst peak-to-trough decline (%)
- `sharpe_ratio`: Risk-adjusted returns
- `profit_factor`: Gross profit / gross loss
- `winrate`: Winning trades / total trades (%)

### Trade Statistics
- `total_trades`: Number of trades
- `winning_trades`: Profitable trades
- `losing_trades`: Unprofitable trades
- `avg_win`: Average winning trade
- `avg_loss`: Average losing trade

### System Health
- `emergency_stops`: ESS activations
- `execution_failures`: Failed orders
- `failed_models`: Models that crashed
- `failed_strategies`: Strategies that failed
- `policy_transitions`: MSC policy changes

### Curves
- `equity_curve`: Equity over time
- `pnl_curve`: Cumulative P&L over time
- `drawdown_curve`: Drawdown over time

## Best Practices

### 1. Start Small
Test with single scenarios before batch tests:
```bash
# Test one scenario first
curl -X POST .../scenarios/run -d '{"name": "Test", ...}'
```

### 2. Use Library Scenarios
Pre-built scenarios are well-tested:
```bash
curl -X POST .../scenarios/batch -d '{
  "scenario_names": ["BTC Flash Crash 2021"],
  "parallel": false
}'
```

### 3. Monitor Results
Check system health metrics:
```python
if result.emergency_stops > 0:
    print("WARNING: Emergency stops triggered")

if result.execution_failures > 10:
    print("WARNING: High execution failure rate")
```

### 4. Iterate and Refine
Use results to improve system:
- Identify weak scenarios
- Adjust risk parameters
- Enhance fail-safes

## Troubleshooting

### Issue: "Scenario not found"
**Solution:** List available scenarios first
```bash
curl http://localhost:8000/api/stress-testing/scenarios/library
```

### Issue: "No results found"
**Solution:** Wait for test to complete (async endpoint)
```bash
# Check status first
curl http://localhost:8000/api/stress-testing/results

# Then get specific result
curl http://localhost:8000/api/stress-testing/results/{name}
```

### Issue: High execution failures
**Solution:** Check exchange simulator settings
- Reduce failure rates in stress scenarios
- Increase latency tolerance
- Adjust slippage caps

### Issue: All scenarios failing
**Solution:** Check component initialization
```bash
curl http://localhost:8000/api/stress-testing/status
```

## Next Steps

1. **Run Comprehensive Suite**
   ```bash
   curl -X POST .../scenarios/batch -d '{
     "scenario_names": [
       "BTC Flash Crash 2021",
       "ETH Liquidity Collapse",
       "SOL Volatility Explosion",
       "Black Swan Combo"
     ],
     "parallel": true
   }'
   ```

2. **Create Custom Scenarios**
   ```python
   scenario = Scenario(
       name="My Custom Test",
       type=ScenarioType.MIXED_CUSTOM,
       parameters={
           "flash_crash_drop_pct": 0.40,
           "volatility_multiplier": 10.0
       }
   )
   ```

3. **Automate Testing**
   - Schedule weekly stress tests
   - Monitor metrics over time
   - Alert on regressions

4. **Extend System**
   - Add new scenario types
   - Enhance component adapters
   - Integrate with CI/CD

## Support

For issues or questions:
- Check logs: `/app/logs/`
- Review documentation: `backend/services/stress_testing/README.md`
- See examples: `backend/services/stress_testing/examples.py`

---

**Integration Status:** ✅ Complete  
**Last Updated:** November 30, 2025
