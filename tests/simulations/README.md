# 🎯 FAILURE SIMULATION - QUICK REFERENCE

## Run Tests

```bash
# All scenarios
pytest tests/simulations/ -v -s

# Master runner with report
python tests/simulations/run_all_scenarios.py

# Individual scenarios
pytest tests/simulations/test_flash_crash.py -v -s
pytest tests/simulations/test_redis_down.py -v -s
pytest tests/simulations/test_binance_down.py -v -s
pytest tests/simulations/test_signal_flood.py -v -s
pytest tests/simulations/test_ess_trigger.py -v -s
```

## Scenario Summary

| Scenario | Config | Key Metrics |
|----------|--------|-------------|
| **Flash Crash** | 15% drop / 60s | drawdown_percent, ess_tripped |
| **Redis Down** | 60s downtime | buffered_events, flush_duration |
| **Binance Down** | -1003/-1015 errors | binance_errors, avg_retries |
| **Signal Flood** | 50 signals / 5s | queue_lag, trade_intents |
| **ESS Trigger** | 12% loss | ess_trips, trades_blocked |

## Expected Results

- ✅ All scenarios PASSED
- ✅ 25+ checks passed
- ✅ 0 checks failed
- ✅ Pass rate: 100%

## Key Files

```
tests/simulations/
├── harness.py              # 900+ lines, core harness
├── run_all_scenarios.py    # Master runner
├── test_*.py               # 5 test files, 7+ tests each
└── simulation_report.json  # Generated report
```

## Harness Usage

```python
from tests.simulations.harness import (
    FailureSimulationHarness,
    FlashCrashConfig
)

# Create harness
harness = FailureSimulationHarness()

# Run scenario
result = await harness.run_flash_crash_scenario(
    FlashCrashConfig(price_drop_percent=15.0)
)

# Check result
assert result.status == ScenarioStatus.PASSED
assert result.checks_failed == 0

# Get summary
summary = harness.get_summary_report()
```

## Customization

```python
# Custom flash crash
config = FlashCrashConfig(
    symbols=["BTCUSDT", "ETHUSDT", "BNBUSDT"],
    price_drop_percent=20.0,  # More severe
    duration_seconds=30.0,     # Faster crash
    ess_drawdown_threshold=15.0 # Higher threshold
)

# Custom signal flood
config = SignalFloodConfig(
    signal_count=100,          # More signals
    publish_interval_ms=50.0,  # Faster rate
    max_queue_lag_seconds=10.0 # Higher tolerance
)
```

## Metrics Reference

```python
# Global metrics tracked across all scenarios
metrics = {
    "events_published": 0,
    "disk_buffer_writes": 0,
    "redis_reconnects": 0,
    "ess_trips": 0,
    "trades_blocked": 0,
    "binance_errors": 0,
    "health_alerts": 0
}
```

## Troubleshooting

**All tests fail with import errors**:
```bash
# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# Windows PowerShell
$env:PYTHONPATH="C:\quantum_trader;$env:PYTHONPATH"
```

**Slow test execution**:
- Reduce `downtime_seconds`, `duration_seconds` in configs
- Run individual scenarios instead of full suite

**Mock errors**:
- Ensure `unittest.mock` is available
- Update mock setup in harness `__init__`

## Next Steps

1. ✅ Run full test suite: `python tests/simulations/run_all_scenarios.py`
2. ⏳ Integrate with CI/CD (GitHub Actions)
3. ⏳ Replace mocks with real components
4. ⏳ Implement advanced scenarios (multi-symbol, cascading failures)
5. ⏳ Add chaos engineering (random failure injection)
