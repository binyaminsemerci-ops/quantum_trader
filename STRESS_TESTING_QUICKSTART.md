# Stress Testing System - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Start the Backend

```bash
cd c:\quantum_trader
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**Expected Output:**
```
INFO: [OK] Stress Testing API endpoints registered (/api/stress-testing/*)
INFO: Application startup complete.
```

### 2. Check System Status

```bash
curl http://localhost:8000/api/stress-testing/status
```

**Response:**
```json
{
  "available": true,
  "executor_initialized": true,
  "running_tests": 0,
  "completed_tests": 0
}
```

### 3. List Available Scenarios

```bash
curl http://localhost:8000/api/stress-testing/scenarios/library
```

**You'll see 10 pre-defined scenarios:**
- BTC Flash Crash 2021
- ETH Liquidity Collapse
- SOL Volatility Explosion
- Mixed Regime Chaos
- Model Failure Simulation
- Data Corruption Shock
- Execution Crisis
- Correlation Breakdown
- Pump & Dump Mania
- Black Swan Combo

### 4. Run Your First Stress Test

```bash
curl -X POST http://localhost:8000/api/stress-testing/scenarios/run \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My First Test",
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
  "scenario_name": "My First Test",
  "message": "Check /results/My First Test for results"
}
```

### 5. Get Results (wait 5 seconds)

```bash
curl "http://localhost:8000/api/stress-testing/results/My%20First%20Test"
```

**Response:**
```json
{
  "scenario_name": "My First Test",
  "success": true,
  "duration_sec": 2.45,
  "total_pnl": -1250.50,
  "total_pnl_pct": -1.25,
  "max_drawdown": 3.2,
  "sharpe_ratio": -0.45,
  "winrate": 45.5,
  "total_trades": 12,
  "emergency_stops": 0,
  "execution_failures": 2
}
```

---

## 🎯 Common Use Cases

### Test Flash Crash

```bash
curl -X POST http://localhost:8000/api/stress-testing/scenarios/run \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Flash Crash Test",
    "type": "flash_crash",
    "parameters": {"drop_pct": 0.30},
    "symbols": ["BTCUSDT", "ETHUSDT"]
  }'
```

### Test High Volatility

```bash
curl -X POST http://localhost:8000/api/stress-testing/scenarios/run \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Volatility Test",
    "type": "volatility_spike",
    "parameters": {"multiplier": 10.0},
    "symbols": ["SOLUSDT"]
  }'
```

### Test Liquidity Crisis

```bash
curl -X POST http://localhost:8000/api/stress-testing/scenarios/run \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Liquidity Test",
    "type": "liquidity_drop",
    "parameters": {"drop_pct": 0.95},
    "symbols": ["BTCUSDT"]
  }'
```

### Run Batch Test (3 scenarios)

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

---

## 📊 Understanding Results

### Key Metrics

| Metric | Good | Warning | Critical |
|--------|------|---------|----------|
| **Max Drawdown** | < 5% | 5-15% | > 15% |
| **Win Rate** | > 55% | 45-55% | < 45% |
| **Profit Factor** | > 1.5 | 1.0-1.5 | < 1.0 |
| **Sharpe Ratio** | > 1.0 | 0-1.0 | < 0 |
| **Emergency Stops** | 0 | 1-2 | > 2 |
| **Execution Failures** | < 5% | 5-15% | > 15% |

### Interpreting Results

**Scenario passed if:**
- ✅ No emergency stops triggered
- ✅ Drawdown within limits (< 15%)
- ✅ Positive profit factor
- ✅ Low execution failure rate

**Scenario failed if:**
- ❌ Emergency stop triggered
- ❌ Drawdown > 20%
- ❌ Profit factor < 0.5
- ❌ High execution failure rate (> 20%)

---

## 🔧 Python Integration

### Basic Usage

```python
import requests
import time

# Run test
response = requests.post(
    "http://localhost:8000/api/stress-testing/scenarios/run",
    json={
        "name": "Python Test",
        "type": "flash_crash",
        "parameters": {"drop_pct": 0.25},
        "symbols": ["BTCUSDT"]
    }
)

print(response.json())

# Wait for completion
time.sleep(5)

# Get results
result = requests.get(
    "http://localhost:8000/api/stress-testing/results/Python%20Test"
).json()

# Analyze
if result["emergency_stops"] > 0:
    print("⚠️ WARNING: Emergency stops triggered!")

if result["max_drawdown"] > 15.0:
    print(f"❌ FAIL: High drawdown: {result['max_drawdown']:.2f}%")
else:
    print(f"✅ PASS: Drawdown: {result['max_drawdown']:.2f}%")

print(f"Win Rate: {result['winrate']:.1f}%")
print(f"Total Trades: {result['total_trades']}")
```

### Batch Testing

```python
# Run multiple scenarios
response = requests.post(
    "http://localhost:8000/api/stress-testing/scenarios/batch",
    json={
        "scenario_names": [
            "BTC Flash Crash 2021",
            "Black Swan Combo"
        ],
        "parallel": True
    }
)

results = response.json()

# Summary
print(f"Total: {results['total_scenarios']}")
print(f"Success: {results['successful']}")
print(f"Failed: {results['failed']}")
print(f"Avg Win Rate: {results['avg_winrate']:.1f}%")
print(f"Worst DD: {results['worst_drawdown']:.2f}%")

# Check each result
for r in results['results']:
    print(f"\n{r['scenario_name']}:")
    print(f"  PnL: ${r['total_pnl']:.2f}")
    print(f"  Max DD: {r['max_drawdown']:.2f}%")
    print(f"  ESS: {r['emergency_stops']}")
```

---

## 📚 Available Scenario Types

### 1. flash_crash
Sudden price drop with recovery.

**Parameters:**
- `drop_pct` (float): Drop percentage (e.g., 0.20 = 20%)
- `duration_bars` (int): Crash duration in bars
- `recovery_bars` (int): Recovery duration in bars

### 2. volatility_spike
Multiply volatility by factor.

**Parameters:**
- `multiplier` (float): Volatility multiplier (e.g., 5.0)
- `duration_bars` (int): Spike duration

### 3. liquidity_drop
Reduce trading volume.

**Parameters:**
- `drop_pct` (float): Volume reduction (e.g., 0.90 = 90% drop)
- `duration_bars` (int): Drop duration

### 4. spread_explosion
Widen bid-ask spreads.

**Parameters:**
- `spread_mult` (float): Spread multiplier (e.g., 10.0)
- `duration_bars` (int): Duration

### 5. data_corruption
Inject data quality issues.

**Parameters:**
- `corruption_pct` (float): Percentage to corrupt (e.g., 0.10)
- `corruption_types` (list): ["nan", "spike", "duplicate"]

### 6. mixed_custom
Combine multiple stresses.

**Parameters:**
- Any combination of above parameters

---

## 🛠️ Troubleshooting

### Issue: Backend won't start

**Solution:**
```bash
# Check if port 8000 is in use
netstat -ano | findstr :8000

# Kill process if needed
taskkill /F /PID <process_id>

# Restart backend
python -m uvicorn backend.main:app --reload
```

### Issue: "No results found"

**Solution:** Wait longer or check status
```bash
# List all results
curl http://localhost:8000/api/stress-testing/results

# Check if test is still running
curl http://localhost:8000/api/stress-testing/status
```

### Issue: Scenarios failing

**Solution:** Check logs
```bash
# Backend logs show detailed errors
# Look for [STRESS-TEST] messages
```

---

## 📖 Next Steps

1. **Read full documentation:**
   - `backend/services/stress_testing/README.md`
   - `STRESS_TESTING_INTEGRATION.md`
   - `STRESS_TESTING_COMPLETE.md`

2. **Run comprehensive tests:**
   ```bash
   curl -X POST .../scenarios/batch -d '{
     "scenario_names": [...all 10 scenarios...],
     "parallel": true
   }'
   ```

3. **Create custom scenarios:**
   - Mix multiple stress conditions
   - Test specific market conditions
   - Validate system improvements

4. **Automate testing:**
   - Schedule weekly runs
   - Monitor metrics over time
   - Alert on regressions

---

## 🎓 Learn More

- **Examples:** `backend/services/stress_testing/examples.py`
- **Tests:** `tests/test_stress_testing.py`
- **API Docs:** http://localhost:8000/api/docs
- **Library Scenarios:** 10 pre-defined scenarios covering common stress conditions

---

**🚀 You're ready to stress test Quantum Trader!**

Start with simple scenarios and gradually increase complexity. The system will validate your trading system's robustness under extreme conditions.

**Happy Testing!** 🎯
