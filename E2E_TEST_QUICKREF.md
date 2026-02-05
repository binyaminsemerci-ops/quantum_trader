# E2E TEST - QUICK COMMAND REFERENCE

## 🚀 Run Full Test (One Command)

```bash
# Set credentials and run
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"
export TESTNET=true

# Run test
python run_e2e_test.py

# Watch output
tail -f e2e_test_report.json
```

---

## 📋 Test Phases at a Glance

| Phase | Command | Expected Time | Success Signal |
|-------|---------|----------------|-----------------|
| **1. Initialization** | Check env & backends | < 5s | ✅ Credentials OK |
| **2. Prediction** | Get AI signals | 5-10s | 🎯 3+ signals |
| **3. Signal Generation** | Filter & size | 2-3s | 📊 Position sizes calc |
| **4. Entry Logic** | Create orders | 1-2s | 🔧 Orders created |
| **5. Order Placement** | Place on exchange | 5-10s | 📤 Order IDs assigned |
| **6. Fill Verification** | Wait for fills | 10-30s | ✔️ Orders filled |
| **7. Position Monitor** | Check open positions | 3-5s | 📍 Positions found |
| **8. Profit Taking** | Set TP/SL | 5-10s | 🎯 TP/SL placed |
| **9. Settlement** | Close positions | 10-20s | 💰 Profit recorded |

**Total Expected Time:** 45-90 seconds

---

## 🔧 Setup Commands

### Pre-Flight Checks
```bash
# 1. Check environment
echo "API Key: $BINANCE_API_KEY"
echo "Testnet: $TESTNET"

# 2. Check backend
curl -s http://localhost:8000/health | python -m json.tool

# 3. Check AI engine
curl -s http://localhost:8001/health | python -m json.tool

# 4. Check Python version
python --version  # Need 3.8+

# 5. Install requirements
pip install -r requirements.txt
```

### Start Services (Required)
```bash
# Terminal 1: Backend
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000

# Terminal 2: AI Engine
cd ai_engine
python main.py
# OR
python -m uvicorn main:app --port 8001

# Terminal 3: Check health
curl http://localhost:8000/health && curl http://localhost:8001/health
```

---

## ✅ Run Test Variations

### Test 1: Single Symbol
```bash
# Edit test file, change to:
self.test_symbols = ["BTCUSDT"]
python test_e2e_prediction_to_profit.py
```

### Test 2: All Symbols
```bash
self.test_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
python run_e2e_test.py
```

### Test 3: With Debugging
```bash
python -u test_e2e_prediction_to_profit.py 2>&1 | tee debug.log
grep "ERROR\|FAIL" debug.log
```

### Test 4: Testnet Only
```bash
export TESTNET=true
python run_e2e_test.py
```

### Test 5: With Custom API Keys
```bash
export BINANCE_API_KEY="pk_test_..."
export BINANCE_API_SECRET="sk_test_..."
export TESTNET=true
python run_e2e_test.py
```

---

## 📊 Monitor Test in Real-Time

### In separate terminal(s):

```bash
# Watch test report JSON
watch -n 1 'cat e2e_test_report.json | python -m json.tool | tail -20'

# Watch backend logs
tail -f backend/logs/trading.log | grep -E "order|position|profit"

# Watch AI predictions
tail -f ai_engine/logs/predictions.log

# Monitor system resources
watch -n 1 'free -h && ps aux | grep python | grep -v grep'

# Check open positions
python check_positions.py --live

# Monitor orders
python check_open_orders.py --live
```

---

## 🎯 Verify Results

```bash
# View summary
cat e2e_test_report.json | python -c "
import json, sys
r = json.load(sys.stdin)
print('Status:', r['status'])
print('Duration:', r['duration_seconds'], 'seconds')
print('Tests Passed:', r['summary']['passed_tests'])
print('Tests Failed:', r['summary']['failed_tests'])
print('Trades Closed:', r['summary']['closed_trades'])
print('Total Profit:', r['summary']['total_profit'])
"

# Extract only summary
python -c "import json; print(json.dumps(json.load(open('e2e_test_report.json'))['summary'], indent=2))"

# Check for failures
grep -i "fail\|error" e2e_test_report.json || echo "✅ No errors"
```

---

## 🐛 Quick Troubleshooting

### Test Won't Start
```bash
# 1. Check backend alive
curl http://localhost:8000/health

# 2. Check Python syntax
python -m py_compile test_e2e_prediction_to_profit.py

# 3. Test imports
python -c "import numpy, pandas, requests, asyncio; print('✅ Imports OK')"
```

### Predictions Fail
```bash
# 1. Check AI engine
curl http://localhost:8001/health

# 2. Check market data available
python -c "import requests; print(requests.get('http://localhost:8000/prices/BTCUSDT').json())"

# 3. Use synthetic predictions (no AI needed)
# Edit test to use built-in synthetic predictions
```

### Orders Don't Fill
```bash
# 1. Check balance
python check_balance.py

# 2. Check bid-ask spread
python check_bid_ask.py BTCUSDT

# 3. Switch to market orders
# Edit test: "order_type": "MARKET"

# 4. Use testnet
export TESTNET=true
```

### Memory/CPU Issues
```bash
# Reduce test scope
# 1. Test one symbol: ["BTCUSDT"]
# 2. Reduce data: closes[:50] instead of closes[:100]
# 3. Sequential instead of parallel
```

---

## 💾 Save & Compare Results

```bash
# Save baseline
cp e2e_test_report.json e2e_test_baseline.json

# Run test again
python run_e2e_test.py

# Compare
diff -u e2e_test_baseline.json e2e_test_report.json

# See only summary changes
python -c "
import json
baseline = json.load(open('e2e_test_baseline.json'))
current = json.load(open('e2e_test_report.json'))
print('Baseline:', baseline['summary'])
print('Current:', current['summary'])
print('Change:', current['summary']['total_profit'] - baseline['summary']['total_profit'])
"
```

---

## 📈 Performance Benchmarks

Target metrics for successful test run:

```
✅ Initialization:      < 5 seconds
✅ Prediction:          5-10 seconds (1-3 predictions)
✅ Signal Generation:   2-3 seconds
✅ Entry Logic:         1-2 seconds
✅ Order Placement:     5-10 seconds (depends on API latency)
✅ Fill Verification:   10-30 seconds (depends on market)
✅ Position Monitor:    3-5 seconds
✅ Profit Taking:       5-10 seconds
✅ Settlement:          10-20 seconds

Total:                  45-90 seconds
```

If slower, check:
- Network latency: `ping api.binance.com`
- Backend performance: `curl http://localhost:8000/metrics`
- Database: `sqlite3 quantum_trader.db "SELECT COUNT(*) FROM trades"`

---

## 🎓 Understanding Output

### Log Messages
```
[INITIALIZATION] Starting initialization           # Phase starting
✅ PASS - Environment Check                       # Test passed
❌ FAIL - Order Placement                         # Test failed
[PREDICTION] Requesting prediction for BTCUSDT    # Sub-phase
Signal: BUY @ 87.50% confidence                   # Result info
Order ENTRY_BTCUSDT_... created                   # Action taken
```

### Report JSON Structure
```json
{
  "status": "SUCCESS|PARTIAL|FAILED",
  "summary": {
    "total_trades": 3,
    "closed_trades": 3,
    "total_profit": 124.56,
    "passed_tests": 18,
    "failed_tests": 0
  },
  "trades": {
    "TRADE_...": {
      "symbol": "BTCUSDT",
      "status": "CLOSED",
      "profit_pnl": 45.67,
      "profit_percent": 0.0234
    }
  }
}
```

---

## 🚨 Emergency Commands

```bash
# STOP ALL TRADING IMMEDIATELY
redis-cli SET quantum:global:kill_switch true

# Check if stopped
redis-cli GET quantum:global:kill_switch

# RESTART AFTER FIX
redis-cli SET quantum:global:kill_switch false

# CLEAR TEST DATA
python -c "
import sqlite3
db = sqlite3.connect('quantum_trader.db')
db.execute('DELETE FROM trades WHERE created_at > datetime(\"now\", \"-1 hour\")')
db.commit()
print('✅ Test trades cleared')
"

# RESET TO CLEAN STATE
rm e2e_test_report.json
rm -f quantum_trader.db
python backend/database_startup_validator.py
```

---

## 📞 Support

### Check Logs
```bash
grep "ERROR\|CRITICAL" backend/logs/*.log
grep "Exception" ai_engine/logs/*.log
```

### Collect Diagnostics
```bash
# Create support bundle
tar -czf quantum_e2e_test_debug.tar.gz \
  backend/logs/ \
  ai_engine/logs/ \
  e2e_test_report.json \
  e2e_test_debug.log \
  .env

echo "📦 Support bundle created: quantum_e2e_test_debug.tar.gz"
```

---

## ✨ Success Checklist

- [ ] Backend running on port 8000
- [ ] AI Engine running on port 8001  
- [ ] API credentials configured
- [ ] Testnet enabled (recommended)
- [ ] Requirements installed
- [ ] Python 3.8+ available
- [ ] Test script executable

**Ready?** Run: `python run_e2e_test.py`

🎯 **Goal:** All phases complete → Status: SUCCESS → Report saved
