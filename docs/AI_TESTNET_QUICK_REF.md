# 🧪 Testnet Simulation - Quick Reference

**Status:** ✅ Deployed (Partial Success)  
**Mode:** Simulation Only (No Real Trades)  
**VPS:** 46.224.116.254  
**Date:** December 17, 2025

---

## 🎯 Quick Start

### Run Simulation

```bash
ssh qt@46.224.116.254

# Copy script to AI Engine container
docker cp ~/quantum_trader/scripts/run_testnet_simulation.py \
  quantum_ai_engine:/tmp/

# Execute simulation
docker exec quantum_ai_engine \
  python3 /tmp/run_testnet_simulation.py
```

**Expected Output:**
- ✅ Exit Brain V3 initialized
- ✅ TP Optimizer V3 initialized
- 🟡 Some components in mock mode
- Overall: PARTIAL SUCCESS

---

## 📊 Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Exit Brain V3 | ✅ Working | Fully initialized |
| TP Optimizer V3 | ✅ Working | Fully initialized |
| RL Environment V3 | 🟡 Mock | Needs `gymnasium` |
| Execution Engine | 🟡 Mock | Module import issue |

**Success Rate:** 56% (5/9 steps)

---

## ⚙️ Configuration

### .env Settings
```bash
GO_LIVE=false
SIMULATION_MODE=true
BINANCE_TESTNET=true
EXIT_MODE=EXIT_BRAIN_V3
EXIT_BRAIN_PROFILE=CHALLENGE_100
RL_DEBUG=true
```

### go_live.yaml Settings
```yaml
environment: testnet
activation_enabled: false
simulate_orders: true
risk_mode: sandbox
```

---

## 🔍 Check Results

### View Simulation Results
```bash
# Latest simulation JSON
cat /home/qt/quantum_trader/status/testnet_simulation_*.json | tail -100

# Show in container
docker exec quantum_ai_engine \
  cat /home/qt/quantum_trader/status/testnet_simulation_*.json
```

### Check Component Status
```bash
# Exit Brain V3
docker exec quantum_ai_engine \
  python3 -c "from backend.domains.exits.exit_brain_v3 import ExitBrainV3; print('✅ Exit Brain V3 OK')"

# TP Optimizer V3
docker exec quantum_ai_engine \
  python3 -c "from backend.services.monitoring.tp_optimizer_v3 import TPOptimizerV3; print('✅ TP Optimizer V3 OK')"
```

---

## 🔧 Fix Issues

### Install Missing Dependencies
```bash
# Install gymnasium for RL Environment
docker exec quantum_ai_engine pip install gymnasium

# Install python-binance for API calls
docker exec quantum_ai_engine pip install python-binance

# Verify installations
docker exec quantum_ai_engine pip list | grep -E '(gymnasium|binance)'
```

### Fix Async Issues
Create async version of simulation script that uses:
```python
async def run_simulation():
    plan = await exit_brain.build_exit_plan(ctx)
    rec = await tp_opt.evaluate_profile(strategy, symbol)
```

---

## 🛡️ Safety Verification

### Check Safety Settings
```bash
# Verify simulation mode
docker exec quantum_ai_engine \
  python3 -c "import os; print('GO_LIVE:', os.getenv('GO_LIVE')); print('SIMULATION_MODE:', os.getenv('SIMULATION_MODE'))"

# Should output:
# GO_LIVE: false
# SIMULATION_MODE: true
```

### Verify No Live Trading
```bash
# Check go_live.yaml
cat ~/quantum_trader/config/go_live.yaml | grep -E '(environment|simulate|risk_mode)'

# Should show:
# environment: testnet
# simulate_orders: true
# risk_mode: sandbox
```

---

## 📈 Monitoring

### Dashboard
http://46.224.116.254:8080

**Check:**
- AI Engine status (should be healthy)
- Recent audit entries
- System metrics

### Logs
```bash
# AI Engine logs
docker logs quantum_ai_engine --tail 50

# Audit log
tail -20 ~/quantum_trader/status/AUTO_REPAIR_AUDIT.log

# Simulation results
ls -lh ~/quantum_trader/status/testnet_simulation_*.json
```

---

## 🎯 Next Steps

### Priority 1: Fix Dependencies
```bash
docker exec quantum_ai_engine pip install gymnasium python-binance
```

### Priority 2: Create Async Version
Update simulation script to use `async/await` properly

### Priority 3: Re-run Simulation
```bash
docker cp ~/quantum_trader/scripts/run_testnet_simulation_v2.py \
  quantum_ai_engine:/tmp/
docker exec quantum_ai_engine \
  python3 /tmp/run_testnet_simulation_v2.py
```

### Priority 4: Validate with Real Testnet
Once all components working, place actual testnet orders

---

## ⚠️ Troubleshooting

### Problem: "Module not found"
```bash
# Check PYTHONPATH
docker exec quantum_ai_engine python3 -c "import sys; print('\n'.join(sys.path))"

# Install missing module
docker exec quantum_ai_engine pip install <module_name>
```

### Problem: "Coroutine was never awaited"
- Add `async def` to function
- Add `await` before async calls
- Use `asyncio.run()` to execute

### Problem: Container not responding
```bash
# Restart container
docker restart quantum_ai_engine

# Check status
docker ps | grep quantum_ai_engine

# View logs
docker logs quantum_ai_engine --tail 20
```

---

## 📋 Sample Output

**Successful Simulation:**
```
🚀 Quantum Trader V3 – Testnet AI Pipeline Simulation

Step 1: ✅ RL Environment V3 initialized
Step 2: ✅ Exit Brain V3 initialized
Step 3: ✅ TP Optimizer V3 initialized
Step 4: ✅ Execution Engine initialized (SIMULATION)
Step 5: ✅ Trading Context created (BTCUSDT LONG)
Step 6: ✅ Exit Plan Generated (SL: $42000, TP1: $43800)
Step 7: ✅ TP Profile Evaluated (momentum_aggressive, 75%)
Step 8: ✅ RL Reward Computed (0.85)
Step 9: ✅ Simulated Trade Executed (SIM-20251217-182401)

✅ Steps Completed: 9/9
🎯 Overall Status: SUCCESS
```

**Current Status (Partial):**
```
✅ Steps Completed: 5/9
🎯 Overall Status: PARTIAL

Components:
- RL Environment: mock (missing gymnasium)
- Exit Brain V3: ✅ success
- TP Optimizer V3: ✅ success
- Execution Engine: mock (import issue)
```

---

## 🔑 Key Commands

| Task | Command |
|------|---------|
| Run simulation | `docker exec quantum_ai_engine python3 /tmp/run_testnet_simulation.py` |
| Check results | `cat ~/quantum_trader/status/testnet_simulation_*.json` |
| View logs | `docker logs quantum_ai_engine --tail 50` |
| Install deps | `docker exec quantum_ai_engine pip install gymnasium` |
| Restart container | `docker restart quantum_ai_engine` |
| Check safety | `cat ~/quantum_trader/config/go_live.yaml` |

---

## ✅ Safety Checklist

- [x] GO_LIVE=false (No live trading)
- [x] SIMULATION_MODE=true (Simulation only)
- [x] BINANCE_TESTNET=true (Testnet API)
- [x] simulate_orders=true (Mock orders)
- [x] risk_mode=sandbox (Sandbox mode)
- [x] activation_enabled=false (No auto-activation)
- [x] Exit Brain V3 functional
- [x] TP Optimizer V3 functional

---

## 📚 Full Documentation

See: `AI_TESTNET_SIMULATION_DEPLOYED.md`

---

**Status:** ✅ Ready for Testing  
**Risk Level:** 🟢 Zero (Simulation Only)  
**Next Action:** Fix dependencies → Re-run → Validate
