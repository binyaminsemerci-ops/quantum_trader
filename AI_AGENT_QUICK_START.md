# AI Agent Integration - Quick Start Guide

**Date:** 2025-12-17  
**Purpose:** Fast-track AI agent validation after Docker startup

---

## ⚡ Quick Start (When Docker is Running)

### Option 1: Automated Validation (Recommended)
```powershell
cd C:\quantum_trader
.\ai_agent_integration_validator.ps1
```

**Duration:** 2-3 minutes  
**Output:** Full validation report with GO LIVE simulation

### Option 2: Manual Step-by-Step

#### Step 1: Build & Start
```powershell
cd C:\quantum_trader
docker compose build backend  # 10-15 min first time
docker compose up -d backend  # 30 sec
```

#### Step 2: Test Module Imports
```powershell
docker exec quantum_backend python3 -c "
import sys; sys.path.insert(0, '/app')
from backend.domains.exits.exit_brain_v3 import dynamic_executor
from backend.services.monitoring import tp_optimizer_v3
from backend.domains.learning.rl_v3 import rl_manager_v3
from backend.services.clm_v3 import orchestrator
from backend.risk import risk_gate_v3
print('✅ All imports successful')
"
```

#### Step 3: Run GO LIVE Simulation
```powershell
docker exec quantum_backend python3 -c "
import sys; sys.path.insert(0, '/app')
from backend.domains.exits.exit_brain_v3.dynamic_executor import DynamicTPExecutor
from backend.services.monitoring.tp_optimizer_v3 import TPOptimizerV3
from backend.domains.learning.rl_v3.rl_manager_v3 import RLManagerV3

print('[SIMULATION] Initializing AI agents...')
executor = DynamicTPExecutor()
print('✓ Exit Brain V3 OK')
optimizer = TPOptimizerV3()
print('✓ TP Optimizer V3 OK')
rl_manager = RLManagerV3()
print('✓ RL Agent V3 OK')
print('✅ GO LIVE simulation complete')
"
```

#### Step 4: Check Logs
```powershell
docker logs quantum_backend --tail 100
# Look for: [ExitBrainV3], [TPOptimizerV3], [RLAgentV3], [RiskGateV3]
```

---

## 🎯 Expected Results

### Successful Validation

```
✅ All AI agents active
⚙️  GO LIVE simulation succeeded
🧠 RL loop verified

Quantum Trader v3 — AI agents verified and GO LIVE simulation successful.
```

### Module Imports
```
[OK] domains.exits.exit_brain_v3
[OK] services.monitoring.tp_optimizer_v3
[OK] domains.learning.rl_v3.env_v3
[OK] domains.learning.rl_v3.reward_v3
[OK] services.clm_v3.orchestrator
[OK] services.monitoring.tp_performance_tracker
```

### GO LIVE Dry Run
```
[GO LIVE SIMULATION] Initializing system...
[1/7] Environment: GO_LIVE=true, PYTHONPATH=/app/backend
[2/7] Exit Brain V3... ✓
[3/7] TP Optimizer V3... ✓
[4/7] RL Agent V3... ✓
[5/7] CLM V3... ✓
[6/7] Risk Gate V3... ✓
[7/7] Execution Engine... ✓
GO LIVE SIMULATION COMPLETE ✅
```

---

## 🚨 Troubleshooting

### Issue: Import fails with ModuleNotFoundError
```powershell
# Check PYTHONPATH
docker exec quantum_backend env | Select-String "PYTHONPATH"
# Should show: PYTHONPATH=/app/backend

# Check volume mount
docker exec quantum_backend ls -la /app/backend/domains/exits/exit_brain_v3/
# Should list Python files

# Rebuild if needed
docker compose build --no-cache backend
```

### Issue: Container exits immediately
```powershell
# Check logs for errors
docker logs quantum_backend

# Check container status
docker ps -a | Select-String "quantum_backend"
# Should show "Up" not "Exited"
```

### Issue: Simulation fails
```powershell
# Check individual components
docker exec quantum_backend python3 -c "from backend.domains.exits.exit_brain_v3 import dynamic_executor; print('OK')"

# Check environment variables
docker exec quantum_backend env | Select-String "GO_LIVE|RL_DEBUG"

# Review full logs
docker logs quantum_backend --tail 200
```

---

## 📋 Validation Checklist

After running validation, verify:

- [ ] Docker Desktop running
- [ ] quantum_backend container running (not exited)
- [ ] All 6 module imports successful
- [ ] Exit Brain V3 initialized
- [ ] TP Optimizer V3 initialized
- [ ] RL Agent V3 initialized
- [ ] Risk Gate V3 initialized
- [ ] CLM V3 initialized
- [ ] GO LIVE simulation completed
- [ ] No ModuleNotFoundError in logs
- [ ] Health endpoint returns 200: `Invoke-WebRequest http://localhost:8000/health`

---

## 🎉 Success Criteria

System is production-ready when:

✅ All module imports succeed  
✅ All AI agents initialize  
✅ GO LIVE simulation completes  
✅ No critical errors in logs  
✅ Health endpoint responsive  
✅ Simulation mode: No real trades executed

---

## 📄 Report Files

- **Full Report:** `AI_AGENT_VALIDATION_REPORT.md`
- **Validator Script:** `ai_agent_integration_validator.ps1`
- **VPS Recovery:** `VPS_RECOVERY_REPORT.md`
- **Build Guide:** `DOCKER_BUILD_INSTRUCTIONS.md`

---

**Next:** After validation succeeds → Monitor for 24h → Enable live trading

**Command:** `.\ai_agent_integration_validator.ps1`
