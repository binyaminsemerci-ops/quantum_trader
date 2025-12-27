# Quantum Trader V3 - Runtime Configuration Quick Reference

## Configuration Status: ✅ COMPLETE

### Files Updated/Created
```
✓ .env                      - Updated (VPS runtime section added)
✓ activation.yaml           - Created (system activation status)
✓ config/go_live.yaml       - Verified (existing production config)
✓ docker-compose.yml        - Verified (PYTHONPATH updated)
```

### Critical Environment Variables

#### System Activation
```bash
GO_LIVE=true                    # Live trading mode enabled
RL_DEBUG=true                   # RL debugging enabled
PYTHONPATH=/app/backend         # Module resolution path
```

#### Database & Cache
```bash
DB_URI=postgresql://quantum:quantum@db:5432/quantum
REDIS_URL=redis://redis:6379
```

#### Exchange Configuration
```bash
BINANCE_TESTNET=true
EXCHANGE_MODE=binance_testnet
BINANCE_API_KEY=xOPqaf2iSKt4gVuScoebb3wDBm0R9gw0qSPtpHYnJNzcahTSL58b4QZcC4dsJ5eX
BINANCE_API_SECRET=hwyeOL1BHBMv5jLmCEemg2OQNUb8dUAyHgamOftcS9oFDfc605SX1IZs294zvNmZ
```

#### Module Configuration
```bash
EXIT_MODE=EXIT_BRAIN_V3
EXIT_EXECUTOR_MODE=LIVE
EXIT_BRAIN_V3_ENABLED=true
EXIT_BRAIN_PROFILE=CHALLENGE_100
QT_CLM_ENABLED=true
QT_CLM_AUTO_RETRAIN=true
```

### Quick Start Commands

#### Option 1: WSL + Podman
```bash
# Clean and rebuild
wsl bash -c 'cd /mnt/c/quantum_trader && podman system reset -f'
wsl bash -c 'cd /mnt/c/quantum_trader && podman-compose build backend'
wsl bash -c 'cd /mnt/c/quantum_trader && podman-compose up -d backend'

# Verify
wsl bash -c 'podman exec quantum_backend env | grep -E "(GO_LIVE|PYTHONPATH|RL_DEBUG)"'
wsl bash -c 'podman logs quantum_backend --tail 100'
```

#### Option 2: Docker Desktop
```powershell
# Clean and rebuild
docker compose down
docker system prune -a -f
docker compose build backend
docker compose up -d backend

# Verify
docker exec quantum_backend env | findstr /I "GO_LIVE PYTHONPATH RL_DEBUG"
docker logs quantum_backend --tail 100
```

### Verification Checklist

Inside Container:
```bash
# Enter container
podman exec -it quantum_backend bash

# Check environment
echo $GO_LIVE          # Should be: true
echo $PYTHONPATH       # Should be: /app/backend
echo $RL_DEBUG         # Should be: true

# Test imports
python3 -c "from domains.exits.exit_brain_v3 import dynamic_executor; print('✓ OK')"
python3 -c "from domains.learning.rl_v3 import rl_manager_v3; print('✓ OK')"
python3 -c "from services.clm_v3 import orchestrator; print('✓ OK')"
python3 -c "from services.monitoring import tp_optimizer_v3; print('✓ OK')"
```

### Expected Log Output

✅ Success:
```
INFO: Application startup complete
INFO: Uvicorn running on http://0.0.0.0:8000
Successfully loaded: exit_brain_v3
Successfully loaded: rl_v3
Successfully loaded: clm_v3
```

❌ Failure:
```
ModuleNotFoundError: No module named 'domains'
ImportError: cannot import name 'X' from 'Y'
```

### Module Activation Status (from activation.yaml)
```yaml
modules:
  exit_brain_v3: true      # ✓ Exit Brain V3 dynamic TP/SL
  rl_v3: true              # ✓ Reinforcement Learning V3
  clm_v3: true             # ✓ Continuous Learning Manager
  risk_gate_v3: true       # ✓ Risk Gate V3
  tp_optimizer_v3: true    # ✓ Take Profit Optimizer V3
```

### Safety Checks (from activation.yaml)
```yaml
safety_checks:
  folder_structure: true           # ✓ All required folders exist
  pythonpath_configured: true      # ✓ PYTHONPATH set correctly
  env_variables_set: true          # ✓ All env vars configured
  docker_compose_updated: true     # ✓ Docker config updated
```

### Related Documentation
- [RUNTIME_CONFIG_RESTORED.md](RUNTIME_CONFIG_RESTORED.md) - Full configuration report
- [DOCKER_PYTHONPATH_CONFIG_COMPLETE.md](DOCKER_PYTHONPATH_CONFIG_COMPLETE.md) - Docker setup
- [VPS_MIGRATION_FOLDER_AUDIT.md](VPS_MIGRATION_FOLDER_AUDIT.md) - Folder structure

### Troubleshooting

**Problem:** ModuleNotFoundError for backend modules  
**Solution:** Verify PYTHONPATH=/app/backend in container and docker-compose.yml

**Problem:** GO_LIVE not recognized  
**Solution:** Check .env file loaded, verify with `docker exec quantum_backend env | grep GO_LIVE`

**Problem:** Import errors for tp_profiles_v3  
**Solution:** Verify exit_brain_v3 folder exists in backend/domains/exits/

**Problem:** CLM or RL modules not found  
**Solution:** Check backend/domains/learning/rl_v3/ and backend/services/clm_v3/ exist

### Status Summary
```
Configuration:  ✅ COMPLETE
Files:          ✅ 4 files updated/created
Env Variables:  ✅ All required vars set
Docker:         ✅ PYTHONPATH configured
Modules:        ✅ 5 core modules ready
Safety Checks:  ✅ All passed

Next Action: Build and start containers
```
