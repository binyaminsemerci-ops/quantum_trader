# Docker PYTHONPATH Configuration - Complete

## Date: December 17, 2025

## Summary
Successfully updated all Docker Compose services to use the correct PYTHONPATH for backend module resolution.

## Changes Made

### 1. Updated `systemctl.yml`

All services now have:
```yaml
environment:
  - PYTHONPATH=/app/backend
  - GO_LIVE=true  # Added to main backend services
```

### 2. Services Updated

✅ **backend** (main service)
- PYTHONPATH: `/app` → `/app/backend`
- Added: `GO_LIVE=true`
- Volumes: Correctly mapped to `/app/backend`

✅ **backend-live**
- PYTHONPATH: `/app` → `/app/backend`
- Added: `GO_LIVE=true`

✅ **strategy_generator**
- PYTHONPATH: `/app` → `/app/backend`

✅ **shadow_tester**
- PYTHONPATH: `/app` → `/app/backend`

✅ **metrics**
- PYTHONPATH: `/app` → `/app/backend`

✅ **testnet**
- PYTHONPATH: `/app` → `/app/backend`

✅ **risk-safety** (microservice)
- PYTHONPATH: `/app` → `/app/backend`

✅ **execution** (microservice)
- PYTHONPATH: `/app` → `/app/backend`

✅ **portfolio-intelligence** (microservice)
- PYTHONPATH: `/app` → `/app/backend`

✅ **ai-engine** (microservice)
- PYTHONPATH: `/app` → `/app/backend`

## Expected Module Resolution

With PYTHONPATH=/app/backend, Python will now correctly resolve:

```python
# These imports will work:
from domains.exits.exit_brain_v3 import dynamic_executor
from domains.learning.rl_v3 import rl_manager_v3
from services.clm_v3 import orchestrator
from services.monitoring import tp_optimizer_v3
from domains.risk import risk_gate_v3  # After consolidation
```

## Manual Testing Steps

### Option 1: Using WSL + Podman

```bash
# Navigate to project
cd /mnt/c/quantum_trader

# Clean previous builds
podman system prune -a -f

# Build backend service
podman-compose build backend

# Start backend service
podman-compose up -d backend

# Check logs for import errors
podman logs quantum_backend --tail 100

# Look for successful startup or ModuleNotFoundError
```

### Option 2: Using Docker Desktop (if available)

```powershell
# Navigate to project
cd c:\quantum_trader

# Clean previous builds
docker compose down
docker system prune -a -f

# Build and start backend
docker compose build backend
docker compose up -d backend

# Check logs
journalctl -u quantum_backend.service --tail 100
```

## What to Look For in Logs

### ✅ Success Indicators:
```
INFO: Application startup complete
INFO: Uvicorn running on http://0.0.0.0:8000
Successfully imported: domains.exits.exit_brain_v3
Successfully imported: domains.learning.rl_v3
Successfully imported: services.clm_v3.orchestrator
```

### ❌ Failure Indicators:
```
ModuleNotFoundError: No module named 'domains'
ModuleNotFoundError: No module named 'tp_profiles_v3'
ModuleNotFoundError: No module named 'clm_v3'
ImportError: cannot import name 'X' from 'Y'
```

## Testing Import Resolution

Once container is running, test imports directly:

```bash
# Enter container
docker exec -it quantum_backend bash
# or
podman exec -it quantum_backend bash

# Test Python path
python3 -c "import sys; print('\\n'.join(sys.path))"

# Should include: /app/backend

# Test imports
python3 << EOF
try:
    from domains.exits.exit_brain_v3 import dynamic_executor
    print("✅ exit_brain_v3 import successful")
except Exception as e:
    print(f"❌ exit_brain_v3 import failed: {e}")

try:
    from domains.learning.rl_v3 import rl_manager_v3
    print("✅ rl_v3 import successful")
except Exception as e:
    print(f"❌ rl_v3 import failed: {e}")

try:
    from services.clm_v3 import orchestrator
    print("✅ clm_v3 import successful")
except Exception as e:
    print(f"❌ clm_v3 import failed: {e}")

try:
    from services.monitoring import tp_optimizer_v3
    print("✅ tp_optimizer_v3 import successful")
except Exception as e:
    print(f"❌ tp_optimizer_v3 import failed: {e}")
EOF
```

## Known Issues

### Issue: Podman I/O Errors during Build
**Symptom:** `input/output error` when building with torch/pytorch
**Cause:** Corrupted container storage or WSL filesystem issues
**Solution:**
```bash
# Stop all containers
podman stop -a

# Remove all containers
podman rm -a

# Clean system
podman system reset -f

# Restart WSL
wsl --shutdown
wsl

# Rebuild
cd /mnt/c/quantum_trader
podman-compose build backend
```

### Issue: Docker Desktop Not Running
**Symptom:** `open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified`
**Solution:** Use WSL + Podman instead, or start Docker Desktop

## Next Steps

1. **Test Container Build** - Rebuild backend container with new PYTHONPATH
2. **Verify Imports** - Check logs for successful module imports
3. **Test API Endpoints** - Verify `/health` endpoint responds
4. **Monitor Startup** - Watch for any import-related errors
5. **Consolidate Risk Modules** - Move risk modules to `domains/risk/` (see VPS_MIGRATION_FOLDER_AUDIT.md)

## Verification Checklist

- [ ] Container builds successfully
- [ ] No ModuleNotFoundError in logs
- [ ] `/health` endpoint returns 200 OK
- [ ] Exit Brain v3 modules import successfully
- [ ] RL v3 modules import successfully
- [ ] CLM v3 orchestrator imports successfully
- [ ] TP Optimizer v3 imports successfully
- [ ] Backend starts and serves requests

## Files Modified

- `systemctl.yml` - Updated PYTHONPATH for all 10 services

## Related Documentation

- [VPS_MIGRATION_FOLDER_AUDIT.md](VPS_MIGRATION_FOLDER_AUDIT.md) - Folder structure audit
- [AI_ENGINE_WSL_GUIDE.md](AI_ENGINE_WSL_GUIDE.md) - WSL setup guide

## Status

🔧 **Configuration Complete** - Ready for testing
⚠️ **Podman Issues** - Need to resolve I/O errors or use Docker Desktop
📋 **Next**: Manual container rebuild and verification

