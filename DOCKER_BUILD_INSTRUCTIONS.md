# Docker Build and Test Instructions
**Date:** 2025-12-17  
**Purpose:** Rebuild quantum_backend container with PYTHONPATH=/app/backend configuration and verify module imports

---

## Configuration Status ✅

All configuration files have been updated:

| File | Status | Key Changes |
|------|--------|-------------|
| `systemctl.yml` | ✅ Updated | PYTHONPATH=/app/backend for all 10 services, GO_LIVE=true added |
| `.env` | ✅ Updated | Added VPS runtime section (GO_LIVE, PYTHONPATH, RL_DEBUG, DB_URI) |
| `activation.yaml` | ✅ Created | Module activation status and safety checks |
| `config/go_live.yaml` | ✅ Verified | Production activation config exists |

---

## Pre-Flight Checklist

### Step 1: Start Docker Desktop

**Option A - Start from Windows:**
1. Open Start Menu
2. Search for "Docker Desktop"
3. Click to start
4. Wait for Docker Desktop to show "Running" status (green icon in system tray)

**Option B - Start from PowerShell:**
```powershell
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
# Wait 30-60 seconds for Docker to initialize
Start-Sleep -Seconds 60
systemctl list-units  # Verify Docker is responsive
```

**Verify Docker is running:**
```powershell
docker --version
# Expected: Docker version 29.1.2, build 890dcca

systemctl list-units
# Should list running containers (may be empty)
```

---

## Build and Test Procedure

### Step 2: Navigate to Project Directory
```powershell
cd C:\quantum_trader
```

### Step 3: Stop Any Existing Containers
```powershell
# Stop all containers
docker compose down

# Verify stopped
systemctl list-units -a | Select-String "quantum_backend"
```

### Step 4: Build Backend Container
```powershell
# Build only the backend service
docker compose build backend

# This will:
# - Use backend/Dockerfile
# - Set PYTHONPATH=/app/backend
# - Install Python dependencies
# - May take 5-15 minutes depending on system
```

**Expected Output:**
```
[+] Building ...
 => [internal] load build definition from Dockerfile
 => => transferring dockerfile: ...
 => [internal] load .dockerignore
 => [stage-1  1/10] FROM docker.io/library/python:3.11-slim
 ...
 => => naming to docker.io/library/quantum_trader-backend
```

**If build fails:**
- Check `build_error.log` if using the PowerShell script
- Look for common errors:
  - `pip install` failures → check requirements.txt
  - `COPY` failures → verify backend/ folder exists
  - Network errors → check internet connection

### Step 5: Start Backend Container
```powershell
# Start in detached mode
docker compose up -d backend

# Verify container is running
systemctl list-units | Select-String "quantum_backend"
```

**Expected Output:**
```
[+] Running 1/1
 ✔ Container quantum_backend  Started
```

**Container should show:**
- STATUS: Up (healthy) or Up (starting)
- PORTS: 0.0.0.0:8000->8000/tcp

### Step 6: Check Container Logs (CRITICAL)
```powershell
# View last 50 lines of logs
journalctl -u quantum_backend.service --tail 50

# Follow logs in real-time (Ctrl+C to stop)
journalctl -u quantum_backend.service --follow
```

**Look for:**

✅ **SUCCESS Indicators:**
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

❌ **ERROR Indicators:**
```
ModuleNotFoundError: No module named 'domains.exits.exit_brain_v3'
ImportError: cannot import name 'dynamic_executor'
ModuleNotFoundError: No module named 'domains.learning.rl_v3'
```

**If you see ModuleNotFoundError:**
- PYTHONPATH is incorrect or not set
- Module files are missing
- systemctl.yml volume mounts are wrong

### Step 7: Verify Environment Variables Inside Container
```powershell
# Check critical environment variables
docker exec quantum_backend env | Select-String "GO_LIVE|PYTHONPATH|RL_DEBUG"
```

**Expected Output:**
```
GO_LIVE=true
PYTHONPATH=/app/backend
RL_DEBUG=true
```

### Step 8: Test Module Imports Directly
```powershell
# Test Exit Brain V3
docker exec quantum_backend python3 -c "from domains.exits.exit_brain_v3 import dynamic_executor; print('✓ Exit Brain V3 OK')"

# Test RL V3
docker exec quantum_backend python3 -c "from domains.learning.rl_v3 import rl_manager_v3; print('✓ RL V3 OK')"

# Test CLM V3
docker exec quantum_backend python3 -c "from services.clm_v3 import orchestrator; print('✓ CLM V3 OK')"

# Test TP Optimizer V3
docker exec quantum_backend python3 -c "from services.monitoring import tp_optimizer_v3; print('✓ TP Optimizer V3 OK')"
```

**Expected Output (for each):**
```
✓ Exit Brain V3 OK
✓ RL V3 OK
✓ CLM V3 OK
✓ TP Optimizer V3 OK
```

### Step 9: Test Backend Health Endpoint
```powershell
# Test HTTP health endpoint
Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing | Select-Object StatusCode, Content
```

**Expected Output:**
```
StatusCode Content
---------- -------
       200 {"status":"healthy","version":"3.0.0"}
```

---

## Automated Test Script

For convenience, use the automated test script:

```powershell
# Ensure Docker Desktop is running first!
# Then run:
.\test-docker-build.ps1
```

This script will:
1. ✓ Verify systemctl.yml configuration
2. ✓ Check Docker availability
3. ✓ Clean previous builds
4. ✓ Build backend container
5. ✓ Start backend container
6. ✓ Check container logs for errors
7. ✓ Verify environment variables
8. ✓ Test module imports
9. ✓ Generate final report

---

## Troubleshooting

### Issue: "Cannot connect to the Docker daemon"
**Solution:**
- Start Docker Desktop (see Step 1)
- Wait 60 seconds for Docker to initialize
- Run `systemctl list-units` to verify

### Issue: Build fails with "network timeout" or "connection refused"
**Solution:**
- Check internet connection
- Check Docker Desktop → Settings → Resources → Network
- Try with Docker Desktop WSL 2 backend enabled

### Issue: "ModuleNotFoundError" in logs
**Solution:**
1. Verify PYTHONPATH in systemctl.yml:
   ```yaml
   environment:
     - PYTHONPATH=/app/backend
   ```
2. Verify volume mount:
   ```yaml
   volumes:
     - ./backend:/app/backend
   ```
3. Rebuild container: `docker compose build --no-cache backend`

### Issue: Container exits immediately (STATUS: Exited)
**Solution:**
- Check logs: `journalctl -u quantum_backend.service`
- Look for Python syntax errors or import errors
- Check if database/redis dependencies are running

### Issue: Build takes too long (>30 minutes)
**Solution:**
- Stop build with Ctrl+C
- Use WSL + Podman instead (if Windows user permissions allow)
- Check if antivirus is scanning Docker layers

---

## Expected Results

After successful build and startup:

| Check | Expected Result |
|-------|----------------|
| Container Status | `Up` (running) |
| GO_LIVE | `true` |
| PYTHONPATH | `/app/backend` |
| Exit Brain V3 Import | ✅ No errors |
| RL V3 Import | ✅ No errors |
| CLM V3 Import | ✅ No errors |
| TP Optimizer V3 Import | ✅ No errors |
| Health Endpoint | HTTP 200 OK |
| Container Logs | "Application startup complete" |

---

## Next Steps After Verification

Once all imports succeed:

1. **Start remaining services:**
   ```powershell
   docker compose up -d
   ```

2. **Monitor all services:**
   ```powershell
   docker compose ps
   docker compose logs -f
   ```

3. **Test trading operations:**
   - Check `/api/docs` (FastAPI Swagger UI)
   - Verify Binance Testnet connection
   - Run shadow testing mode

4. **Review documentation:**
   - `VPS_MIGRATION_FOLDER_AUDIT.md` - Folder structure status
   - `DOCKER_PYTHONPATH_CONFIG_COMPLETE.md` - Docker configuration details
   - `RUNTIME_CONFIG_RESTORED.md` - Runtime configuration summary
   - `RUNTIME_CONFIG_QUICKREF.md` - Quick reference guide

---

## Success Criteria

✅ **VERIFICATION COMPLETE** when:
- [ ] Docker Desktop is running
- [ ] `docker compose build backend` completes without errors
- [ ] `docker compose up -d backend` starts container successfully
- [ ] `journalctl -u quantum_backend.service` shows "Application startup complete"
- [ ] No `ModuleNotFoundError` in logs
- [ ] All 4 module imports succeed (Exit Brain V3, RL V3, CLM V3, TP Optimizer V3)
- [ ] Environment variables are correct (GO_LIVE=true, PYTHONPATH=/app/backend)
- [ ] Health endpoint returns HTTP 200

---

**Last Updated:** 2025-12-17  
**Configuration Version:** VPS Migration v1.0  
**Contact:** Check `RUNTIME_CONFIG_QUICKREF.md` for troubleshooting

