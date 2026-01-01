# Dashboard Cleanup Plan - Remove Unused Copies

## Current Situation Analysis

### Active Services (KEEP)
1. **dashboard_v4/** - ACTIVE dashboard used by docker-compose.yml
   - `dashboard_v4/backend/` → quantum_dashboard_backend (port 8025)
   - `dashboard_v4/frontend/` → quantum_dashboard_frontend (port 8889)
   - Size: ~1.2MB
   - Status: ✅ IN USE

2. **microservices/rl_dashboard/** - ACTIVE RL monitoring dashboard
   - Used by docker-compose.vps.yml
   - Container: quantum_rl_dashboard (port 8026)
   - Purpose: Flask RL monitoring (reads exitbrain.pnl stream)
   - Status: ✅ IN USE

### Unused/Duplicate (DELETE)
1. **dashboard/** - OLD Flask dashboard
   - Size: 40KB
   - Status: ❌ NOT IN USE - superseded by dashboard_v4
   - Files: `app.py`, `Dockerfile`, `requirements.txt`, `static/index.html`

2. **quantumfond_backend/** - DUPLICATE backend
   - Size: 26MB (!!)
   - Status: ❌ NOT IN USE - duplicate of dashboard_v4/backend
   - This is causing space waste

3. **quantumfond_frontend/** - DUPLICATE frontend
   - Size: 468KB
   - Status: ❌ NOT IN USE - duplicate of dashboard_v4/frontend

4. **docker-compose.quantumfond.yml** - OLD compose file
   - Status: ❌ NOT IN USE - functionality moved to docker-compose.yml

5. **quantumfond-backend.service** - OLD systemd service
   - Status: ❌ NOT IN USE - using Docker now

## Files to Delete

### Local (c:\quantum_trader\)
```bash
# Old dashboard
dashboard/

# Old deployment scripts (duplicates)
deploy-quantumfond.ps1
deploy-quantumfond.sh
deploy_quantumfond.sh
deploy_quantumfond_simple.sh
deploy-quantumfond-vps.sh
deploy-quantumfond-vps-docker.sh

# Old compose file
docker-compose.quantumfond.yml

# Old nginx configs (duplicates)
nginx-quantumfond.conf
quantumfond.nginx.conf
```

### VPS (/home/qt/quantum_trader/)
```bash
# Old dashboard
dashboard/

# Duplicate backends/frontends (26MB waste!)
quantumfond_backend/
quantumfond_frontend/

# Old compose file
docker-compose.quantumfond.yml

# Old systemd service
quantumfond-backend.service

# Old nginx configs
nginx-quantumfond.conf

# Old deployment scripts
deploy-quantumfond.ps1
deploy-quantumfond.sh
deploy_quantumfond.sh
deploy_quantumfond_simple.sh
deploy-quantumfond-vps.sh
deploy-quantumfond-vps-docker.sh
```

## Space Savings
- VPS: ~26MB (quantumfond_backend) + 468KB (quantumfond_frontend) + 40KB (old dashboard) = **~26.5MB**
- Local: ~100KB (old files)
- Git repo: Reduced clutter, easier navigation

## Execution Plan

### Step 1: Verify Active Services
```bash
# On VPS
docker ps | grep dashboard
# Should show:
# - quantum_dashboard_backend (port 8025)
# - quantum_dashboard_frontend (port 8889)
# - quantum_rl_dashboard (port 8026)
```

### Step 2: Remove from VPS
```bash
ssh root@46.224.116.254
cd /home/qt/quantum_trader

# Backup first (just in case)
tar -czf ~/dashboard_backup_$(date +%Y%m%d).tar.gz \
  dashboard/ quantumfond_backend/ quantumfond_frontend/

# Remove old dashboard
rm -rf dashboard/

# Remove duplicates (26MB!)
rm -rf quantumfond_backend/
rm -rf quantumfond_frontend/

# Remove old config files
rm -f docker-compose.quantumfond.yml
rm -f quantumfond-backend.service
rm -f nginx-quantumfond.conf

# Remove old deployment scripts
rm -f deploy-quantumfond*.ps1
rm -f deploy-quantumfond*.sh
rm -f deploy_quantumfond*.sh
```

### Step 3: Remove from Local Repo
```powershell
cd C:\quantum_trader

# Remove old dashboard
Remove-Item -Recurse -Force dashboard/

# Remove old deployment scripts
Remove-Item deploy-quantumfond.ps1
Remove-Item deploy-quantumfond.sh
Remove-Item deploy_quantumfond.sh
Remove-Item deploy_quantumfond_simple.sh
Remove-Item deploy-quantumfond-vps.sh
Remove-Item deploy-quantumfond-vps-docker.sh

# Remove old compose file
Remove-Item docker-compose.quantumfond.yml

# Remove old nginx configs
Remove-Item nginx-quantumfond.conf
Remove-Item quantumfond.nginx.conf
```

### Step 4: Commit Cleanup
```bash
git add -A
git commit -m "chore: Remove old/duplicate dashboard files (saved 26MB)"
git push origin main
```

### Step 5: Verify Services Still Work
```bash
# Test dashboard API
curl -s http://localhost:8025/rl-dashboard/ | head -20

# Test frontend
curl -I http://localhost:8889

# Test RL dashboard
curl -I http://localhost:8026
```

## Safety Checklist
- [x] Identified active services (dashboard_v4, rl_dashboard)
- [x] Identified duplicates (quantumfond_backend, quantumfond_frontend)
- [x] Created backup before deletion (5.7MB tar.gz at ~/dashboard_backup_20260101.tar.gz)
- [x] Removed old files from VPS
- [x] Removed old files from local repo
- [x] Committed changes to git (commit bc11a6a9)
- [x] Verified services still work
- [x] Pulled changes to VPS (git pull)

## Result ✅
- ✅ **26.5MB disk space freed on VPS** (compressed to 5.7MB backup)
- ✅ **Cleaner codebase** (15 files removed, 1880 lines deleted)
- ✅ **No confusion about which dashboard to use**
- ✅ **Easier navigation in project**
- ✅ **All services still running:**
  - quantum_dashboard_backend: Up 3 minutes (healthy)
  - quantum_dashboard_frontend: Up 48 minutes
  - quantum_rl_dashboard: Up 13 minutes

## Remaining Dashboard Files
- `dashboard_v4/` (1.2MB) - Active dashboard
- `microservices/rl_dashboard/` (24KB) - Active RL monitoring

## Backup Location
VPS: `/root/dashboard_backup_20260101.tar.gz` (5.7MB)
- Can be restored if needed with: `tar -xzf ~/dashboard_backup_20260101.tar.gz -C /home/qt/quantum_trader/`
