# 🚨 VPS Deployment Status - Manual Intervention Required

**Date:** December 22, 2025  
**Status:** ⚠️ BLOCKED - Root Ownership & Missing Dependencies  

---

## 🔍 Current VPS State

### Files Status
```bash
# ExitBrain v3.5 files exist but owned by root:
-rw-r--r-- 1 root root  9933 Dec 21 17:59 adaptive_leverage_engine.py (OLD VERSION with numpy)
-rw-r--r-- 1 root root 12350 Dec 21 17:59 exit_brain.py (OLD VERSION without monitoring)
```

### Issues Identified
1. ❌ Files owned by `root` (cannot overwrite without sudo)
2. ❌ Old adaptive_leverage_engine.py uses numpy (not our new version)
3. ❌ Old exit_brain.py missing monitoring logs and Redis streaming
4. ❌ Missing Python dependencies: `redis` module
5. ❌ `redis-cli` command not found (Redis may not be installed/configured)
6. ❌ No logs in `logs/exitbrain_v3.log` (service may not be logging there)

### Services Running
```bash
# Backend running on port 8000 (as root)
root 609296 uvicorn backend.main:app --host 0.0.0.0 --port 8000

# AI Engine running on port 8001 (as root)
root 470357 python -m uvicorn microservices.ai_engine.main:app --port 8001
```

### Files Successfully Copied (No Sudo Required)
✅ `monitor_adaptive_leverage.py` → ~/quantum_trader/
✅ `ADAPTIVE_LEVERAGE_USAGE_GUIDE.md` → ~/quantum_trader/

---

## 🛠️ Required Manual Steps

You need SSH access with sudo privileges to complete deployment. Here's what needs to be done:

### Option A: Complete Manual Deployment (Recommended)

**Step 1: SSH to VPS**
```bash
ssh qt@46.224.116.254
```

**Step 2: Install Python Dependencies**
```bash
# Install redis-py module
pip3 install redis

# Or if using requirements.txt:
cd ~/quantum_trader
pip3 install -r requirements.txt
```

**Step 3: Backup Existing Files**
```bash
cd ~/quantum_trader
sudo cp microservices/exitbrain_v3_5/exit_brain.py microservices/exitbrain_v3_5/exit_brain.py.backup.$(date +%Y%m%d)
sudo cp microservices/exitbrain_v3_5/adaptive_leverage_engine.py microservices/exitbrain_v3_5/adaptive_leverage_engine.py.backup.$(date +%Y%m%d)
```

**Step 4: Copy New Files from /tmp**
```bash
sudo cp /tmp/exit_brain.py microservices/exitbrain_v3_5/exit_brain.py
sudo cp /tmp/adaptive_leverage_config.py microservices/exitbrain_v3_5/adaptive_leverage_config.py
```

**Step 5: Fix Permissions**
```bash
sudo chown qt:qt microservices/exitbrain_v3_5/exit_brain.py
sudo chown qt:qt microservices/exitbrain_v3_5/adaptive_leverage_config.py
sudo chmod 644 microservices/exitbrain_v3_5/exit_brain.py
sudo chmod 644 microservices/exitbrain_v3_5/adaptive_leverage_config.py
```

**Step 6: Update adaptive_leverage_engine.py**
```bash
# Our new version doesn't use numpy, but old version does
# Check if numpy is needed elsewhere:
grep -r "import numpy" microservices/exitbrain_v3_5/ --exclude-dir=__pycache__

# If only adaptive_leverage_engine.py uses numpy, we can replace it:
# Create our new version (177 lines, no numpy dependency)
cat > /tmp/new_adaptive_engine.py << 'NEWFILE'
[PASTE CONTENT FROM LOCAL FILE]
NEWFILE

sudo cp /tmp/new_adaptive_engine.py microservices/exitbrain_v3_5/adaptive_leverage_engine.py
sudo chown qt:qt microservices/exitbrain_v3_5/adaptive_leverage_engine.py
```

**Step 7: Validate Imports**
```bash
cd ~/quantum_trader
python3 -c "from microservices.exitbrain_v3_5.adaptive_leverage_engine import AdaptiveLeverageEngine; print('✅ Import OK')"
python3 -c "from microservices.exitbrain_v3_5.adaptive_leverage_config import get_config; print('✅ Config OK')"
```

**Step 8: Run Unit Tests**
```bash
cd ~/quantum_trader
python3 -m pytest microservices/exitbrain_v3_5/tests/test_adaptive_leverage_engine.py -v
```

**Step 9: Check/Install Redis**
```bash
# Check if Redis is running
redis-cli ping

# If not installed:
sudo apt-get update
sudo apt-get install redis-server redis-tools -y
sudo systemctl start redis
sudo systemctl enable redis
```

**Step 10: Restart Services**
```bash
# Find what's managing the services (systemd, docker, supervisor?)
sudo systemctl list-units | grep -E 'exitbrain|quantum'

# If using docker:
docker ps | grep quantum

# If using systemd:
sudo systemctl restart quantum_trader
# or
sudo systemctl restart backend

# If manual processes:
sudo pkill -f "uvicorn backend.main:app"
# Then start service again (check startup scripts)
```

**Step 11: Monitor Logs**
```bash
# Find where logs are actually written:
find ~/quantum_trader -name "*.log" -type f

# Watch for adaptive levels:
tail -f [LOG_FILE_PATH] | grep "Adaptive Levels"
```

**Step 12: Test Monitoring Script**
```bash
cd ~/quantum_trader
python3 monitor_adaptive_leverage.py 10
```

---

### Option B: Git Pull Method (If Permissions Can Be Fixed)

**Step 1: Fix Git Repository Ownership**
```bash
ssh qt@46.224.116.254
cd ~/quantum_trader
sudo chown -R qt:qt .git
sudo chown -R qt:qt microservices/
```

**Step 2: Pull Latest Code**
```bash
git fetch origin main
git reset --hard origin/main
```

**Step 3: Install Dependencies & Restart (Steps 2, 9, 10 from Option A)**

---

## 📊 What's Blocked

### Cannot Auto-Deploy Because:
1. SSH doesn't allow interactive sudo (needs tty)
2. Files owned by root require sudo to modify
3. Missing redis Python module
4. Unknown log file location
5. Unknown service management method (docker/systemd/supervisor?)

### What Works:
✅ Monitoring script copied to ~/quantum_trader/
✅ Documentation copied to ~/quantum_trader/
✅ New files ready in /tmp/ directory
✅ Services are running (backend on 8000, AI engine on 8001)

---

## 🎯 Simplified Quick Start

If you have sudo access, run this single command block:

```bash
ssh qt@46.224.116.254 << 'ENDSSH'
cd ~/quantum_trader

# Install dependencies
pip3 install redis

# Backup & deploy
sudo cp microservices/exitbrain_v3_5/exit_brain.py microservices/exitbrain_v3_5/exit_brain.py.backup
sudo cp /tmp/exit_brain.py microservices/exitbrain_v3_5/exit_brain.py
sudo cp /tmp/adaptive_leverage_config.py microservices/exitbrain_v3_5/adaptive_leverage_config.py
sudo chown qt:qt microservices/exitbrain_v3_5/exit_brain.py microservices/exitbrain_v3_5/adaptive_leverage_config.py

# Test
python3 -c "from microservices.exitbrain_v3_5.adaptive_leverage_config import get_config; print('✅ Config loaded')"

echo "✅ Files deployed. Now restart services with: sudo systemctl restart quantum_trader"
ENDSSH
```

---

## 💡 Alternative: Wait for Root User

If you don't have sudo access, you'll need to:
1. Contact VPS administrator (root user)
2. Share this document with deployment steps
3. Ask them to run Option A steps 1-12

Or:
1. Fix git repository permissions once: `sudo chown -R qt:qt ~/quantum_trader`
2. Then use git pull for future deployments

---

## 📋 Files Ready for Deployment (in /tmp/)

- `/tmp/exit_brain.py` (16KB) - Enhanced with monitoring + Redis streaming
- `/tmp/adaptive_leverage_config.py` (4.4KB) - Tunable configuration
- `/tmp/monitor_adaptive_leverage.py` (9KB) - Production monitoring
- `/tmp/ADAPTIVE_LEVERAGE_USAGE_GUIDE.md` (9.9KB) - Documentation
- `/tmp/deploy_adaptive.sh` (1.8KB) - Deployment script

**Status:** ✅ All files uploaded, waiting for sudo execution

---

## 🆘 Next Actions

**Immediate:**
1. SSH to VPS: `ssh qt@46.224.116.254`
2. Run: `sudo pip3 install redis`
3. Run: `sudo bash /tmp/deploy_adaptive.sh`
4. If that fails, follow Option A steps manually

**After Deployment:**
1. Verify imports work
2. Restart services
3. Monitor logs for "Adaptive Levels"
4. Run monitoring script

---

**Current Status:** ⏸️ PAUSED - Waiting for manual sudo access  
**Blocker:** File permissions + missing dependencies  
**Resolution:** Execute Option A or B steps above with sudo privileges  
**ETA:** 10-15 minutes once sudo access available  
