# 🔐 API Key Rotation Execution Checklist

**Date Started:** January 28, 2026  
**Operator:** ________________  
**Status:** 🔴 IN PROGRESS

---

## ✅ Pre-Rotation (COMPLETE)

- [x] Secrets scrubbed from local repo (commit 2d90a3b3)
- [x] Security guide created (SECURITY_API_KEY_ROTATION_REQUIRED.md)
- [x] Rotation scripts prepared (rotate_testnet_keys.sh, scrub_git_history.sh)
- [x] VPS key locations identified (4 files in /etc/quantum/)

---

## 🔥 Step 1: Rotate on Binance Testnet (YOU DO THIS MANUALLY)

**URL:** https://testnet.binancefuture.com/

- [ ] **1.1** Log in to Binance Testnet
- [ ] **1.2** Navigate to: Account → API Management
- [ ] **1.3** **Delete/Disable old keys:**
  - Old API Key: `e9ZqWh...pzUPD` (51 chars)
  - Confirm deletion: YES
- [ ] **1.4** **Create new API keys:**
  - Label: `quantum_trader_v2_jan2026`
  - Permissions: ✅ Enable Futures Trading, ❌ Disable others
  - IP Whitelist: `46.224.116.254` (if supported)
  - Click "Create"
- [ ] **1.5** **Copy new keys securely:**
  - API Key: __________________ (save in password manager)
  - Secret Key: __________________ (save in password manager)
  - ⚠️ **DO NOT paste in terminal or logs**

**Verification:**
- [ ] Old keys show "Disabled" or deleted in API Management
- [ ] New keys show "Active" with Futures permission
- [ ] Keys saved in secure location (password manager)

---

## 🔧 Step 2: Update VPS Configuration

**Files to update (4 total):**
- `/etc/quantum/apply-layer.env`
- `/etc/quantum/governor.env`
- `/etc/quantum/intent-executor.env`
- `/etc/quantum/testnet.env`

### Option A: Automated (Recommended)

```bash
# Deploy rotation script to VPS
scp -i ~/.ssh/hetzner_fresh scripts/rotate_testnet_keys.sh root@46.224.116.254:/tmp/

# SSH to VPS
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254

# Run rotation script (paste your NEW keys)
bash /tmp/rotate_testnet_keys.sh 'NEW_API_KEY_HERE' 'NEW_API_SECRET_HERE'

# Verify (shows key lengths only, no actual keys)
# Should show ~51 chars for KEY, ~64 chars for SECRET
```

### Option B: Manual

```bash
# SSH to VPS
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254

# Update each file
nano /etc/quantum/governor.env
# Change:
#   BINANCE_TESTNET_API_KEY=NEW_KEY_HERE
#   BINANCE_TESTNET_API_SECRET=NEW_SECRET_HERE

# Repeat for:
nano /etc/quantum/apply-layer.env
nano /etc/quantum/intent-executor.env
nano /etc/quantum/testnet.env  # Note: Also update BINANCE_TESTNET_SECRET_KEY here
```

**Checklist:**
- [ ] **2.1** Backed up old config files
- [ ] **2.2** Updated `/etc/quantum/governor.env`
- [ ] **2.3** Updated `/etc/quantum/apply-layer.env`
- [ ] **2.4** Updated `/etc/quantum/intent-executor.env`
- [ ] **2.5** Updated `/etc/quantum/testnet.env`
- [ ] **2.6** Verified no quotes or extra spaces in values
- [ ] **2.7** Verified file permissions: `ls -la /etc/quantum/*.env` (should be 600 or 640)

---

## 🔄 Step 3: Restart Services

```bash
# Restart services that use testnet keys
systemctl restart quantum-governor
systemctl restart quantum-apply-layer
systemctl restart quantum-intent-executor

# Optional: If you have execution/AI services using testnet
systemctl restart quantum-execution || true
systemctl restart quantum-ai-engine || true

# Check all are active
systemctl is-active quantum-governor quantum-apply-layer quantum-intent-executor
```

**Checklist:**
- [ ] **3.1** Restarted quantum-governor
- [ ] **3.2** Restarted quantum-apply-layer
- [ ] **3.3** Restarted quantum-intent-executor
- [ ] **3.4** All services show "active"

---

## ✅ Step 4: Verify Without Exposing Secrets

### 4.1 Test Exchange Dump Script

```bash
cd /home/qt/quantum_trader
python3 scripts/dump_exchange_positions.py
```

**Expected output:**
```
================================================================================
BINANCE TESTNET FUTURES - POSITION DUMP
================================================================================
Total symbols checked: 674
Active positions (abs(qty) > 1e-08): X
...
✅ Direct exchange position dump complete
```

**❌ If you see:**
```
❌ ERROR: Missing BINANCE_TESTNET_API_KEY
```
→ Script not loading from env. Check EnvironmentFile or export manually.

**❌ If you see 401/authentication errors:**
→ Keys incorrect. Verify copy-paste from Binance (no extra spaces).

### 4.2 Check Governor Logs

```bash
# Recent logs (should show no auth errors)
journalctl -u quantum-governor --since "3 minutes ago" --no-pager | tail -50

# Search for auth errors
journalctl -u quantum-governor --since "5 minutes ago" --no-pager | grep -iE "401|403|auth|invalid.*api"
```

**Good signs:**
- `Governor starting main loop`
- `Testnet P2.9 gate ENABLED`
- No `401 Unauthorized` or `-2015` errors

**Bad signs:**
- `401 Unauthorized`
- `-2015: Invalid API-key`
- `Signature verification failed`

### 4.3 Check Apply Layer Logs

```bash
journalctl -u quantum-apply-layer --since "3 minutes ago" --no-pager | tail -30 | grep -iE "auth|401|api"
```

**Checklist:**
- [ ] **4.1** `dump_exchange_positions.py` ran successfully
- [ ] **4.2** Governor logs show no auth errors
- [ ] **4.3** Apply layer logs show no auth errors
- [ ] **4.4** Services able to query Binance testnet

---

## 🧹 Step 5: Clean VPS Repo

```bash
# Pull latest (scrubbed) code
cd /home/qt/quantum_trader
git pull

# Verify no hardcoded secrets in repo
grep -RIn "e9ZqWh\|ZowBZE" . 2>/dev/null | grep -v ".git" | head -20
```

**Expected:** 0 results (or only in .git/ which is ok before history scrub)

**Checklist:**
- [ ] **5.1** VPS repo pulled latest code
- [ ] **5.2** No hardcoded secrets in working tree files
- [ ] **5.3** Only `/etc/quantum/*.env` contains actual keys (correct)

---

## 🗑️ Step 6: Git History Scrubbing (OPTIONAL BUT RECOMMENDED)

**⚠️ WARNING:** This rewrites history. Requires force-push.

### Install git-filter-repo

```bash
# Windows (PowerShell)
pip install git-filter-repo

# Verify
git-filter-repo --version
```

### Run Scrubbing Script

```bash
# From local repo root (Windows)
bash scripts/scrub_git_history.sh

# Follow prompts:
# - Type "yes" to confirm
# - Script creates backup tag
# - Rewrites history
# - Shows force-push command
```

### Force Push

```bash
# Push cleaned history
git push --force --all
git push --force --tags
```

**Checklist:**
- [ ] **6.1** git-filter-repo installed
- [ ] **6.2** Backup tag created (git tag | grep backup)
- [ ] **6.3** History rewritten locally
- [ ] **6.4** Force-pushed to GitHub
- [ ] **6.5** VPS re-cloned fresh (if needed)

**Alternative: Skip history scrub**
- [ ] **6.ALT** Decided to skip (keys rotated, old keys dead = acceptable risk)

---

## 📋 Final Verification Checklist

### Security
- [ ] Old testnet keys deleted/disabled on Binance
- [ ] New testnet keys active with correct permissions
- [ ] New keys ONLY in `/etc/quantum/*.env` (not in repo)
- [ ] No hardcoded secrets in git working tree
- [ ] (Optional) Git history scrubbed and force-pushed

### Functionality
- [ ] All services restarted successfully
- [ ] No auth errors in logs (governor, apply-layer, intent-executor)
- [ ] `dump_exchange_positions.py` works
- [ ] P2.9 gate active and checking positions
- [ ] Can query Binance testnet API

### Documentation
- [ ] Rotation logged in this checklist
- [ ] New keys saved in password manager
- [ ] Backup configs saved in `/etc/quantum/backup_*`
- [ ] Team notified (if applicable)

---

## 🎯 Completion

**Date Completed:** ________________  
**Operator:** ________________  
**New Key Version:** quantum_trader_v2_jan2026  
**Status:** ⚠️ PENDING EXECUTION

### Sign-Off

- [ ] All steps completed
- [ ] Services verified working
- [ ] No auth errors
- [ ] Keys rotated successfully

**Final Status:** ⚠️ AWAITING USER EXECUTION  
**Next Action:** User must rotate keys on Binance UI (Step 1)

---

## 📞 Troubleshooting

### Auth Errors After Rotation

**Error: "401 Unauthorized"**
- Check keys copied correctly (no extra spaces)
- Verify IP whitelist includes VPS IP (if enabled)
- Confirm new keys have Futures permissions

**Error: "-2015 Invalid API-key"**
- Old keys still in config? Check all 4 files
- Services not restarted? Run Step 3 again

**Error: Script can't find keys**
- Export manually: `export $(grep BINANCE_TESTNET /etc/quantum/governor.env | xargs)`
- Or use systemd-run with EnvironmentFile

### Services Won't Start

```bash
# Check service status
systemctl status quantum-governor -l

# Check for syntax errors in env files
cat /etc/quantum/governor.env | grep BINANCE_TESTNET
```

### Rollback

```bash
# Restore backup configs
ls /etc/quantum/backup_*  # Find backup dir
cp /etc/quantum/backup_TIMESTAMP/governor.env /etc/quantum/
systemctl restart quantum-governor

# Or restore old keys (if not deleted on Binance)
nano /etc/quantum/governor.env
# Put old keys back temporarily
```

---

*Checklist created: January 28, 2026*  
*Scripts ready: rotate_testnet_keys.sh, scrub_git_history.sh*  
*Status: Ready for execution - awaiting Step 1 (Binance UI rotation)*
