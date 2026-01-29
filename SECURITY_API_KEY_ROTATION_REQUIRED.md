# 🔐 URGENT: API Key Rotation Required

**Date:** January 28, 2026  
**Status:** 🔴 CRITICAL - Hardcoded keys found and scrubbed  
**Action:** Rotate keys immediately

---

## 🚨 What Happened

During testnet flatten verification, API keys were used in command-line arguments and subsequently appeared in:

1. **Git Repository** (15 files with hardcoded keys)
2. **Conversation logs** (AI assistant session)
3. **Command history** (WSL ssh commands with inline credentials)

**Keys Exposed:**
- `BINANCE_TESTNET_API_KEY`: `e9ZqWh...pzUPD` (51 chars)
- `BINANCE_TESTNET_API_SECRET`: `ZowBZE...CFoja` (64 chars)

---

## ✅ Immediate Actions Taken

### 1. Repository Scrubbing (COMPLETE)
- Created `scripts/scrub_secrets.sh` to replace all hardcoded keys
- Scrubbed 15 files with placeholders: `your_binance_testnet_api_key_here`
- Committed: `2d90a3b3` "security: scrub hardcoded API keys from repo"
- **Git history still contains old commits with keys** (see History Scrubbing below)

### 2. Script Security Hardening (COMPLETE)
- Enhanced `scripts/dump_exchange_positions.py`:
  * Now requires credentials via environment only (no CLI args)
  * Added security warning in docstring
  * Recommends systemd EnvironmentFile usage
  * Never prints credentials

### 3. VPS File Check (COMPLETE)
- Checked `/home/qt/quantum_trader`: Same files, will be updated on next pull
- Checked `/etc/quantum/*.env`: Keys present (correct - protected config files)

---

## 🔥 REQUIRED: Key Rotation Steps

### Step 1: Generate New Testnet Keys

**On Binance Testnet:**
1. Go to https://testnet.binancefuture.com/
2. Log in with your account
3. Navigate to API Management
4. **Delete old keys:**
   - API Key: `e9ZqWh...pzUPD`
5. **Create new keys:**
   - Label: "quantum_trader_testnet_v2"
   - Enable "Futures" permissions
   - Save API Key and Secret securely (password manager)

### Step 2: Update VPS Configuration

```bash
# SSH to VPS
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254

# Update all config files with new keys
NEW_KEY="your_new_testnet_api_key_here"
NEW_SECRET="your_new_testnet_secret_here"

# Update each config file
sudo sed -i "s/BINANCE_TESTNET_API_KEY=.*/BINANCE_TESTNET_API_KEY=$NEW_KEY/" /etc/quantum/apply-layer.env
sudo sed -i "s/BINANCE_TESTNET_API_SECRET=.*/BINANCE_TESTNET_API_SECRET=$NEW_SECRET/" /etc/quantum/apply-layer.env

sudo sed -i "s/BINANCE_TESTNET_API_KEY=.*/BINANCE_TESTNET_API_KEY=$NEW_KEY/" /etc/quantum/governor.env
sudo sed -i "s/BINANCE_TESTNET_API_SECRET=.*/BINANCE_TESTNET_API_SECRET=$NEW_SECRET/" /etc/quantum/governor.env

sudo sed -i "s/BINANCE_TESTNET_API_KEY=.*/BINANCE_TESTNET_API_KEY=$NEW_KEY/" /etc/quantum/intent-executor.env
sudo sed -i "s/BINANCE_TESTNET_API_SECRET=.*/BINANCE_TESTNET_API_SECRET=$NEW_SECRET/" /etc/quantum/intent-executor.env

sudo sed -i "s/BINANCE_TESTNET_API_KEY=.*/BINANCE_TESTNET_API_KEY=$NEW_KEY/" /etc/quantum/testnet.env
sudo sed -i "s/BINANCE_TESTNET_SECRET_KEY=.*/BINANCE_TESTNET_SECRET_KEY=$NEW_SECRET/" /etc/quantum/testnet.env

sudo sed -i "s/BINANCE_API_KEY=.*/BINANCE_API_KEY=$NEW_KEY/" /etc/quantum/binance-pnl-tracker.env
sudo sed -i "s/BINANCE_API_SECRET=.*/BINANCE_API_SECRET=$NEW_SECRET/" /etc/quantum/binance-pnl-tracker.env
```

### Step 3: Restart Services

```bash
# Restart all services that use testnet keys
sudo systemctl restart quantum-apply-layer
sudo systemctl restart quantum-governor
sudo systemctl restart quantum-intent-executor
sudo systemctl restart binance-pnl-tracker

# Verify no errors
sudo journalctl -u quantum-governor -n 20 --no-pager | grep -i "auth\|api\|401\|403"
```

### Step 4: Verify New Keys Work

```bash
# Test with dump script (safe - no key exposure)
cd /home/qt/quantum_trader
export $(grep BINANCE_TESTNET /etc/quantum/governor.env | xargs)
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

---

## 🧹 Git History Scrubbing (ADVANCED - OPTIONAL)

**WARNING:** This rewrites Git history. Only do if you control the repo and can force-push.

### Option A: BFG Repo-Cleaner (Recommended)

```bash
# Install BFG
# Windows: Download from https://rtyley.github.io/bfg-repo-cleaner/
# Linux: sudo apt install bfg

# Clone fresh copy
git clone --mirror https://github.com/binyaminsemerci-ops/quantum_trader.git
cd quantum_trader.git

# Create passwords file
echo "e9ZqWhGhAEhDPfNBfQMiJv8zULKJZBIwaaJdfbbUQ8ZNj1WUMumrjenHoRzpzUPD" > ../secrets.txt
echo "ZowBZEfL1R1ValcYLkbxjMfZ1tOxfEDRW4eloWRGGjk5etn0vSFFSU3gCTdCFoja" >> ../secrets.txt

# Scrub history
bfg --replace-text ../secrets.txt

# Force push (DESTRUCTIVE)
git reflog expire --expire=now --all && git gc --prune=now --aggressive
git push --force
```

### Option B: git-filter-repo

```bash
pip install git-filter-repo

# Clone fresh copy
git clone https://github.com/binyaminsemerci-ops/quantum_trader.git
cd quantum_trader

# Create filter expressions
git filter-repo --replace-text <(echo "e9ZqWhGhAEhDPfNBfQMiJv8zULKJZBIwaaJdfbbUQ8ZNj1WUMumrjenHoRzpzUPD==>REDACTED_TESTNET_KEY")
git filter-repo --replace-text <(echo "ZowBZEfL1R1ValcYLkbxjMfZ1tOxfEDRW4eloWRGGjk5etn0vSFFSU3gCTdCFoja==>REDACTED_TESTNET_SECRET")

# Force push
git push --force --all
```

### Option C: Accept Risk & Rotate Only

If history scrubbing is too risky:
1. ✅ Rotate keys (Step 1-4 above)
2. ✅ Old keys now invalid (can't be used)
3. ⚠️  Old commits still contain dead keys (no risk if rotated)
4. 📝 Document in security log

**Recommendation:** Option C is sufficient if keys are rotated promptly.

---

## 🛡️ Prevention: Security Best Practices

### 1. Never Use Inline Credentials
```bash
# ❌ BAD - exposes keys
BINANCE_KEY="xxx" python3 script.py

# ✅ GOOD - via env file
export $(grep BINANCE /etc/quantum/governor.env | xargs)
python3 script.py
```

### 2. Use systemd EnvironmentFile
```bash
# ✅ BEST - systemd loads from protected file
sudo systemd-run --unit=temp-task \
  --setenv=EnvironmentFile=/etc/quantum/governor.env \
  python3 /home/qt/quantum_trader/scripts/dump_exchange_positions.py
```

### 3. Never Print Credentials
```python
# ❌ BAD
print(f"API Key: {api_key}")

# ✅ GOOD
logger.info("API authentication successful")
```

### 4. Use .gitignore for Sensitive Files
```gitignore
# Already in .gitignore
.env
*.env
/config/*.env
```

### 5. Scan Before Commit
```bash
# Install git-secrets
# https://github.com/awslabs/git-secrets

git secrets --scan
git secrets --install
```

---

## 📋 Verification Checklist

- [ ] Old testnet keys deleted on Binance
- [ ] New testnet keys generated
- [ ] All 5 VPS config files updated
- [ ] Services restarted successfully
- [ ] `dump_exchange_positions.py` test passed
- [ ] No auth errors in logs
- [ ] Git history scrubbed (optional)
- [ ] Security practices documented
- [ ] Team notified of key rotation

---

## 📊 Impact Assessment

**Severity:** Medium (Testnet keys only, no mainnet exposure)

**Exposure Window:** January 28, 2026 (today)

**Affected Systems:**
- ✅ Testnet trading (not mainnet)
- ✅ Development/testing environment
- ❌ Production trading (unaffected)
- ❌ Real funds (no risk)

**Risk Level:**
- If keys rotated within 24h: **LOW**
- If keys not rotated: **MEDIUM** (testnet compromise possible)

---

## 🎯 Lessons Learned

1. **Never pass secrets via CLI** - Always use environment files
2. **systemd EnvironmentFile preferred** - Secure, no shell exposure
3. **Script validation needed** - Review credential handling before execution
4. **Testnet keys still sensitive** - Treat as production hygiene practice
5. **AI assistance requires caution** - Verify commands don't expose secrets

---

## 📞 Questions?

If key rotation fails or authentication issues occur:
1. Check `/etc/quantum/*.env` file permissions: Should be `600` or `640`
2. Verify no typos in new keys (copy-paste from Binance)
3. Check service logs: `journalctl -u quantum-governor -n 50`
4. Test with single service first before restarting all

**Status after rotation:** Update this document with:
- [ ] Rotation complete: `[DATE]`
- [ ] New keys verified: `[DATE]`
- [ ] Old keys confirmed invalid: `[DATE]`

---

*Report created: January 28, 2026*  
*Keys scrubbed from repo: Commit 2d90a3b3*  
*Action required: Key rotation within 24 hours*
