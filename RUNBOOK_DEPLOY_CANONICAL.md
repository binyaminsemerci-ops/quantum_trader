# Canonical Deploy Runbook
**Purpose:** Standard deployment flow for quantum_trader production  
**Status:** Active (established 2026-02-03)  
**Owner:** Operations Team

---

## Standard Deploy Flow

**Use this for all normal deployments. Avoid manual file copy unless emergency.**

### Prerequisites
- ✅ Changes committed on development machine
- ✅ Tests passing locally (if applicable)
- ✅ VPS SSH access configured (`~/.ssh/hetzner_fresh`)

---

## Step 1: Push to Origin (Development Machine)

```bash
cd c:\quantum_trader  # or /path/to/quantum_trader on Linux/Mac
git status            # Verify clean state
git add <files>
git commit -m "descriptive message"
git push origin main
```

**Verification:**
```bash
git log --oneline -1 origin/main
# Should show your commit at top
```

---

## Step 2: Pull on VPS (Production)

```bash
# SSH to VPS
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254

# Navigate to quantum_trader
cd /home/qt/quantum_trader

# Fetch and reset to origin/main (safest)
git fetch origin
git reset --hard origin/main

# Verify commit
git log --oneline -1
```

**Expected:** Should match origin/main commit from Step 1

---

## Step 3: Restart Service (VPS)

### For Intent Executor
```bash
systemctl restart quantum-intent-executor
sleep 3
systemctl status quantum-intent-executor --no-pager -l | head -40
```

### For Other Services
Replace service name as needed:
- `quantum-apply-layer`
- `quantum-position-state-brain`
- `quantum-intent-bridge`
- etc.

**Verification:**
```bash
# Check service is active
systemctl is-active quantum-intent-executor

# View recent logs
journalctl -u quantum-intent-executor --since "5 minutes ago" --no-pager | tail -50
```

---

## Step 4: Run Proof Script (VPS)

```bash
cd /home/qt/quantum_trader

# Run relevant proof script
bash scripts/proof_intent_executor_exit_owner.sh

# Or other proofs as needed
# bash scripts/proof_exit_owner_gate.sh
# bash scripts/proof_policy_refresh.sh
```

**Expected:** All tests PASS

**If proof fails:**
1. Check service logs: `journalctl -u <service> -n 100`
2. Verify git commit: `git log --oneline -3`
3. Check for syntax errors: `python3 -m py_compile <file>.py`
4. Rollback if needed (see Emergency Rollback section)

---

## Alignment Verification (Optional)

Ensure all 3 sources of truth are synchronized:

```bash
# On VPS
cd /home/qt/quantum_trader
git log --oneline -1

# On development machine
git log --oneline -1 origin/main

# Compare manually - should be identical commit SHA
```

---

## Emergency Procedures

### Emergency Rollback
```bash
# On VPS
cd /home/qt/quantum_trader
git log --oneline -5  # Find previous good commit
git reset --hard <commit-sha>
systemctl restart <service>
bash scripts/proof_<relevant>.sh
```

### Emergency Deploy (When git push broken)
**Use ONLY when git pipeline is unavailable:**

```bash
# On development machine (Windows with WSL)
cd /mnt/c/quantum_trader
cat <file> | ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'cat > /home/qt/quantum_trader/<file>'

# On VPS - commit the change
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'cd /home/qt/quantum_trader && git add <file> && git commit -m "emergency: <reason>"'

# Restart and verify
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'systemctl restart <service> && bash /home/qt/quantum_trader/scripts/proof_*.sh'
```

**After emergency deploy:** Push VPS commit to origin to restore alignment:
```bash
# On VPS
cd /home/qt/quantum_trader
git push origin HEAD:main
```

---

## Health Checks

### Quick Sanity Check
```bash
# Check service status
systemctl is-active quantum-intent-executor

# Check recent logs (last 10 minutes)
journalctl -u quantum-intent-executor --since "10 minutes ago" | tail -50

# Check for errors
journalctl -u quantum-intent-executor --since "10 minutes ago" | grep -i error

# Check for DENY events (exit ownership enforcement)
journalctl -u quantum-intent-executor --since "10 minutes ago" | grep "DENY_NOT_EXIT_OWNER exec_boundary"
```

### Full Health Verification
```bash
# All services
systemctl status quantum-* --no-pager | grep "Active:"

# Redis connectivity
redis-cli PING

# Check metrics
redis-cli HGETALL quantum:metrics:intent_executor

# Run all relevant proofs
cd /home/qt/quantum_trader
bash scripts/proof_intent_executor_exit_owner.sh
bash scripts/proof_exit_owner_gate.sh
bash scripts/proof_policy_refresh.sh
```

---

## Common Issues

### Issue: "Permission denied (publickey)" on git push
**Solution:** Use VPS as push bridge or set up SSH key on dev machine

### Issue: Proof script fails on "service not running"
**Solution:** Normal on non-Linux dev machines. Proof will skip Test 1 automatically.

### Issue: Merge conflict on VPS
**Solution:** Use `git reset --hard origin/main` (destructive but safe for VPS)

### Issue: Service fails to start after deploy
**Check:**
1. Syntax errors: `python3 -m py_compile microservices/<service>/main.py`
2. Import errors: `python3 -c "import microservices.<service>.main"`
3. Config errors: Check env vars in systemd service file
4. Rollback to previous commit

---

## Service-Specific Notes

### Intent Executor
- **Proof:** `scripts/proof_intent_executor_exit_owner.sh` (3 tests)
- **Key metric:** `quantum:metrics:intent_executor` hash
- **Critical log:** `DENY_NOT_EXIT_OWNER exec_boundary` (exit gate enforcement)

### Apply Layer
- **Proof:** `scripts/proof_exit_owner_gate.sh` (6 tests)
- **Key feature:** Exit ownership enforcement at plan creation

### Policy Services
- **Proof:** `scripts/proof_policy_refresh.sh` (10 tests)
- **Timers:** policy-refresh.timer (30min), exit-owner-watch.timer (5min)
- **Check timers:** `systemctl list-timers policy-refresh exit-owner-watch`

---

## Notes

- **Git as source of truth:** Always deploy via git. Manual file copy is emergency-only.
- **VPS can push:** VPS has SSH key for GitHub. Use for emergency commits.
- **Proof scripts are portable:** Will skip service checks on non-Linux machines.
- **Reset is safe:** `git reset --hard origin/main` is safe on VPS (stateless code deploy).

---

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-02-03 | Initial runbook created | GitHub Copilot |
| 2026-02-03 | Added emergency procedures | GitHub Copilot |
| 2026-02-03 | Added health checks section | GitHub Copilot |

---

**Last Updated:** 2026-02-03  
**Next Review:** 2026-03-03 (or after major architecture change)
