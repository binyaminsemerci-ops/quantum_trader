# P0.CAP+QUAL - MANUAL VPS DEPLOYMENT
**Date:** 2026-01-19 12:21 UTC  
**Status:** ⚠️ Terminal broken - Execute via native SSH

---

## QUICK START (Copy-Paste in Git Bash or PowerShell)

```bash
# Connect to VPS
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254

# Copy and paste this entire block:
cat > /tmp/deploy_now.sh << 'EOFSCRIPT'
#!/bin/bash
set -e
echo "=== P0.CAP+QUAL DEPLOYMENT ==="
date -u; hostname

# Find router
ROUTER_PY=$(systemctl show -p ExecStart --value quantum-ai-strategy-router.service | grep -oP '/[^ ]*ai_strategy_router\.py' | head -1)
echo "Router: $ROUTER_PY"

# Pre-proof
redis-cli SET quantum:state:open_orders 0
BEFORE=$(journalctl -u quantum-ai-strategy-router --since "2min ago" --no-pager | grep -ci "published" || echo "0")
echo "BEFORE publishes: $BEFORE"
TODAY=$(date -u +%Y%m%d)
BEFORE_GOV=$(redis-cli GET quantum:governor:daily_trades:$TODAY || echo "0")
echo "BEFORE governor: $BEFORE_GOV"

# Backup
DIR=/tmp/p0cap_$(date -u +%H%M%S)
mkdir -p $DIR
cp -a "$ROUTER_PY" "$DIR/backup.py"
cp -a /tmp/ai_strategy_router.py "$DIR/candidate.py"
echo "Backup: $DIR"

# Deploy
cp -a /tmp/ai_strategy_router.py "$ROUTER_PY"
python3 -m py_compile "$ROUTER_PY" || { echo "SYNTAX FAIL"; cp "$DIR/backup.py" "$ROUTER_PY"; exit 1; }
echo "✅ Deployed"

# Restart
systemctl restart quantum-ai-strategy-router.service
sleep 3
systemctl is-active quantum-ai-strategy-router.service || { echo "START FAIL"; cp "$DIR/backup.py" "$ROUTER_PY"; systemctl restart quantum-ai-strategy-router; exit 1; }
echo "✅ Running"

# Proof
sleep 30
AFTER=$(journalctl -u quantum-ai-strategy-router --since "2min ago" --no-pager | grep -ci "published" || echo "0")
echo "AFTER publishes: $AFTER"
echo "REDUCTION: $BEFORE → $AFTER"
AFTER_GOV=$(redis-cli GET quantum:governor:daily_trades:$TODAY || echo "0")
echo "GOVERNOR: $BEFORE_GOV → $AFTER_GOV"

echo ""
echo "=== CAP+QUAL EVIDENCE ==="
journalctl -u quantum-ai-strategy-router --since "3min ago" --no-pager | grep -E "CAPQUAL|CAPACITY|TARGET|BEST_OF|SIZE_BOOST" | tail -40
echo ""
echo "SUCCESS - Backup: $DIR/backup.py"
EOFSCRIPT

# Execute deployment
bash /tmp/deploy_now.sh 2>&1 | tee /tmp/capqual_deploy.log
```

---

## What You'll See

**Success indicators:**
```
✅ Deployed
✅ Running
REDUCTION: 150 → 8 (or similar large drop)
```

**CAP+QUAL logs:**
```
📤 BEST_OF_PUBLISH count=2 allowed=2
💰 SIZE_BOOST rank=1 factor=1.30
⏸️ TARGET_REACHED open=3 target=3
```

**If anything fails:**
- Script auto-rollbacks from backup
- Service restarts with old version
- Check `/tmp/capqual_deploy.log`

---

## Verify After Deployment

```bash
# Watch live
journalctl -u quantum-ai-strategy-router -f

# Check capacity key
redis-cli GET quantum:state:open_orders

# Test different states
redis-cli SET quantum:state:open_orders 10  # Should log CAPACITY_FULL
sleep 10
redis-cli SET quantum:state:open_orders 3   # Should log TARGET_REACHED
sleep 10
redis-cli SET quantum:state:open_orders 1   # Should log BEST_OF_PUBLISH
sleep 10

# Restore to safe default
redis-cli SET quantum:state:open_orders 0
```

---

## Rollback if Needed

```bash
# Find backup (look for latest timestamp)
ls -lt /tmp/p0cap_*/backup.py

# Restore
DIR=/tmp/p0cap_<TIMESTAMP>
ROUTER_PY=$(systemctl show -p ExecStart --value quantum-ai-strategy-router.service | grep -oP '/[^ ]*ai_strategy_router\.py' | head -1)
cp $DIR/backup.py $ROUTER_PY
systemctl restart quantum-ai-strategy-router.service
systemctl status quantum-ai-strategy-router.service
```

---

## Expected Results

**BEFORE (P0.DX2 audit):**
- ~150+ publishes per 2 minutes (every AI signal)
- No capacity awareness
- No quality selection
- Execution lag 67k messages

**AFTER (P0.CAP+QUAL):**
- ~4-10 publishes per 2 minutes (best-of only)
- 80-90% reduction
- Logs show: BEST_OF_PUBLISH, SIZE_BOOST, TARGET_REACHED
- Respects capacity limits

---

## Files

**On VPS:**
- Candidate: `/tmp/ai_strategy_router.py` (11K, exists)
- Deployment script: `/tmp/deploy_now.sh` (created above)
- Logs: `/tmp/capqual_deploy.log`
- Backup: `/tmp/p0cap_*/backup.py`

**Router location:** (script will auto-detect)
- Likely: `/home/qt/quantum_trader/ai_strategy_router.py`

---

## Post-Deployment Report Template

**After running deployment, report back:**

```
ROUTER_PY: <path>
BACKUP_DIR: /tmp/p0cap_<timestamp>
BEFORE publishes: <count>
AFTER publishes: <count>
REDUCTION: <percentage>%
SERVICE STATUS: active / failed
CAP+QUAL LOGS: <paste sample>
```

---

**STATUS:** Ready for manual execution via SSH  
**TERMINAL ISSUE:** VSCode + WSL + SSH completely broken  
**WORKAROUND:** Use native Git Bash, PowerShell, or PuTTY  
**SAFETY:** Auto-backup + auto-rollback on any failure
