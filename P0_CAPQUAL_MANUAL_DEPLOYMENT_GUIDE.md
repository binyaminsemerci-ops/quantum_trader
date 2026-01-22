# P0.CAP+QUAL Manual Deployment Guide
**Date:** 2026-01-19 12:17 UTC  
**Status:** ⚠️ READY - Terminal broken, manual execution required

## What P0.CAP+QUAL Does

**CURRENT PROBLEM (from P0.DX2 audit):**
- Router publishes **EVERY** AI signal → 10,001 intents in stream
- No capacity awareness → execution service has 67,448 message lag
- No quality selection → safe mode triggers frequently
- Governor count explodes

**P0.CAP+QUAL SOLUTION:**
- ✅ **Capacity awareness:** Reads open_orders from Redis, fails-closed if missing
- ✅ **Target-open policy:** Aims for 3 orders (not always filling to max 10)
- ✅ **Best-of selection:** Buffers 3s, publishes only top 2-3 by score
- ✅ **Quality filtering:** Min 75% confidence to enter buffer
- ✅ **Position size boost:** Rank 1 (+30%), Rank 2 (+10%)
- ✅ **Expected reduction:** 80-90% fewer intents published

---

## Files Ready for Deployment

**Local (Windows):**
- `c:\quantum_trader\ai_strategy_router.py` (497 lines, P0.CAP+QUAL complete)
- `c:\quantum_trader\deploy_p0_capqual.sh` (automated deployment script)

**Backup created:**
- `c:\quantum_trader\ai_strategy_router.py.backup_p0capqual_*`

---

## Option 1: One-Command Deployment (RECOMMENDED)

**Open PowerShell (not WSL) and run:**

```powershell
# Step 1: Upload files
scp -i C:\Users\<YOUR_USERNAME>\.ssh\hetzner_fresh C:\quantum_trader\deploy_p0_capqual.sh root@46.224.116.254:/tmp/
scp -i C:\Users\<YOUR_USERNAME>\.ssh\hetzner_fresh C:\quantum_trader\ai_strategy_router.py root@46.224.116.254:/home/qt/quantum_trader/ai_strategy_router.py.new

# Step 2: Execute deployment
ssh -i C:\Users\<YOUR_USERNAME>\.ssh\hetzner_fresh root@46.224.116.254 'chmod +x /tmp/deploy_p0_capqual.sh && bash /tmp/deploy_p0_capqual.sh 2>&1 | tee /tmp/deploy_output.log'
```

**What the script does automatically:**
1. ✅ Gather BEFORE metrics (publish rate, governor count, capacity)
2. ✅ Create backup directory with timestamp
3. ✅ Verify uploaded file syntax
4. ✅ Deploy file (move .new → actual)
5. ✅ Set capacity key to 0 (safe default)
6. ✅ Restart router service
7. ✅ Rollback automatically if service fails
8. ✅ Wait 30s and gather AFTER metrics
9. ✅ Compare before/after for proof

---

## Option 2: Manual Step-by-Step

**SSH into VPS:**
```bash
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254
```

### Phase 0: Reconnaissance (BEFORE metrics)

```bash
# System check
date -u
systemctl is-active quantum-ai-strategy-router.service

# Find router location
systemctl cat quantum-ai-strategy-router.service | grep ExecStart

# BEFORE: Publish rate
journalctl -u quantum-ai-strategy-router.service --since "2 minutes ago" --no-pager | grep -c "Trade Intent published"

# BEFORE: Governor count
TODAY=$(date -u +%Y%m%d)
redis-cli GET quantum:governor:daily_trades:$TODAY
redis-cli TTL quantum:governor:daily_trades:$TODAY

# BEFORE: Capacity key
redis-cli GET quantum:state:open_orders

# BEFORE: Dedup keys
redis-cli --scan --pattern "quantum:dedup:trade_intent:*" | wc -l
```

**SAVE THESE NUMBERS** - you'll compare them after deployment.

### Phase 1: Backup

```bash
BACKUP_DIR=/tmp/p0_capqual_$(date -u +%Y%m%d_%H%M%S)
mkdir -p $BACKUP_DIR/backup
cp -a /home/qt/quantum_trader/ai_strategy_router.py $BACKUP_DIR/backup/
echo "Backup: $BACKUP_DIR/backup/ai_strategy_router.py"
```

### Phase 2: Upload Modified File

**From Windows PowerShell:**
```powershell
scp -i C:\Users\<YOUR_USERNAME>\.ssh\hetzner_fresh C:\quantum_trader\ai_strategy_router.py root@46.224.116.254:/home/qt/quantum_trader/ai_strategy_router.py.new
```

**Back on VPS:**
```bash
# Verify syntax
python3 -m py_compile /home/qt/quantum_trader/ai_strategy_router.py.new

# Create diff (optional)
diff -u /home/qt/quantum_trader/ai_strategy_router.py /home/qt/quantum_trader/ai_strategy_router.py.new | head -100

# Deploy
mv /home/qt/quantum_trader/ai_strategy_router.py.new /home/qt/quantum_trader/ai_strategy_router.py
```

### Phase 3: Set Capacity Key (CRITICAL)

```bash
# Set to 0 (safe default - will fail-closed if system is down)
redis-cli SET quantum:state:open_orders 0

# Verify
redis-cli GET quantum:state:open_orders
```

**IMPORTANT:** This key must exist before router starts, or it will publish 0 intents (fail-closed).

### Phase 4: Restart Router

```bash
systemctl restart quantum-ai-strategy-router.service
sleep 3
systemctl status quantum-ai-strategy-router.service --no-pager | head -40
```

**If service FAILS:**
```bash
# Rollback immediately
cp $BACKUP_DIR/backup/ai_strategy_router.py /home/qt/quantum_trader/ai_strategy_router.py
systemctl restart quantum-ai-strategy-router.service
echo "ROLLBACK COMPLETE - Check logs: journalctl -u quantum-ai-strategy-router -n 100"
```

### Phase 5: Wait for Proof

```bash
# Wait 30 seconds for service to process some signals
sleep 30
```

### Phase 6: PROOF AFTER

```bash
# AFTER: Publish rate
journalctl -u quantum-ai-strategy-router.service --since "2 minutes ago" --no-pager | grep -c "Trade Intent published"

# AFTER: P0.CAPQUAL logs (should see new features)
journalctl -u quantum-ai-strategy-router.service --since "2 minutes ago" --no-pager | grep -E "P0.CAP|CAPACITY|TARGET|BEST_OF|SIZE_BOOST" | tail -40

# AFTER: Governor count
TODAY=$(date -u +%Y%m%d)
redis-cli GET quantum:governor:daily_trades:$TODAY

# AFTER: Dedup keys
redis-cli --scan --pattern "quantum:dedup:trade_intent:*" | wc -l

# AFTER: Capacity key
redis-cli GET quantum:state:open_orders
```

**Compare BEFORE vs AFTER:**
- Publish count should be **80-90% lower**
- Governor count should grow **much slower**
- Logs should show: `📤 BEST_OF_PUBLISH`, `💰 SIZE_BOOST`, `⏸️ TARGET_REACHED`

---

## Success Criteria

✅ Service is active after restart  
✅ Publish rate decreased by 80-90%  
✅ Logs show `BEST_OF_PUBLISH` and `SIZE_BOOST`  
✅ No `CAPACITY_UNKNOWN` warnings (means key exists)  
✅ Governor count growth slowed  

---

## Testing Different Capacity States

### Test 1: Capacity Full (should publish 0)
```bash
redis-cli SET quantum:state:open_orders 10
sleep 10
journalctl -u quantum-ai-strategy-router -n 20 | grep CAPACITY_FULL
```
**Expected:** `🔴 CAPACITY_FULL open=10 max=10`

### Test 2: At Target (should publish only super-confidence signals)
```bash
redis-cli SET quantum:state:open_orders 3
sleep 10
journalctl -u quantum-ai-strategy-router -n 20 | grep TARGET_REACHED
```
**Expected:** `⏸️ TARGET_REACHED open=3 target=3 (awaiting super-signal)`

### Test 3: Below Target (should publish top 2)
```bash
redis-cli SET quantum:state:open_orders 1
sleep 10
journalctl -u quantum-ai-strategy-router -n 40 | grep BEST_OF_PUBLISH
```
**Expected:** `📤 BEST_OF_PUBLISH count=2 allowed=2 open=1 slots=9 top=[...]`

### Test 4: Capacity Unknown (should fail-closed)
```bash
redis-cli DEL quantum:state:open_orders
sleep 10
journalctl -u quantum-ai-strategy-router -n 20 | grep CAPACITY_UNKNOWN
```
**Expected:** `🔴 CAPACITY_UNKNOWN: quantum:state:open_orders missing → fail-closed`

**After testing, restore to 0:**
```bash
redis-cli SET quantum:state:open_orders 0
```

---

## Rollback Plan

**If anything goes wrong:**

```bash
# Find your backup
BACKUP_DIR=/tmp/p0_capqual_<YOUR_TIMESTAMP>
ls -la $BACKUP_DIR/backup/

# Restore old file
cp $BACKUP_DIR/backup/ai_strategy_router.py /home/qt/quantum_trader/ai_strategy_router.py

# Restart service
systemctl restart quantum-ai-strategy-router.service

# Verify
systemctl status quantum-ai-strategy-router.service
journalctl -u quantum-ai-strategy-router -n 50
```

---

## What Changed in Code

**NEW CONSTANTS (env overridable):**
```python
MAX_OPEN_ORDERS = 10          # Hard cap
TARGET_OPEN_ORDERS = 3        # Desired normal state
MAX_NEW_PER_CYCLE = 2         # Burst protection
CANDIDATE_WINDOW_SEC = 3      # Best-of window
MIN_CONFIDENCE = 0.75         # Quality gate
SUPER_CONFIDENCE = 0.90       # Bypass target policy
```

**NEW METHODS:**
1. `_read_capacity()` - Read open_orders from Redis (fail-closed if missing)
2. `_compute_allowed_slots()` - Target-open policy logic
3. `_score_candidate()` - Score = confidence + bonuses
4. `_apply_size_boost()` - Rank 1: +30%, Rank 2: +10%
5. `buffer_candidate()` - Non-blocking candidate collection
6. `process_best_of_batch()` - Select + publish top-N
7. `publish_trade_intent()` - Replaces old route_decision()

**MAIN LOOP CHANGE:**
- **BEFORE:** Read → process immediately → ACK
- **AFTER:** Read → buffer → every 3s select best-of → publish top-N → ACK all

**TOTAL MODIFICATIONS:** ~180 lines added/changed (497 lines total)

---

## Monitoring After Deployment

**Watch live logs:**
```bash
journalctl -u quantum-ai-strategy-router.service -f
```

**Look for:**
- ✅ `📤 BEST_OF_PUBLISH count=2 allowed=2` - working
- ✅ `💰 SIZE_BOOST rank=1 factor=1.30` - rank bonuses working
- ✅ `⏸️ TARGET_REACHED open=3` - target policy working
- ❌ `🔴 CAPACITY_UNKNOWN` - capacity key missing (fix: set key)
- ❌ `🔴 CAPACITY_FULL` - too many open orders (normal when at max)

---

## Key Redis Requirements

**MUST exist before deployment:**
```bash
redis-cli SET quantum:state:open_orders 0
```

**Will be updated by execution service:**
- When trade opens → increment
- When trade closes → decrement

**Router reads this key every 3s** to decide how many intents to publish.

---

## Expected Impact

**Current State (BEFORE):**
- 10,001 trade intents in stream
- Router publishes EVERY AI signal
- Execution service has 67,448 message lag
- Safe mode triggers frequently

**After P0.CAP+QUAL (EXPECTED):**
- ~2-3 intents per 3s cycle (only best-of)
- 80-90% reduction in publish rate
- Execution lag should decrease
- Safe mode triggers less often
- Top signals get bigger position sizes (+30%)
- System respects TARGET_OPEN_ORDERS=3 (not always filling to 10)

---

## Terminal Issue Note

VSCode terminal (PowerShell + WSL) is completely broken - all SSH commands return "DifferenceObject[N]" errors. This is why I created this manual guide.

**Use alternative terminals:**
- Windows PowerShell (native, not WSL)
- Git Bash
- PuTTY
- Windows Terminal with SSH

**All files are ready** - just need working SSH access.

---

## Contact Points

**Files:**
- Modified router: `c:\quantum_trader\ai_strategy_router.py`
- Deployment script: `c:\quantum_trader\deploy_p0_capqual.sh`
- Local backup: `c:\quantum_trader\ai_strategy_router.py.backup_p0capqual_*`

**VPS:**
- Server: `root@46.224.116.254`
- SSH key: `~/.ssh/hetzner_fresh`
- Service: `quantum-ai-strategy-router.service`
- Router file: `/home/qt/quantum_trader/ai_strategy_router.py`

**Redis Keys:**
- Capacity: `quantum:state:open_orders` (MUST BE SET)
- Governor: `quantum:governor:daily_trades:YYYYMMDD`
- Dedup: `quantum:dedup:trade_intent:*`

---

## After Deployment - Report Back

**Please send me:**
1. ✅ BEFORE publish count (from Phase 0)
2. ✅ AFTER publish count (from Phase 6)
3. ✅ Sample of P0.CAPQUAL logs
4. ✅ Governor count before/after
5. ✅ Service status after restart

**I'll analyze and confirm:**
- Reduction percentage
- Feature activation
- Any tuning needed
- Follow-up risks

---

**STATUS:** ⚠️ **READY FOR MANUAL DEPLOYMENT**  
**BLOCKER:** Terminal failure - use native PowerShell or Git Bash  
**SAFETY:** Backups created, rollback plan ready, fail-closed if key missing
