# Exit Brain V3: Dynamic TP Deployment Checklist

## ✅ Pre-Deployment Verification

### Code Quality
- [x] No syntax errors in modified files
- [x] All imports resolved correctly
- [x] Type hints consistent
- [x] Docstrings complete

### Testing
- [x] 15+ unit tests created
- [x] Test coverage for:
  - [x] Dynamic partial TP execution
  - [x] SL ratcheting (LONG and SHORT)
  - [x] Loss guard triggering
  - [x] TP fraction normalization
  - [x] State field initialization

### Documentation
- [x] Full implementation guide created
- [x] Quick reference guide created
- [x] Log patterns documented
- [x] Configuration reference complete

### Backward Compatibility
- [x] No breaking changes to existing code
- [x] All new fields have safe defaults
- [x] Features can be disabled via flags
- [x] Existing behavior preserved when features disabled

---

## 🚀 Deployment Steps

### Step 1: Backup Current State
```bash
# Commit current state before deployment
cd c:/quantum_trader
git add .
git commit -m "Checkpoint before Exit Brain V3 dynamic TP deployment"
git push origin main

# Tag current version
git tag -a exit-brain-v3-pre-dynamic-tp -m "Pre dynamic TP deployment"
git push origin exit-brain-v3-pre-dynamic-tp
```

### Step 2: Verify File Changes
```bash
# List modified files
git status

# Expected files:
# - backend/domains/exits/exit_brain_v3/types.py (modified)
# - backend/domains/exits/exit_brain_v3/dynamic_executor.py (modified)
# - backend/domains/exits/exit_brain_v3/test_dynamic_executor_partial_tp.py (new)
# - AI_EXIT_BRAIN_DYNAMIC_TP_IMPLEMENTATION.md (new)
# - AI_EXIT_BRAIN_DYNAMIC_TP_QUICKREF.md (new)
```

### Step 3: Run Unit Tests (Optional but Recommended)
```bash
# Run dynamic TP tests
pytest backend/domains/exits/exit_brain_v3/test_dynamic_executor_partial_tp.py -v

# Expected: All tests pass
# If any fail, investigate before deploying
```

### Step 4: Deploy to Backend
```bash
# No special deployment needed - files already in place
# Just restart the backend container

systemctl restart quantum_backend

# Wait for restart (usually 10-15 seconds)
sleep 15
```

### Step 5: Verify Backend Started
```bash
# Check container is running
systemctl list-units | grep quantum_backend

# Check logs for initialization
journalctl -u quantum_backend.service --tail 100 | grep "EXIT_BRAIN_EXECUTOR"

# Expected log:
# [EXIT_BRAIN_EXECUTOR] Initialized in LIVE/SHADOW MODE
```

---

## 🔍 Post-Deployment Validation

### Check 1: Loss Guard Active
```bash
# Monitor for loss guard checks (should see periodically)
docker logs -f quantum_backend 2>&1 | grep "EXIT_LOSS_GUARD"

# If no logs appear after 2-3 minutes, loss guard is working silently
# (Only logs when triggered or when checking positions)
```

### Check 2: TP Execution Flow
```bash
# Wait for a TP to trigger on an open position, then check logs
journalctl -u quantum_backend.service --tail 200 | grep -E "EXIT_TP_TRIGGER|remaining_size|tp_hits_count"

# Expected pattern:
# [EXIT_TP_TRIGGER] 🎯 SYMBOL SIDE: TP{N} HIT @ $...
# [EXIT_ORDER] ✅ TP{N} MARKET ... executed
# [EXIT_TP_TRIGGER] 📊 SYMBOL SIDE: Updated state after TP{N} - remaining_size=..., tp_hits_count=...
```

### Check 3: SL Ratcheting
```bash
# After TP trigger, check for ratcheting
journalctl -u quantum_backend.service --tail 200 | grep "EXIT_RATCHET_SL"

# Expected after TP1:
# [EXIT_RATCHET_SL] 🎯 SYMBOL SIDE: SL ratcheted $... → $... - breakeven after TP1 (tp_hits=1)

# Expected after TP2:
# [EXIT_RATCHET_SL] 🎯 SYMBOL SIDE: SL ratcheted $... → $... - TP1 price after TP2 hit (tp_hits=2)
```

### Check 4: No Errors
```bash
# Check for any errors related to new features
journalctl -u quantum_backend.service --tail 500 | grep -i "error" | grep -E "remaining_size|tp_hits|loss_guard|ratchet"

# Expected: No errors
```

### Check 5: State Initialization
```bash
# Check that new positions have dynamic fields initialized
journalctl -u quantum_backend.service --tail 200 | grep "Created new state"

# Expected:
# [EXIT_BRAIN_EXECUTOR] Created new state for SYMBOL:SIDE - entry=$..., initial_size=...
```

---

## 🎯 Validation Tests (Manual)

### Test 1: Monitor Position Through Full TP Ladder

**Setup:**
- Wait for a position to have 2-3 TP levels set
- Note the initial state

**Validate:**
1. TP1 triggers → Check logs:
   ```
   [EXIT_TP_TRIGGER] TP1 HIT
   remaining_size decreased by TP1 size_pct
   tp_hits_count = 1
   SL ratcheted to breakeven
   ```

2. TP2 triggers → Check logs:
   ```
   [EXIT_TP_TRIGGER] TP2 HIT
   remaining_size decreased by TP2 size_pct (of REMAINING, not original)
   tp_hits_count = 2
   SL ratcheted to TP1 price
   ```

**Pass Criteria:**
- ✅ remaining_size decreases correctly after each TP
- ✅ tp_hits_count increments
- ✅ SL ratchets at expected thresholds

### Test 2: Loss Guard Under Extreme Loss

**Setup:**
- Wait for a position to have unrealized PnL approaching -12.5%
- Monitor closely

**Validate:**
- If PnL <= -12.5%, check logs:
  ```
  [EXIT_LOSS_GUARD] 🚨 LOSS GUARD TRIGGERED - unrealized_pnl=-XX% <= -12.5%
  [EXIT_LOSS_GUARD] 🚨 Closing FULL position with MARKET
  [EXIT_ORDER] ✅ LOSS GUARD MARKET ... executed
  ```

**Pass Criteria:**
- ✅ Loss guard triggers at threshold
- ✅ Full position closed
- ✅ State cleared (no duplicate triggers)

### Test 3: TP Fraction Normalization

**Setup:**
- Wait for UPDATE_TP_LIMITS decision with fractions summing > 1.0

**Validate:**
- Check logs:
  ```
  [EXIT_BRAIN_STATE] SYMBOL SIDE: TP fractions sum=X.XX > 1.0, normalized to 1.0
  ```

**Pass Criteria:**
- ✅ Warning logged
- ✅ TP levels set correctly
- ✅ No execution errors

---

## 🚨 Rollback Procedure

### If Issues Detected:

**Quick Rollback:**
```bash
# Revert to pre-deployment state
git reset --hard exit-brain-v3-pre-dynamic-tp

# Restart backend
systemctl restart quantum_backend

# Verify rollback
journalctl -u quantum_backend.service --tail 50 | grep "EXIT_BRAIN_EXECUTOR"
```

**Alternative (Disable Features):**
```python
# In dynamic_executor.py, set:
RATCHET_SL_ENABLED = False
MAX_UNREALIZED_LOSS_PCT_PER_POSITION = 999.9  # Effectively disabled

# Restart backend
systemctl restart quantum_backend
```

### Rollback Validation:
```bash
# Verify old behavior restored
journalctl -u quantum_backend.service --tail 100 | grep -E "EXIT_RATCHET_SL|EXIT_LOSS_GUARD"

# Expected: No new feature logs
```

---

## 📊 Monitoring (First 24 Hours)

### Key Metrics to Watch:

**TP Execution Rate:**
```bash
# Count TP triggers
journalctl -u quantum_backend.service 2>&1 | grep "EXIT_TP_TRIGGER.*HIT" | wc -l
```

**SL Ratcheting Events:**
```bash
# Count ratchet events
journalctl -u quantum_backend.service 2>&1 | grep "EXIT_RATCHET_SL.*ratcheted" | wc -l
```

**Loss Guard Triggers:**
```bash
# Count loss guard fires
journalctl -u quantum_backend.service 2>&1 | grep "LOSS GUARD TRIGGERED" | wc -l
```

**Errors Related to New Features:**
```bash
# Check for errors
journalctl -u quantum_backend.service 2>&1 | grep -i "error" | grep -E "remaining_size|tp_hits|loss_guard|ratchet"
```

### Alert Thresholds:
- 🟢 **Normal:** 0-5 loss guard triggers per day
- 🟡 **Warning:** 5-10 loss guard triggers per day
- 🔴 **Critical:** >10 loss guard triggers per day (investigate market conditions)

---

## 🔧 Configuration Tuning (Post-Deployment)

### If Loss Guard Too Sensitive:
```python
# Widen threshold
MAX_UNREALIZED_LOSS_PCT_PER_POSITION = 15.0  # Was 12.5
```

### If Loss Guard Too Loose:
```python
# Tighten threshold
MAX_UNREALIZED_LOSS_PCT_PER_POSITION = 10.0  # Was 12.5
```

### If Ratcheting Too Aggressive:
```python
# Disable temporarily
RATCHET_SL_ENABLED = False

# Or adjust rules in _recompute_dynamic_tp_and_sl()
# (requires code change)
```

### If Need More TP Levels:
```python
# Increase cap
MAX_TP_LEVELS = 6  # Was 4
```

---

## 📝 Success Criteria

### Deployment is successful if:
- ✅ Backend starts without errors
- ✅ No syntax/import errors in logs
- ✅ Existing positions continue to be monitored
- ✅ TP triggers update remaining_size correctly
- ✅ SL ratcheting fires after TP hits (when enabled)
- ✅ Loss guard checks run without errors
- ✅ No regressions in existing functionality

### Features are working correctly if:
- ✅ TP fractions apply to remaining size (not original)
- ✅ SL moves to breakeven after first TP
- ✅ SL moves to TP1 price after second TP
- ✅ Loss guard triggers at -12.5% PnL threshold
- ✅ All logs use correct tags (EXIT_TP_TRIGGER, EXIT_RATCHET_SL, EXIT_LOSS_GUARD)

---

## 📞 Support Contacts

**For issues:**
1. Check logs first (patterns in quick reference guide)
2. Review test cases for expected behavior
3. Verify configuration values
4. If needed, rollback and investigate

**Documentation:**
- Full guide: `AI_EXIT_BRAIN_DYNAMIC_TP_IMPLEMENTATION.md`
- Quick ref: `AI_EXIT_BRAIN_DYNAMIC_TP_QUICKREF.md`
- Test file: `test_dynamic_executor_partial_tp.py`

---

## 🎉 Post-Deployment Tasks

### After 24 Hours:
- [ ] Review logs for any unexpected patterns
- [ ] Check loss guard trigger frequency
- [ ] Verify SL ratcheting behavior matches expectations
- [ ] Confirm no performance degradation
- [ ] Update team on deployment status

### After 7 Days:
- [ ] Analyze effectiveness:
  - Average PnL per position with dynamic TP vs without
  - Number of loss guard saves (prevented large losses)
  - Profit locked via SL ratcheting
- [ ] Tune thresholds if needed
- [ ] Consider enabling features on more positions

### Long-term:
- [ ] Add Grafana dashboards for TP/ratcheting metrics
- [ ] Implement dynamic TP price adjustment (future enhancement)
- [ ] Add configurable ratchet offsets
- [ ] Create admin API for live threshold adjustments

---

**Deployment Date:** _________  
**Deployed By:** _________  
**Backend Version:** Exit Brain V3 + Dynamic TP v1.0  
**Status:** ⬜ Ready for Deployment

---

## ✅ Final Checklist

Before deployment, confirm:
- [ ] All code changes reviewed
- [ ] Unit tests created and passing
- [ ] Documentation complete
- [ ] Backup/rollback plan ready
- [ ] Monitoring plan established
- [ ] Team notified of deployment

**Sign-off:**
- Developer: _________________ Date: _________
- Reviewer: _________________ Date: _________

