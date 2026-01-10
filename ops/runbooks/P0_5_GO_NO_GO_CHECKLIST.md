# PATCHTST P0.5 — GO / NO-GO CHECKLIST

**Date**: ________________  
**Time**: ________________ UTC  
**Operator**: ________________  
**Environment**: VPS systemd-only

---

## SECTION 1: PRE-ACTIVATION CHECKS

| # | Check | Status | Notes |
|---|-------|--------|-------|
| 1 | Shadow mode active (`grep PATCHTST_SHADOW_ONLY=true /etc/quantum/ai-engine.env`) | ☐ PASS ☐ FAIL | |
| 2 | Gate evaluation completed (document exists with ≥3/4 passed) | ☐ PASS ☐ FAIL | |
| 3 | Zero consensus=4 in last 100 events (`redis-cli XREVRANGE ... \| grep -c 'consensus_count": 4'`) | ☐ PASS ☐ FAIL | Count: ___ |
| 4 | Service stable (`systemctl status quantum-ai-engine.service`) | ☐ PASS ☐ FAIL | Uptime: ___ |
| 5 | Payload p95 <1500 bytes (measure from 20 recent events) | ☐ PASS ☐ FAIL | P95: ___ bytes |
| 6 | Backup env file created (`ls -l /etc/quantum/ai-engine.env.bak.*`) | ☐ PASS ☐ FAIL | Path: ___ |
| 7 | Monitoring scripts deployed (`activation_monitor.sh` exists) | ☐ PASS ☐ FAIL | |
| 8 | Emergency rollback tested (dry-run executed) | ☐ PASS ☐ FAIL | |
| 9 | Team approval obtained (Slack/email confirmation) | ☐ PASS ☐ FAIL | Approver: ___ |
| 10 | Emergency contacts available (ops team on-call) | ☐ PASS ☐ FAIL | |

**Pre-Activation Result**: ☐ ALL PASS → Proceed to Section 2  
**Pre-Activation Result**: ☐ ANY FAIL → STOP, investigate failures

---

## SECTION 2: HARD BLOCKERS

**Any YES = IMMEDIATE STOP**

| # | Blocker | Check | Result | Action |
|---|---------|-------|--------|--------|
| 1 | Consensus=4 in shadow mode | `redis-cli ... \| grep -c '"consensus_count": 4'` | ☐ 0 (OK) ☐ >0 (STOP) | If >0: Shadow broken, investigate |
| 2 | Service restart loop | `systemctl status ... \| grep -c "restarts"` | ☐ 0 (OK) ☐ >0 (STOP) | If >0: Fix stability first |
| 3 | <3/4 gates passed | (From gate eval) | ☐ ≥3/4 (OK) ☐ <3/4 (STOP) | If <3: Re-train required |
| 4 | Payload bloat (p95 >1500) | (From check #5 above) | ☐ <1500 (OK) ☐ ≥1500 (STOP) | If bloat: Optimize payload |
| 5 | High hard disagreement (>30%) in shadow | `redis-cli ... \| awk ...` | ☐ <30% (OK) ☐ ≥30% (STOP) | If high: Calibration issue |

**Blocker Result**: ☐ ALL OK → Proceed to Section 3  
**Blocker Result**: ☐ ANY STOP → ABORT ACTIVATION

---

## SECTION 3: ACTIVATION PARAMETERS

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Initial Weight** | _____% | (Recommended: 10%, conservative start) |
| **Confidence Cap** | _____ | (Recommended: 0.75 or none) |
| **Observation Window** | _____h | (Recommended: 6h, critical monitoring) |
| **Rollback Threshold** | _____s | (Recommended: 60s, emergency revert) |
| **Monitoring Frequency** | T+0-30min: ___min, T+30min-6h: ___min | (Recommended: 5min, 15min) |

**Parameters Approved By**: ________________  
**Date/Time**: ________________

---

## SECTION 4: ACTIVATION EXECUTION

### Commands to Execute

```bash
# 1. Create backup
cp /etc/quantum/ai-engine.env /etc/quantum/ai-engine.env.bak.activation.$(date +%Y%m%d_%H%M%S)

# 2. Remove shadow flag
sed -i '/^PATCHTST_SHADOW_ONLY=/d' /etc/quantum/ai-engine.env

# 3. Verify removal
grep "^PATCHTST_SHADOW_ONLY" /etc/quantum/ai-engine.env
# Expected: (empty)

# 4. Restart service
systemctl restart quantum-ai-engine.service

# 5. Wait for init
sleep 10

# 6. Verify active
systemctl is-active quantum-ai-engine.service
# Expected: active
```

### Execution Log

| Step | Time (UTC) | Status | Notes |
|------|------------|--------|-------|
| Backup created | _____:_____ | ☐ OK ☐ FAIL | Path: ___ |
| Shadow flag removed | _____:_____ | ☐ OK ☐ FAIL | |
| Verification passed | _____:_____ | ☐ OK ☐ FAIL | |
| Service restarted | _____:_____ | ☐ OK ☐ FAIL | |
| Service active | _____:_____ | ☐ OK ☐ FAIL | |

**Activation Completed**: ☐ YES at _____:_____ UTC  
**Activation Failed**: ☐ YES, reason: ________________

---

## SECTION 5: POST-ACTIVATION VERIFICATION (T+5 MIN)

| # | Check | Command | Result | Notes |
|---|-------|---------|--------|-------|
| 1 | PatchTST loaded without shadow | `journalctl ... \| grep patchtst` | ☐ OK ☐ FAIL | |
| 2 | Consensus=4 appearing | `redis-cli XREVRANGE ... COUNT 5` | ☐ OK ☐ FAIL | Found: ___/5 |
| 3 | No shadow flag in payload | `redis-cli ... \| grep '"shadow": true'` | ☐ Empty (OK) ☐ Found (FAIL) | |
| 4 | PatchTST in model_breakdown | `redis-cli ... \| grep '"patchtst":'` | ☐ OK ☐ FAIL | |

**T+5min Result**: ☐ ALL OK → Continue monitoring  
**T+5min Result**: ☐ ANY FAIL → ROLLBACK IMMEDIATELY

---

## SECTION 6: MONITORING CHECKPOINTS

### T+30 Minutes

| Metric | Command | Value | Threshold | Status |
|--------|---------|-------|-----------|--------|
| Service uptime | `systemctl status ...` | ___min | >25min | ☐ OK ☐ FAIL |
| Consensus=4 % | `redis-cli ... \| awk ...` | ___% | 40-70% | ☐ OK ☐ FAIL |
| Agreement rate | `redis-cli ... \| awk ...` | ___% | 50-80% | ☐ OK ☐ FAIL |
| Hard disagreement | `redis-cli ... \| awk ...` | ___% | <25% | ☐ OK ☐ FAIL |

**T+30min Decision**: ☐ Continue ☐ Rollback (reason: _______________)

### T+1 Hour

| Metric | Value | Threshold | Status | Notes |
|--------|-------|-----------|--------|-------|
| Service uptime | ___h | >55min | ☐ OK ☐ FAIL | |
| Consensus=4 % | ___% | 40-70% | ☐ OK ☐ FAIL | |
| Agreement rate | ___% | 50-80% | ☐ OK ☐ FAIL | |
| Confidence range (P10-P90) | _____ | ≥0.05 | ☐ OK ☐ FAIL | |

**T+1h Decision**: ☐ Continue ☐ Rollback (reason: _______________)

### T+2 Hours

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Service uptime | ___h | >115min | ☐ OK ☐ FAIL |
| Consensus distribution stable | ☐ YES ☐ NO | Stable | ☐ OK ☐ FAIL |
| Agreement rate stable | ___% | 50-80% | ☐ OK ☐ FAIL |
| No stop conditions triggered | ☐ YES ☐ NO | None | ☐ OK ☐ FAIL |

**T+2h Decision**: ☐ Continue ☐ Rollback (reason: _______________)

### T+6 Hours (Final Critical Window)

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Service uptime | ___h | >5.9h | ☐ OK ☐ FAIL |
| Total predictions | _____ | >200 | ☐ OK ☐ FAIL |
| Consensus=4 % (6h avg) | ___% | 40-70% | ☐ OK ☐ FAIL |
| Agreement rate (6h avg) | ___% | 50-80% | ☐ OK ☐ FAIL |
| Hard disagreement (6h avg) | ___% | <25% | ☐ OK ☐ FAIL |
| Confidence range | _____ | ≥0.05 | ☐ OK ☐ FAIL |
| PnL impact | ☐ Neutral ☐ Positive ☐ Negative | Neutral/Positive | ☐ OK ☐ FAIL |

**T+6h Final Decision**:
- ☐ SUCCESS → Continue to 24h observation, reduce monitoring frequency
- ☐ ROLLBACK → Execute emergency rollback, document reasons

---

## SECTION 7: IMMEDIATE STOP CONDITIONS

**ROLLBACK IMMEDIATELY if any of these occur:**

| Stop Condition | Detected | Time (UTC) | Action Taken |
|----------------|----------|------------|--------------|
| Service crash or restart loop | ☐ YES ☐ NO | ___:___ | |
| Consensus=4 >85% sustained | ☐ YES ☐ NO | ___:___ | |
| Hard disagreement >30% | ☐ YES ☐ NO | ___:___ | |
| Confidence collapse (range <0.01) | ☐ YES ☐ NO | ___:___ | |
| Sudden PnL degradation (>10% drop) | ☐ YES ☐ NO | ___:___ | |

**Emergency Rollback Executed**: ☐ YES at ___:___ UTC  
**Rollback Reason**: ________________

---

## SECTION 8: EMERGENCY ROLLBACK (IF NEEDED)

### Rollback Command

```bash
echo "PATCHTST_SHADOW_ONLY=true" >> /etc/quantum/ai-engine.env && \
systemctl restart quantum-ai-engine.service && \
sleep 10 && \
journalctl -u quantum-ai-engine.service --since "1 minute ago" | grep "\[SHADOW\] PatchTST"
```

### Rollback Verification

| Check | Result | Notes |
|-------|--------|-------|
| Shadow flag present | ☐ OK ☐ FAIL | |
| Shadow logs returning | ☐ OK ☐ FAIL | |
| Consensus back to 0-3 (not 4) | ☐ OK ☐ FAIL | |
| Service stable | ☐ OK ☐ FAIL | |

**Rollback Verified**: ☐ YES at ___:___ UTC  
**Rollback Failed**: ☐ YES, escalate to: ________________

---

## SECTION 9: FINAL STATUS

### Activation Outcome

☐ **SUCCESS** — PatchTST active, stable after 6h, continue monitoring  
☐ **ROLLED BACK** — Activation failed, returned to shadow mode  
☐ **PENDING** — Still in observation window (fill out at T+6h)

### Summary

| Field | Value |
|-------|-------|
| **Activation Start Time** | ___:___ UTC |
| **Activation End Time** (T+6h or rollback) | ___:___ UTC |
| **Total Predictions (6h)** | _____ |
| **Average Consensus=4 %** | ___% |
| **Average Agreement Rate** | ___% |
| **Stop Conditions Triggered** | ☐ None ☐ List: _______________ |
| **Rollback Executed** | ☐ NO ☐ YES at ___:___ |
| **Final Status** | ☐ Active ☐ Shadow ☐ Disabled |

### Lessons Learned

1. ________________________________________________________
2. ________________________________________________________
3. ________________________________________________________

### Follow-Up Actions

- [ ] ________________________________________________________
- [ ] ________________________________________________________
- [ ] ________________________________________________________

### Approval & Sign-Off

| Role | Name | Signature | Date/Time |
|------|------|-----------|-----------|
| **Operator** | | | |
| **Approver** | | | |
| **Reviewer** | | | |

---

**Checklist Version**: 1.0  
**Last Updated**: 2026-01-10  
**Environment**: VPS systemd-only (NO DOCKER)

---

## QUICK REFERENCE COMMANDS

```bash
# Monitor service
systemctl status quantum-ai-engine.service

# Check consensus distribution
redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 100 | awk '/^payload$/ {getline; if (match($0, /"consensus_count": ([0-9]+)/, m)) print m[1]}' | sort | uniq -c

# Run monitoring snapshot
/home/qt/quantum_trader/ops/runbooks/activation_monitor.sh

# Emergency rollback (single command)
echo "PATCHTST_SHADOW_ONLY=true" >> /etc/quantum/ai-engine.env && systemctl restart quantum-ai-engine.service
```

---

**END OF CHECKLIST**

**Status**: 📋 READY TO USE  
**Print this page for manual tracking during activation**
