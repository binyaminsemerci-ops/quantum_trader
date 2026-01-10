# P0.5 PATCHTST ACTIVATION RUNBOOK (SYSTEMD-ONLY)

**Environment**: VPS systemd-only (NO DOCKER)  
**Service**: quantum-ai-engine.service  
**User**: qt  
**Redis**: localhost:6379 (systemd)

---

## 1. PURPOSE & PRECONDITIONS

### When to Use This Runbook
- PatchTST has completed shadow observation (24h+ recommended)
- Gate evaluation shows ≥3/4 gates passed (or ≥2/3 if Gate 4 deferred)
- Team approved for production voting activation

### Required Preconditions
✅ **Shadow mode verified operational** (PATCHTST_SHADOW_ONLY=true active)  
✅ **Zero consensus=4 events** in last 24h (voting exclusion confirmed)  
✅ **Service stability** (no restart loops, uptime ≥24h)  
✅ **Gate evaluation completed** (documented results)  
✅ **Rollback plan validated** (backup env file exists)

---

## 2. HARD BLOCKERS (STOP CONDITIONS)

**DO NOT ACTIVATE** if any of these conditions exist:

| Blocker | Check Command | Threshold | Action |
|---------|---------------|-----------|--------|
| **Consensus=4 in shadow** | `redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 100 \| grep -c '"consensus_count": 4'` | 0 | ❌ STOP: Shadow mode broken |
| **PatchTST voting while flag on** | `grep "^PATCHTST_SHADOW_ONLY=true" /etc/quantum/ai-engine.env && redis-cli XREVRANGE ... \| grep consensus_count` | Shadow flag present + consensus≠3 | ❌ STOP: Logic error |
| **Payload p95 spike** | `redis-cli XREVRANGE ... COUNT 100` → measure sizes | <1500 bytes | ❌ STOP: Payload bloat |
| **AI engine restart loop** | `systemctl status quantum-ai-engine.service \| grep "Active:"` | active (running) | ❌ STOP: Service unstable |
| **Gate failures** | (From gate evaluation) | ≥3/4 passed | ❌ STOP: Model not ready |

**If any blocker present**: STOP. Investigate root cause. Extend shadow observation or re-train.

---

## 3. ACTIVATION STRATEGY

### Initial Configuration
- **Voting Weight**: 10% (conservative, vs 20% default)
- **Confidence Cap**: 0.75 (optional safety limit)
- **Observation Window**: First 6h critical monitoring
- **Rollback Threshold**: <60s emergency revert if issues

### Strategy Rationale
**Why 10% weight?**
- Lower than trained weight (20%) to reduce initial impact
- Allows ensemble to dominate if PatchTST misbehaves
- Can increase to 15-20% after 24h if stable

**Why confidence cap?**
- Prevents over-confident predictions from swaying ensemble
- Can remove after calibration validation

**Why 6h observation?**
- Long enough to see consensus patterns across multiple regimes
- Short enough to catch issues before significant PnL impact

---

## 4. EXACT ACTIVATION COMMANDS (SYSTEMD)

### Pre-Activation Verification

```bash
# SSH to VPS
ssh root@46.224.116.254

# 1. Verify shadow mode currently active
grep "^PATCHTST_SHADOW_ONLY=true" /etc/quantum/ai-engine.env
# Expected: PATCHTST_SHADOW_ONLY=true

# 2. Check latest shadow log
journalctl -u quantum-ai-engine.service --since "5 minutes ago" | grep "\[SHADOW\] PatchTST"
# Expected: Recent shadow logs present

# 3. Verify zero consensus=4
redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 50 | awk '/^payload$/ {getline; if (match($0, /"consensus_count": 4/)) print}' | wc -l
# Expected: 0

# 4. Confirm service stable
systemctl is-active quantum-ai-engine.service
# Expected: active
```

### Activation (3-Step Process)

```bash
# STEP 1: Backup current config
cp /etc/quantum/ai-engine.env /etc/quantum/ai-engine.env.bak.activation.$(date +%Y%m%d_%H%M%S)

# STEP 2: Remove shadow flag (use anchored match)
sed -i '/^PATCHTST_SHADOW_ONLY=/d' /etc/quantum/ai-engine.env

# STEP 3: Verify removal
grep "^PATCHTST_SHADOW_ONLY" /etc/quantum/ai-engine.env
# Expected: (empty output)

# STEP 4: Verify MODEL_PATH still present
grep "^PATCHTST_MODEL_PATH" /etc/quantum/ai-engine.env
# Expected: PATCHTST_MODEL_PATH=/opt/quantum/ai_engine/models/patchtst_v20260109_233444.pth

# STEP 5: Restart service
systemctl restart quantum-ai-engine.service

# STEP 6: Wait for initialization
sleep 10

# STEP 7: Verify service active
systemctl is-active quantum-ai-engine.service
# Expected: active
```

### Post-Activation Verification (T+0 to T+5 min)

```bash
# 1. Check PatchTST loaded without shadow mode
journalctl -u quantum-ai-engine.service --since "2 minutes ago" | grep -i "patchtst"
# Expected: "[OK] PatchTST agent loaded (weight: 20.0%)" OR similar
# NOT expected: "[SHADOW]" logs

# 2. Check for consensus=4 (PatchTST now voting)
sleep 60  # Wait for first predictions
redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 5
# Expected: Some events with "consensus_count": 4 (includes PatchTST)

# 3. Verify no shadow flag in payload
redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 1 | grep '"shadow": true'
# Expected: (empty - shadow flag should be absent)

# 4. Confirm PatchTST in model_breakdown
redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 1 | grep '"patchtst":'
# Expected: Present, but "model" field should NOT say "patchtst_shadow"
```

**Activation Complete**: PatchTST now participates in voting with configured weight.

---

## 5. POST-ACTIVATION MONITORING (0-6 HOURS)

### Monitoring Schedule

| Window | Check Frequency | Focus |
|--------|----------------|-------|
| **T+0 to T+30min** | Every 5 min | Consensus distribution, service stability |
| **T+30min to T+2h** | Every 15 min | Agreement rate, confidence patterns |
| **T+2h to T+6h** | Every 30 min | PnL impact, regime changes |

### Critical Metrics

#### Metric 1: Consensus Distribution

```bash
# Extract consensus_count distribution
redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 100 | \
  awk '/^payload$/ {getline; if (match($0, /"consensus_count": ([0-9]+)/, m)) print m[1]}' | \
  sort | uniq -c | sort -rn

# Expected (healthy):
#   40-60  4    ← PatchTST agrees with others
#   20-40  3    ← 1 model disagrees
#   10-20  2    ← Split vote
#    0-5   1    ← Rare

# RED FLAG:
#   >80% consensus=4 → PatchTST is redundant (just copying others)
#   >30% consensus=0-1 → Severe disagreement, likely broken
```

**Threshold**: 40-70% consensus=4 is healthy.

#### Metric 2: PatchTST Agreement with Ensemble

```bash
# Compare PatchTST action to ensemble final side
redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 100 | \
  awk '
    BEGIN {total=0; agree=0}
    /^payload$/ {
      getline payload
      if (match(payload, /"side": "([^"]+)"/, ens) && \
          match(payload, /"patchtst": \{[^}]*"action": "([^"]+)"/, pt)) {
        total++
        if (ens[1] == pt[1]) agree++
      }
    }
    END {
      if (total > 0) {
        rate = int(agree*100/total)
        print "Agreement:", agree "/" total, "(" rate "%)"
        if (rate < 40 || rate > 90) print "⚠️ WARNING: Rate outside 40-90% range"
        else print "✅ OK"
      }
    }
  '

# Expected: 55-75% (same as shadow mode)
# RED FLAG: <40% (contrarian) OR >90% (redundant/copying)
```

**Threshold**: 50-80% agreement.

#### Metric 3: Hard Disagreement Rate

```bash
# Count events where PatchTST predicts opposite of ensemble
redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 100 | \
  awk '
    BEGIN {total=0; opposite=0}
    /^payload$/ {
      getline payload
      if (match(payload, /"side": "([^"]+)"/, ens) && \
          match(payload, /"patchtst": \{[^}]*"action": "([^"]+)"/, pt)) {
        total++
        ensemble = ens[1]
        patchtst = pt[1]
        # Check for direct opposites: BUY vs SELL
        if ((ensemble == "BUY" && patchtst == "SELL") || \
            (ensemble == "SELL" && patchtst == "BUY")) {
          opposite++
        }
      }
    }
    END {
      if (total > 0) {
        rate = int(opposite*100/total)
        print "Hard Disagree:", opposite "/" total, "(" rate "%)"
        if (rate > 25) print "⚠️ WARNING: High contention"
        else print "✅ OK"
      }
    }
  '

# Expected: <20% (low contention)
# RED FLAG: >25% (high conflict, may increase execution volatility)
```

**Threshold**: <25% hard disagreement.

#### Metric 4: Confidence Distribution

```bash
# Check PatchTST confidence range
redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 100 | \
  awk '
    BEGIN {n=0}
    /^payload$/ {
      getline payload
      if (match(payload, /"patchtst": \{[^}]*"confidence": ([0-9.]+)/, conf)) {
        n++
        vals[n] = conf[1]
        sum += conf[1]
      }
    }
    END {
      if (n > 0) {
        # Sort
        for (i=1; i<=n; i++) {
          for (j=i+1; j<=n; j++) {
            if (vals[i] > vals[j]) {tmp = vals[i]; vals[i] = vals[j]; vals[j] = tmp}
          }
        }
        mean = sum / n
        p10 = vals[int(n * 0.10)]
        p50 = vals[int(n * 0.50)]
        p90 = vals[int(n * 0.90)]
        range_p10_p90 = p90 - p10
        
        printf "Mean: %.4f | P10: %.4f | P50: %.4f | P90: %.4f | Range: %.4f\n", mean, p10, p50, p90, range_p10_p90
        
        if (range_p10_p90 < 0.02) print "⚠️ WARNING: Confidence flatlined"
        else print "✅ OK"
      }
    }
  '

# Expected: P10-P90 range > 0.05 (some diversity)
# RED FLAG: Range < 0.02 (flatlined, not learning from features)
```

**Threshold**: P10-P90 range ≥ 0.05.

### Immediate Stop Conditions

**STOP IMMEDIATELY** (rollback within 60s) if:

1. **Service crashes or restart loop**
   - Check: `systemctl status quantum-ai-engine.service`
   - Action: Emergency rollback (Section 6)

2. **Consensus=4 >85% sustained**
   - Check: Consensus distribution command above
   - Reason: PatchTST is redundant, wasting compute
   - Action: Rollback, investigate feature set

3. **Hard disagreement >30%**
   - Check: Hard disagreement command above
   - Reason: High execution conflict, may increase slippage
   - Action: Rollback, review calibration

4. **Confidence collapse (range <0.01)**
   - Check: Confidence distribution command above
   - Reason: Model predicting same value regardless of input
   - Action: Rollback, re-train with regularization

5. **Sudden PnL degradation**
   - Check: Compare 1h PnL post-activation vs 24h pre-activation baseline
   - Threshold: >10% drop in win rate OR >2x increase in loss magnitude
   - Action: Rollback immediately, investigate execution logs

### Monitoring Snapshot Script

Create `/home/qt/quantum_trader/ops/runbooks/activation_monitor.sh`:

```bash
#!/bin/bash
echo "=== T+$(date +%s) PATCHTST ACTIVATION MONITOR ==="
echo "Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%S UTC")"
echo ""

echo "1. Service Status"
systemctl is-active quantum-ai-engine.service
echo ""

echo "2. Consensus Distribution (last 50 events)"
redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 50 | \
  awk '/^payload$/ {getline; if (match($0, /"consensus_count": ([0-9]+)/, m)) print m[1]}' | \
  sort | uniq -c | sort -rn
echo ""

echo "3. Agreement Rate (last 50 events)"
redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 50 | \
  awk 'BEGIN {total=0; agree=0} /^payload$/ {getline; if (match($0, /"side": "([^"]+)"/, ens) && match($0, /"patchtst": \{[^}]*"action": "([^"]+)"/, pt)) {total++; if (ens[1] == pt[1]) agree++}} END {print agree "/" total, "(" int(agree*100/total) "%)"}'
echo ""

echo "4. Hard Disagreement (last 50 events)"
redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 50 | \
  awk 'BEGIN {total=0; opp=0} /^payload$/ {getline; if (match($0, /"side": "([^"]+)"/, ens) && match($0, /"patchtst": \{[^}]*"action": "([^"]+)"/, pt)) {total++; if ((ens[1] == "BUY" && pt[1] == "SELL") || (ens[1] == "SELL" && pt[1] == "BUY")) opp++}} END {print opp "/" total, "(" int(opp*100/total) "%)"}'
echo ""

echo "5. Latest Event"
redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 1 | awk '/^payload$/ {getline; print}' | python3 -m json.tool | grep -E '"symbol"|"side"|"consensus_count"|"total_models"|"patchtst"' | head -10
echo ""

echo "=== END SNAPSHOT ==="
```

Run every 5-15 min during first 6h.

---

## 6. EMERGENCY ROLLBACK (<60 SECONDS)

### Trigger Conditions
- Any immediate stop condition met (Section 5)
- Team decision to abort activation
- Unexpected behavior in production

### Single-Command Rollback

```bash
# Restore shadow mode
echo "PATCHTST_SHADOW_ONLY=true" >> /etc/quantum/ai-engine.env && \
systemctl restart quantum-ai-engine.service && \
sleep 10 && \
echo "Rollback complete. Verifying..." && \
journalctl -u quantum-ai-engine.service --since "1 minute ago" | grep "\[SHADOW\] PatchTST" && \
echo "✅ Shadow mode restored"
```

**Execution Time**: <60 seconds (restart included)

### Rollback Verification

```bash
# 1. Verify shadow flag present
grep "^PATCHTST_SHADOW_ONLY=true" /etc/quantum/ai-engine.env
# Expected: PATCHTST_SHADOW_ONLY=true

# 2. Check shadow logs returning
journalctl -u quantum-ai-engine.service --since "2 minutes ago" | grep "\[SHADOW\] PatchTST"
# Expected: Shadow logs present

# 3. Verify consensus back to 0-3 (not 4)
sleep 60  # Wait for predictions
redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 5 | grep '"consensus_count":'
# Expected: consensus_count = 0, 1, 2, or 3 (NOT 4)
```

### Post-Rollback Actions

1. **Document trigger**: Record which stop condition triggered rollback
2. **Collect logs**: Save last 1h of journal logs for analysis
   ```bash
   journalctl -u quantum-ai-engine.service --since "1 hour ago" > /tmp/rollback_$(date +%Y%m%d_%H%M%S).log
   ```
3. **Analyze root cause**: Review metrics leading to rollback
4. **Update gate simulation**: Adjust thresholds based on findings
5. **Schedule re-attempt**: Coordinate with team before next activation

---

## 7. ACTIVATION DECISION LOG (TEMPLATE)

Copy this template to track activation attempt:

```markdown
# PATCHTST P0.5 ACTIVATION — DECISION LOG

## Pre-Activation
- **Date/Time**: [YYYY-MM-DD HH:MM UTC]
- **Shadow Mode Duration**: [X hours]
- **Gate Status**:
  - Gate 1 (Action Diversity): [PASS/FAIL]
  - Gate 2 (Confidence Spread): [PASS/FAIL]
  - Gate 3 (Agreement): [PASS/FAIL]
  - Gate 4 (Calibration): [PASS/FAIL/DEFER]
  - **Total**: [X/4 or X/3]
- **Hard Blockers**: [None / List any]
- **Approved By**: [Name/Role]
- **Strategy**: Weight=[X]%, Conf Cap=[Y], Observation=[Z]h

## Activation
- **Executed At**: [YYYY-MM-DD HH:MM UTC]
- **Executed By**: [Name/Username]
- **Commands Run**: [List or reference]
- **Initial Verification**: [PASS/FAIL]
- **Notes**: [Any issues during activation]

## Post-Activation Observations

### T+1 Hour
- **Service Status**: [active/crashed/restarted]
- **Consensus Distribution**: [X% at 4, Y% at 3, etc.]
- **Agreement Rate**: [X%]
- **Hard Disagreement**: [X%]
- **Confidence Range**: [P10-P90 = X]
- **Issues**: [None / List]
- **Action**: [Continue / Rollback]

### T+6 Hours
- **Service Uptime**: [X hours]
- **Consensus 4%**: [X%]
- **Agreement Rate**: [X%]
- **PnL Impact**: [+X% / -X% / No measurable change]
- **Decision**: [Continue / Adjust Weight / Rollback]
- **Notes**: [Observations]

### T+24 Hours (Optional)
- **Service Uptime**: [X hours]
- **Long-term Stability**: [Stable / Issues]
- **Ensemble Impact**: [Positive / Neutral / Negative]
- **Next Steps**: [Increase weight / Maintain / Deactivate]

## Rollback (If Applicable)
- **Rollback Executed**: [YES/NO]
- **Rollback Time**: [YYYY-MM-DD HH:MM UTC]
- **Trigger**: [Which stop condition]
- **Verification**: [PASS/FAIL]
- **Root Cause**: [Brief technical explanation]
- **Next Action**: [Re-train / Adjust gates / Extended shadow]

## Conclusion
- **Activation Status**: [SUCCESS / ROLLED BACK / PENDING]
- **Current State**: [Active voting / Shadow mode / Disabled]
- **Lessons Learned**: [Key takeaways]
- **Follow-up Items**: [Action items]
```

---

## 8. POST-ACTIVATION WEIGHT ADJUSTMENT (OPTIONAL)

If activation successful and stable after 24h, consider increasing weight:

### Weight Increase Criteria
✅ **Service uptime >24h** (no crashes)  
✅ **Consensus distribution healthy** (40-70% at consensus=4)  
✅ **Agreement rate stable** (55-75%)  
✅ **PnL neutral or positive** (no significant degradation)

### Adjustment Commands

```bash
# Currently: Weight is hardcoded in agent initialization
# To adjust: Modify ai_engine/ensemble_manager.py

# For now: Weight change requires code modification and service restart
# Future: Add PATCHTST_WEIGHT env variable for dynamic control

# Recommended progression:
# - 10% (initial, conservative)
# - 15% (after 24h if stable)
# - 20% (after 7d if contributing positively)
# - 25% (maximum, only if calibration excellent)
```

**Note**: Weight adjustment requires code change in current implementation. Consider adding `PATCHTST_WEIGHT` env variable in future enhancement.

---

## 9. GO / NO-GO CHECKLIST (ONE-PAGE)

Use this checklist immediately before activation:

```
╔══════════════════════════════════════════════════════════════════╗
║              PATCHTST P0.5 ACTIVATION — GO/NO-GO                ║
╠══════════════════════════════════════════════════════════════════╣
║ PRE-ACTIVATION CHECKLIST                                         ║
║                                                                  ║
║ [ ] Shadow mode active and verified (PATCHTST_SHADOW_ONLY=true) ║
║ [ ] Gate evaluation completed (≥3/4 or ≥2/3 passed)             ║
║ [ ] Zero consensus=4 events in last 24h                          ║
║ [ ] Service stable (no restart loops, uptime ≥24h)              ║
║ [ ] Payload p95 <1500 bytes                                     ║
║ [ ] Rollback procedure validated                                ║
║ [ ] Backup env file created                                     ║
║ [ ] Monitoring scripts ready                                    ║
║ [ ] Team approval obtained                                      ║
║ [ ] Emergency contacts available                                ║
║                                                                  ║
║ HARD BLOCKERS (Any YES = STOP)                                  ║
║                                                                  ║
║ [ ] Consensus=4 found in shadow mode?          [YES = STOP]     ║
║ [ ] Service in restart loop?                   [YES = STOP]     ║
║ [ ] <3/4 gates passed?                         [YES = STOP]     ║
║ [ ] Payload bloat detected?                    [YES = STOP]     ║
║ [ ] High hard disagreement (>30%) in shadow?   [YES = STOP]     ║
║                                                                  ║
║ ACTIVATION PARAMETERS                                            ║
║                                                                  ║
║ [ ] Initial weight: [___]% (recommended: 10%)                   ║
║ [ ] Confidence cap: [___] (recommended: 0.75 or none)           ║
║ [ ] Observation window: [___]h (recommended: 6h)                ║
║ [ ] Rollback threshold: [___]s (recommended: 60s)               ║
║                                                                  ║
║ POST-ACTIVATION MONITORING                                       ║
║                                                                  ║
║ [ ] T+5min: Service active, consensus=4 appearing               ║
║ [ ] T+30min: Consensus distribution healthy (40-70% at 4)       ║
║ [ ] T+1h: Agreement rate 50-80%, hard disagree <25%             ║
║ [ ] T+2h: No stop conditions triggered                          ║
║ [ ] T+6h: Stable operation, no rollback needed                  ║
║                                                                  ║
║ DECISION                                                         ║
║                                                                  ║
║ [ ] GO — All checks passed, activate PatchTST                   ║
║ [ ] NO-GO — Blockers present, extend shadow or re-train         ║
║                                                                  ║
║ Approved By: ________________  Date: ________________            ║
║                                                                  ║
║ Notes:                                                           ║
║ ________________________________________________________________ ║
║ ________________________________________________________________ ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 10. SUMMARY

| Phase | Duration | Key Action | Success Criteria |
|-------|----------|------------|------------------|
| **Pre-Activation** | 15 min | Run checklist, verify gates | All checks ✅, no blockers |
| **Activation** | 5 min | Remove shadow flag, restart | Service active, consensus=4 appearing |
| **Critical Window** | 0-30 min | Monitor every 5 min | No crashes, healthy consensus |
| **Stabilization** | 30min-6h | Monitor every 15-30 min | Metrics within thresholds |
| **Decision Point** | T+6h | Continue or rollback | If stable: continue, increase monitoring interval |
| **Long-term** | 6h-7d | Daily checks | If successful: consider weight increase |

**Rollback Time**: <60s at any point  
**Emergency Contact**: [Define escalation path]  
**Runbook Version**: 1.0  
**Last Updated**: 2026-01-10

---

**END OF RUNBOOK**

**Status**: 📋 READY (Not yet executed)  
**Environment**: VPS systemd-only  
**Model**: patchtst_v20260109_233444.pth (P0.4)
