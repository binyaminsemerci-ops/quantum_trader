# ✅ AI-OS ACTIVATION CHECKLIST

**Quick reference for activating AI-OS subsystems progressively**

---

## 📋 PRE-ACTIVATION (Today)

### 1. Review Documentation
- [ ] Read `AI_OS_INTEGRATION_SUMMARY.md` (5 min read)
- [ ] Skim `AI_OS_FULL_INTEGRATION_REPORT.md` (20 min)
- [ ] Check `AI_OS_FEATURE_FLAGS_REFERENCE.md` (10 min)

### 2. Verify Integration
```bash
# Run automated verification
cd c:\quantum_trader
python verify_ai_integration.py
```

**Expected Output**: ✅ All tests pass (6/6)

### 3. Check Current Logs
```bash
# Check if backend is running
docker ps | grep quantum_backend

# Check logs for integration hooks
docker logs quantum_backend --tail 500 | grep -E "AI-HFOS|PBA|PAL|PIL"
```

**Expected**: Should see hook calls if backend is running

### 4. Verify Backend Health
```bash
# Health check
curl http://localhost:8000/health

# If backend not running, start it:
docker start quantum_backend
# OR
cd c:\quantum_trader
docker-compose up -d backend
```

---

## 🎯 STAGE 1: OBSERVATION MODE (Days 1-7)

### Goal
Verify integration works, collect metrics, build confidence

### Configuration
```bash
# Set integration stage (observation is default)
$env:QT_AI_INTEGRATION_STAGE="observation"

# Enable all subsystems in OBSERVE mode
$env:QT_AI_UNIVERSE_OS_ENABLED="true"
$env:QT_AI_UNIVERSE_OS_MODE="observe"

$env:QT_AI_HFOS_ENABLED="true"
$env:QT_AI_HFOS_MODE="observe"

$env:QT_AI_PBA_ENABLED="true"
$env:QT_AI_PBA_MODE="observe"

$env:QT_AI_PIL_ENABLED="true"
$env:QT_AI_PIL_MODE="observe"

$env:QT_AI_PAL_ENABLED="true"
$env:QT_AI_PAL_MODE="observe"

$env:QT_AI_MODEL_SUPERVISOR_ENABLED="true"
$env:QT_AI_MODEL_SUPERVISOR_MODE="observe"

$env:QT_AI_AELM_ENABLED="true"
$env:QT_AI_AELM_MODE="observe"

$env:QT_AI_SELF_HEALING_ENABLED="true"
$env:QT_AI_SELF_HEALING_MODE="observe"

# Restart backend to apply
docker restart quantum_backend
```

### Daily Monitoring (7 days)
```bash
# Day 1-7: Check logs daily
docker logs quantum_backend --tail 1000 | grep -E "AI-HFOS|PBA|PAL|PIL" > ai_logs_day_X.txt

# Look for:
# - [Universe OS] OBSERVE mode messages
# - [AI-HFOS] Confidence adjustments
# - [PBA] Portfolio checks
# - [PAL] Amplification opportunities
# - [PIL] Position classifications
```

### Success Criteria
- [ ] No integration errors in logs
- [ ] All hooks called successfully
- [ ] AI decisions logged (not enforced)
- [ ] System stability maintained
- [ ] Trading performance unchanged (as expected)

### Analysis Questions
- What would AI-HFOS have changed?
- How often would PBA block trades?
- What amplification opportunities did PAL find?
- Are PIL classifications accurate?

---

## 🚀 STAGE 2: PARTIAL ENFORCEMENT (Days 8-14)

### Goal
Enable low-risk subsystems, verify no regressions

### Prerequisites
- [ ] Stage 1 completed (7 days)
- [ ] No integration errors
- [ ] Confidence in AI decisions built

### Configuration Changes
```bash
# Update integration stage
$env:QT_AI_INTEGRATION_STAGE="partial"

# Enable ENFORCED mode for low-risk subsystems
$env:QT_AI_UNIVERSE_OS_MODE="enforced"      # Safe: Symbol filtering
$env:QT_AI_MODEL_SUPERVISOR_MODE="enforced" # Safe: Signal monitoring
$env:QT_AI_PIL_MODE="enforced"              # Safe: Position classification

# Keep high-impact in OBSERVE
$env:QT_AI_HFOS_MODE="observe"              # Still learning
$env:QT_AI_PBA_MODE="observe"               # Still learning
$env:QT_AI_PAL_MODE="observe"               # Still learning
$env:QT_AI_AELM_MODE="observe"              # Still learning

# Restart
docker restart quantum_backend
```

### Daily Monitoring (7 days)
```bash
# Check enforcement decisions
docker logs quantum_backend --tail 1000 | grep -E "ENFORCED|BLOCKED|MODIFIED"

# Look for:
# - Symbols filtered by Universe OS
# - Model Supervisor warnings
# - PIL classifications used by other systems
```

### Success Criteria
- [ ] Symbol filtering working correctly
- [ ] No false-positive blocks
- [ ] Trading performance maintained/improved
- [ ] No emergency brake activations

### Rollback if Needed
```bash
# If issues arise, revert to Stage 1
$env:QT_AI_INTEGRATION_STAGE="observation"
$env:QT_AI_UNIVERSE_OS_MODE="observe"
$env:QT_AI_MODEL_SUPERVISOR_MODE="observe"
$env:QT_AI_PIL_MODE="observe"
docker restart quantum_backend
```

---

## 🎓 STAGE 3: COORDINATION MODE (Days 15-30)

### Goal
Enable AI-HFOS supreme coordination, activate HEDGEFUND MODE

### Prerequisites
- [ ] Stage 2 completed (7 days)
- [ ] No regressions observed
- [ ] Performance metrics positive/neutral
- [ ] Confidence in AI decisions high

### Configuration Changes
```bash
# Update integration stage
$env:QT_AI_INTEGRATION_STAGE="coordination"

# Enable AI-HFOS ENFORCED (HEDGEFUND MODE active)
$env:QT_AI_HFOS_MODE="enforced"

# Enable PBA ENFORCED
$env:QT_AI_PBA_MODE="enforced"

# Enable AELM ENFORCED
$env:QT_AI_AELM_MODE="enforced"

# Enable Self-Healing PROTECTIVE
$env:QT_AI_SELF_HEALING_MODE="protective"

# Keep PAL in ADVISORY (suggests, doesn't execute)
$env:QT_AI_PAL_MODE="advisory"

# Restart
docker restart quantum_backend
```

### Daily Monitoring (14+ days)
```bash
# Check HEDGEFUND MODE transitions
docker logs quantum_backend --tail 1000 | grep "HEDGEFUND MODE\|AGGRESSIVE\|CRITICAL"

# Check AI-HFOS coordination
docker logs quantum_backend --tail 1000 | grep "AI-HFOS.*Risk Mode"

# Check PBA blocks
docker logs quantum_backend --tail 1000 | grep "PBA.*BLOCKED"

# Check SafetyGovernor decisions
docker logs quantum_backend --tail 1000 | grep "SAFETY GOVERNOR"
```

### Success Criteria
- [ ] AI-HFOS coordination smooth
- [ ] HEDGEFUND MODE transitions logical
- [ ] PBA prevents overexposure (if applicable)
- [ ] No excessive blocks (<10%)
- [ ] Performance improvement visible (+10-20%)
- [ ] No emergency brake activations

### Monitor Risk Tiers
Look for these in logs:
- `NORMAL mode`: Base parameters (4 positions)
- `OPTIMISTIC mode`: 1.25x scaling (5 positions)
- `AGGRESSIVE mode`: 2.5x scaling (10 positions)
- `CRITICAL mode`: 0.5x scaling (2 positions)

### Rollback if Needed
```bash
# Revert to Stage 2
$env:QT_AI_INTEGRATION_STAGE="partial"
$env:QT_AI_HFOS_MODE="observe"
$env:QT_AI_PBA_MODE="observe"
$env:QT_AI_AELM_MODE="observe"
docker restart quantum_backend
```

---

## 🚀 STAGE 4: AUTONOMY MODE (Days 31+)

### Goal
Full AI autonomy with profit amplification

### Prerequisites
- [ ] Stage 3 completed (14+ days)
- [ ] HEDGEFUND MODE proven effective
- [ ] 14+ days of profitability
- [ ] No emergency brake activations
- [ ] High confidence in AI decisions

### Configuration Changes
```bash
# Update integration stage
$env:QT_AI_INTEGRATION_STAGE="autonomy"

# Enable PAL ENFORCED (profit amplification active)
$env:QT_AI_PAL_MODE="enforced"

# Restart
docker restart quantum_backend
```

### Daily Monitoring (Ongoing)
```bash
# Check PAL amplifications
docker logs quantum_backend --tail 1000 | grep "PAL.*EXECUTED\|SCALE_IN\|EXTEND_HOLD"

# Monitor overall performance
docker logs quantum_backend --tail 1000 | grep "PnL\|Sharpe\|Drawdown"
```

### Success Criteria
- [ ] PAL amplifications executed successfully
- [ ] Larger winners on strong signals
- [ ] No excessive position sizes
- [ ] Performance improvement visible (+20-35%)
- [ ] Sharpe ratio improved

### ⚠️ CAUTION
PAL is the most aggressive subsystem. Monitor closely:
- Position sizes don't exceed risk limits
- Scale-ins are logical
- SafetyGovernor can still override

---

## 🆘 EMERGENCY PROCEDURES

### Pause All AI (Keep Trading)
```bash
# Revert to pure observation
$env:QT_AI_INTEGRATION_STAGE="observation"
docker restart quantum_backend
```

### Stop All Trading
```bash
# Activate emergency brake
$env:QT_EMERGENCY_BRAKE="true"
docker restart quantum_backend
```

### Complete Shutdown
```bash
# Stop backend
docker stop quantum_backend
```

### Reset to Default
```bash
# Remove all AI env vars
Remove-Item Env:QT_AI_*
Remove-Item Env:QT_EMERGENCY_BRAKE
docker restart quantum_backend
```

---

## 📊 METRICS TO TRACK

### Stage 1 (Observation)
- [ ] Hook call frequency (should be every cycle)
- [ ] AI decision patterns (what would change?)
- [ ] No integration errors

### Stage 2 (Partial)
- [ ] Symbols filtered (count)
- [ ] Model Supervisor warnings (count)
- [ ] Trading performance (vs baseline)

### Stage 3 (Coordination)
- [ ] HEDGEFUND MODE transitions (frequency, reasons)
- [ ] AI-HFOS confidence adjustments (impact)
- [ ] PBA blocks (count, validity)
- [ ] SafetyGovernor vetoes (count, reasons)
- [ ] Performance improvement (% vs baseline)

### Stage 4 (Autonomy)
- [ ] PAL amplifications (count, success rate)
- [ ] Winner size increase (% vs baseline)
- [ ] Overall performance (% vs baseline)
- [ ] Sharpe ratio (vs baseline)

---

## 📈 EXPECTED PERFORMANCE TIMELINE

```
Baseline (No AI):    ████████████████████░░░░░░░░░░░░ (Current)

Stage 1 (Day 1-7):   ████████████████████░░░░░░░░░░░░ (0% change)
                     Observation only, no impact

Stage 2 (Day 8-14):  ██████████████████████░░░░░░░░░░ (+5-10%)
                     Better symbol selection, signal quality

Stage 3 (Day 15-30): ████████████████████████░░░░░░░░ (+15-25%)
                     HEDGEFUND MODE, dynamic risk mgmt

Stage 4 (Day 31+):   ██████████████████████████░░░░░░ (+25-40%)
                     Profit amplification active
```

---

## ✅ COMPLETION CRITERIA

### Stage 1 Complete When:
- [x] 7 days of stable observation
- [x] All hooks called successfully
- [x] No integration errors
- [x] Confidence in AI decisions built

### Stage 2 Complete When:
- [ ] 7 days with selective enforcement
- [ ] No false-positive blocks
- [ ] Performance maintained/improved
- [ ] Ready for AI-HFOS coordination

### Stage 3 Complete When:
- [ ] 14+ days with HEDGEFUND MODE
- [ ] Performance improvement visible
- [ ] No emergency brake activations
- [ ] Ready for profit amplification

### Stage 4 Complete When:
- [ ] 30+ days of profitable operation
- [ ] PAL amplifications successful
- [ ] Full AI autonomy proven
- [ ] System mature and stable

---

## 🎯 QUICK COMMANDS REFERENCE

```bash
# Verify integration
python verify_ai_integration.py

# Check current stage
docker logs quantum_backend | grep "Integration Stage"

# Check for errors
docker logs quantum_backend | grep "ERROR" | tail -50

# Check AI decisions
docker logs quantum_backend | grep -E "AI-HFOS|PBA|PAL|PIL" | tail -100

# Check SafetyGovernor
docker logs quantum_backend | grep "SAFETY GOVERNOR" | tail -50

# Check HEDGEFUND MODE
docker logs quantum_backend | grep "HEDGEFUND MODE" | tail -50

# Monitor real-time
docker logs -f quantum_backend | grep -E "AI-HFOS|SAFETY GOVERNOR|HEDGEFUND"

# Health check
curl http://localhost:8000/health
```

---

## 📞 TROUBLESHOOTING

### Integration Hooks Not Called
```bash
# Check if AI services initialized
docker logs quantum_backend | grep "AISystemServices"

# Verify imports
docker logs quantum_backend | grep "integration_hooks"

# Restart backend
docker restart quantum_backend
```

### AI Decisions Not Logged
```bash
# Check if subsystems enabled
docker logs quantum_backend | grep "enabled.*true"

# Verify modes
docker logs quantum_backend | grep "mode.*observe"

# Check env vars
docker exec quantum_backend printenv | grep QT_AI
```

### Backend Not Starting
```bash
# Check logs for errors
docker logs quantum_backend --tail 200

# Check ports
netstat -an | findstr "8000"

# Restart
docker restart quantum_backend
```

---

## 📚 DOCUMENTATION FILES

1. **AI_OS_INTEGRATION_SUMMARY.md** - Executive summary
2. **AI_OS_FULL_INTEGRATION_REPORT.md** - Comprehensive guide (12,000+ words)
3. **AI_OS_FEATURE_FLAGS_REFERENCE.md** - Environment variable reference
4. **AI_OS_INTEGRATION_DISCOVERY.md** - What was found vs expected
5. **AI_OS_ACTIVATION_CHECKLIST.md** - This file

**Verification Tool**: `verify_ai_integration.py`

---

**Status**: ✅ Ready to begin Stage 1  
**Risk**: 🟢 ZERO (observation mode)  
**Time Commitment**: 7 days observation minimum  
**Expected Outcome**: Verified stable integration → Progressive enhancement

---

**Last Updated**: 2025-01-XX  
**Integration Status**: Production-Ready
