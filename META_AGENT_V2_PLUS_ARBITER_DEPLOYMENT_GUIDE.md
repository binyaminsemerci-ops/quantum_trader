# 🚀 Meta-Agent V2 + Arbiter: Deployment Guide

**Architecture Implemented:** Feb 6, 2026  
**Ready for:** VPS Deployment + Production Testing

---

## ✅ Implementation Status

### **COMPLETED**

✅ **Arbiter Agent #5** (`ai_engine/agents/arbiter_agent.py`)
- Market understanding with technical indicators
- Gating logic (confidence ≥ 0.70, action != HOLD)
- Statistics tracking
- **Lines:** 450

✅ **Meta-Agent V2 Refactor** (`ai_engine/meta/meta_agent_v2.py`)
- Changed from direct trading decisions → policy layer
- Returns DEFER or ESCALATE (not action)
- New `_analyze_disagreement()` method
- Escalation triggers: split vote, high disagreement, low confidence, high entropy
- **Lines modified:** ~150

✅ **Ensemble Manager Integration** (`ai_engine/ensemble_manager.py`)
- Imports Arbiter Agent
- 3-layer decision flow implemented
- Clear logging for each decision stage
- **Lines modified:** ~180

✅ **Unit Tests** (`ai_engine/tests/test_arbiter_agent.py`)
- 20+ test cases for Arbiter
- Tests all technical indicators (RSI, MACD, BB, volume, momentum)
- Tests gating logic
- Tests statistics tracking
- **Lines:** 450

✅ **Integration Tests** (`test_meta_v2_arbiter_integration.py`)
- 6 test scenarios covering full decision flow
- Validates Meta defer/escalate logic
- Validates Arbiter gating
- Statistics validation
- **Lines:** 430

✅ **Documentation** (`META_AGENT_V2_PLUS_ARBITER_ARCHITECTURE.md`)
- Complete architecture guide
- Decision flow diagrams
- Example scenarios
- Monitoring guide
- **Lines:** 750

---

## 📁 Files Changed

```
NEW FILES (3):
✅ ai_engine/agents/arbiter_agent.py                    (450 lines)
✅ ai_engine/tests/test_arbiter_agent.py                (450 lines)
✅ test_meta_v2_arbiter_integration.py                  (430 lines)
✅ META_AGENT_V2_PLUS_ARBITER_ARCHITECTURE.md           (750 lines)
✅ META_AGENT_V2_PLUS_ARBITER_DEPLOYMENT_GUIDE.md       (this file)

MODIFIED FILES (2):
✅ ai_engine/meta/meta_agent_v2.py                      (~150 lines changed)
✅ ai_engine/ensemble_manager.py                        (~180 lines changed)

TOTAL: 5 new files, 2 modified files, ~2,410 lines of code
```

---

## 🔧 Deployment Steps

### **STEP 1: Upload Files to VPS**

```bash
# From local machine (C:\quantum_trader)
scp -r ai_engine/agents/arbiter_agent.py root@46.224.116.254:/home/qt/quantum_trader/ai_engine/agents/
scp ai_engine/meta/meta_agent_v2.py root@46.224.116.254:/home/qt/quantum_trader/ai_engine/meta/
scp ai_engine/ensemble_manager.py root@46.224.116.254:/home/qt/quantum_trader/ai_engine/
scp ai_engine/tests/test_arbiter_agent.py root@46.224.116.254:/home/qt/quantum_trader/ai_engine/tests/
scp test_meta_v2_arbiter_integration.py root@46.224.116.254:/home/qt/quantum_trader/
scp META_AGENT_V2_PLUS_ARBITER_ARCHITECTURE.md root@46.224.116.254:/home/qt/quantum_trader/docs/
```

**Windows PowerShell (using WSL):**
```powershell
wsl scp -i ~/.ssh/hetzner_fresh -r /mnt/c/quantum_trader/ai_engine/agents/arbiter_agent.py root@46.224.116.254:/home/qt/quantum_trader/ai_engine/agents/

wsl scp -i ~/.ssh/hetzner_fresh /mnt/c/quantum_trader/ai_engine/meta/meta_agent_v2.py root@46.224.116.254:/home/qt/quantum_trader/ai_engine/meta/

wsl scp -i ~/.ssh/hetzner_fresh /mnt/c/quantum_trader/ai_engine/ensemble_manager.py root@46.224.116.254:/home/qt/quantum_trader/ai_engine/

wsl scp -i ~/.ssh/hetzner_fresh /mnt/c/quantum_trader/test_meta_v2_arbiter_integration.py root@46.224.116.254:/home/qt/quantum_trader/
```

---

### **STEP 2: Run Integration Tests (VPS)**

```bash
ssh root@46.224.116.254

cd /home/qt/quantum_trader

# Run integration test
/opt/quantum/venvs/ai-engine/bin/python test_meta_v2_arbiter_integration.py
```

**Expected Output:**
```
======================================================================
        META-AGENT V2 + ARBITER INTEGRATION TESTS
======================================================================

──────────────────────────────────────────────────────────────────────
  SCENARIO 1: Strong Consensus (3 BUY, 1 HOLD)
──────────────────────────────────────────────────────────────────────
✅ PASS: Meta DEFERS to base ensemble (strong consensus)

──────────────────────────────────────────────────────────────────────
  SCENARIO 2: Split Vote (2 BUY, 2 SELL)
──────────────────────────────────────────────────────────────────────
✅ PASS: Meta ESCALATES (split vote detected)
✅ PASS: Arbiter provides decision: BUY @ 0.780

──────────────────────────────────────────────────────────────────────
        TEST SUMMARY
──────────────────────────────────────────────────────────────────────
ALL TESTS PASSED (6/6)
```

---

### **STEP 3: Enable Meta-V2 (First Stage)**

Meta-V2 is a **policy layer** only. It decides IF we need escalation.  
Arbiter is NOT enabled yet in this stage.

```bash
ssh root@46.224.116.254

# Edit service file
sudo nano /etc/systemd/system/quantum-ai-engine.service
```

**Add/Update environment variables:**
```ini
[Service]
Environment="META_AGENT_ENABLED=true"
Environment="ARBITER_ENABLED=false"
```

**Restart service:**
```bash
sudo systemctl daemon-reload
sudo systemctl restart quantum-ai-engine
sudo systemctl status quantum-ai-engine
```

**Monitor logs:**
```bash
journalctl -u quantum-ai-engine -f | grep -E 'Meta-V2-Policy|DEFER|ESCALATE'
```

**Expected behavior:**
- `[Meta-V2-Policy] DEFER`: ~70-80% (most cases)
- `[Meta-V2-Policy] ESCALATE`: ~20-30% (split votes, disagreements)
- When escalated WITHOUT Arbiter enabled: Falls back to base ensemble

**Validation:**
```bash
# Check Meta-V2 is loading
journalctl -u quantum-ai-engine -n 50 | grep 'META-V2'

# Should see:
# [✅ META-V2] Meta-learning agent loaded (5th layer)
#    └─ Override threshold: 0.65
#    └─ Version: v2
```

---

### **STEP 4: Enable Arbiter (Second Stage)**

After Meta-V2 runs successfully for 1-2 hours, enable Arbiter.

```bash
sudo nano /etc/systemd/system/quantum-ai-engine.service
```

**Update:**
```ini
Environment="ARBITER_ENABLED=true"
Environment="ARBITER_THRESHOLD=0.70"
```

**Restart:**
```bash
sudo systemctl daemon-reload
sudo systemctl restart quantum-ai-engine
```

**Monitor:**
```bash
journalctl -u quantum-ai-engine -f | grep -E 'Arbiter|OVERRIDE|DEFER'
```

**Expected behavior:**
- `[Arbiter] ⚖️  INVOKED`: Called when Meta escalates
- `[Arbiter] ✅ OVERRIDE`: ~10-15% (high confidence + active trade)
- `[Arbiter] ⏸️  DEFER`: ~15-20% (low confidence or HOLD)

---

### **STEP 5: Validation Checklist**

After deployment, validate:

#### **5.1 Service Health**
```bash
# Service running
systemctl is-active quantum-ai-engine
# Output: active

# No restart errors
journalctl -u quantum-ai-engine --since "5 minutes ago" | grep -i error
# Output: (empty or no critical errors)
```

#### **5.2 Meta-V2 Policy Working**
```bash
# Check defer rate (~70-80%)
journalctl -u quantum-ai-engine --since "1 hour ago" | grep -c 'Meta-V2-Policy.*DEFER'

# Check escalation rate (~20-30%)
journalctl -u quantum-ai-engine --since "1 hour ago" | grep -c 'Meta-V2-Policy.*ESCALATE'
```

#### **5.3 Arbiter Invocation**
```bash
# Check Arbiter is being called
journalctl -u quantum-ai-engine --since "1 hour ago" | grep 'Arbiter.*INVOKED'

# Check Arbiter override rate
journalctl -u quantum-ai-engine --since "1 hour ago" | grep -c 'Arbiter.*OVERRIDE'

# Check Arbiter defer rate
journalctl -u quantum-ai-engine --since "1 hour ago" | grep -c 'Arbiter.*DEFER'
```

#### **5.4 Decision Distribution**
```bash
# Last 20 predictions (should show hierarchy)
journalctl -u quantum-ai-engine --since "10 minutes ago" | grep -E 'ENSEMBLE.*BTCUSDT|Meta-V2-Policy|Arbiter' | tail -20
```

Expected pattern:
```
[ENSEMBLE] BTCUSDT: BUY 0.72 | XGB:BUY/0.75 LGBM:BUY/0.70 NH:HOLD/0.65 PT:BUY/0.72
[Meta-V2-Policy] DEFER: using base ensemble BUY@0.720 | Reason: strong_consensus_buy

[ENSEMBLE] ETHUSDT: SELL 0.68 | XGB:BUY/0.70 LGBM:SELL/0.72 NH:SELL/0.66 PT:SELL/0.68
[Meta-V2-Policy] ESCALATE: split_vote → Calling Arbiter
[Arbiter] INVOKED (escalated from Meta-V2)
[Arbiter] OVERRIDE: SELL@0.680 → SELL@0.760 | Reason: sell_signal: overbought_rsi
```

---

## 📊 Performance Monitoring

### **Key Metrics Dashboard**

Create monitoring queries:

```bash
# Escalation rate (target: 20-30%)
ESCALATIONS=$(journalctl -u quantum-ai-engine --since "1 hour ago" | grep -c 'ESCALATE')
TOTAL=$(journalctl -u quantum-ai-engine --since "1 hour ago" | grep -c 'Meta-V2-Policy')
RATE=$(echo "scale=2; $ESCALATIONS * 100 / $TOTAL" | bc)
echo "Escalation rate: $RATE%"

# Arbiter override rate when called (target: 40-60%)
ARBITER_CALLS=$(journalctl -u quantum-ai-engine --since "1 hour ago" | grep -c 'Arbiter.*INVOKED')
ARBITER_OVERRIDES=$(journalctl -u quantum-ai-engine --since "1 hour ago" | grep -c 'Arbiter.*OVERRIDE')
OVERRIDE_RATE=$(echo "scale=2; $ARBITER_OVERRIDES * 100 / $ARBITER_CALLS" | bc)
echo "Arbiter override rate: $OVERRIDE_RATE%"
```

### **Grafana Integration** (Optional)

If you have Grafana:

1. Parse logs → Prometheus metrics
2. Track:
   - `meta_v2_escalation_rate`
   - `arbiter_invocation_count`
   - `arbiter_override_rate`
   - `final_decision_source` (ensemble vs arbiter)

---

## 🔄 Rollback Plan

If issues occur:

### **Quick Rollback (Disable Arbiter)**
```bash
sudo nano /etc/systemd/system/quantum-ai-engine.service

# Change:
Environment="ARBITER_ENABLED=false"

sudo systemctl daemon-reload
sudo systemctl restart quantum-ai-engine
```

System will continue with Meta-V2 policy (defer/escalate) but fall back to base ensemble on escalation.

### **Full Rollback (Disable Meta-V2)**
```bash
# Disable Meta-V2
Environment="META_AGENT_ENABLED=false"
Environment="ARBITER_ENABLED=false"

sudo systemctl daemon-reload
sudo systemctl restart quantum-ai-engine
```

System returns to pure base ensemble (4 models, weighted voting).

### **Emergency: Restore Previous Files**

```bash
cd /home/qt/quantum_trader

# Git restore (if committed before deployment)
git checkout HEAD -- ai_engine/meta/meta_agent_v2.py
git checkout HEAD -- ai_engine/ensemble_manager.py

# Or restore from backup
cp ai_engine/meta/meta_agent_v2.py.backup ai_engine/meta/meta_agent_v2.py
cp ai_engine/ensemble_manager.py.backup ai_engine/ensemble_manager.py

sudo systemctl restart quantum-ai-engine
```

---

## 🐛 Troubleshooting

### **Issue 1: Meta-V2 Not Escalating**

**Symptom:** 100% defer rate

**Check:**
```bash
# Verify META_AGENT_ENABLED
systemctl show quantum-ai-engine | grep META_AGENT_ENABLED

# Check base predictions
journalctl -u quantum-ai-engine -n 100 | grep 'ENSEMBLE.*BTCUSDT'
```

**Likely cause:** All predictions have strong consensus or low disagreement

**Solution:** Normal behavior if market is clear. Monitor for 1-2 hours.

---

### **Issue 2: Arbiter Never Overrides**

**Symptom:** Arbiter called but always defers

**Check:**
```bash
# Look for defer reasons
journalctl -u quantum-ai-engine | grep 'Arbiter.*DEFER' | tail -20
```

**Likely causes:**
- Arbiter confidence too low (<0.70)
- Arbiter returning HOLD (by design)
- Threshold too high

**Solution:**
```bash
# Lower threshold temporarily
Environment="ARBITER_THRESHOLD=0.65"

sudo systemctl daemon-reload
sudo systemctl restart quantum-ai-engine
```

---

### **Issue 3: High Escalation Rate (>50%)**

**Symptom:** Meta escalates more than 50% of cases

**Check:**
```bash
# Review disagreement patterns
journalctl -u quantum-ai-engine | grep 'disagreement_ratio' | tail -20
```

**Likely cause:** High market volatility or model disagreement

**Solution:** 
- Monitor for stability over 24 hours
- If persistent: retrain base models
- Consider adjusting consensus threshold

---

## 📈 Expected Performance Improvements

### **Before (Base Ensemble Only)**
- Decision source: Weighted voting (4 models)
- Edge cases: Sometimes unclear (split votes)
- Confidence: Moderate in uncertain markets

### **After (Meta-V2 + Arbiter)**
- **~75% cases:** Base ensemble (proven, stable)
- **~20% escalations:** Policy check identifies uncertainty
- **~10-15% Arbiter overrides:** Market understanding in edge cases
- **Result:** Better handling of split votes and uncertain markets

---

## 🎯 Success Criteria

**Week 1** (Meta-V2 only):
- ✅ Escalation rate: 20-30%
- ✅ Zero prediction errors
- ✅ Defer rate: 70-80%

**Week 2** (Meta-V2 + Arbiter):
- ✅ Arbiter called: 20-30% of predictions
- ✅ Arbiter override rate: 40-60% when called
- ✅ Final decision source: ~85% ensemble, ~15% arbiter

**Week 3** (Performance):
- ✅ No degradation in base ensemble performance
- ✅ Improved handling of split votes
- ✅ Arbiter adds value in uncertain markets

---

## 📞 Support Commands

```bash
# Real-time monitoring
journalctl -u quantum-ai-engine -f

# Last 50 decisions
journalctl -u quantum-ai-engine --since "10 minutes ago" | grep 'ENSEMBLE\|Meta-V2\|Arbiter'

# Error checking
journalctl -u quantum-ai-engine --since "1 hour ago" | grep -i error

# Restart if needed
sudo systemctl restart quantum-ai-engine && journalctl -u quantum-ai-engine -f
```

---

## ✅ Deployment Checklist

- [ ] Files uploaded to VPS
- [ ] Integration tests passed
- [ ] Service file updated with META_AGENT_ENABLED=true
- [ ] Service restarted successfully
- [ ] Meta-V2 loading confirmed in logs
- [ ] Defer/escalate behavior observed
- [ ] Wait 1-2 hours, monitor stability
- [ ] Enable ARBITER_ENABLED=true
- [ ] Arbiter invocation confirmed
- [ ] Decision hierarchy validated
- [ ] Performance metrics collected
- [ ] Monitoring dashboard updated

---

**Ready for deployment!** 🚀

For questions or issues, check:
- `META_AGENT_V2_PLUS_ARBITER_ARCHITECTURE.md` (architecture details)
- `ai_engine/meta/meta_agent_v2.py` (implementation)
- `ai_engine/agents/arbiter_agent.py` (implementation)
- Integration test results
