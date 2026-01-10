# PATCHTST SHADOW MODE DEPLOYMENT GUIDE

**Mission**: Deploy retrained PatchTST in shadow-only observation mode  
**Impact**: Zero ensemble voting impact, full telemetry collection  
**Rollback**: <2 minutes

---

## 🎯 WHAT IS SHADOW MODE?

Shadow mode allows PatchTST to:
- ✅ Run full inference on every prediction
- ✅ Log detailed metrics (action, confidence, logits)
- ✅ Appear in `trade.intent` telemetry with `"shadow": true`
- ❌ **NOT contribute to ensemble voting** (consensus remains 3/4 of XGB/LGBM/NHiTS)
- ❌ **NOT affect execution decisions**

### Why Shadow Mode?

The retrained PatchTST model (P0.4) shows:
- ✅ **450 unique confidence values** (vs baseline's 2)
- ❌ **100% WIN prediction bias** (class imbalance issue)

Shadow mode lets us:
1. Observe real-world behavior without risk
2. Collect calibration data
3. Validate eval gates before activation

---

## 🚨 POLICY PRESERVATION GUARANTEES

### What CANNOT Change (Enforced)
- ❌ Ensemble weights (XGB 25 / LGBM 25 / NHiTS 30 / PatchTST 20)
- ❌ Consensus rules (3/4 required, 4/4 boosts confidence)
- ❌ Execution gates (governor kill=1 still active)
- ❌ Vote counting logic

### What DOES Change
- ✅ PatchTST predictions excluded from `model_actions` count
- ✅ PatchTST marked with `"shadow": true` in telemetry
- ✅ Enhanced logging every 30s in shadow mode

**Verification**: Consensus count will show 3 total models instead of 4 when shadow mode active.

---

## 📋 PRE-DEPLOYMENT CHECKLIST

### 1. Verify Code Changes Deployed
```bash
# On VPS
ssh root@46.224.116.254

# Check patchtst_agent.py has shadow code
grep -n "PATCHTST_SHADOW_ONLY" /home/qt/quantum_trader/ai_engine/agents/patchtst_agent.py
# Expected: Line showing os.getenv check

# Check ensemble_manager.py has filtering
grep -n "shadow_predictions" /home/qt/quantum_trader/ai_engine/ensemble_manager.py
# Expected: Line showing shadow prediction filter
```

### 2. Stage New Model
```bash
# Copy trained model from P0.4 retrain
sudo cp /tmp/patchtst_retrain/20260109_233419/patchtst_v20260109_233444.pth \
    /opt/quantum/ai_engine/models/patchtst_v20260109_233444.pth

sudo chown qt:qt /opt/quantum/ai_engine/models/patchtst_v20260109_233444.pth

# Verify file exists
ls -lh /opt/quantum/ai_engine/models/patchtst_v20260109_233444.pth
# Expected: 2.4 MB file
```

### 3. Backup Current Configuration
```bash
# Backup baseline model
sudo cp /opt/quantum/ai_engine/models/patchtst_model.pth \
    /opt/quantum/ai_engine/models/patchtst_model_BASELINE_20260110.pth

# Backup current env config
sudo cp /etc/quantum/ai-engine.env /etc/quantum/ai-engine.env.backup.20260110
```

---

## 🚀 DEPLOYMENT PROCEDURE

### Step 1: Enable Shadow Mode + New Model (2 minutes)

```bash
# Set both flags (order matters: model path first, then shadow mode)
cat << 'EOF' | sudo tee -a /etc/quantum/ai-engine.env
PATCHTST_MODEL_PATH=/opt/quantum/ai_engine/models/patchtst_v20260109_233444.pth
PATCHTST_SHADOW_ONLY=true
EOF

# Verify config
sudo cat /etc/quantum/ai-engine.env | grep PATCHTST
# Expected output:
# PATCHTST_MODEL_PATH=/opt/quantum/ai_engine/models/patchtst_v20260109_233444.pth
# PATCHTST_SHADOW_ONLY=true
```

### Step 2: Restart AI Engine
```bash
sudo systemctl restart quantum-ai-engine.service

# Wait 10 seconds for initialization
sleep 10
```

### Step 3: Verify Shadow Mode Active (1 minute)

```bash
# Check model loaded with shadow mode
sudo journalctl -u quantum-ai-engine.service --since "1 minute ago" | grep -E "PatchTST.*SHADOW|PATCHTST_MODEL_PATH"

# Expected output:
# [PatchTST] Using PATCHTST_MODEL_PATH env var: /opt/quantum/ai_engine/models/patchtst_v20260109_233444.pth
# [PatchTST] ✅ Model weights loaded from patchtst_v20260109_233444.pth

# Check for shadow logs (wait 30s for rate-limited log)
sleep 30
sudo journalctl -u quantum-ai-engine.service --since "1 minute ago" | grep "SHADOW"

# Expected output (every ~30s):
# [SHADOW] PatchTST | BTCUSDT | action=BUY conf=0.6207 | prob=0.6207 logit=0.5012 | mode=SHADOW_ONLY
```

### Step 4: Verify Voting Exclusion (2 minutes)

```bash
# Extract recent ensemble decisions
sudo journalctl -u quantum-ai-engine.service --since "2 minutes ago" | grep "model_breakdown" | tail -5

# Check each payload for:
# 1. "patchtst" key exists
# 2. "shadow": true present in patchtst entry
# 3. consensus_count is 2-3 (not 4, since PatchTST excluded)

# Example expected structure:
# 'patchtst': {'action': 'BUY', 'confidence': 0.6207, 'model': 'patchtst_shadow', 'shadow': True}
# 'consensus_count': 2  # (XGB + LGBM, NHiTS disagreed)
```

---

## 📊 MONITORING & VALIDATION

### Immediate Checks (T+0 to T+10 min)

```bash
# 1. Service health
sudo systemctl status quantum-ai-engine.service
# Expected: active (running)

# 2. No error spam
sudo journalctl -u quantum-ai-engine.service --since "10 minutes ago" | grep -i error | wc -l
# Expected: 0-2 errors (transient network issues OK)

# 3. Publish rate stable
sudo journalctl -u quantum-ai-engine.service --since "10 minutes ago" | grep "Publishing trade.intent" | wc -l
# Expected: ~10-20 events (1-2 per minute)
```

### Shadow Metrics Analysis (T+2 hours)

```bash
# Run analysis script
cd /home/qt/quantum_trader
python3 ops/analysis/analyze_shadow_metrics.py --hours 2

# Expected output sections:
# 1. Action Distribution (BUY/SELL/HOLD %)
# 2. Confidence Statistics (mean, stddev, histogram)
# 3. Shadow vs Ensemble correlation
# 4. Evaluation Gates (4 checks)
```

### Success Criteria (T+24 hours)

#### ✅ PASS Criteria:
- No service crashes or restarts
- Shadow logs appear every ~30s
- All `trade.intent` events include `patchtst` with `"shadow": true`
- Consensus count never includes PatchTST (remains 2-3, not 4)
- `analyze_shadow_metrics.py` shows ≥3/4 eval gates passed

#### ❌ FAIL Criteria (Trigger Rollback):
- Service crashes >2 times in 24h
- Shadow predictions affecting consensus (count = 4)
- Disk usage increase >5 GB
- Log spam >1000 lines/minute

---

## 🔁 ROLLBACK PROCEDURE (<2 minutes)

### Fast Rollback (Remove Shadow Mode, Keep New Model)
```bash
# Remove shadow flag only
sudo sed -i '/PATCHTST_SHADOW_ONLY/d' /etc/quantum/ai-engine.env

# This makes new model active voter (use if shadow metrics look good)
sudo systemctl restart quantum-ai-engine.service
```

### Full Rollback (Restore Baseline Everything)
```bash
# Remove both flags
sudo sed -i '/PATCHTST_MODEL_PATH/d' /etc/quantum/ai-engine.env
sudo sed -i '/PATCHTST_SHADOW_ONLY/d' /etc/quantum/ai-engine.env

# Optionally restore baseline model explicitly
echo 'PATCHTST_MODEL_PATH=/opt/quantum/ai_engine/models/patchtst_model_BASELINE_20260110.pth' | \
    sudo tee -a /etc/quantum/ai-engine.env

# Restart
sudo systemctl restart quantum-ai-engine.service

# Verify baseline active
sudo journalctl -u quantum-ai-engine.service --since "1 minute ago" | grep "PatchTST.*baseline"
```

### Nuclear Rollback (Restore Config Backup)
```bash
sudo cp /etc/quantum/ai-engine.env.backup.20260110 /etc/quantum/ai-engine.env
sudo systemctl restart quantum-ai-engine.service
```

---

## 🧪 EVALUATION GATES (For Activation)

PatchTST can exit shadow mode **ONLY IF** ≥3/4 gates pass:

### Gate 1: Action Diversity ✅
**Test**: No single action >70% of predictions  
**Command**: `python3 ops/analysis/analyze_shadow_metrics.py | grep "Action Diversity"`  
**Pass**: BUY ≤70%, SELL ≤70%, HOLD ≤70%

### Gate 2: Confidence Spread ✅
**Test**: stddev(confidence) ≥ 0.05  
**Command**: `python3 ops/analysis/analyze_shadow_metrics.py | grep "Stddev"`  
**Pass**: σ ≥ 0.05 (shows model isn't flatlined)

### Gate 3: Shadow Correlation ✅
**Test**: ≥55% agreement with ensemble  
**Command**: `python3 ops/analysis/analyze_shadow_metrics.py | grep "Agreement Rate"`  
**Pass**: Agreement ≥55% (model aligns with ensemble majority)

### Gate 4: Calibration ✅
**Test**: Higher confidence → higher accuracy (monotonic)  
**Command**: `python3 ops/analysis/analyze_shadow_metrics.py | grep "Calibration"`  
**Pass**: Accuracy increases with confidence buckets

### Running All Gates
```bash
# Automated gate check
cd /home/qt/quantum_trader
python3 ops/analysis/analyze_shadow_metrics.py --hours 24

# Look for final summary:
# ✅ READY FOR ACTIVATION (≥3/4 gates passed)
# OR
# ❌ NOT READY (need ≥3/4 gates)
```

---

## 🎓 ACTIVATION PROCESS (Post-Shadow)

**IF** eval gates pass after 7 days of shadow observation:

### Option A: Activate at 20% Weight (Full Participation)
```bash
# Remove shadow flag, keep new model
sudo sed -i '/PATCHTST_SHADOW_ONLY/d' /etc/quantum/ai-engine.env
sudo systemctl restart quantum-ai-engine.service

# PatchTST now votes with 20% weight
```

### Option B: Activate at 5% Weight (Conservative)
```bash
# 1. Edit ensemble_manager.py weights
sudo nano /home/qt/quantum_trader/ai_engine/ensemble_manager.py

# Find: self.weights = {...}
# Change: 'patchtst': 20 → 'patchtst': 5
# Adjust others to sum to 100: XGB 27.5 / LGBM 27.5 / NHiTS 40 / PatchTST 5

# 2. Remove shadow flag
sudo sed -i '/PATCHTST_SHADOW_ONLY/d' /etc/quantum/ai-engine.env

# 3. Restart
sudo systemctl restart quantum-ai-engine.service

# 4. Monitor for 7 more days, then consider increasing to 20%
```

### Option C: Keep in Shadow (Gates Failed)
```bash
# Do nothing - shadow mode continues indefinitely
# Re-evaluate after collecting more diverse training data
```

---

## 🔍 DEBUGGING COMMON ISSUES

### Issue 1: Shadow Logs Not Appearing
```bash
# Check env vars loaded
sudo systemctl show quantum-ai-engine.service --property=Environment
# Should show PATCHTST_SHADOW_ONLY=true

# Force a prediction by checking recent activity
sudo journalctl -u quantum-ai-engine.service --since "5 minutes ago" | grep "trade.intent" | wc -l
# If 0, AI engine might be paused or no market activity
```

### Issue 2: PatchTST Still Voting (Shadow Not Working)
```bash
# Check consensus_count in payloads
sudo journalctl -u quantum-ai-engine.service --since "5 minutes ago" | grep "consensus_count.*4"
# If found, shadow filtering broken - rollback immediately

# Emergency rollback
sudo sed -i '/PATCHTST_SHADOW_ONLY/d' /etc/quantum/ai-engine.env
sudo systemctl restart quantum-ai-engine.service
```

### Issue 3: Model Load Failure
```bash
# Check model file exists and readable
ls -lh /opt/quantum/ai_engine/models/patchtst_v20260109_233444.pth
# Should show 2.4 MB, owned by qt:qt

# Check logs for load errors
sudo journalctl -u quantum-ai-engine.service --since "10 minutes ago" | grep -i "patchtst.*error"

# If model corrupted, re-copy from /tmp/patchtst_retrain/
```

---

## 📁 FILE REFERENCE

### Modified Files (Shadow Mode)
```
ai_engine/agents/patchtst_agent.py
├─ Added: PATCHTST_SHADOW_ONLY env check
├─ Added: Enhanced shadow logging (rate-limited 30s)
└─ Changed: Return 'patchtst_shadow' model name when in shadow mode

ai_engine/ensemble_manager.py
├─ Added: shadow_predictions dict
├─ Added: Vote filtering for shadow models
└─ Changed: Re-add shadow predictions to telemetry with 'shadow': True flag

ops/analysis/analyze_shadow_metrics.py
├─ Purpose: Analyze shadow predictions vs ensemble
├─ Metrics: Action dist, confidence stats, correlation, calibration
└─ Output: 4 eval gates pass/fail
```

### Environment Variables
```bash
# /etc/quantum/ai-engine.env
PATCHTST_MODEL_PATH=/opt/quantum/ai_engine/models/patchtst_v20260109_233444.pth
PATCHTST_SHADOW_ONLY=true
```

---

## 🎯 DECISION TREE

```
┌─ Deploy Shadow Mode ─────────────────────────────────┐
│                                                       │
│  Wait 7 days, collect ~10,000 predictions            │
│                                                       │
│  Run: analyze_shadow_metrics.py                      │
│          │                                            │
│          ├─ ≥3/4 Gates Pass? ────── YES ──┐          │
│          │                                 │          │
│          NO                                │          │
│          │                                 ▼          │
│          ▼                                            │
│   Option 1: Keep Shadow                   Activate!  │
│   Wait for more diverse data              (Option A  │
│   Re-train with class balancing           or B)      │
│                                                       │
│   Option 2: Re-train Now                             │
│   Use class weights or SMOTE                         │
│   Deploy new model in shadow                         │
│                                                       │
└───────────────────────────────────────────────────────┘
```

---

## 📞 SUPPORT & NEXT STEPS

### If Shadow Mode Works Well (Gates Pass)
1. Document findings in P0.4 report addendum
2. Activate at 5-20% weight (decision based on correlation %)
3. Monitor for 7 days post-activation
4. Consider integrating with continuous learning pipeline

### If Shadow Mode Reveals Issues
1. Keep in shadow indefinitely
2. Collect calibration data for 30+ days
3. Re-train with:
   - Balanced class sampling (50/50 WIN/LOSS per batch)
   - True time series input (OHLCV sequences, not tabular)
   - More features (20+ technical indicators)
4. Deploy new model in shadow again

### Emergency Contacts
- Rollback: Any team member with SSH access
- Analysis: Run `analyze_shadow_metrics.py --hours <N>`
- Questions: Refer to [PATCHTST_P0_4_REPORT.md](PATCHTST_P0_4_REPORT.md)

---

**Deployment Checklist**:
- [ ] Code changes pulled from main branch
- [ ] New model copied to `/opt/quantum/ai_engine/models/`
- [ ] Baseline backed up
- [ ] Env flags set (MODEL_PATH + SHADOW_ONLY)
- [ ] Service restarted
- [ ] Shadow logs verified
- [ ] Consensus exclusion confirmed
- [ ] Monitoring scheduled (T+2h, T+24h, T+7d)
- [ ] Analysis script tested

**Last Updated**: 2026-01-10  
**Version**: Shadow Mode v1.0 (P0.4)  
**Rollback Time**: <2 minutes
