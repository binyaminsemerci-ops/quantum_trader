# QSC FAIL-CLOSED DEPLOYMENT RUNBOOK

**Date:** 2026-01-11  
**Version:** QSC v2.0 with Fail-Closed Hardening  
**Commit:** d13e87bd

---

## 🛡️ FAIL-CLOSED PHILOSOPHY

**"Hvis systemet stopper deg – så har det reddet deg"**

**Changes from QSC v1:**
- ✅ Feature sanity check validates pipeline before any deployment
- ✅ XGB/PatchTST raise exceptions (INACTIVE) instead of silent fallback
- ✅ Degeneracy detection prevents constant-output models from "passing"
- ✅ Collection vs Canary modes prevent accidental safety bypass
- ✅ QSC mode blocks collection-mode reports

---

## 📋 PREREQUISITES

1. **VPS Access:** ssh root@46.224.116.254 (key: ~/.ssh/hetzner_fresh)
2. **Git repo:** /home/qt/quantum_trader (pull latest)
3. **Python venv:** /opt/quantum/venvs/ai-engine/bin/python
4. **Testnet active:** BINANCE_TESTNET=true in /etc/quantum/ai-engine.env
5. **Rate limits raised:** MAX_SIGNALS_PER_MINUTE=60, SYMBOL_COOLDOWN_SECONDS=10

---

## 🔬 STEP 1: Deploy Code (VPS)

```bash
# SSH to VPS
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254

# Pull latest commit
cd /home/qt/quantum_trader
git fetch origin
git checkout main
git pull origin main

# Verify commit
git log --oneline -1
# Should show: d13e87bd QSC FAIL-CLOSED HARDENING

# Restart AI engine
sudo systemctl restart quantum-ai-engine.service

# Check startup logs
journalctl -u quantum-ai-engine.service --since "30 seconds ago" | tail -50

# Look for:
# - [XGB-INIT] Model file: ...
# - [PatchTST-INIT] Model file: ...
# - [XGB] QSC FAIL-CLOSED: ... (if models fail to load)
```

**Expected behavior after restart:**
- XGB/PatchTST log model metadata at startup
- If models missing/corrupted: RuntimeError raised, ensemble marks INACTIVE
- No silent fallback to HOLD 0.5

---

## 🔬 STEP 2: Run Feature Sanity Check

```bash
# Record new cutover timestamp from restart
CUTOVER=$(systemctl show -p ActiveEnterTimestamp quantum-ai-engine.service | awk '{print $2" "$3}')
CUTOVER_ISO=$(date -d "$CUTOVER" -u +%Y-%m-%dT%H:%M:%SZ)
echo "Cutover: $CUTOVER_ISO"

# Wait 2-3 minutes for data collection
sleep 180

# Run feature sanity check (requires >=50 events)
cd /home/qt/quantum_trader
/opt/quantum/venvs/ai-engine/bin/python ops/model_safety/feature_sanity.py \
  --after "$CUTOVER_ISO" \
  --count 200

# Check exit code
echo "Exit code: $?"
```

**Exit codes:**
- `0` = PASS (features healthy)
- `2` = FAIL (NaN/Inf/flatlines detected)

**If FAIL:**
- Check which features are flatlined
- If >30% flatlined: Feature pipeline broken, investigate feature engineering
- If NaN/Inf: Data feed issue or normalization bug
- **DO NOT PROCEED** to quality gate until features fixed

---

## 🔬 STEP 3: Run Quality Gate (Collection Mode First)

```bash
# Collection mode: min_events=100, gathers data, exit 3 (never promotes)
/opt/quantum/venvs/ai-engine/bin/python ops/model_safety/quality_gate.py \
  --mode collection \
  --after "$CUTOVER_ISO"

# Check output:
# - "📦 COLLECTION MODE: min_events=100, will exit 3 (NO PROMOTION)"
# - Event count (should be >=100)
# - Model analysis results

# Exit codes:
# 0 = Impossible in collection mode
# 2 = FAIL (blockers or insufficient data)
# 3 = Collection complete (expected)
```

**Collection mode purpose:**
- Gather telemetry with relaxed threshold (100 events vs 200)
- Diagnose model behavior without risk of accidental promotion
- Report saved with `_collection` suffix

**If RC=2 (FAIL):**
- Check "INSUFFICIENT DATA" (need >=100 events, wait longer)
- Check "Quality violations detected" (model collapse)
- Read report: `reports/safety/quality_gate_YYYYMMDD_HHMMSS_collection_post_cutover.md`

---

## 🔬 STEP 4: Diagnose Model Collapse (If Quality Gate Fails)

**If XGB or PatchTST show constant output:**

```bash
# Check recent intents for actual model outputs
redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 10 | grep -A 5 "model_breakdown"

# Check for degeneracy detection in logs
journalctl -u quantum-ai-engine.service --since "$CUTOVER" | grep -i "degenerate"

# Check for fail-closed exceptions
journalctl -u quantum-ai-engine.service --since "$CUTOVER" | grep "QSC FAIL-CLOSED"

# Expected if models degraded:
# - "[XGB] QSC FAIL-CLOSED: Degenerate output detected"
# - "[PatchTST] QSC FAIL-CLOSED: Input tensor contains NaN values"
# - Ensemble manager logs: "inactive_reason=degenerate_output"
```

**Root causes:**
1. **Feature drift (PSI=1.000):** Features out-of-distribution → models collapse to buy bias
2. **Feature dimension mismatch:** NHiTS 14 != 12 (already fail-closed, ignored)
3. **Model weights corrupted:** Unlikely (SHA256 logged at init)
4. **Frozen model output:** PatchTST always 0.6150 → fail-closed now raises exception

**Next actions if collapse confirmed:**
- Retrain XGB on recent data (last 30 days)
- Fix feature engineering to reduce drift
- Update scalers/normalizers
- **DO NOT** lower quality gate thresholds

---

## 🚀 STEP 5: Run Quality Gate (Canary Mode - Deployment)

**Only after:**
- ✅ Feature sanity passed
- ✅ Collection mode showed >100 events
- ✅ >=200 events accumulated

```bash
# Canary mode: min_events=200, exit 0 enables activation
/opt/quantum/venvs/ai-engine/bin/python ops/model_safety/quality_gate.py \
  --mode canary \
  --after "$CUTOVER_ISO"

# Check output:
# - "🚀 CANARY MODE: min_events=200, exit 0 enables promotion"
# - Event count (must be >=200)
# - Model analysis with no blockers

# Exit codes:
# 0 = PASS (safe to activate canary)
# 2 = FAIL (blockers detected)
# 3 = Impossible in canary mode
```

**If RC=0 (PASS):**
- Report saved: `reports/safety/quality_gate_YYYYMMDD_HHMMSS_post_cutover.md`
- Read report to verify which model passed
- Proceed to canary activation (Step 6)

**If RC=2 (FAIL):**
- Check "Quality violations detected"
- Models still showing constant output
- **DO NOT** activate canary
- Loop back to Step 4 (diagnose)

---

## 🎯 STEP 6: Activate Canary (Only if Quality Gate RC=0)

```bash
# Choose model that PASSED quality gate (e.g., xgb or patchtst)
CANARY_MODEL="patchtst"  # Or "xgb" if it passed

# Dry run first
/opt/quantum/venvs/ai-engine/bin/python ops/model_safety/qsc_mode.py \
  --model "$CANARY_MODEL" \
  --cutover "$CUTOVER_ISO" \
  --dry-run

# Actual activation
/opt/quantum/venvs/ai-engine/bin/python ops/model_safety/qsc_mode.py \
  --model "$CANARY_MODEL" \
  --cutover "$CUTOVER_ISO"

# This will:
# - Re-run quality gate in canary mode (verification)
# - Save baseline weights
# - Create canary weights (model=10%, others scaled to 90%)
# - Create systemd override: /etc/systemd/system/quantum-ai-engine.service.d/canary_weights.conf
# - Log activation to logs/qsc_canary.jsonl
# - Show rollback command

# Restart with canary config
sudo systemctl daemon-reload
sudo systemctl restart quantum-ai-engine.service
```

**Post-activation verification:**
```bash
# Check canary is active
cat /etc/systemd/system/quantum-ai-engine.service.d/canary_weights.conf

# Check logs
tail -20 logs/qsc_canary.jsonl

# Check ensemble manager logs
journalctl -u quantum-ai-engine.service --since "1 minute ago" | grep -i "ensemble\|canary\|weight"
```

---

## 🔬 STEP 7: Start 6-Hour Monitoring

```bash
# Run QSC monitor (checks scoreboard every 30s for 6 hours)
screen -S qsc_monitor
cd /home/qt/quantum_trader
/opt/quantum/venvs/ai-engine/bin/python ops/model_safety/qsc_monitor.py

# Detach: Ctrl+A, D
# Reattach: screen -r qsc_monitor

# Monitor checks:
# - Runs scoreboard.py every 30s
# - 8 violation types checked
# - ANY violation → immediate rollback
# - 6 hours with NO violations → promotion eligible
```

**Violation types:**
1. Constant output (std < 0.01)
2. >85% BUY/SELL bias
3. >85% HOLD dead zone
4. Narrow range (P10-P90 < 0.12)
5. Action distribution collapse
6. Confidence distribution collapse
7. Error rate spike
8. Latency spike

---

## 🚨 ROLLBACK (If Violation or Manual Stop)

```bash
# Immediate rollback
bash ops/model_safety/qsc_rollback.sh

# This will:
# - Remove systemd override
# - Restore baseline weights
# - Restart service
# - Log rollback event

# Verify rollback
systemctl show -p Environment quantum-ai-engine.service | grep CANARY
# Should be empty (no canary vars)

journalctl -u quantum-ai-engine.service --since "1 minute ago" | grep -i rollback
```

---

## 📊 STEP 8: Post-6h Promotion (Manual)

**If monitor completes 6 hours with NO violations:**

```bash
# Read monitor report
cat reports/safety/qsc_monitor_final_YYYYMMDD_HHMMSS.md

# Manual decision to promote:
# - Increase canary weight from 10% → 50% → 100%
# - Update baseline weights permanently
# - Remove QSC monitoring
# - Mark as production-stable

# Promotion NOT automated (policy decision)
```

---

## 🛡️ FAIL-CLOSED SAFETY GUARANTEES

1. **Feature Sanity:** RC=2 blocks deployment if features flat/NaN/Inf
2. **Model Loading:** RuntimeError raised if model file missing/corrupted
3. **Degeneracy Detection:** XGB raises if >95% same action + low conf variance
4. **Collection Mode:** RC=3 never promotes, only gathers data
5. **QSC Mode:** Rejects collection-mode reports (RC=3 check)
6. **No Silent Fallback:** Exceptions raised, ensemble marks INACTIVE

---

## 📝 TROUBLESHOOTING

**"Feature sanity fails with >30% flatlines"**
- Feature engineering broken or data feed stale
- Check Redis stream: `redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 5`
- Check market_data timestamps in payloads
- Restart data-feed-orchestrator if stale

**"Quality gate RC=2 with constant output"**
- Model collapse due to drift (PSI=1.000)
- Retrain model on recent data
- Check drift logs: `journalctl | grep "DRIFT DETECTED"`

**"Quality gate RC=3 in canary mode (impossible)"**
- Bug in quality_gate.py --mode parsing
- Verify: `python quality_gate.py --help`

**"QSC mode rejects activation"**
- Quality gate must return RC=0 in canary mode
- Check: `echo $?` after quality_gate.py run
- If RC=3: You ran collection mode, rerun with `--mode canary`

**"Canary activation succeeds but no weight change"**
- Check systemd override: `cat /etc/systemd/system/quantum-ai-engine.service.d/canary_weights.conf`
- Ensemble manager may not read env var yet (restart service)

---

## 🎯 SUCCESS CRITERIA

✅ Feature sanity RC=0  
✅ Quality gate (canary mode) RC=0 with >=200 events  
✅ At least ONE model passes (varied predictions, healthy confidence distribution)  
✅ Canary activated at 10% traffic  
✅ 6-hour monitoring with ZERO violations  

**Then:** Manual promotion decision
