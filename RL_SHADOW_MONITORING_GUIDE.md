# RL Shadow Monitoring Guide (cooldown=30s)

## Baseline Snapshot
**Date**: 2026-01-15T22:47:45Z  
**Configuration**: cooldown=30s, COUNT=2000

### Performance Metrics
- **Pass Rate**: 12.8% (256/2000)
- **Eligible Rate**: 32.1% (642/2000)  
- **Cooldown Blocking**: 19.3% (386/2000)
- **Ens Conf (pass)**: 0.720
- **Ens Conf (fail)**: 0.584

### Top Performers
1. **STXUSDT**: 63.3% pass rate, 100% eligible
2. **ARBUSDT**: 37.0% pass rate, 69.6% eligible
3. **OPUSDT**: 37.0% pass rate, 65.4% eligible
4. **BTCUSDT**: 37.5% pass rate, 48.2% eligible (RL conf 0.66!)
5. **DOTUSDT**: 35.4% pass rate, 71.7% eligible

---

## Operational Status

### Active Services
```bash
# RL Policy Publisher
systemctl status quantum-rl-policy-publisher.service
# PID: 3654417, running since 10:16:38 UTC
# Publishing 10 policies every 30s

# AI Engine  
systemctl status quantum-ai-engine.service
# Cooldown: 30s (verified in logs)

# Scorecard Timer
systemctl status quantum-rl-shadow-scorecard.timer  
# Runs every 15 minutes
```

### Configuration Files
```bash
/etc/quantum/ai-engine.env:
  RL_INFLUENCE_COOLDOWN_SEC=30

/etc/quantum/rl-shadow-scorecard.env:
  COUNT=2000
  STREAM=quantum:stream:trade.intent
  TOPN=10

/etc/quantum/rl-policy-publisher.env:
  SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOTUSDT,OPUSDT,ARBUSDT,INJUSDT,BNBUSDT,STXUSDT
  INTERVAL_SEC=30
```

### Log Management
```bash
# Logrotate configured
/etc/logrotate.d/quantum-rl-shadow-scorecard
# Rotates daily, keeps 14 days, compresses old logs
```

---

## 24-48 Hour Monitoring Commands

### Quick Status Check (Run Daily - 30 seconds)
```bash
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 '
echo "=== SCORECARD HEADLINES ==="
grep -E "Timestamp:|SUMMARY:|ELIGIBLE:" /var/log/quantum/rl_shadow_scorecard.log | tail -40
echo ""
echo "=== SERVICE STATUS ==="
systemctl is-active quantum-rl-policy-publisher.service
systemctl is-active quantum-ai-engine.service
systemctl is-active quantum-rl-shadow-scorecard.timer
'
```

### Full Latest Report
```bash
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 '
tail -120 /var/log/quantum/rl_shadow_scorecard.log
'
```

### Historical Trend (Last 24 Scorecards)
```bash
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 '
echo "=== LAST 24 SCORECARD HEADLINES ==="
grep -E "SUMMARY:|ELIGIBLE:" /var/log/quantum/rl_shadow_scorecard.log | tail -50
'
```

### Detailed Trend Analysis
```bash
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 '
echo "=== PASS RATE TREND ==="
grep "SUMMARY:" /var/log/quantum/rl_shadow_scorecard.log | 
tail -20 | 
awk "{print \$5, \$7}" | 
sed "s/|//g"

echo ""
echo "=== COOLDOWN BLOCKING TREND ==="
grep "cooldown_active" /var/log/quantum/rl_shadow_scorecard.log | 
grep "Global" | 
tail -20 | 
awk "{print \$3, \$5}"
'
```

### Service Health Check
```bash
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 '
echo "=== SERVICE STATUS ==="
systemctl is-active quantum-rl-policy-publisher.service
systemctl is-active quantum-ai-engine.service  
systemctl is-active quantum-rl-shadow-scorecard.timer

echo ""
echo "=== LAST SCORECARD UPDATE ==="
ls -lh /var/log/quantum/rl_shadow_scorecard.log | awk "{print \$6, \$7, \$8}"

echo ""
echo "=== PUBLISHER ACTIVITY (last 5 min) ==="
journalctl -u quantum-rl-policy-publisher.service --since "5 minutes ago" --no-pager | grep "Published" | tail -3
'
```

---

## What To Monitor

### ✅ Success Indicators
- **Pass rate stable or rising** (baseline: 12.8%)
- **Cooldown blocking ≤ 20%** (baseline: 19.3%)
- **Eligible rate 30-40%** (baseline: 32.1%)
- **ens_conf_avg_when_pass ≈ 0.72** (stable)
- **Top performers maintain 30%+ pass rates**

### ⚠️ Warning Signs
- **Pass rate drops < 10%** → investigate gate failures
- **Cooldown blocking > 25%** → may need further tuning
- **Eligible rate drops < 25%** → check other gate conditions  
- **ens_conf_avg_when_pass < 0.70** → confidence degradation
- **Service inactive** → restart required

### 🚨 Critical Issues
- **Pass rate < 5%** → major regression
- **Scorecard log not updating > 20 min** → timer/service failure
- **Publisher inactive** → policy staleness (60s+ age)
- **AI Engine crash** → no RL influence at all

---

## Performance Evolution

| Phase | Cooldown | Pass Rate | Cooldown Blocking | vs Baseline |
|-------|----------|-----------|-------------------|-------------|
| **Baseline** | 300s | 5.2% | 32.9% | - |
| **Phase 1** | 60s | 9.2% | 30.0% | +77% |
| **Phase 2** | **30s** | **12.8%** | **19.3%** | **+146%** |

---

## Next Steps Decision Tree

### After 24-48 Hours:

**If pass rate holds 12%+ and cooldown blocking stays < 22%:**
- ✅ **Option A**: Promote to production (increase RL_INFLUENCE_WEIGHT 0.05 → 0.10)
- ✅ **Option B**: Test cooldown=15s for further optimization
- ✅ **Option C**: Implement selective cooldown (only on would_flip)

**If pass rate drops to 8-11%:**
- ⚠️ Investigate gate reason distribution changes
- ⚠️ Check for policy staleness or symbol distribution shifts
- ⚠️ May need to adjust other gate conditions (not cooldown)

**If pass rate drops < 8%:**
- 🚨 Revert to cooldown=60s (proven stable)
- 🚨 Analyze root cause before further tuning
- 🚨 Check for system-wide issues (Redis, AI Engine, etc.)

---

## Contact Points

- **Log Location**: `/var/log/quantum/rl_shadow_scorecard.log`
- **Config Backups**: `/etc/quantum/*.env.bak.*`
- **Service User**: `qt` (RL Policy Publisher), `root` (AI Engine, Scorecard)
- **Git Commits**: 
  - Scorecard v3: `43f09e57`
  - Scorecard v2: `5396fdea`
  - Policy Publisher: `d8fbfb13`
  - RL_PROOF Logging: `9c641d52`
