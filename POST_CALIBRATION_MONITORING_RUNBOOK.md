# Post-Calibration Monitoring Runbook

**Operasjonell guide for første 24-72 timer etter calibration deploy**

---

## 🚀 Pre-Flight (Før calibration kjøres)

### 1. Start background monitor

```bash
# SSH til VPS
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254

# Opprett logs directory hvis ikke finnes
mkdir -p /root/logs

# Start continuous monitor
cd /root/quantum_trader
python3 post_calibration_monitor.py > /root/logs/monitor_console.log 2>&1 &
echo $! > /root/logs/monitor.pid

# Bekreft at den kjører
ps aux | grep post_calibration_monitor
```

### 2. Ta baseline snapshot (PRE-calibration)

```bash
# Disse verdiene brukes som referanse
echo "=== PRE-CALIBRATION BASELINE ===" | tee /root/logs/pre_calibration_baseline.txt

# Equity state
redis-cli GET quantum:equity:current | jq '.' | tee -a /root/logs/pre_calibration_baseline.txt

# Decision behavior (last 100)
redis-cli LRANGE quantum:decisions:history 0 99 | wc -l | tee -a /root/logs/pre_calibration_baseline.txt

# System health
systemctl status quantum-ai-engine | grep Active | tee -a /root/logs/pre_calibration_baseline.txt
systemctl status quantum-harvest-consumer | grep Active | tee -a /root/logs/pre_calibration_baseline.txt

echo "Baseline saved to /root/logs/pre_calibration_baseline.txt"
```

### 3. Verifiser rollback er klar

```bash
# Test at rollback script er executable
test -x /root/quantum_trader/rollback_calibration.sh && echo "✅ Rollback ready" || echo "❌ Rollback NOT ready"

# Bekreft pre-calibration config backup finnes
test -f /root/quantum_trader/config/calibration_backup.json && echo "✅ Backup exists" || echo "❌ No backup!"
```

---

## 📊 FASE A: 0-30 minutter (Critical Watch)

**Mål**: Oppdage fatale feil umiddelbart  
**Frekvens**: Hver 5. minutt  
**Lokasjoner**: Terminal + Background monitor

### Commands

```bash
# 1. Bekreft AI Engine lever
systemctl status quantum-ai-engine | grep "Active: active"

# 2. Sjekk for crashes
journalctl -u quantum-ai-engine -n 50 --no-pager | grep -i "error\|exception\|fatal"

# 3. Verifiser calibration loaded (kun én gang)
redis-cli GET quantum:calibration:status | jq '.loaded, .load_count'

# 4. Sjekk decisions fortsetter å flyte
redis-cli GET quantum:decision:latest | jq '.timestamp, .action, .confidence'
```

### Success Criteria (30 min)

- ✅ AI Engine: `Active: active (running)`
- ✅ No calibration exceptions
- ✅ CalibrationLoader: `load_count = 1`
- ✅ Decisions continue flowing (timestamps updating)

### STOP Criteria

🛑 **Umiddelbar rollback hvis**:
- AI Engine restarts
- Calibration exceptions i logs
- `load_count > 1` (reloading)
- All confidence = 0 eller = 1

```bash
# Kjør rollback
cd /root/quantum_trader
./rollback_calibration.sh
```

---

## 🧪 FASE B: 0-6 timer (Behavioral Sanity)

**Mål**: Systemet oppfører seg normalt  
**Frekvens**: Hver 30. minutt  
**Fokus**: Decision distribution, ikke PnL

### Commands

```bash
# 1. Decision distribution (siste 100 decisions)
redis-cli LRANGE quantum:decisions:history 0 99 | \
jq -r '.action' | \
sort | uniq -c | \
awk '{printf "%s: %.1f%%\n", $2, ($1/100)*100}'

# 2. Average confidence
redis-cli LRANGE quantum:decisions:history 0 99 | \
jq -r '.confidence' | \
awk '{sum+=$1; n++} END {print "Avg conf:", sum/n}'

# 3. Meta-Agent overrides
redis-cli GET quantum:meta:override_count

# 4. Risk Guards (skal være uendret)
redis-cli GET quantum:risk:active_guards | jq '.triggered'
```

### Success Criteria (6 timer)

- ✅ BUY/SELL/HOLD distribution ≈ pre-calibration (±10-15%)
- ✅ Average confidence: 0.45-0.75 (ikke flat)
- ✅ Normal override rate (ikke eksplosjon)
- ✅ Risk Guards trigges ~samme frekvens

### Warning Signs

⚠️ **Undersøk nærmere hvis**:
- HOLD > 85% (for konservativ)
- BUY or SELL > 85% (for aggressiv)
- Avg confidence < 0.30 or > 0.85 (collapsed)
- Risk Guards trigges 2× oftere

---

## 🛡️ FASE C: 6-24 timer (Risk Validation)

**Mål**: Risk profile uendret  
**Frekvens**: Hver time  
**Fokus**: Drawdown, volatility, guards

### Commands

```bash
# 1. Current drawdown
redis-cli GET quantum:equity:current | jq '.equity, .peak_equity' | \
awk 'NR==1{eq=$1} NR==2{peak=$1} END {print "DD:", ((peak-eq)/peak)*100 "%"}'

# 2. Equity volatility (last 24 samples = 12 hours @ 30min intervals)
redis-cli LRANGE quantum:equity:history 0 23 | \
jq -r '.equity' | \
awk '{sum+=$1; sumsq+=$1*$1; n++} END {mean=sum/n; print "StdDev:", sqrt(sumsq/n - mean*mean)}'

# 3. Stop-loss / Take-profit triggers
redis-cli LRANGE quantum:stream:trade.closed 0 99 | \
jq -r '.exit_reason' | \
sort | uniq -c

# 4. Risk Guard frequency
journalctl -u quantum-harvest-consumer --since "6 hours ago" | \
grep "BLOCKED" | wc -l
```

### Success Criteria (24 timer)

- ✅ Drawdown ≤ 1.5× normal range
- ✅ Equity volatility ~same as pre-calibration
- ✅ Stop-loss triggered at normal rate
- ✅ Risk Guards blocking ≤ 1.5× normal

### STOP Criteria

🛑 **Rollback hvis**:
- Drawdown > 2× normal
- Risk Guards blocking 2× oftere
- Equity SwingS > 2× normal volatility

```bash
# Kjør rollback
cd /root/quantum_trader
./rollback_calibration.sh
```

---

## ✅ FASE D: 24-72 timer (Acceptance)

**Mål**: Etablere ny baseline  
**Frekvens**: Hver 6. time  
**Fokus**: Long-term stability, CLM improvement

### Commands

```bash
# 1. Confidence-Outcome alignment (closed trades)
redis-cli LRANGE quantum:stream:trade.closed 0 199 | \
jq -r '[.entry_confidence, .outcome] | @csv' > /tmp/confidence_outcome.csv

# Analyze in Python or manually
# Look for: HIGH confidence → WIN, LOW confidence → LOSS (improvement)

# 2. CLM data quality
redis-cli LLEN quantum:stream:trade.closed
redis-cli LLEN quantum:stream:trade.metrics

# Should see steady growth

# 3. System stability (no degradation)
uptime
systemctl status quantum-ai-engine | grep "Active:"
systemctl status quantum-harvest-consumer | grep "Active:"

# 4. Average performance (not goal, but sanity check)
redis-cli LRANGE quantum:stream:trade.closed 0 199 | \
jq -r '.pnl' | \
awk '{sum+=$1; n++} END {print "Avg PnL:", sum/n}'
```

### Success Criteria (72 timer)

- ✅ System stable (no restarts, no exceptions)
- ✅ Confidence distribution looks "healthier" (less extreme)
- ✅ CLM data accumulating normally
- ✅ No risk degradation

### Acceptance

🎉 **Calibration accepted if**:
- All stability metrics green
- No rollback triggers
- Decision behavior unchanged (strategically)
- Confidence-outcome gap narrowing

---

## 🚨 Emergency Rollback Procedure

### When to Rollback (NO hesitation)

1. Drawdown > 2× normal range
2. AI Engine instability (crashes, exceptions)
3. Confidence collapse (all ~0 or ~1)
4. Risk Guards blocking abnormally
5. CalibrationLoader reloading multiple times

### Rollback Steps

```bash
# 1. SSH to VPS
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254

# 2. Execute atomic rollback
cd /root/quantum_trader
./rollback_calibration.sh

# 3. Verify services restarted
systemctl status quantum-ai-engine
systemctl status quantum-harvest-consumer

# 4. Confirm calibration reverted
redis-cli GET quantum:calibration:status | jq '.loaded'
# Should be: false or pre-calibration config

# 5. Monitor for 30 minutes
# Follow FASE A protocol again
```

### Post-Rollback

- Document trigger that caused rollback
- Review `/root/logs/post_calibration_monitor.log`
- Analyze what went wrong
- Fix before re-attempting calibration

---

## 📋 Checklist Summary

### Pre-Deployment
- [ ] Background monitor running
- [ ] Baseline snapshot saved
- [ ] Rollback script tested

### FASE A (0-30 min)
- [ ] AI Engine stable
- [ ] No calibration exceptions
- [ ] CalibrationLoader loaded once
- [ ] Decisions flowing

### FASE B (0-6 timer)
- [ ] Decision distribution normal
- [ ] Confidence not collapsed
- [ ] Meta-Agent behaving normally

### FASE C (6-24 timer)
- [ ] Drawdown ≤ 1.5× normal
- [ ] Risk Guards unchanged
- [ ] Equity volatility normal

### FASE D (24-72 timer)
- [ ] System stable
- [ ] CLM data growing
- [ ] No rollback triggers
- [ ] New baseline established

---

## 📝 Logging & Review

### Monitor Logs

```bash
# Real-time tail
tail -f /root/logs/post_calibration_monitor.log

# Phase-specific review
grep "FASE A" /root/logs/post_calibration_monitor.log
grep "FASE B" /root/logs/post_calibration_monitor.log
grep "WARNING" /root/logs/post_calibration_monitor.log
grep "CRITICAL" /root/logs/post_calibration_monitor.log
```

### Stop Monitor (after 72 hours)

```bash
# Kill background monitor
kill $(cat /root/logs/monitor.pid)
rm /root/logs/monitor.pid

# Archive logs
mkdir -p /root/logs/archive
mv /root/logs/post_calibration_monitor.log /root/logs/archive/$(date +%Y%m%d)_calibration_monitor.log
```

---

## 🔒 Locked Principles

1. **Post-calibration success ≠ høyere PnL**  
   Success = systemet er stabilt, forutsigbart, ærlig

2. **Flat PnL første døgn er OK**  
   Hvis systemet er rolig → calibration fungerer

3. **Rollback er forventet, ikke feil**  
   Det er en del av prosessen

4. **Confidence skal forbedre SAMSVAR, ikke strategi**  
   BUY/SELL/HOLD rate skal forbli ~lik

5. **Observation > Reaction**  
   24 timer observasjon før konklusjoner

---

**Runbook klar. Alle commands testet. Rollback klar. Monitoring infrastruktur deployet.**

**Når Cadence signalerer `ready=true` → kjør calibration → følg denne runbooken.**
