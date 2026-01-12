# QSC MODE - Quality Safeguard Canary

**Status:** ✅ PRODUCTION READY

Quick-start guide for safe AI model deployment with automated quality gates and instant rollback.

---

## 🚀 Quick Start

### One-Line Deployment

```bash
bash qsc_quick_start.sh patchtst
```

### Manual Steps

```bash
# 1. Get cutover timestamp
systemctl show quantum-ai_engine.service -p ActiveEnterTimestamp

# 2. Activate canary (checks quality gate automatically)
python3 ops/model_safety/qsc_mode.py \
  --model patchtst \
  --cutover 2026-01-10T05:43:15Z

# 3. Restart AI engine
sudo systemctl restart quantum-ai_engine.service

# 4. Monitor for 6 hours (auto-rollback on violations)
python3 ops/model_safety/qsc_monitor.py
```

---

## 📋 Requirements

✅ quality_gate.py exits 0  
✅ ≥200 post-cutover events  
✅ ONE model at 10% traffic  
✅ systemd integration  
✅ Logged rollback command  
✅ 6-hour monitoring  
✅ Auto-rollback on violations  
❌ NO retraining  
❌ NO auto-scale

---

## 📊 What It Does

1. **Quality Gate** - Validates model safety with ≥200 events
2. **Canary Deploy** - Routes 10% traffic to new model
3. **Monitor** - Watches for violations every 30s for 6h
4. **Auto-Rollback** - Instant revert to baseline on any issue

---

## 🔍 Violation Detection

- Action collapse (>70%)
- Dead zone (HOLD >85%)
- Flat predictions (std <0.05)
- Narrow range (P10-P90 <0.12)
- Ensemble dysfunction (agreement <55% or >80%)
- Model chaos (hard disagree >20%)

---

## 📁 Files

| File | Purpose | Size |
|------|---------|------|
| [qsc_mode.py](ops/model_safety/qsc_mode.py) | Activation | 11KB |
| [qsc_monitor.py](ops/model_safety/qsc_monitor.py) | Monitoring | 14KB |
| [qsc_rollback.sh](ops/model_safety/qsc_rollback.sh) | Rollback | 3KB |
| [qsc_quick_start.sh](qsc_quick_start.sh) | Quick start | 4KB |

---

## 🔄 Status & Logs

```bash
# Monitor output
tail -f logs/qsc_canary.jsonl | jq .

# View scoreboard
cat reports/safety/scoreboard_latest.md

# AI engine status
systemctl status quantum-ai_engine.service
```

---

## 🛡️ Safety

- **Fail-closed:** No activation without ≥200 events + quality pass
- **Logged rollback:** Pre-computed command in every activation log
- **Auto-rollback:** No human approval needed for safety
- **Baseline preserved:** Original weights saved before changes
- **Full audit:** All events logged to qsc_canary.jsonl

---

## 📚 Documentation

- **Full Guide:** [QSC_MODE_DOCUMENTATION.md](QSC_MODE_DOCUMENTATION.md)
- **Summary:** [QSC_MODE_SUMMARY.md](QSC_MODE_SUMMARY.md)
- **Test Suite:** [ops/model_safety/qsc_test.py](ops/model_safety/qsc_test.py)

---

## 🔧 Manual Rollback

```bash
bash ops/model_safety/qsc_rollback.sh
```

---

**Created:** 2026-01-10  
**Version:** 1.0
