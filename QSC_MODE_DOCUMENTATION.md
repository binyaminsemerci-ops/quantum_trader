# QSC MODE - Quality Safeguard Canary Deployment

**Purpose:** Safely deploy new AI models using automated quality gates, 10% canary traffic, and continuous monitoring with instant rollback.

---

## 📋 Overview

QSC (Quality Safeguard Canary) is a production-safe deployment system with:

1. ✅ **Quality Gate** - Validates ≥200 post-cutover events pass all safety checks
2. 🔸 **10% Canary** - Routes 10% traffic to new model via systemd override
3. 👁️ **6-Hour Monitor** - Detects violations and auto-rollbacks
4. 🔄 **Instant Rollback** - Reverts to baseline weights on any violation

**Key Rules:**
- NO retraining during canary
- NO auto-scaling
- NO manual intervention required (fully automated)

---

## 🚀 Quick Start

### Prerequisites

```bash
# 1. Ensure quality_gate.py exists
ls ops/model_safety/quality_gate.py

# 2. Get cutover timestamp (AI engine restart time)
systemctl show quantum-ai_engine.service -p ActiveEnterTimestamp

# Example output:
# ActiveEnterTimestamp=Fri 2026-01-10 05:43:15 UTC
# Convert to ISO: 2026-01-10T05:43:15Z
```

### Step 1: Activate Canary (with Quality Gate Check)

```bash
# Dry run (quality gate check only)
python3 ops/model_safety/qsc_mode.py \
  --model patchtst \
  --cutover 2026-01-10T05:43:15Z \
  --dry-run

# Activate canary if quality gate passes
python3 ops/model_safety/qsc_mode.py \
  --model patchtst \
  --cutover 2026-01-10T05:43:15Z
```

**What happens:**
1. Runs `quality_gate.py --after 2026-01-10T05:43:15Z`
2. Checks exit code = 0 AND event count ≥ 200
3. Saves baseline weights to `data/baseline_model_weights.json`
4. Creates canary weights (10% to target model, 90% split among others)
5. Writes systemd override to `/etc/systemd/system/quantum-ai_engine.service.d/qsc_canary.conf`
6. Logs activation to `logs/qsc_canary.jsonl`

**Output example:**
```
🚀 CANARY ACTIVATED
================================================================================
  Model:         patchtst
  Weight:        10.0%
  Start Time:    2026-01-10T12:00:00Z
  Monitor For:   6 hours

Weights:
  🔸 patchtst    10.0%  (canary)
     xgb         22.5%
     lgbm        22.5%
     nhits       45.0%

Rollback Command:
  python3 ops/model_safety/qsc_rollback.sh
================================================================================
```

### Step 2: Restart AI Engine

```bash
# Apply canary weights
sudo systemctl restart quantum-ai_engine.service

# Verify canary is active
sudo journalctl -u quantum-ai_engine.service -n 50 | grep -i canary
```

### Step 3: Start Monitoring (Auto-Rollback)

```bash
# Start 6-hour monitoring daemon
python3 ops/model_safety/qsc_monitor.py

# Or use systemd (auto-stops after 6h)
sudo systemctl start quantum-qsc-monitor.service
sudo journalctl -u quantum-qsc-monitor.service -f
```

**What happens:**
- Checks scoreboard every 30 seconds for 6 hours
- Detects violations (collapse, flat predictions, ensemble dysfunction)
- **Executes immediate rollback** if any violation detected
- Logs all checks and violations to `logs/qsc_canary.jsonl`

### Step 4: Manual Rollback (if needed)

```bash
# Immediate rollback to baseline
bash ops/model_safety/qsc_rollback.sh

# Or via systemctl
sudo systemctl restart quantum-ai_engine.service
```

---

## 📊 Monitoring & Logs

### Real-Time Monitoring

```bash
# Monitor daemon output
sudo journalctl -u quantum-qsc-monitor.service -f

# Check canary log
tail -f logs/qsc_canary.jsonl | jq .

# View scoreboard
cat reports/safety/scoreboard_latest.md
```

### Log Entries

**Canary Activation:**
```json
{
  "timestamp": "2026-01-10T12:00:00Z",
  "action": "canary_activated",
  "canary_model": "patchtst",
  "canary_weight": 0.10,
  "weights": {"xgb": 0.225, "lgbm": 0.225, "nhits": 0.45, "patchtst": 0.10},
  "cutover_ts": "2026-01-10T05:43:15Z",
  "quality_gate_events": 342,
  "monitoring_duration_hours": 6,
  "rollback_cmd": "python3 ops/model_safety/qsc_rollback.sh"
}
```

**Rollback Event:**
```json
{
  "timestamp": "2026-01-10T14:30:00Z",
  "action": "rollback_executed",
  "canary_model": "patchtst",
  "violations": [
    "Action collapse: HOLD=78.3% (>70%)",
    "Flat predictions: conf_std=0.0412 (<0.05)"
  ],
  "trigger": "qsc_monitor_violation"
}
```

**Monitoring Success:**
```json
{
  "timestamp": "2026-01-10T18:00:00Z",
  "action": "monitoring_completed",
  "canary_model": "patchtst",
  "duration_hours": 6,
  "checks_performed": 720,
  "violations": null,
  "result": "safe"
}
```

---

## 🔍 Violation Criteria

Monitor checks for these violations (from `quality_gate.py`):

1. **Action Collapse** - Any class >70%
2. **Dead Zone** - HOLD >85%
3. **Flat Predictions** - Confidence std <0.05
4. **Narrow Range** - P10-P90 range <0.12
5. **Quality Gate Fail** - Model marked NO-GO in scoreboard
6. **Ensemble Dysfunction** - Agreement <55% or >80%
7. **Model Chaos** - Hard disagree >20%

**Any single violation triggers immediate rollback.**

---

## 📁 File Structure

```
ops/model_safety/
├── qsc_mode.py          # Main activation script
├── qsc_monitor.py       # 6-hour monitoring daemon
├── qsc_rollback.sh      # Rollback script
├── quality_gate.py      # Quality gate checker (dependency)
└── scoreboard.py        # Status checker (dependency)

ops/systemd/
└── quantum-qsc-monitor.service  # Systemd service for monitoring

data/
├── baseline_model_weights.json   # Original weights (for rollback)
├── qsc_canary_weights.json       # Canary weights (10% split)
└── systemd_overrides/
    └── qsc_canary.conf           # Local copy of override

logs/
└── qsc_canary.jsonl              # Activation/rollback/monitoring log

/etc/systemd/system/quantum-ai_engine.service.d/
└── qsc_canary.conf               # Systemd override (active)
```

---

## 🔧 Advanced Usage

### Custom Duration

```bash
# Monitor for 3 hours instead of 6
python3 ops/model_safety/qsc_monitor.py --duration 3
```

### Different Model

```bash
# Deploy N-HiTS as canary
python3 ops/model_safety/qsc_mode.py \
  --model nhits \
  --cutover 2026-01-10T05:43:15Z
```

### Quality Gate Only

```bash
# Check quality gate without activating canary
python3 ops/model_safety/qsc_mode.py \
  --dry-run \
  --cutover 2026-01-10T05:43:15Z
```

### Manual Weight Override

```bash
# Edit canary weights before activation
vim data/qsc_canary_weights.json

# Apply manually
sudo cp data/systemd_overrides/qsc_canary.conf \
  /etc/systemd/system/quantum-ai_engine.service.d/

sudo systemctl daemon-reload
sudo systemctl restart quantum-ai_engine.service
```

---

## 🛡️ Safety Guarantees

1. **Fail-Closed** - Quality gate blocks activation if <200 events or violations
2. **Logged Rollback** - Every activation includes logged rollback command
3. **Auto-Rollback** - Monitor executes rollback without human intervention
4. **Baseline Preserved** - Original weights saved before any changes
5. **Systemd Isolation** - Override config separated from main service

---

## 📈 Post-Canary Workflow

### If Monitoring Completes (No Violations)

```bash
# Canary is safe! Promote to full production:

# 1. Update baseline weights to make canary permanent
cp data/qsc_canary_weights.json data/baseline_model_weights.json

# 2. Optional: Increase canary weight gradually
#    (e.g., 10% → 25% → 50% → 100%)

# 3. Remove canary override (now using baseline)
sudo rm /etc/systemd/system/quantum-ai_engine.service.d/qsc_canary.conf
sudo systemctl daemon-reload
sudo systemctl restart quantum-ai_engine.service
```

### If Rollback Executed

```bash
# 1. Analyze violations
cat logs/qsc_canary.jsonl | jq 'select(.action=="rollback_executed")'

# 2. Re-run quality gate for diagnosis
python3 ops/model_safety/quality_gate.py \
  --after 2026-01-10T05:43:15Z > reports/safety/quality_gate_post_rollback.txt

# 3. Fix model/data issues before next attempt
```

---

## 🔄 Integration with Existing Systems

### With Continuous Learning

QSC MODE is the final deployment gate after CLM training:

1. CLM trains new model → `models/trained/`
2. Run quality gate → `python3 ops/model_safety/quality_gate.py`
3. If pass → Activate canary → `qsc_mode.py`
4. Monitor 6h → Auto-rollback if violations
5. If safe → Promote to production

### With Shadow Models

QSC can deploy shadow models to canary:

```bash
# Shadow model passed validation
# Deploy as 10% canary for real traffic
python3 ops/model_safety/qsc_mode.py \
  --model patchtst_shadow \
  --cutover 2026-01-10T05:43:15Z
```

---

## ❓ Troubleshooting

### "Quality gate failed"

```bash
# Check quality gate output
python3 ops/model_safety/quality_gate.py --after <cutover_ts>

# Common issues:
# - <200 events: Wait longer for data
# - Collapse detected: Model needs retraining
# - Flat predictions: Check model confidence calibration
```

### "Systemd override not installed"

```bash
# Need sudo for systemd
sudo cp data/systemd_overrides/qsc_canary.conf \
  /etc/systemd/system/quantum-ai_engine.service.d/

sudo systemctl daemon-reload
```

### "Monitoring check failed"

```bash
# Check scoreboard manually
python3 ops/model_safety/scoreboard.py

# Check Redis stream
redis-cli XLEN quantum:stream:trade.intent
```

### "Rollback doesn't work"

```bash
# Manual rollback steps:
sudo rm /etc/systemd/system/quantum-ai_engine.service.d/qsc_canary.conf
sudo systemctl daemon-reload
sudo systemctl restart quantum-ai_engine.service

# Verify baseline restored
sudo journalctl -u quantum-ai_engine.service -n 50 | grep -i weight
```

---

## 📚 References

- [quality_gate.py](ops/model_safety/quality_gate.py) - Quality gate implementation
- [scoreboard.py](ops/model_safety/scoreboard.py) - Model status checker
- [CUTOVER_BASED_GATING_IMPLEMENTATION.md](../CUTOVER_BASED_GATING_IMPLEMENTATION.md) - Quality gate design
- [CONTINUOUS_LEARNING_DOCUMENTATION.md](../CONTINUOUS_LEARNING_DOCUMENTATION.md) - CLM integration

---

## ✅ Status

- ✅ Quality gate cutover filtering
- ✅ QSC mode activation script
- ✅ 6-hour monitoring daemon
- ✅ Automatic rollback on violations
- ✅ Systemd integration
- ✅ JSONL logging
- ⏸️ Production testing (pending)

**Created:** 2026-01-10  
**Last Updated:** 2026-01-10  
**Author:** QSC Implementation Team
