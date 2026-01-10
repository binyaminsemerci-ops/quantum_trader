# Telemetry-Only Quality Gates - VPS Testing Commands

**Commit:** e2fbf602  
**Date:** 2025-01-01

## Overview

Refactored quality_gate.py and scoreboard.py to be **TELEMETRY-ONLY (FAIL-CLOSED)**:

- ✅ **NEVER load model files** - uses Redis stream telemetry only
- ✅ **Production predictions** - analyzes actual trade.intent events
- ✅ **FAIL-CLOSED** - <200 events → exit 2 (BLOCKER)
- ✅ **Per-model breakdown** - extracts model_breakdown JSON from ensemble
- ✅ **Hard gates** - majority>70%, conf_std<0.05, p10-p90<0.12, HOLD>85%, constant

**CRITICAL RULE:**  
> Manglende bevis = ingen aktivering (Missing data = no activation)

---

## VPS Testing Commands

### Step 1: Pull Latest Code

```bash
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254

cd /home/qt/quantum_trader
git pull origin main
```

**Expected output:**
```
From https://github.com/binyaminsemerci-ops/quantum_trader
   df265cd6..e2fbf602  main       -> origin/main
Updating df265cd6..e2fbf602
 ops/model_safety/quality_gate.py | 637 +++++++++++++++++++++++++++++---------
 ops/model_safety/scoreboard.py   | 380 ++++++----------------
 2 files changed, 637 insertions(+), 380 deletions(-)
```

---

### Step 2: Verify Redis Stream Exists

```bash
# Check if stream exists
redis-cli EXISTS quantum:stream:trade.intent

# Should return: 1 (exists) or 0 (missing)
```

```bash
# Count events in stream
redis-cli XLEN quantum:stream:trade.intent

# Should return: number of events (need at least 200)
```

```bash
# Peek at last 3 events
redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 3
```

**Expected format:**
```
1) "1736478123456-0"
2) 1) "action"
   2) "BUY"
   3) "confidence"
   4) "0.75"
   5) "model_breakdown"
   6) "{\"xgboost\": {\"action\": \"BUY\", \"confidence\": 0.8}, ...}"
```

---

### Step 3: Run Quality Gate (Telemetry-Only)

```bash
cd /home/qt/quantum_trader
make quality-gate
```

**Expected outcomes:**

**PASS (exit 0):**
```
📊 Reading telemetry from Redis stream: quantum:stream:trade.intent
📦 Parsed 1847 events
✅ Sufficient data for analysis (1847 ≥ 200)

📊 Per-Model Analysis:
  - xgboost: 1847 predictions
  - lightgbm: 1847 predictions
  - nhits: 1843 predictions

🟢 Quality Gate: PASS

All models pass quality checks
No collapse detected
Confidence spread adequate

✅ Report saved: reports/safety/quality_gate_2025-01-01T12:34:56Z.md
```

**FAIL (exit 2):**
```
📊 Reading telemetry from Redis stream: quantum:stream:trade.intent
📦 Parsed 150 events
❌ INSUFFICIENT DATA (FAIL-CLOSED)

Need at least 200 events for reliable analysis
Found: 150 events

🔴 Quality Gate: FAIL (BLOCKER)

❌ Report saved: reports/safety/quality_gate_2025-01-01T12:34:56Z.md
```

---

### Step 4: View Quality Gate Report

```bash
# Find latest report
ls -lh reports/safety/quality_gate_*.md

# Read report
cat reports/safety/quality_gate_2025-01-01T*.md
```

**Expected sections:**
- Telemetry Info (stream, event count, models)
- Per-Model Analysis (action%, conf_stats, quality checks)
- Overall Status (PASS/FAIL)
- Failure details (if any)

---

### Step 5: Run Scoreboard (Telemetry-Only)

```bash
cd /home/qt/quantum_trader
make scoreboard
```

**Expected outcomes:**

**SUCCESS:**
```
📊 Reading telemetry from Redis stream: quantum:stream:trade.intent
📦 Parsed 1847 events
✅ Sufficient data for scoreboard

📊 Per-Model Status:
  - xgboost: GO ✅
  - lightgbm: WAIT ⏳
  - nhits: GO ✅

🟢 Overall Status: ALL-GO

✅ Scoreboard saved: reports/safety/scoreboard_latest.md
```

**INSUFFICIENT DATA:**
```
📊 Reading telemetry from Redis stream: quantum:stream:trade.intent
📦 Parsed 150 events
⚠️  WARNING: Only 150 events (need 200)
⚠️  Scoreboard may be inaccurate

✅ Scoreboard saved: reports/safety/scoreboard_latest.md
```

---

### Step 6: View Scoreboard Report

```bash
cat reports/safety/scoreboard_latest.md
```

**Expected sections:**
- Telemetry Info (stream, event count)
- Overall Status (ALL-GO/WAIT/NO-GO)
- Ensemble Agreement (agreement%, hard_disagree%)
- Per-Model Status (action%, conf_stats, quality gate)

**Status meanings:**
- **GO** 🟢: Passes quality gate + agreement 55-80% + hard_disagree <20%
- **WAIT** 🟡: Passes gate but outside agreement range or insufficient data
- **NO-GO** 🔴: Fails quality gate (BLOCKER)

---

## Troubleshooting

### Redis Stream Missing

If `redis-cli XLEN quantum:stream:trade.intent` returns 0 or error:

```bash
# Check if Redis is running
systemctl status redis

# Check if AI Engine is running (produces events)
systemctl status quantum-ai-engine

# Check AI Engine logs
journalctl -u quantum-ai-engine --since "5 minutes ago"
```

### Insufficient Events (<200)

If stream has <200 events:

```bash
# Check when AI Engine last started
systemctl status quantum-ai-engine | grep "since"

# Let it run for 10-15 minutes to accumulate events
sleep 600

# Recheck event count
redis-cli XLEN quantum:stream:trade.intent
```

### Parse Errors

If `make quality-gate` fails with parse errors:

```bash
# Check redis-cli is installed
which redis-cli

# Test redis-cli manually
redis-cli PING
# Should return: PONG

# Check Python numpy is available
/opt/quantum/venvs/ai-engine/bin/python3 -c "import numpy; print(numpy.__version__)"
```

---

## Exit Codes

- **0** = PASS (safe to proceed)
- **1** = ERROR (Redis failure, parse error, etc.)
- **2** = FAIL (BLOCKER - quality gate failed OR insufficient data)

**FAIL-CLOSED RULE:**  
Exit code 2 means **NO ACTIVATION** - insufficient data or safety violation.

---

## Safety Confirmation

**AFTER TESTING:**

1. ✅ Quality gate runs without loading model files
2. ✅ Scoreboard runs without database dependency
3. ✅ Both use Redis stream telemetry only
4. ✅ <200 events triggers FAIL-CLOSED (exit 2)
5. ✅ Reports generated with per-model breakdown
6. ✅ NO ACTIVATION PERFORMED

**Next steps:**
- Wait for production to accumulate ≥200 events
- Run `make quality-gate` to verify gates work
- Run `make scoreboard` to check model status
- **DO NOT activate any model** until gates pass

---

## Git Proof

```bash
git log --oneline -3
```

**Expected:**
```
e2fbf602 (HEAD -> main, origin/main) Refactor quality gates to telemetry-only (FAIL-CLOSED)
df265cd6 Million-safe model lifecycle with quality gates (NO ACTIVATION)
...
```

```bash
git show e2fbf602 --stat
```

**Expected:**
```
ops/model_safety/quality_gate.py  | 637 insertions(+), 380 deletions(-)
ops/model_safety/scoreboard.py    | 380 insertions(+), ...
```

---

## NO ACTIVATION CONFIRMATION

**THIS IS A GATE-ONLY UPDATE.**

No model activation commands were run:
- ❌ NO `canary_activate.sh` execution
- ❌ NO `.env.model_config` edits
- ❌ NO systemd restart commands
- ✅ ONLY quality gate and scoreboard refactor

**STATUS:** Testing phase only - waiting for production telemetry to accumulate.
