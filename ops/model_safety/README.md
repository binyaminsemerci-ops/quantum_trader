# Model Safety Operations

**Million-Safe Model Lifecycle - SYSTEMD ONLY**

## Hard Rules

- **FAIL-CLOSED**: Unclear = blocked
- **ONE MODEL AT A TIME**: Canary deployments only
- **ALL CHANGES**: Backup + proof required
- **NO AUTOMATION**: Manual activation only

## Components

### 1. Quality Gate (BLOCKER)
**Script:** `ops/model_safety/quality_gate.py`

**Detects:**
- Constant output (std<0.01 or p10==p90)
- HOLD collapse (HOLD>85% + prob in [0.4,0.6])
- Feature parse/shape mismatch

**Fails if:**
- Any class >70%
- conf_std <0.05 or p10-p90 <0.12
- Constant/dead-zone collapse detected
- Missing fields / parsing errors

**Exit codes:**
- `0` = PASS
- `2` = FAIL (BLOCKER)

**Usage:**
```bash
cd /home/qt/quantum_trader
python3 ops/model_safety/quality_gate.py
```

**Output:** `reports/safety/quality_gate_<timestamp>.md`

---

### 2. Scoreboard (Overview)
**Script:** `ops/model_safety/scoreboard.py`

**Shows:**
- Per-model: action%, conf_std, p10-p90
- Ensemble: agreement%, hard_disagree%
- Status: GO/WAIT/NO-GO

**Status logic:**
- **GO** ✅: Passes quality gate + agreement 55-80% + hard_disagree <20%
- **WAIT** ⏳: Passes gate but outside agreement range
- **NO-GO** ❌: Fails quality gate (BLOCKER)

**Usage:**
```bash
python3 ops/model_safety/scoreboard.py
```

**Output:** `reports/safety/scoreboard_latest.md`

---

### 3. Canary Activation (MANUAL)
**Script:** `ops/model_safety/canary_activate.sh`

**Steps:**
1. Run quality gate → FAIL = abort
2. Backup `/etc/quantum/ai-engine.env`
3. Get git hash (traceability)
4. Update env key (anchored edit)
5. Restart `quantum-ai-engine` service
6. Journal proof

**Usage:**
```bash
# Example: Activate PatchTST
sudo ops/model_safety/canary_activate.sh patchtst PATCHTST_SHADOW_ONLY

# Backup location: /opt/quantum/backups/model_activations/
```

**Proof required:**
- Journal logs (model loaded + voting)
- trade_intents table (predictions from activated model)

---

### 4. Rollback (UNDO)
**Script:** `ops/model_safety/rollback_last.sh`

**Steps:**
1. Find most recent backup
2. Restore `/etc/quantum/ai-engine.env`
3. Restart service
4. Journal proof

**Usage:**
```bash
sudo ops/model_safety/rollback_last.sh
```

---

### 5. PatchTST Sequence Dataset Builder
**Script:** `scripts/build_patchtst_sequence_dataset.py`

**Purpose:** Build proper temporal sequences (T×F format) for PatchTST training

**Input:**
- Historical OHLCV from database
- Lookback window (default: 64 timesteps)
- Prediction horizon (default: 4h)

**Output:**
- `data/patchtst_sequences_<timestamp>.npz`
  - X: (N, T, F) sequences
  - y: (N,) labels (WIN/LOSS)
  - metadata: features, lookback, horizon, label_rule

**Usage:**
```bash
python3 scripts/build_patchtst_sequence_dataset.py
```

**Output:** 
- Dataset: `data/patchtst_sequences_<timestamp>.npz`
- Report: `reports/safety/patchtst_sequence_dataset_<timestamp>.md`

---

## Makefile Targets

Add to `Makefile`:

```makefile
.PHONY: quality-gate scoreboard build-patchtst-dataset

quality-gate:
	python3 ops/model_safety/quality_gate.py

scoreboard:
	python3 ops/model_safety/scoreboard.py

build-patchtst-dataset:
	python3 scripts/build_patchtst_sequence_dataset.py
```

**Usage:**
```bash
make quality-gate       # Check model quality (BLOCKER)
make scoreboard         # View all models status
make build-patchtst-dataset  # Build temporal sequences
```

---

## Execution Workflow (VPS)

**1. Check model quality:**
```bash
cd /home/qt/quantum_trader
make quality-gate
cat reports/safety/quality_gate_*.md | tail -50
```

**2. View scoreboard:**
```bash
make scoreboard
cat reports/safety/scoreboard_latest.md
```

**3. Build PatchTST dataset (for retraining):**
```bash
make build-patchtst-dataset
ls -lh data/patchtst_sequences_*.npz
cat reports/safety/patchtst_sequence_dataset_*.md
```

**4. Activate model (MANUAL ONLY):**
```bash
# ONLY if quality gate passed
sudo ops/model_safety/canary_activate.sh <model_name> <env_key>

# Example:
sudo ops/model_safety/canary_activate.sh patchtst PATCHTST_SHADOW_ONLY
```

**5. Monitor:**
```bash
# Journal
journalctl -u quantum-ai-engine -f

# Trade intents
sqlite3 /opt/quantum/data/quantum_trader.db \
  "SELECT * FROM trade_intents WHERE model_source LIKE '%patchtst%' ORDER BY created_at DESC LIMIT 10;"
```

**6. Rollback (if issues):**
```bash
sudo ops/model_safety/rollback_last.sh
```

---

## Critical Safety Notes

### HOLD Dead-Zone Detection
- **Mapping:** prob ∈ [0.4, 0.6] → HOLD
- **Collapse:** HOLD >85% + constant prob ~0.5
- **Action:** Quality gate FAILS (blocks deployment)

### No Automatic Changes
- **Manual only:** All activations require explicit command
- **Proof required:** Backup + journal + git hash
- **Fail-closed:** Unclear status = blocked

### Ensemble Protection
- **Agreement range:** 55-80% (healthy)
- **Hard disagree:** <20% (BUY vs SELL conflict)
- **Outside range:** Status = WAIT (monitor, don't activate)

---

## Report Locations

- **Quality Gate:** `reports/safety/quality_gate_<timestamp>.md`
- **Scoreboard:** `reports/safety/scoreboard_latest.md`
- **Dataset Report:** `reports/safety/patchtst_sequence_dataset_<timestamp>.md`
- **Backups:** `/opt/quantum/backups/model_activations/`

---

## Emergency Contacts

If model causes losses:
1. `sudo ops/model_safety/rollback_last.sh` (immediate)
2. Check journal: `journalctl -u quantum-ai-engine -n 100`
3. Verify rollback: `make scoreboard`
4. Review trade_intents table for damage assessment
