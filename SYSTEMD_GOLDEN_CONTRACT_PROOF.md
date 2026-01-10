# SYSTEMD GOLDEN CONTRACT - EXECUTION PROOF

**Commit:** 1923072b  
**Date:** 2026-01-10  
**Authority:** Principal Systems Architect  
**Scope:** Complete systemd standardization (NO Docker)

---

## ✅ DELIVERABLES COMPLETED

### 1️⃣ STANDARD_CONTRACT.md (docs/)

**File:** `docs/STANDARD_CONTRACT.md`  
**Size:** 37,822 bytes (500+ lines)  
**Status:** ✅ DELIVERED

**Contents:**
- **Section A:** Filesystem Layout (repo, data, models, logs)
- **Section B:** Python Runtime (venv paths, service mapping)
- **Section C:** Environment Configuration (env files, required vars)
- **Section D:** systemd Unit Rules (mandatory fields, dependencies)
- **Section E:** Redis / Dependencies (localhost only, validation)
- **Section F:** Safety Rules (shadow→canary→promote, quality gate)
- **Section G:** Ops Execution Model (wrapper, Makefile integration)
- **Section H:** Audit & Compliance (audit script, violations)
- **Section I:** Immutability Checklist (deployment verification)
- **Section J:** Principle of Least Privilege (User=qt rationale)
- **Section K:** Repeatability Guarantee (clean slate test)
- **Section L:** Anti-Patterns (forbidden practices)
- **Section M:** Change Log
- **Section N:** Signature

**Key Rules Enforced:**
```
REPO_ROOT = /home/qt/quantum_trader
VENV_PATH = /opt/quantum/venvs/<service>/bin/python
ENV_FILE  = /etc/quantum/<service>.env
USER      = qt (NEVER root)
REDIS     = localhost:6379 (systemd service)
WORKING_DIR = /home/qt/quantum_trader (IMMUTABLE)
```

---

### 2️⃣ quantum-golden.service (systemd/templates/)

**File:** `systemd/templates/quantum-golden.service`  
**Size:** 7,429 bytes  
**Status:** ✅ DELIVERED

**Structure:**
- `[Unit]` section: Description, After, Wants, BindsTo
- `[Service]` section: Type, User, WorkingDirectory, EnvironmentFile, ExecStart, Restart, Logging
- `[Install]` section: WantedBy
- Commented customization checklist
- Common service examples (AI Engine, Backend, Execution, RL Agent)
- Ops job examples (Training, Quality Gate)
- Forbidden patterns guide
- Troubleshooting decision trees

**Mandatory Fields (DO NOT CHANGE):**
```ini
WorkingDirectory=/home/qt/quantum_trader
User=qt
Group=qt
After=network-online.target redis-server.service
StandardOutput=journal
StandardError=journal
```

**Customizable Fields:**
```ini
Description=Quantum Trading System - <SERVICE_NAME>
EnvironmentFile=/etc/quantum/<service>.env
ExecStart=/opt/quantum/venvs/<service>/bin/python <SCRIPT>
Restart=always  # or "no" for ops jobs
```

---

### 3️⃣ ops/run.sh

**File:** `ops/run.sh`  
**Size:** 5,236 bytes  
**Status:** ✅ DELIVERED + EXECUTABLE

**Purpose:** MANDATORY wrapper for ALL ops jobs.

**Usage:**
```bash
ops/run.sh <service> <script> [args...]

# Examples:
ops/run.sh ai-engine ops/model_safety/quality_gate.py
ops/run.sh ai-engine ops/training/train_patchtst.py --epochs 100
```

**Validation Steps (FAIL-CLOSED):**
1. ✅ Check repo root exists: `/home/qt/quantum_trader`
2. ✅ Check venv exists: `/opt/quantum/venvs/<service>`
3. ✅ Check Python executable: `<venv>/bin/python`
4. ✅ Check env file exists: `/etc/quantum/<service>.env`
5. ✅ Check script exists: `<repo>/<script>`
6. ✅ Check Redis responds: `redis-cli PING`
7. ✅ Check database exists: `/opt/quantum/data/quantum_trader.db` (warn if missing)
8. ✅ Check data directory: `/opt/quantum/data`

**Environment Setup:**
- Source `/etc/quantum/<service>.env`
- Export `REPO_ROOT=/home/qt/quantum_trader`
- Export `PYTHONPATH=$REPO_ROOT:$PYTHONPATH`
- Export `PATH=$VENV/bin:$PATH`
- Export `PYTHONUNBUFFERED=1`

**Execution:**
```bash
cd $REPO_ROOT
exec $VENV/bin/python $SCRIPT $ARGS
```

**FAIL-CLOSED:** If ANY check fails → exit 1 immediately.

---

### 4️⃣ Makefile (Updated)

**File:** `Makefile`  
**Size:** 1,421 bytes  
**Status:** ✅ DELIVERED

**Changes:**
```makefile
# OLD (VIOLATED CONTRACT):
PYTHON := /opt/quantum/venvs/ai-engine/bin/python
quality-gate:
	$(PYTHON) ops/model_safety/quality_gate.py

# NEW (COMPLIANT):
SERVICE := ai-engine
RUN := ops/run.sh $(SERVICE)
quality-gate:
	$(RUN) ops/model_safety/quality_gate.py
```

**All Targets:**
- `make quality-gate` → `ops/run.sh ai-engine ops/model_safety/quality_gate.py`
- `make scoreboard` → `ops/run.sh ai-engine ops/model_safety/scoreboard.py`
- `make build-patchtst-dataset` → `ops/run.sh ai-engine scripts/build_patchtst_sequence_dataset.py`
- `make train-patchtst` → `ops/run.sh ai-engine ops/training/train_patchtst.py`
- `make audit` → `ops/run.sh ai-engine ops/audit_contract.py`

**NO direct Python calls.** ALL targets use `$(RUN)` wrapper.

---

### 5️⃣ ops/audit_contract.py

**File:** `ops/audit_contract.py`  
**Size:** 10,562 bytes  
**Status:** ✅ DELIVERED + EXECUTABLE

**Purpose:** Validate Golden Contract compliance.

**Checks Performed:**

**Infrastructure:**
- ✅ Repo exists: `/home/qt/quantum_trader` (with .git)
- ✅ Redis running: `redis-cli PING`
- ✅ Database exists: `/opt/quantum/data/quantum_trader.db`
- ✅ Ops wrapper exists and executable: `ops/run.sh`
- ✅ Makefile uses `ops/run.sh` (no direct Python calls)

**Per-Service (ai-engine, backend, execution, rl-agent):**
- ✅ Venv exists: `/opt/quantum/venvs/<service>/bin/python`
- ✅ Env file exists: `/etc/quantum/<service>.env`
- ✅ Env file permissions: 640 or 644
- ✅ Env file has required vars: `REPO_ROOT`, `DATA_DIR`, `SERVICE_NAME`, `REDIS_HOST`, `REDIS_PORT`
- ✅ systemd unit exists: `/etc/systemd/system/quantum-<service>.service`
- ✅ Unit has `WorkingDirectory=/home/qt/quantum_trader`
- ✅ Unit has `EnvironmentFile=/etc/quantum/<service>.env`
- ✅ Unit has `User=qt`
- ✅ Unit uses venv path in `ExecStart`
- ✅ Unit has `After=redis-server.service`
- ✅ Unit has `StandardOutput=journal`

**Exit Codes:**
- `0` = ALL PASS (compliant)
- `1` = VIOLATIONS FOUND (non-compliant)
- `2` = CRITICAL ERROR (cannot audit)

**Output:** Color-coded with ✅/❌ per check.

---

### 6️⃣ SYSTEMD_STANDARD_README.md (docs/)

**File:** `docs/SYSTEMD_STANDARD_README.md`  
**Size:** 15,683 bytes  
**Status:** ✅ DELIVERED

**Sections:**

1. **How to Add a New Service**
   - Step-by-step: venv, env file, systemd unit, enable, verify
   - Complete command examples
   - Compliance checklist

2. **How to Run Ops Safely**
   - Golden Rule: ALWAYS use `ops/run.sh` or `make <target>`
   - Common operations: quality-gate, scoreboard, training, audit
   - Manual execution patterns
   - What `ops/run.sh` does

3. **What Breaks the Contract**
   - Filesystem violations (wrong paths, hardcoded paths)
   - Python runtime violations (system python, wrong venv)
   - Environment violations (.env in repo, missing vars)
   - systemd violations (User=root, missing fields)
   - Redis violations (Docker, remote, containers)
   - Safety violations (auto-activation, skip gate)
   - Ops violations (bypass wrapper, direct Python)

4. **Rollback Rules**
   - When to rollback (triggers)
   - Rollback procedure (`ops/model_safety/rollback_last.sh`)
   - Post-rollback verification

5. **Quick Troubleshooting**
   - Service won't start
   - Service crashes immediately
   - Permission errors
   - Audit fails
   - Quality gate fails
   - Makefile target fails

**Emergency Procedures:**
- Stop service
- Check logs
- Run audit
- Rollback if unsafe
- Document incident

---

## 📊 FILE MANIFEST

```
docs/
├── STANDARD_CONTRACT.md         (37,822 bytes) - Single source of truth
└── SYSTEMD_STANDARD_README.md   (15,683 bytes) - Quick reference

systemd/templates/
└── quantum-golden.service       (7,429 bytes)  - systemd unit template

ops/
├── run.sh                       (5,236 bytes)  - Mandatory wrapper (executable)
└── audit_contract.py            (10,562 bytes) - Compliance validator (executable)

Makefile                         (1,421 bytes)  - Updated to use wrapper
```

**Total:** 6 files, 77,153 bytes, 1,869 insertions, 9 deletions

---

## 🔒 HARD CONSTRAINTS ENFORCED

### Filesystem (IMMUTABLE)

```
✅ REPO:   /home/qt/quantum_trader
✅ DATA:   /opt/quantum/data
✅ MODELS: /opt/quantum/ai_engine/models
✅ VENVS:  /opt/quantum/venvs/<service>
✅ ENV:    /etc/quantum/<service>.env

❌ NO ~/quantum_trader
❌ NO /tmp or /var/tmp for persistent data
❌ NO .env in repo
❌ NO hardcoded paths
```

### Python Runtime (MANDATORY)

```
✅ ALWAYS: /opt/quantum/venvs/<service>/bin/python
❌ NEVER:  python, python3, /usr/bin/python3
```

### systemd (GOLDEN TEMPLATE)

```
✅ WorkingDirectory=/home/qt/quantum_trader
✅ EnvironmentFile=/etc/quantum/<service>.env
✅ User=qt (NEVER root)
✅ After=redis-server.service
✅ StandardOutput=journal
✅ ExecStart with full venv path
```

### Redis (LOCALHOST ONLY)

```
✅ systemd service: redis-server.service
✅ Connection: localhost:6379
❌ NO Docker: quantum_redis
❌ NO remote Redis
❌ NO container names
```

### Ops Execution (WRAPPER ONLY)

```
✅ ops/run.sh <service> <script>
✅ make <target>
❌ python ops/script.py
❌ Direct venv calls
❌ Bypass wrapper
```

### Safety (FAIL-CLOSED)

```
✅ Quality gate MUST pass (exit 0)
✅ Manual activation ONLY (canary_activate.sh)
✅ Backup before EVERY model change
✅ Audit before EVERY deployment
❌ NO auto-activation
❌ NO skip gate
❌ NO bypass audit
```

---

## 🧪 REPEATABILITY GUARANTEE

### Clean Slate Test

```bash
# 1. Delete repo
sudo rm -rf /home/qt/quantum_trader

# 2. Clone fresh
cd /home/qt
git clone https://github.com/binyaminsemerci-ops/quantum_trader.git

# 3. Install systemd units
sudo cp quantum_trader/systemd/templates/quantum-golden.service \
        /etc/systemd/system/quantum-ai-engine.service
# (customize Description, EnvironmentFile, ExecStart)
sudo systemctl daemon-reload
sudo systemctl enable quantum-ai-engine

# 4. Start service
sudo systemctl start quantum-ai-engine

# 5. Run audit
cd /home/qt/quantum_trader
make audit
```

**Expected Result:** ✅ ALL CHECKS PASS

**IF FAILS → CONTRACT IS BROKEN**

---

## 🚨 ANTI-PATTERNS (FORBIDDEN)

```bash
# ❌ WRONG: System python
python ops/quality_gate.py

# ✅ CORRECT: Wrapper
ops/run.sh ai-engine ops/model_safety/quality_gate.py

# ❌ WRONG: Direct venv call
/opt/quantum/venvs/ai-engine/bin/python ops/training/train.py

# ✅ CORRECT: Wrapper
ops/run.sh ai-engine ops/training/train.py

# ❌ WRONG: Env in repo
source .env && python app.py

# ✅ CORRECT: Env in /etc
# (systemd unit sources /etc/quantum/<service>.env automatically)

# ❌ WRONG: Docker Redis
docker exec quantum_redis redis-cli

# ✅ CORRECT: systemd Redis
redis-cli -h localhost -p 6379

# ❌ WRONG: Hardcoded path
DB_PATH = "/home/user/data/db.sqlite"

# ✅ CORRECT: Env var
DB_PATH = os.getenv('DB_PATH')

# ❌ WRONG: Root user
User=root

# ✅ CORRECT: Least privilege
User=qt

# ❌ WRONG: Missing WorkingDirectory
# (implicit cwd)

# ✅ CORRECT: Explicit
WorkingDirectory=/home/qt/quantum_trader

# ❌ WRONG: Auto-activation
if quality_gate_pass:
    activate_model()

# ✅ CORRECT: Manual only
# Human runs: ops/model_safety/canary_activate.sh
```

---

## ✅ SUCCESS CRITERIA

### Validation Checklist

- [x] All 6 deliverables created
- [x] STANDARD_CONTRACT.md (500+ lines, 14 sections)
- [x] quantum-golden.service (template with examples)
- [x] ops/run.sh (FAIL-CLOSED wrapper, executable)
- [x] Makefile updated (ALL targets use wrapper)
- [x] ops/audit_contract.py (compliance validator, executable)
- [x] SYSTEMD_STANDARD_README.md (quick reference)
- [x] No Docker references
- [x] No Compose references
- [x] No implicit defaults
- [x] No auto-activation logic
- [x] Everything explicit
- [x] Everything repeatable
- [x] Everything audit-ready
- [x] Committed to git (1923072b)
- [x] Pushed to GitHub

### Deployment Test (VPS)

```bash
# On VPS:
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254

cd /home/qt/quantum_trader
git pull origin main

# Expected files:
ls -l docs/STANDARD_CONTRACT.md
ls -l docs/SYSTEMD_STANDARD_README.md
ls -l systemd/templates/quantum-golden.service
ls -l ops/run.sh
ls -l ops/audit_contract.py
ls -l Makefile

# Test wrapper
ops/run.sh ai-engine ops/model_safety/quality_gate.py

# Test Makefile
make quality-gate
make scoreboard
make audit

# Expected: ALL PASS or clear error messages (FAIL-CLOSED)
```

---

## 🎯 PRINCIPLE

**This is the foundation for the ensemble system.**

- **Precision > Speed**
- **Fail-Closed > "It Works"**
- **Explicit > Implicit**
- **Repeatable > Manual**
- **Auditable > Convenient**

**One Standard. No Exceptions.**

---

## 🔐 SIGNATURE

**Principal Systems Architect**  
**Date:** 2026-01-10  
**Commit:** 1923072b  
**Status:** ✅ DELIVERED IN FULL

**All deliverables completed as specified.**  
**No Docker. No Compose. No deviations.**  
**Ready for production enforcement.**

---

## 📝 CHANGE LOG

| Commit | Date | Change |
|--------|------|--------|
| 1923072b | 2026-01-10 | Initial Golden Contract implementation |

---

## 🚀 NEXT STEPS

1. **Deploy on VPS:**
   ```bash
   git pull origin main
   make audit  # Verify compliance
   ```

2. **Update Existing Services:**
   - Audit current units: `make audit`
   - Fix violations per SYSTEMD_STANDARD_README.md
   - Re-audit: `make audit` → must exit 0

3. **Add New Services:**
   - Follow docs/SYSTEMD_STANDARD_README.md section "How to Add a New Service"
   - Copy quantum-golden.service template
   - Run audit: `make audit`

4. **Enforce Contract:**
   - Run `make audit` before EVERY deployment
   - Exit 1 = DO NOT DEPLOY (fix violations first)
   - Exit 0 = DEPLOY APPROVED

5. **Training:**
   - Read STANDARD_CONTRACT.md (single source of truth)
   - Study SYSTEMD_STANDARD_README.md (quick reference)
   - Practice: Add dummy service, run audit, rollback

---

**END OF PROOF**
