# SYSTEMD GOLDEN CONTRACT - QUICK REFERENCE

**For:** Developers, SREs, Ops Engineers  
**Purpose:** How to work with the systemd-only Quantum system  
**Authority:** [STANDARD_CONTRACT.md](STANDARD_CONTRACT.md)

---

## 📖 Table of Contents

1. [How to Add a New Service](#how-to-add-a-new-service)
2. [How to Run Ops Safely](#how-to-run-ops-safely)
3. [What Breaks the Contract](#what-breaks-the-contract)
4. [Rollback Rules](#rollback-rules)
5. [Quick Troubleshooting](#quick-troubleshooting)

---

## How to Add a New Service

### Step 1: Create Virtual Environment

```bash
sudo mkdir -p /opt/quantum/venvs
sudo python3 -m venv /opt/quantum/venvs/<service>
sudo chown -R qt:qt /opt/quantum/venvs/<service>
```

### Step 2: Install Dependencies

```bash
sudo -u qt /opt/quantum/venvs/<service>/bin/pip install -r requirements.txt
```

### Step 3: Create Environment File

```bash
sudo nano /etc/quantum/<service>.env
```

**Required variables:**
```bash
REPO_ROOT=/home/qt/quantum_trader
DATA_DIR=/opt/quantum/data
MODEL_DIR=/opt/quantum/ai_engine/models
SERVICE_NAME=<service>
LOG_LEVEL=INFO
PYTHONUNBUFFERED=1
REDIS_HOST=localhost
REDIS_PORT=6379
DB_PATH=/opt/quantum/data/quantum_trader.db
AUTO_ACTIVATION=disabled
```

**Set permissions:**
```bash
sudo chown root:qt /etc/quantum/<service>.env
sudo chmod 640 /etc/quantum/<service>.env
```

### Step 4: Create systemd Unit

```bash
# Copy golden template
sudo cp /home/qt/quantum_trader/systemd/templates/quantum-golden.service \
        /etc/systemd/system/quantum-<service>.service

# Edit (customize Description, EnvironmentFile, ExecStart)
sudo nano /etc/systemd/system/quantum-<service>.service
```

**Customize these fields:**
```ini
Description=Quantum Trading System - <YOUR SERVICE NAME>
EnvironmentFile=/etc/quantum/<service>.env
ExecStart=/opt/quantum/venvs/<service>/bin/python <SCRIPT_PATH>
```

**Keep these fields as-is:**
```ini
WorkingDirectory=/home/qt/quantum_trader
User=qt
Group=qt
After=network-online.target redis-server.service
StandardOutput=journal
StandardError=journal
```

### Step 5: Enable and Start

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable (start on boot)
sudo systemctl enable quantum-<service>

# Start now
sudo systemctl start quantum-<service>

# Check status
sudo systemctl status quantum-<service>

# Follow logs
journalctl -u quantum-<service> -f
```

### Step 6: Verify Compliance

```bash
cd /home/qt/quantum_trader
make audit
# Should exit 0 (all checks pass)
```

---

## How to Run Ops Safely

### Golden Rule

**NEVER:**
```bash
python ops/quality_gate.py                    # ❌ WRONG
python3 ops/training/train_patchtst.py        # ❌ WRONG
/opt/quantum/venvs/ai-engine/bin/python ...   # ❌ WRONG
```

**ALWAYS:**
```bash
ops/run.sh ai-engine ops/model_safety/quality_gate.py    # ✅ CORRECT
make quality-gate                                         # ✅ CORRECT
```

### Common Operations

**Quality Gate:**
```bash
make quality-gate
# Checks model quality from telemetry
# Exit 0 = PASS, Exit 2 = FAIL (BLOCKER)
```

**Scoreboard:**
```bash
make scoreboard
# Shows all models' status (GO/WAIT/NO-GO)
```

**Training:**
```bash
make train-patchtst
# Trains PatchTST model (does NOT activate)
```

**Dataset Builder:**
```bash
make build-patchtst-dataset
# Builds temporal sequences for PatchTST
```

**Audit:**
```bash
make audit
# Validates Golden Contract compliance
# Exit 0 = compliant, Exit 1 = violations
```

### Manual Ops Execution

```bash
# Generic pattern:
ops/run.sh <service> <script> [args...]

# Examples:
ops/run.sh ai-engine ops/model_safety/quality_gate.py
ops/run.sh ai-engine ops/training/train_patchtst.py --epochs 100
ops/run.sh backend scripts/backup_database.py
```

### What ops/run.sh Does

1. ✅ Validates repo root exists
2. ✅ Validates venv exists
3. ✅ Validates env file exists
4. ✅ Validates script exists
5. ✅ Checks Redis is running
6. ✅ Checks database exists
7. ✅ Sources /etc/quantum/<service>.env
8. ✅ Sets PYTHONPATH and PATH
9. ✅ Executes with correct Python interpreter

**FAIL-CLOSED:** If ANY check fails → exits immediately.

---

## What Breaks the Contract

### Filesystem Violations

❌ Using `~/quantum_trader` instead of `/home/qt/quantum_trader`  
❌ Storing data in repo instead of `/opt/quantum/data`  
❌ Hardcoding paths like `/home/user/models`  
❌ Using `/tmp` or `/var/tmp` for persistent data  

### Python Runtime Violations

❌ Using `python` or `python3` commands  
❌ Using system python: `/usr/bin/python3`  
❌ Wrong venv: `/opt/quantum/venvs/backend/bin/python` for AI Engine ops  
❌ No venv path in systemd ExecStart  

### Environment Violations

❌ `.env` files in repo  
❌ Hardcoded environment variables in code  
❌ Missing `/etc/quantum/<service>.env`  
❌ Wrong permissions on env files (should be 640)  
❌ Missing required variables (REPO_ROOT, DATA_DIR, etc.)  

### systemd Violations

❌ `User=root` (should be `User=qt`)  
❌ Missing `WorkingDirectory`  
❌ Wrong `WorkingDirectory` (not /home/qt/quantum_trader)  
❌ Missing `EnvironmentFile`  
❌ Missing `After=redis-server.service`  
❌ `StandardOutput=file` (should be journal)  

### Redis Violations

❌ Docker Redis containers  
❌ Remote Redis (not localhost)  
❌ Container names like `quantum_redis`  
❌ Missing Redis dependency check  

### Safety Violations

❌ Auto-activation of models  
❌ Skipping quality gate  
❌ No backup before model change  
❌ Overwriting active models without archive  
❌ Activating without manual confirmation  

### Ops Violations

❌ Bypassing ops/run.sh wrapper  
❌ Direct Python calls in Makefile  
❌ Running ops without env file  
❌ Missing dependency validation  

---

## Rollback Rules

### When to Rollback

**Mandatory rollback triggers:**
- Quality gate fails AFTER activation
- P&L drops >5% in 1 hour
- Error rate >1% in production
- systemd service fails to start
- Audit fails (exit 1)

### Rollback Procedure

```bash
# Automatic rollback script
cd /home/qt/quantum_trader
ops/model_safety/rollback_last.sh

# Manual confirmation required
# Script will:
# 1. Show current config
# 2. Show backup config
# 3. Ask for confirmation
# 4. Restore backup
# 5. Restart service
# 6. Log to journal
```

### Post-Rollback Verification

```bash
# Check service status
sudo systemctl status quantum-ai-engine

# Check logs for clean startup
journalctl -u quantum-ai-engine --since "1 minute ago"

# Run quality gate
make quality-gate

# Run scoreboard
make scoreboard

# Run audit
make audit
```

**All checks must pass before proceeding.**

---

## Quick Troubleshooting

### Service Won't Start

```bash
# 1. Check env file
ls -l /etc/quantum/<service>.env
sudo cat /etc/quantum/<service>.env | grep REPO_ROOT

# 2. Check venv
ls -l /opt/quantum/venvs/<service>/bin/python

# 3. Check Redis
redis-cli PING
sudo systemctl status redis-server

# 4. Check logs
journalctl -u quantum-<service> -n 50

# 5. Run audit
make audit
```

### Service Crashes Immediately

```bash
# 1. Check Python syntax
/opt/quantum/venvs/<service>/bin/python -m py_compile <script.py>

# 2. Check dependencies
/opt/quantum/venvs/<service>/bin/pip list

# 3. Check Redis connection
redis-cli -h localhost -p 6379 PING

# 4. Check database
ls -l /opt/quantum/data/quantum_trader.db

# 5. Test manually
ops/run.sh <service> <script.py>
```

### Permission Errors

```bash
# Check ownership
ls -la /home/qt/quantum_trader
ls -la /opt/quantum/data

# Fix ownership
sudo chown -R qt:qt /home/qt/quantum_trader
sudo chown -R qt:qt /opt/quantum/data
sudo chown -R qt:qt /opt/quantum/ai_engine/models

# Check env file permissions
ls -l /etc/quantum/<service>.env
sudo chmod 640 /etc/quantum/<service>.env
sudo chown root:qt /etc/quantum/<service>.env

# Verify qt user exists
id qt
```

### Audit Fails

```bash
# Run audit with details
make audit

# Check specific violations
# Example output:
# ❌ Unit: WorkingDirectory not /home/qt/quantum_trader
# ❌ Env: Missing vars: REPO_ROOT, DATA_DIR
# ❌ Makefile: Direct Python calls found

# Fix violations one by one
# Re-run audit after each fix
make audit
```

### Quality Gate Fails

```bash
# Check Redis stream
redis-cli XLEN quantum:stream:trade.intent
# Need at least 200 events

# Check event format
redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 1

# Run quality gate manually
make quality-gate
cat reports/safety/quality_gate_*.md

# Wait for more events to accumulate
sleep 600  # 10 minutes
make quality-gate
```

### Makefile Target Fails

```bash
# Check wrapper exists and is executable
ls -l ops/run.sh
chmod +x ops/run.sh

# Test wrapper manually
ops/run.sh ai-engine ops/model_safety/quality_gate.py

# Check Makefile uses wrapper
grep "ops/run.sh" Makefile

# Verify no direct Python calls
grep -E "python |python3 " Makefile
# Should ONLY show in comments, NOT in targets
```

---

## Emergency Contacts

**Production incidents:**
1. Stop service: `sudo systemctl stop quantum-<service>`
2. Check logs: `journalctl -u quantum-<service> --since "10 minutes ago"`
3. Run audit: `make audit`
4. If unsafe: Rollback: `ops/model_safety/rollback_last.sh`
5. Document incident in `docs/incidents/<date>.md`

**Contract violations:**
1. Do NOT deploy
2. Fix violation
3. Run audit: `make audit` (must exit 0)
4. Get approval from SRE team
5. Deploy with monitoring

---

## Reference Documentation

- **Full Contract:** [docs/STANDARD_CONTRACT.md](STANDARD_CONTRACT.md)
- **Template:** [systemd/templates/quantum-golden.service](../systemd/templates/quantum-golden.service)
- **Wrapper:** [ops/run.sh](../ops/run.sh)
- **Audit:** [ops/audit_contract.py](../ops/audit_contract.py)

---

**Questions?** See [STANDARD_CONTRACT.md](STANDARD_CONTRACT.md) for authoritative answers.

**Remember:** Precision > Speed. Fail-Closed > "It Works".
