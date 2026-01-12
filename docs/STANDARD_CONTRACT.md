# SYSTEMD GOLDEN CONTRACT

**Version:** 1.0.0  
**Effective:** 2026-01-10  
**Authority:** Principal Systems Architect  
**Scope:** ALL Quantum Trading System Components

> **THIS IS THE SINGLE SOURCE OF TRUTH**  
> Any deviation from this contract is a production incident.

---

## A. FILESYSTEM LAYOUT

### Repository Location

```
/home/qt/quantum_trader
```

**IMMUTABLE RULES:**
- ✅ ALL code lives here
- ✅ ALL relative paths start from this root
- ✅ Git working directory = this path
- ❌ NEVER use ~/quantum_trader or symbolic links
- ❌ NEVER clone to /opt or /var

### Data Directory

```
/opt/quantum/data/
├── quantum_trader.db         # SQLite main database
├── redis_dumps/              # Redis persistence
├── training_data/            # Historical OHLCV
├── model_checkpoints/        # Training outputs
└── backups/                  # System backups
```

**IMMUTABLE RULES:**
- ✅ ALL persistent state lives here
- ✅ Must survive repo deletion
- ✅ Owned by qt:qt (755)
- ❌ NEVER store in repo
- ❌ NEVER use /tmp or /var/tmp

### Models Directory

```
/opt/quantum/ai_engine/models/
├── active/
│   ├── xgboost_P0.4.pkl
│   ├── lightgbm_P0.5.pkl
│   └── nhits_P0.2.pt
├── shadow/
│   └── patchtst_P0.7.pt
└── archive/
    └── <timestamp>_<model>_<version>
```

**IMMUTABLE RULES:**
- ✅ Models deployed via canary_activate.sh ONLY
- ✅ active/ = production models
- ✅ shadow/ = shadow-mode evaluation
- ✅ archive/ = timestamped backups
- ❌ NEVER overwrite active models without backup
- ❌ NEVER auto-promote from shadow

### Logs Directory

```
Primary: journalctl -u quantum-<service>
Fallback: /var/log/quantum/<service>.log
```

**IMMUTABLE RULES:**
- ✅ systemd journal is primary source
- ✅ Structured logging with JSON
- ✅ Log rotation via logrotate
- ❌ NEVER log to repo directory
- ❌ NEVER log credentials

---

## B. PYTHON RUNTIME

### Virtual Environment Paths

```
/opt/quantum/venvs/
├── ai-engine/        # AI Engine + training + quality gates
├── backend/          # FastAPI backend
├── execution/        # Position execution service
└── rl-agent/         # RL sizing + monitor
```

**GOLDEN RULE:**

```bash
# ❌ NEVER USE:
python script.py
python3 script.py
/usr/bin/python3 script.py

# ✅ ALWAYS USE:
/opt/quantum/venvs/<service>/bin/python script.py
```

### Service-to-Venv Mapping

| Service | Venv | Why |
|---------|------|-----|
| quantum-ai-engine | ai-engine | PyTorch, PatchTST, training |
| quantum-backend | backend | FastAPI, SQLAlchemy |
| quantum-execution | execution | ccxt, async |
| quantum-rl-* | rl-agent | Gymnasium, stable-baselines3 |
| ops/training/* | ai-engine | Shared with AI Engine |
| ops/model_safety/* | ai-engine | Needs numpy, redis |

**IMMUTABLE RULES:**
- ✅ Each venv is isolated
- ✅ Ops scripts use same venv as runtime service
- ✅ Venv paths are absolute in systemd units
- ❌ NEVER use system python
- ❌ NEVER mix venvs (e.g., backend importing ai-engine deps)

---

## C. ENVIRONMENT CONFIGURATION

### Environment File Location

```
/etc/quantum/
├── ai-engine.env
├── backend.env
├── execution.env
├── rl-agent.env
└── common.env        # Shared vars
```

**GOLDEN RULE:**

```bash
# ❌ NEVER USE:
- Hardcoded paths in code
- .env files in repo
- Environment variables set in shell
- Docker Compose env files

# ✅ ALWAYS USE:
- /etc/quantum/<service>.env
- EnvironmentFile= in systemd unit
- source /etc/quantum/<service>.env in ops scripts
```

### Required Variables (ALL Services)

```bash
# /etc/quantum/<service>.env

# Paths (ABSOLUTE)
REPO_ROOT=/home/qt/quantum_trader
DATA_DIR=/opt/quantum/data
MODEL_DIR=/opt/quantum/ai_engine/models

# Runtime
SERVICE_NAME=<service>
LOG_LEVEL=INFO
PYTHONUNBUFFERED=1

# Redis (LOCALHOST ONLY)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Database
DB_PATH=/opt/quantum/data/quantum_trader.db

# Safety
AUTO_ACTIVATION=disabled
SHADOW_MODE=enabled
```

**IMMUTABLE RULES:**
- ✅ ALL paths are absolute
- ✅ ALL services share common Redis
- ✅ Env files owned by root:qt (640)
- ✅ NO secrets in env files (use /opt/quantum/secrets/)
- ❌ NEVER commit env files to git
- ❌ NEVER use relative paths

---

## D. SYSTEMD UNIT RULES

### Mandatory Fields

```ini
[Unit]
Description=<Service Name>
After=network-online.target redis-server.service
Wants=network-online.target
BindsTo=redis-server.service  # For services that REQUIRE Redis

[Service]
Type=simple
User=qt
Group=qt
WorkingDirectory=/home/qt/quantum_trader
EnvironmentFile=/etc/quantum/<service>.env
ExecStart=/opt/quantum/venvs/<service>/bin/python <script>
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=quantum-<service>

[Install]
WantedBy=multi-user.target
```

**IMMUTABLE RULES:**

1. **WorkingDirectory** - ALWAYS /home/qt/quantum_trader
2. **EnvironmentFile** - ALWAYS /etc/quantum/<service>.env
3. **User** - ALWAYS qt (NEVER root - principle of least privilege)
4. **ExecStart** - ALWAYS full venv path (NEVER bare python)
5. **Restart** - ALWAYS "always" for runtime services
6. **After** - ALWAYS redis-server.service for services using Redis
7. **StandardOutput/StandardError** - ALWAYS journal (NOT file)

### Service Types

**Runtime Services:**
- Type=simple
- Restart=always
- Long-running processes (FastAPI, AI Engine, etc.)

**Ops Jobs:**
- Type=oneshot
- RemainAfterExit=no
- Triggered manually or via timer

### Dependency Chain

```
redis-server.service (system package)
  ↓
quantum-backend.service
  ↓
quantum-ai-engine.service
  ↓
quantum-execution.service
  ↓
quantum-rl-*.service
```

**IMMUTABLE RULES:**
- ✅ Redis MUST be running before any Quantum service
- ✅ Backend MUST be running before AI Engine
- ✅ Use After= and BindsTo= to enforce dependencies
- ❌ NEVER start services out of order
- ❌ NEVER ignore dependency failures

---

## E. REDIS / DEPENDENCIES

### Redis Service

```bash
# System service (NOT container)
systemctl status redis-server.service

# Connection
Host: localhost
Port: 6379
Socket: /var/run/redis/redis-server.sock (optional)
```

**IMMUTABLE RULES:**
- ✅ Redis runs as systemd service (apt install redis-server)
- ✅ ALL services connect to localhost:6379
- ✅ NO container names (quantum_redis, redis, etc.)
- ✅ Persistence enabled (appendonly yes)
- ❌ NEVER use Docker Redis
- ❌ NEVER use remote Redis (security risk)

### Dependency Validation

**Before ANY service starts:**

```bash
# Check Redis
redis-cli PING || exit 1

# Check Database
test -f /opt/quantum/data/quantum_trader.db || exit 1

# Check Venv
test -d /opt/quantum/venvs/<service> || exit 1

# Check Env File
test -f /etc/quantum/<service>.env || exit 1
```

**FAIL-CLOSED RULE:**  
If ANY dependency check fails → EXIT IMMEDIATELY with non-zero code.

---

## F. SAFETY RULES

### Model Activation

```
Shadow → Canary → Manual Promote ONLY
```

**GOLDEN WORKFLOW:**

1. Train new model → archive checkpoint
2. Deploy to shadow/ → passive evaluation (no trades)
3. Run quality_gate.py → MUST PASS (exit 0)
4. Run canary_activate.sh → Manual confirmation required
5. Update .env.model_config → Explicit model path
6. Restart service → journalctl proof
7. Monitor → scoreboard.py every hour

**IMMUTABLE RULES:**
- ✅ Quality gate MUST pass (exit 0) before activation
- ✅ Canary activation requires manual confirmation
- ✅ Backup created before EVERY model change
- ✅ Git hash recorded in activation log
- ❌ NEVER auto-activate from training
- ❌ NEVER skip quality gate
- ❌ NEVER overwrite backups

### Quality Gate (BLOCKER)

```bash
make quality-gate
# Exit 0 = PASS (safe to proceed)
# Exit 2 = FAIL (BLOCKER - stop immediately)
```

**FAIL-CLOSED RULES:**
- <200 telemetry events → FAIL
- Any model collapse (>70% one class) → FAIL
- Confidence std <0.05 → FAIL
- P10-P90 range <0.12 → FAIL
- HOLD >85% → FAIL

**Manglende bevis = ingen aktivering** (Missing proof = no activation)

### Rollback Protocol

```bash
ops/model_safety/rollback_last.sh
```

**Mandatory when:**
- Quality gate fails AFTER activation
- P&L drops >5% in 1 hour
- Error rate >1% in production
- Any systemd service fails to start

**IMMUTABLE RULES:**
- ✅ Rollback script restores previous .env.model_config
- ✅ Git hash logged for audit
- ✅ Restart service automatically
- ✅ Notify via journal
- ❌ NEVER rollback without backup
- ❌ NEVER skip restart

---

## G. OPS EXECUTION MODEL

### Ops Wrapper (MANDATORY)

```bash
# ❌ NEVER RUN DIRECTLY:
python ops/training/train_patchtst.py

# ✅ ALWAYS USE:
ops/run.sh ai-engine ops/training/train_patchtst.py
```

**ops/run.sh responsibilities:**
1. Source /etc/quantum/<service>.env
2. Export PATH with venv bin/
3. Set PYTHONPATH=/home/qt/quantum_trader
4. Validate Redis connection
5. Validate Database exists
6. Validate Venv exists
7. Abort if ANY check fails
8. Execute script with full venv python path

### Makefile Integration

```makefile
# ❌ OLD WAY:
quality-gate:
	python ops/model_safety/quality_gate.py

# ✅ NEW WAY:
quality-gate:
	ops/run.sh ai-engine ops/model_safety/quality_gate.py
```

**IMMUTABLE RULES:**
- ✅ ALL Makefile targets use ops/run.sh
- ✅ NO direct python calls in Makefile
- ✅ Service name passed as first arg
- ❌ NEVER bypass wrapper

---

## H. AUDIT & COMPLIANCE

### Audit Script

```bash
ops/audit_contract.py
# Exit 0 = compliant
# Exit 1 = violations found
```

**Checks performed:**
1. systemd unit file syntax
2. WorkingDirectory = /home/qt/quantum_trader
3. EnvironmentFile exists
4. User = qt
5. Venv path matches service
6. Env file has required vars
7. Redis is running
8. Database exists

**Run frequency:**
- Before EVERY deployment
- After EVERY config change
- Daily via cron

### Contract Violations

**Examples of violations:**

❌ Using `python` instead of `/opt/quantum/venvs/<service>/bin/python`  
❌ Hardcoded paths in code (e.g., `/home/user/models`)  
❌ Missing EnvironmentFile in systemd unit  
❌ WorkingDirectory pointing to wrong path  
❌ Env file in repo instead of /etc/quantum/  
❌ Docker/Compose references  
❌ Auto-activation logic in code  

**Remediation:**
1. Stop service immediately
2. Fix violation
3. Run audit_contract.py
4. Restart service
5. Verify in journalctl

---

## I. IMMUTABILITY CHECKLIST

Before deploying ANY change:

- [ ] systemd unit follows golden template
- [ ] Python calls use full venv path
- [ ] WorkingDirectory = /home/qt/quantum_trader
- [ ] EnvironmentFile = /etc/quantum/<service>.env
- [ ] No hardcoded paths in code
- [ ] Ops scripts use ops/run.sh wrapper
- [ ] Quality gate runs before model activation
- [ ] Backup created before changes
- [ ] Git hash recorded
- [ ] Audit script passes (exit 0)
- [ ] journalctl shows clean startup

**IF ANY CHECKBOX FAILS → DO NOT DEPLOY**

---

## J. PRINCIPLE OF LEAST PRIVILEGE

### Why User=qt (NOT root)?

1. **Security**: Compromised service cannot modify system files
2. **Isolation**: Service cannot interfere with other users
3. **Auditability**: Clear ownership of files and processes
4. **Compliance**: Industry best practice for production services

**What qt user CAN do:**
- Read/write /home/qt/quantum_trader
- Read/write /opt/quantum/data
- Read/write /opt/quantum/ai_engine/models
- Connect to Redis localhost
- Write logs via journalctl

**What qt user CANNOT do:**
- Modify systemd units (requires sudo)
- Install packages (requires sudo)
- Modify /etc/quantum/ env files (requires sudo)
- Access other users' files

**IMMUTABLE RULE:**  
If a service needs root → REDESIGN THE SERVICE (it's wrong).

---

## K. REPEATABILITY GUARANTEE

### Clean Slate Test

```bash
# 1. Delete repo
sudo rm -rf /home/qt/quantum_trader

# 2. Clone fresh
cd /home/qt
git clone <repo_url> quantum_trader

# 3. Install systemd units
sudo cp systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload

# 4. Start services
sudo systemctl start quantum-backend
sudo systemctl start quantum-ai-engine

# 5. Run audit
ops/audit_contract.py

# Expected: ✅ ALL PASS
```

**SUCCESS CRITERIA:**
- All services start without errors
- All ops scripts run without modification
- All paths resolve correctly
- Audit passes with exit 0

**IF THIS FAILS → CONTRACT IS BROKEN**

---

## L. ANTI-PATTERNS (FORBIDDEN)

### DO NOT DO THIS:

```bash
# ❌ Relative paths
cd models && python train.py

# ❌ System python
python3 train.py

# ❌ Env in repo
source .env && python app.py

# ❌ Docker references
docker exec quantum_redis redis-cli

# ❌ Hardcoded paths
DB_PATH = "/home/user/data/db.sqlite"

# ❌ Auto-activation
if quality_gate_pass:
    activate_model()  # NO!

# ❌ Root user
User=root  # NEVER!

# ❌ Missing WorkingDirectory
# (relies on implicit cwd)

# ❌ Bypassing wrapper
python ops/quality_gate.py  # NO!
```

### DO THIS INSTEAD:

```bash
# ✅ Absolute venv path
/opt/quantum/venvs/ai-engine/bin/python /home/qt/quantum_trader/ops/training/train.py

# ✅ Env from /etc
source /etc/quantum/ai-engine.env

# ✅ Systemd Redis
redis-cli -h localhost -p 6379

# ✅ Env vars for paths
DB_PATH = os.getenv('DB_PATH')

# ✅ Manual activation
# Human runs: ops/model_safety/canary_activate.sh

# ✅ Least privilege
User=qt

# ✅ Explicit WorkingDirectory
WorkingDirectory=/home/qt/quantum_trader

# ✅ Wrapper usage
ops/run.sh ai-engine ops/model_safety/quality_gate.py
```

---

## M. CHANGE LOG

| Version | Date | Change | Author |
|---------|------|--------|--------|
| 1.0.0 | 2026-01-10 | Initial contract | SRE Team |

---

## N. SIGNATURE

This contract is **IMMUTABLE** unless approved by Principal Systems Architect.

**Effective Date:** 2026-01-10  
**Next Review:** 2026-04-10 (quarterly)

**Acknowledgment:**  
By deploying to this system, you agree to follow this contract without exception.

---

**END OF CONTRACT**
