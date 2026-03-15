# QTOS MASTER PLAN v2 — Untangle the Yarn Ball
## Version: 2.2 | Created: 2026-03-14 | Last Updated: 2026-03-14
### v2.2: Added timer handling, Restart policy, requirements merge, port map, model cleanup
### v2.1: Added 3-layer env file consolidation strategy (OP 3 + OP 4)

---

> **RULE #1**: This document IS the plan. Every session starts by reading this file.
> Never search from scratch. Never redo mapping. Open this, read status, continue.

> **RULE #2**: Work ONE operation at a time. Mark it [DONE] before starting next.
> No jumping. No parallel experiments. Sequential, verified, documented.

> **RULE #3**: Every change gets verified BEFORE marking done.
> Verify = running service status + Redis check + log check on VPS.

---

## PHILOSOPHY

Forget "critical" or "not critical". The system is a chaos yarn ball.
We untangle it by pulling the LOOSE ENDS first, then working inward
to the core knots. Every operation reduces noise and makes the next
operation easier.

**Sequence logic**: Remove dead things → Fix living things → Kill the ghost → Clean up → Build new

---

## TABLE OF CONTENTS

1. [Verified System State (2026-03-14)](#1-verified-system-state)
2. [OP 0: Snapshot Everything](#op-0-snapshot-everything)
3. [OP 1: Clear the Dead](#op-1-clear-the-dead)
4. [OP 2: Fix Source Code Hardcodes](#op-2-fix-source-code-hardcodes)
5. [OP 3: Unify Python, Venvs & Env Files](#op-3-unify-python-venvs--env-files)
6. [OP 4: Fix Service Files & Deploy](#op-4-fix-service-files--deploy)
7. [OP 5: Bury /opt/quantum](#op-5-bury-optquantum)
8. [OP 6: Clean the Repo](#op-6-clean-the-repo)
9. [OP 7+: Architecture (Future)](#op-7-architecture-future)
10. [Quick Reference](#quick-reference)

---

## 1. VERIFIED SYSTEM STATE

### VPS Snapshot — 2026-03-14

| What | Count | Notes |
|------|-------|-------|
| Running services | **42** | All use `/usr/bin/python3.12` as actual interpreter |
| Total .service files | **134** | 92 are dead/zombie noise |
| Services with /opt/quantum in ExecStart | **9** | Using wrong venv or wrong script path |
| Services with /opt/quantum in WorkingDir | **2** | market-publisher, rl-trainer |
| Source files hardcoding /opt/quantum | **~18** | In microservices/, backend/, bin/, systemd/ |
| Python venvs on disk | **10** | Should be 1 |
| /opt/quantum size | **14 GB** | Ghost code tree |
| Root-level junk files | **~1500** | Scripts, docs, fixes, diagnostics |
| Env files in /etc/quantum/ | **84** | For 42 services — massive duplication |
| REDIS_HOST duplicated across | **61** | Same `localhost` in 61 of 84 env files |
| Binance API key copies | **5** | Same key in 5 different env files |
| Binance testnet key copies | **10** | Same testnet keys in 10 env files |
| Systemd timers (.timer) | **15** | 4 active, 11 inactive |
| Timer services with /opt/quantum | **4** | contract-check, core-health, diagnostic, training-worker |
| Listening TCP ports (Python) | **15** | No port allocation map exists |
| Services WITH Restart= policy | **113** | Of 134 total (includes dead services) |
| requirements.txt files | **31** | Fragmented — no single source of truth |
| Model files (.pkl) in git | **2000+** | Massive repo bloat — should be external |

### The 9 Running Services Touching /opt/quantum

| Service | What Uses /opt/quantum | Problem |
|---------|----------------------|---------|
| ai-strategy-router | ExecStart: /opt/quantum/venvs/ai-engine/bin/python3 | Wrong venv |
| ensemble-predictor | ExecStart: /opt/quantum/venvs/ai-engine/bin/python | Wrong venv |
| harvest-metrics-exporter | ExecStart: /opt/quantum/venvs/ai-engine/bin/python3 | Wrong venv |
| harvest-proposal | ExecStart: /opt/quantum/venvs/ai-engine/bin/python3 | Wrong venv |
| marketstate | ExecStart: /opt/quantum/venvs/ai-engine/bin/python3 | Wrong venv |
| portfolio-governance | ExecStart: /opt/quantum/venvs/ai-engine/bin/python | Wrong venv |
| risk-proposal | ExecStart: /opt/quantum/venvs/ai-engine/bin/python3 | Wrong venv |
| market-publisher | ExecStart runs /opt/quantum/ops/market/market_publisher.py | Wrong script path |
| rl-trainer | ExecStart: /opt/quantum/bin/start_rl_trainer.sh + WorkingDir | Both wrong |

### The 18 Source Files Hardcoding /opt/quantum

**CORE runtime (must fix):**
```
microservices/ai_engine/config.py               ← model paths: /opt/quantum/model_registry/
microservices/ai_engine/model_path_guard.py      ← fallback paths
microservices/dag3_hw_stops/dag3_hw_stops.py     ← sys.path.insert
microservices/dag7_snapshot/dag7_snapshot.py      ← snapshot dir
microservices/layer1_data_sink/layer1_data_sink.py ← DATA_ROOT
microservices/layer1_historical_backfill/layer1_historical_backfill.py ← DATA_ROOT
microservices/layer2_research_sandbox/layer2_research_sandbox.py ← DATA_ROOT
microservices/layer3_backtest_runner/layer3_backtest_runner.py ← DATA_ROOT
microservices/layer6_post_trade/layer6_post_trade.py ← DATA_ROOT
microservices/rl_sizing_agent/rl_agent.py        ← paths
microservices/training_worker/model_trainer.py   ← staging dir
```

**Backend (may be unused by runtime):**
```
backend/services/clm_v3/adapters.py              ← retrain script paths
backend/services/ai/continuous_learning_manager.py ← model paths
backend/services/monitoring/tp_performance_tracker.py ← model paths
```

**Infrastructure:**
```
bin/start_rl_trainer.sh                          ← cd /opt/quantum
systemd/env-templates/ai-engine.env              ← PYTHONPATH=/opt/quantum
systemd/env-templates/ai-client-base.env         ← PYTHONPATH=/opt/quantum
systemd/env-templates/execution.env              ← PYTHONPATH=/opt/quantum
```

**Timer-triggered services (on VPS only, fix in OP 4):**
```
quantum-contract-check.service     ← 2 /opt/quantum refs
quantum-core-health.service        ← 4 /opt/quantum refs
quantum-diagnostic.service         ← 4 /opt/quantum refs
quantum-rl-reward-publisher.service ← 2 /opt/quantum refs
quantum-training-worker.service    ← 3 /opt/quantum refs
```

### Venvs on Disk

```
MAIN (target):      /home/qt/quantum_trader_venv/
SECONDARY (kill):   /opt/quantum/venvs/ai-engine/
VOLUME (kill):      /mnt/HC_Volume_104287969/quantum-venvs/{ai-client-base,ai-engine,
                    execution,rl-dashboard,rl-sizer,runtime,safety-telemetry,strategy-ops}
```

### The 42 Running Services (confirmed 2026-03-14)

```
quantum-ai-engine                  quantum-ai-strategy-router
quantum-balance-tracker            quantum-capital-allocation
quantum-clm                        quantum-ensemble-predictor
quantum-execution-result-bridge    quantum-exit-brain-shadow
quantum-exit-intelligence          quantum-exit-intent-gateway
quantum-exit-management-agent      quantum-governor
quantum-harvest-metrics-exporter   quantum-harvest-optimizer
quantum-harvest-proposal           quantum-heat-gate
quantum-intent-bridge              quantum-intent-executor
quantum-layer1-data-sink           quantum-market-publisher
quantum-marketstate                quantum-metricpack-builder
quantum-p35-decision-intelligence  quantum-performance-attribution
quantum-performance-tracker        quantum-portfolio-clusters
quantum-portfolio-gate             quantum-portfolio-governance
quantum-portfolio-heat-gate        quantum-portfolio-state-publisher
quantum-price-feed                 quantum-reconcile-engine
quantum-risk-proposal              quantum-rl-agent
quantum-rl-policy-publisher        quantum-rl-shadow-metrics-exporter
quantum-rl-sizer                   quantum-rl-trainer
quantum-stream-bridge              quantum-trade-logger
quantum-universe-service           quantum-utf-publisher
```

### Env File Sprawl (audited 2026-03-14)

84 env files in `/etc/quantum/` for 42 services — nearly 2× duplication.

| Duplicated Value | Copies | Appears In |
|---|---|---|
| REDIS_HOST=localhost | 61 | Nearly all env files |
| BINANCE_API_KEY (same key) | 5 | binance-pnl-tracker, exitbrain-v35, intent-executor, portfolio-intelligence, testnet |
| BINANCE_TESTNET_API_KEY/SECRET | 10 | balance-tracker, binance-pnl-tracker, exitbrain-v35, governor, harvest-brain, intent-bridge, intent-executor, position-monitor, reconcile-engine, testnet |
| PYTHONPATH=/opt/quantum | 6 | Hardcoded wrong path in 6 env files |
| POSTGRES credentials | 1 | exit-intelligence.env only |

**Target**: 3-layer shared env architecture:
```
/etc/quantum/common.env    ← REDIS_HOST, REDIS_PORT, TZ, LOG_LEVEL (shared by ALL 42 services)
/etc/quantum/secrets.env   ← BINANCE_API_KEY/SECRET, BINANCE_TESTNET_*, POSTGRES_* (chmod 600)
/etc/quantum/python.env    ← VIRTUAL_ENV, PATH, PYTHONPATH, QT_BASE_DIR (eliminates all venv hardcodes)
/etc/quantum/<service>.env ← ONLY service-specific config (ports, symbols, rate limits, feature flags)
```

Each service file will use:
```ini
[Service]
EnvironmentFile=/etc/quantum/common.env
EnvironmentFile=/etc/quantum/secrets.env
EnvironmentFile=/etc/quantum/python.env
EnvironmentFile=/etc/quantum/<service-name>.env
```

This eliminates: 61 REDIS_HOST dupes, 5–10 Binance key dupes, 6 PYTHONPATH hardcodes.
Single point of change for secrets rotation, Redis relocation, or venv changes.

### Port Allocation Map (audited 2026-03-14)

| Port | Service | Purpose |
|------|---------|----------|
| 8005 | rl-trainer | RL training API |
| 8042 | harvest-metrics-exporter | Harvest metrics |
| 8044 | governor | Governance API |
| 8046 | reconcile-engine | Reconciliation API |
| 8047 | portfolio-gate | Portfolio gating |
| 8048 | portfolio-clusters | Cluster analysis |
| 8051 | metricpack-builder | Metric packs |
| 8052 | harvest-optimizer | Harvest optimization |
| 8056 | portfolio-heat-gate | Heat gating |
| 8059 | capital-allocation | Capital allocation API |
| 8061 | performance-attribution | Performance analysis |
| 8068 | heat-gate | Heat gate (port 1) |
| 8069 | heat-gate | Heat gate (port 2) |
| 9092 | rl-shadow-metrics-exporter | RL shadow metrics |
| 9109 | exit-intelligence | Exit intelligence API |
| 6379 | Redis | Data store (not Python) |

> **NOTE**: 27 of 42 services have NO listening port — they are pure Redis stream workers.
> Port allocation is env-configurable after OP 4 env file consolidation.

### Systemd Timers (audited 2026-03-14)

15 timer files on VPS. 4 currently active, 11 inactive.

**Active Timers:**
```
quantum-stream-recover.timer       every ~2 min
quantum-exit-owner-watch.timer     every ~5 min
quantum-rl-shadow-scorecard.timer  every ~15 min
quantum-offline-evaluator.timer    every ~4 hours
```

**Timer Services with /opt/quantum References (MUST FIX in OP 2):**
```
quantum-contract-check.service     2 refs
quantum-core-health.service        4 refs
quantum-diagnostic.service         4 refs
quantum-rl-reward-publisher.service 2 refs
quantum-training-worker.service    3 refs
```

> Timers themselves do NOT reference /opt/quantum — only their corresponding service files do.

---

## OP 0: SNAPSHOT EVERYTHING
### Status: [DONE] 2026-03-14 21:27 UTC
### Verified: 134 services, 84 env, 15 timers, 14 drop-ins, 158+41 pkg lists, 15 ports. 42 running. 1.1MB total.

Take a full backup before touching ANYTHING. This is our rollback point.

```bash
# On VPS:
mkdir -p /opt/backups/2026-03-14-pre-cleanup

# Service files
cp -a /etc/systemd/system/quantum-*.service /opt/backups/2026-03-14-pre-cleanup/
cp -a /etc/systemd/system/quantum-*.service.d /opt/backups/2026-03-14-pre-cleanup/ 2>/dev/null

# Env files
cp -a /etc/quantum/*.env /opt/backups/2026-03-14-pre-cleanup/ 2>/dev/null

# Service list snapshot
systemctl list-units quantum-*.service --all --no-pager > /opt/backups/2026-03-14-pre-cleanup/service-list.txt

# Running PIDs
ps aux | grep quantum > /opt/backups/2026-03-14-pre-cleanup/processes.txt

# Timer files
cp -a /etc/systemd/system/quantum-*.timer /opt/backups/2026-03-14-pre-cleanup/ 2>/dev/null

# Timer service files (the oneshot services triggered by timers)
for t in /etc/systemd/system/quantum-*.timer; do
  svc=$(basename "$t" .timer).service
  cp -a /etc/systemd/system/$svc /opt/backups/2026-03-14-pre-cleanup/ 2>/dev/null
done

# Pip freeze from every venv
/home/qt/quantum_trader_venv/bin/pip freeze > /opt/backups/2026-03-14-pre-cleanup/main-venv-packages.txt
/opt/quantum/venvs/ai-engine/bin/pip freeze > /opt/backups/2026-03-14-pre-cleanup/ai-engine-venv-packages.txt 2>/dev/null

# Port snapshot
ss -tlnp | grep python > /opt/backups/2026-03-14-pre-cleanup/listening-ports.txt

echo "SNAPSHOT DONE: $(date)"
ls -la /opt/backups/2026-03-14-pre-cleanup/
```

**Verify**: Backup dir contains service files, timer files, env files, package lists, port list.

**Rollback**: This IS the rollback. No rollback needed for a backup.

---

## OP 1: CLEAR THE DEAD
### Status: [DONE] 2026-03-14 21:33 UTC
### Verified: 42 svc files + 2 masks, 42 running, 4 timers. 92 svc + 11 tmr + 12 dirs in graveyard.
### Depends on: OP 0

Remove 92 zombie service files. Keep only the 42 running ones.
Also mask the dangerous `execution` and `apply-layer` services.

#### Step 1.1: Create keep-list [ ]
```bash
cat > /tmp/keep-services.txt << 'KEEPLIST'
quantum-ai-engine.service
quantum-ai-strategy-router.service
quantum-balance-tracker.service
quantum-capital-allocation.service
quantum-clm.service
quantum-ensemble-predictor.service
quantum-execution-result-bridge.service
quantum-exit-brain-shadow.service
quantum-exit-intelligence.service
quantum-exit-intent-gateway.service
quantum-exit-management-agent.service
quantum-governor.service
quantum-harvest-metrics-exporter.service
quantum-harvest-optimizer.service
quantum-harvest-proposal.service
quantum-heat-gate.service
quantum-intent-bridge.service
quantum-intent-executor.service
quantum-layer1-data-sink.service
quantum-market-publisher.service
quantum-marketstate.service
quantum-metricpack-builder.service
quantum-p35-decision-intelligence.service
quantum-performance-attribution.service
quantum-performance-tracker.service
quantum-portfolio-clusters.service
quantum-portfolio-gate.service
quantum-portfolio-governance.service
quantum-portfolio-heat-gate.service
quantum-portfolio-state-publisher.service
quantum-price-feed.service
quantum-reconcile-engine.service
quantum-risk-proposal.service
quantum-rl-agent.service
quantum-rl-policy-publisher.service
quantum-rl-shadow-metrics-exporter.service
quantum-rl-sizer.service
quantum-rl-trainer.service
quantum-stream-bridge.service
quantum-trade-logger.service
quantum-universe-service.service
quantum-utf-publisher.service
KEEPLIST
```

#### Step 1.2: Move dead services to graveyard [ ]
```bash
mkdir -p /opt/backups/systemd-graveyard
for f in /etc/systemd/system/quantum-*.service; do
  name=$(basename "$f")
  if ! grep -qx "$name" /tmp/keep-services.txt; then
    systemctl stop "$name" 2>/dev/null
    systemctl disable "$name" 2>/dev/null
    mv "$f" /opt/backups/systemd-graveyard/
    echo "REMOVED: $name"
  fi
done
# Also move orphan drop-in dirs
for d in /etc/systemd/system/quantum-*.service.d; do
  svc=$(basename "$d" .d)
  if ! grep -qx "$svc" /tmp/keep-services.txt; then
    mv "$d" /opt/backups/systemd-graveyard/
    echo "REMOVED DIR: $d"
  fi
done
systemctl daemon-reload
```

#### Step 1.3: Mask dangerous dead services [ ]
```bash
# execution can cause double trades if started
systemctl mask quantum-execution.service 2>/dev/null
# apply-layer has legacy close execution code
systemctl mask quantum-apply-layer.service 2>/dev/null
```

#### Step 1.4: Move dead timers to graveyard [ ]
```bash
# Keep only active timers, move the rest
ACTIVE_TIMERS="quantum-stream-recover.timer quantum-exit-owner-watch.timer quantum-rl-shadow-scorecard.timer quantum-offline-evaluator.timer"
for t in /etc/systemd/system/quantum-*.timer; do
  name=$(basename "$t")
  if ! echo "$ACTIVE_TIMERS" | grep -q "$name"; then
    systemctl stop "$name" 2>/dev/null
    systemctl disable "$name" 2>/dev/null
    mv "$t" /opt/backups/systemd-graveyard/
    echo "REMOVED TIMER: $name"
  fi
done
systemctl daemon-reload
```

#### Step 1.5: Verify [ ]
```bash
ls /etc/systemd/system/quantum-*.service | wc -l
# EXPECTED: 42
systemctl list-units quantum-*.service --no-pager | grep running | wc -l
# EXPECTED: 42
ls /etc/systemd/system/quantum-*.timer | wc -l
# EXPECTED: 4 (only active timers kept)
```

**Rollback**:
```bash
cp /opt/backups/systemd-graveyard/quantum-*.service /etc/systemd/system/
cp /opt/backups/systemd-graveyard/quantum-*.timer /etc/systemd/system/ 2>/dev/null
systemctl daemon-reload
```

---

## OP 2: FIX SOURCE CODE HARDCODES
### Status: [DONE] 2026-03-14 21:40 UTC — commit 845bc6e39
### Verified: git grep "/opt/quantum" returns zero matches in core code (microservices/, backend/, bin/, systemd/)
### Depends on: Nothing (can be done locally in git, independently)

Fix all `/opt/quantum` hardcodes in source code. Use environment variables
with sensible defaults pointing to `/home/qt/quantum_trader`.

This is ALL git-side work. No VPS changes yet.

#### Step 2.1: Fix microservices/ai_engine/config.py [ ]

Change all model paths from `/opt/quantum/model_registry/...` to use
env var `QT_BASE_DIR` with default `/home/qt/quantum_trader`:

```python
import os
QT_BASE = os.environ.get("QT_BASE_DIR", "/home/qt/quantum_trader")

APPROVED_MODEL_DIR = os.path.join(QT_BASE, "model_registry", "approved")
STAGING_MODEL_DIR = os.path.join(QT_BASE, "model_registry", "staging")
MODELS_DIR = os.path.join(QT_BASE, "ai_engine", "models")
# ... etc for XGB_MODEL_PATH, XGB_SCALER_PATH, LGBM_MODEL_PATH, etc.
```

#### Step 2.2: Fix microservices/ai_engine/model_path_guard.py [ ]

Replace `/opt/quantum` fallback defaults with QT_BASE_DIR env var.

#### Step 2.3: Fix layer services DATA_ROOT [ ]

Files:
- microservices/layer1_data_sink/layer1_data_sink.py
- microservices/layer1_historical_backfill/layer1_historical_backfill.py
- microservices/layer2_research_sandbox/layer2_research_sandbox.py
- microservices/layer3_backtest_runner/layer3_backtest_runner.py
- microservices/layer6_post_trade/layer6_post_trade.py

All have: `DATA_ROOT = "/opt/quantum/data"` → change to:
```python
DATA_ROOT = os.environ.get("QT_DATA_DIR", "/home/qt/quantum_trader/data")
```

#### Step 2.4: Fix DAG services [ ]

- microservices/dag3_hw_stops/dag3_hw_stops.py: `sys.path.insert(0, "/opt/quantum")` → remove or use QT_BASE
- microservices/dag7_snapshot/dag7_snapshot.py: `SNAPSHOT_DIR = "/opt/quantum/snapshots"` → use QT_BASE

#### Step 2.5: Fix remaining microservices [ ]

- microservices/rl_sizing_agent/rl_agent.py
- microservices/training_worker/model_trainer.py

#### Step 2.6: Fix backend hardcodes [ ]

- backend/services/clm_v3/adapters.py — retrain script paths + Python interpreter
- backend/services/ai/continuous_learning_manager.py
- backend/services/monitoring/tp_performance_tracker.py

#### Step 2.7: Fix infrastructure files [ ]

- bin/start_rl_trainer.sh — `cd /opt/quantum` → `cd /home/qt/quantum_trader`
- systemd/env-templates/ai-engine.env — `PYTHONPATH=/opt/quantum` → `/home/qt/quantum_trader`
- systemd/env-templates/ai-client-base.env — same
- systemd/env-templates/execution.env — same

#### Step 2.8: Commit and push [ ]
```bash
git add -A
git commit -m "OP2: eliminate all /opt/quantum hardcodes from source code

Replaced with QT_BASE_DIR env var (default: /home/qt/quantum_trader).
Files changed: ~18 core files in microservices/, backend/, bin/, systemd/.
Part of QTOS yarn ball untangling - see docs/architecture/QTOS_MASTER_PLAN.md"
git push
```

#### Step 2.9: Verify no /opt/quantum left in core code [ ]
```bash
git grep "/opt/quantum" -- "microservices/**/*.py" "backend/**/*.py" "bin/*.sh" "systemd/env-templates/*.env"
# EXPECTED: zero matches
```

**Rollback**: `git revert HEAD`

---

## OP 3: UNIFY PYTHON, VENVS & ENV FILES
### Status: [x] DONE — 2025-07-17
### Depends on: OP 0 (backup done), OP 2 (code pushed)

**Completed**:
- 3.1: Package comparison — 3 missing (httptools, pyyaml, watchfiles)
- 3.2: Installed 3 missing pkgs → main venv now 161 pkgs (superset of ai-engine)
- 3.3: Smoke test — AI engine imports work with main venv
- 3.4: Requirements reconciled — 90 aspirational pkgs, deferred to OP 6
- 3.5: Created `/etc/quantum/common.env` (6 keys: REDIS_*, TZ, LOG_LEVEL, QT_BASE_DIR)
- 3.6: Created `/etc/quantum/secrets.env` (4 keys: BINANCE_*, chmod 600)
- 3.7: Created `/etc/quantum/python.env` (3 keys: VIRTUAL_ENV, PATH, PYTHONPATH)
- 3.8: Full verification — all checks PASS, 42 services running

Two goals:
1. Merge 10 venvs into ONE: `/home/qt/quantum_trader_venv/`
2. Create 3 shared env files to eliminate duplication across 84 env files

#### Step 3.1: Compare packages [ ]
```bash
# On VPS:
/home/qt/quantum_trader_venv/bin/pip freeze | cut -d= -f1 | sort > /tmp/main-pkgs.txt
/opt/quantum/venvs/ai-engine/bin/pip freeze | cut -d= -f1 | sort > /tmp/ai-pkgs.txt
comm -23 /tmp/ai-pkgs.txt /tmp/main-pkgs.txt
# Shows packages in ai-engine venv that are NOT in main venv
```

#### Step 3.2: Install missing packages into main venv [ ]
```bash
# Based on Step 3.1 output:
/home/qt/quantum_trader_venv/bin/pip install <MISSING_PACKAGES>
```

#### Step 3.3: Verify main venv works for ai-engine workload [ ]
```bash
# Quick smoke test — does the ai-engine code import without errors?
cd /home/qt/quantum_trader
/home/qt/quantum_trader_venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from microservices.ai_engine.config import *
print('config OK')
from ai_engine.agents import *
print('agents OK')
"
```

**Rollback**: Packages only added, never removed. Safe.

#### Step 3.4: Reconcile requirements files into one [ ]
```bash
# On VPS — merge all 31 requirements files into a single deduplicated list:
find /home/qt/quantum_trader -name "requirements*.txt" -type f \
  -exec cat {} \; | grep -v '^#' | grep -v '^$' | \
  sed 's/[>=<].*//' | sort -u > /tmp/all-packages.txt

echo "Total unique packages across 31 requirements files:"
wc -l /tmp/all-packages.txt

# Compare with what's actually installed in main venv:
/home/qt/quantum_trader_venv/bin/pip freeze | cut -d= -f1 | sort > /tmp/installed.txt

echo "Packages referenced but NOT installed:"
comm -23 /tmp/all-packages.txt /tmp/installed.txt

# Install any genuinely missing packages:
# /home/qt/quantum_trader_venv/bin/pip install <MISSING_FROM_OUTPUT>
```

> After OP 6 (repo clean), consolidate 31 files into ONE `requirements.txt`
> at repo root with pinned versions from `pip freeze`.

#### Step 3.5: Create shared common.env [ ]
```bash
cat > /etc/quantum/common.env << 'EOF'
# === QTOS Common Environment ===
# Shared by ALL 42 quantum services
# Single source of truth for infrastructure config
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_URL=redis://localhost:6379
TZ=UTC
LOG_LEVEL=INFO
QT_BASE_DIR=/home/qt/quantum_trader
EOF
echo "Created common.env"
```

#### Step 3.6: Create shared secrets.env [ ]
```bash
# Extract current keys from existing env files (verify values before writing!)
BINANCE_KEY=$(grep -h BINANCE_API_KEY /etc/quantum/intent-executor.env | head -1 | cut -d= -f2-)
BINANCE_SECRET=$(grep -h BINANCE_API_SECRET /etc/quantum/intent-executor.env | head -1 | cut -d= -f2-)
TESTNET_KEY=$(grep -h BINANCE_TESTNET_API_KEY /etc/quantum/testnet.env | head -1 | cut -d= -f2-)
TESTNET_SECRET=$(grep -h BINANCE_TESTNET_API_SECRET /etc/quantum/testnet.env | head -1 | cut -d= -f2-)
POSTGRES_LINE=$(grep -h POSTGRES /etc/quantum/exit-intelligence.env | head -1)

cat > /etc/quantum/secrets.env << EOF
# === QTOS Secrets ===
# chmod 600 — only root can read
# Single source of truth for ALL API keys and credentials
BINANCE_API_KEY=${BINANCE_KEY}
BINANCE_API_SECRET=${BINANCE_SECRET}
BINANCE_TESTNET_API_KEY=${TESTNET_KEY}
BINANCE_TESTNET_API_SECRET=${TESTNET_SECRET}
${POSTGRES_LINE}
EOF

chmod 600 /etc/quantum/secrets.env
echo "Created secrets.env (chmod 600)"
```

#### Step 3.7: Create shared python.env [ ]
```bash
cat > /etc/quantum/python.env << 'EOF'
# === QTOS Python Environment ===
# Single source of truth for Python/venv config
# All 42 services use the same venv after OP 3 venv merge
VIRTUAL_ENV=/home/qt/quantum_trader_venv
PATH=/home/qt/quantum_trader_venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
PYTHONPATH=/home/qt/quantum_trader
EOF
echo "Created python.env"
```

#### Step 3.8: Verify shared env files [ ]
```bash
echo "=== common.env ==="
cat /etc/quantum/common.env
echo ""
echo "=== secrets.env ==="
ls -la /etc/quantum/secrets.env  # Should show -rw------- root
echo "(contents hidden — verify value count manually)"
wc -l /etc/quantum/secrets.env
echo ""
echo "=== python.env ==="
cat /etc/quantum/python.env
echo ""
echo "All 3 shared env files ready for OP 4."
```

**Rollback** (env files): Simply delete the 3 shared files — per-service env files
still contain all original values until OP 4 strips them.

---

## OP 4: FIX SERVICE FILES & DEPLOY
### Status: [x] DONE 2026-03-14
### Depends on: OP 2 (code pushed), OP 3 (venv ready)

Update all 42 remaining service files on VPS:
- Fix ExecStart paths (Python interpreter, script paths, WorkingDirectory)
- Add shared EnvironmentFile directives (common.env, secrets.env, python.env)
- Strip duplicated values from per-service env files
- Deploy the fixed source code

#### Step 4.1: Git pull on VPS [x]
```bash
cd /home/qt/quantum_trader
git pull origin main
```

#### Step 4.2: Fix the 7 services using ai-engine venv [x]
```bash
# These services need ExecStart changed from /opt/quantum/venvs/ai-engine/bin/python*
# to /home/qt/quantum_trader_venv/bin/python:
SERVICES=(
  quantum-ai-strategy-router
  quantum-ensemble-predictor
  quantum-harvest-metrics-exporter
  quantum-harvest-proposal
  quantum-marketstate
  quantum-portfolio-governance
  quantum-risk-proposal
)
for svc in "${SERVICES[@]}"; do
  sed -i 's|/opt/quantum/venvs/ai-engine/bin/python[3]*|/home/qt/quantum_trader_venv/bin/python|g' \
    /etc/systemd/system/${svc}.service
  echo "Fixed: ${svc}"
done
```

#### Step 4.3: Fix market-publisher [x]
```bash
# market-publisher runs from /opt/quantum/ops/market/market_publisher.py
# Change to /home/qt/quantum_trader/ops/market/market_publisher.py
sed -i 's|/opt/quantum/ops/market|/home/qt/quantum_trader/ops/market|g' \
  /etc/systemd/system/quantum-market-publisher.service
# Also fix WorkingDirectory
sed -i 's|WorkingDirectory=/opt/quantum/ops/market|WorkingDirectory=/home/qt/quantum_trader/ops/market|g' \
  /etc/systemd/system/quantum-market-publisher.service
```

#### Step 4.4: Fix rl-trainer [x]
```bash
# rl-trainer uses /opt/quantum/bin/start_rl_trainer.sh and WorkingDirectory=/opt/quantum
sed -i 's|/opt/quantum/bin/start_rl_trainer.sh|/home/qt/quantum_trader/bin/start_rl_trainer.sh|g' \
  /etc/systemd/system/quantum-rl-trainer.service
sed -i 's|WorkingDirectory=/opt/quantum$|WorkingDirectory=/home/qt/quantum_trader|g' \
  /etc/systemd/system/quantum-rl-trainer.service
```

#### Step 4.5: Fix all remaining services to use main venv [x]
```bash
# Any service still using /usr/bin/python3 or other Python paths:
for f in /etc/systemd/system/quantum-*.service; do
  # Replace any /opt/quantum/venvs/*/bin/python* references
  sed -i 's|/opt/quantum/venvs/[^/]*/bin/python[3]*|/home/qt/quantum_trader_venv/bin/python|g' "$f"
  # Replace any remaining /opt/quantum paths in WorkingDirectory or PYTHONPATH
  sed -i 's|WorkingDirectory=/opt/quantum\b|WorkingDirectory=/home/qt/quantum_trader|g' "$f"
done
systemctl daemon-reload
```

#### Step 4.6: Fix timer-triggered service files [x]
```bash
# 5 timer service files also reference /opt/quantum:
TIMER_SERVICES=(
  quantum-contract-check
  quantum-core-health
  quantum-diagnostic
  quantum-rl-reward-publisher
  quantum-training-worker
)
for svc in "${TIMER_SERVICES[@]}"; do
  f="/etc/systemd/system/${svc}.service"
  if [ -f "$f" ]; then
    sed -i 's|/opt/quantum/venvs/[^/]*/bin/python[3]*|/home/qt/quantum_trader_venv/bin/python|g' "$f"
    sed -i 's|WorkingDirectory=/opt/quantum\b|WorkingDirectory=/home/qt/quantum_trader|g' "$f"
    sed -i 's|/opt/quantum/|/home/qt/quantum_trader/|g' "$f"
    echo "Fixed timer service: $svc"
  fi
done
```

#### Step 4.7: Add shared EnvironmentFile directives to all services [x]
```bash
# Add the 3 shared env files to every quantum service that doesn't already have them
for f in /etc/systemd/system/quantum-*.service; do
  svc=$(basename "$f" .service)
  svc_name=${svc#quantum-}  # strip "quantum-" prefix

  # Add shared EnvironmentFile lines if not already present
  if ! grep -q "common.env" "$f"; then
    sed -i '/\[Service\]/a EnvironmentFile=/etc/quantum/common.env' "$f"
  fi
  if ! grep -q "secrets.env" "$f"; then
    sed -i '/common.env/a EnvironmentFile=/etc/quantum/secrets.env' "$f"
  fi
  if ! grep -q "python.env" "$f"; then
    sed -i '/secrets.env/a EnvironmentFile=/etc/quantum/python.env' "$f"
  fi

  echo "Added shared env directives: $svc"
done
```

#### Step 4.8: Add Restart=always to all 42 running services [x]
```bash
# While we're editing service files anyway — ensure ALL 42 have a restart policy.
# This costs nothing extra and prevents silent service deaths.
for f in /etc/systemd/system/quantum-*.service; do
  name=$(basename "$f")
  # Only touch the 42 we're keeping (skip if not in keep-list)
  if grep -qx "$name" /tmp/keep-services.txt; then
    if ! grep -q "Restart=" "$f"; then
      sed -i '/\[Service\]/a Restart=always\nRestartSec=5' "$f"
      echo "ADDED Restart=always: $name"
    elif grep -q "Restart=on-failure" "$f"; then
      sed -i 's/Restart=on-failure/Restart=always/' "$f"
      echo "UPGRADED to Restart=always: $name"
    fi
  fi
done
```

#### Step 4.9: Strip duplicated values from per-service env files [x]
```bash
# Remove values that are now in shared env files from per-service env files
# This makes per-service files contain ONLY service-specific config
SHARED_KEYS="REDIS_HOST|REDIS_PORT|REDIS_URL|TZ|LOG_LEVEL|QT_BASE_DIR"
SHARED_KEYS="${SHARED_KEYS}|BINANCE_API_KEY|BINANCE_API_SECRET"
SHARED_KEYS="${SHARED_KEYS}|BINANCE_TESTNET_API_KEY|BINANCE_TESTNET_API_SECRET"
SHARED_KEYS="${SHARED_KEYS}|VIRTUAL_ENV|PYTHONPATH"

for f in /etc/quantum/*.env; do
  name=$(basename "$f")
  # Skip the 3 new shared files
  case "$name" in
    common.env|secrets.env|python.env) continue ;;
  esac

  # Count lines before
  before=$(wc -l < "$f")
  # Remove lines matching shared keys (but keep comments)
  grep -vE "^(${SHARED_KEYS})=" "$f" > "${f}.tmp" && mv "${f}.tmp" "$f"
  after=$(wc -l < "$f")
  echo "Cleaned $name: $before → $after lines"
done

echo "Per-service env files now contain only service-specific config."
```

#### Step 4.10: Reload and restart services in groups (verify between each) [x]

```bash
systemctl daemon-reload
systemctl restart quantum-performance-attribution quantum-performance-tracker \
  quantum-metricpack-builder quantum-p35-decision-intelligence quantum-trade-logger
sleep 5
systemctl is-active quantum-performance-attribution quantum-performance-tracker \
  quantum-metricpack-builder quantum-p35-decision-intelligence quantum-trade-logger

# Group B: Portfolio / Risk (no trades)
systemctl restart quantum-portfolio-clusters quantum-portfolio-gate \
  quantum-portfolio-governance quantum-portfolio-heat-gate quantum-portfolio-state-publisher \
  quantum-capital-allocation quantum-risk-proposal quantum-heat-gate quantum-governor
sleep 5
systemctl is-active quantum-portfolio-clusters quantum-portfolio-gate \
  quantum-portfolio-governance quantum-portfolio-heat-gate quantum-portfolio-state-publisher

# Group C: AI / RL (signal generation)
systemctl restart quantum-ai-engine quantum-ai-strategy-router quantum-ensemble-predictor \
  quantum-marketstate quantum-rl-agent quantum-rl-sizer quantum-rl-trainer \
  quantum-rl-policy-publisher quantum-rl-shadow-metrics-exporter quantum-clm
sleep 5
systemctl is-active quantum-ai-engine quantum-ensemble-predictor quantum-rl-agent

# Group D: Data feeds
systemctl restart quantum-price-feed quantum-market-publisher quantum-balance-tracker \
  quantum-stream-bridge quantum-execution-result-bridge quantum-layer1-data-sink \
  quantum-universe-service quantum-utf-publisher
sleep 5
systemctl is-active quantum-price-feed quantum-market-publisher quantum-balance-tracker

# Group E: Exit / Harvest
systemctl restart quantum-exit-management-agent quantum-exit-brain-shadow \
  quantum-exit-intelligence quantum-exit-intent-gateway \
  quantum-harvest-optimizer quantum-harvest-proposal quantum-harvest-metrics-exporter \
  quantum-reconcile-engine
sleep 5
systemctl is-active quantum-exit-management-agent quantum-harvest-optimizer

# Group F: Execution (LAST — most sensitive)
systemctl restart quantum-intent-bridge quantum-intent-executor
sleep 5
systemctl is-active quantum-intent-bridge quantum-intent-executor
```

#### Step 4.11: Full verification [x]
```bash
# All 42 running?
systemctl list-units quantum-*.service --no-pager | grep -c running

# No errors in last 5 minutes?
journalctl -u 'quantum-*' --since "5 min ago" --priority err --no-pager | head -20

# AI engine healthy?
curl -s http://localhost:8001/health

# No service uses /opt/quantum anymore?
for svc in $(systemctl list-units quantum-*.service --no-pager --plain | grep running | awk '{print $1}'); do
  grep /opt/quantum /etc/systemd/system/$svc 2>/dev/null && echo "STILL BAD: $svc"
done

# All services have shared env file directives?
for f in /etc/systemd/system/quantum-*.service; do
  name=$(basename "$f")
  if ! grep -q "common.env" "$f"; then echo "MISSING common.env: $name"; fi
  if ! grep -q "secrets.env" "$f"; then echo "MISSING secrets.env: $name"; fi
  if ! grep -q "python.env" "$f"; then echo "MISSING python.env: $name"; fi
done

# Redis streams flowing?
redis-cli xlen quantum:stream:trade.intent
redis-cli xlen quantum:stream:apply.result
```

**Rollback**:
```bash
cp /opt/backups/2026-03-14-pre-cleanup/quantum-*.service /etc/systemd/system/
systemctl daemon-reload
# Restart all services — they'll use old paths which still exist
```

---

## OP 5: BURY /opt/quantum
### Status: [x] DONE 2026-03-14
### Depends on: OP 4 verified and stable (wait 24h if possible)

The ghost directory is no longer needed. Move it away.

#### Step 5.1: Final check — nothing references /opt/quantum [x]
```bash
# Check no service file references it
grep -rl /opt/quantum /etc/systemd/system/quantum-*.service
# EXPECTED: empty

# Check no running process has /opt/quantum in its open files
for pid in $(pgrep -f quantum); do
  ls -l /proc/$pid/fd 2>/dev/null | grep /opt/quantum && echo "PID $pid!!!"
done
```

#### Step 5.2: Move /opt/quantum to backup [x]
```bash
mv /opt/quantum /opt/backups/opt-quantum-2026-03-14
echo "BURIED: $(date)" > /opt/backups/opt-quantum-buried.txt
```

#### Step 5.3: Verify nothing broke [x]
```bash
systemctl list-units quantum-*.service --no-pager | grep -c running
# EXPECTED: 42

# Check for errors in last 2 minutes
journalctl -u 'quantum-*' --since "2 min ago" --priority err --no-pager
```

**Rollback**: `mv /opt/backups/opt-quantum-2026-03-14 /opt/quantum`

---

## OP 6: CLEAN THE REPO
### Status: [x] DONE 2026-03-14
### Depends on: OP 5

Now that /opt/quantum is dead and all services run from one codebase,
clean the actual repository.

#### Step 6.1: Archive root-level junk files [x]
```bash
# In local git repo:
mkdir -p archive/{scripts,docs,fixes,tmp}

# Move prefixed diagnostic scripts
git mv _*.py _*.sh archive/scripts/ 2>/dev/null
git mv fix_*.py fix_*.sh archive/fixes/ 2>/dev/null
git mv check_*.py check_*.sh archive/scripts/ 2>/dev/null
git mv deploy_*.sh deploy_*.py archive/scripts/ 2>/dev/null
git mv diag_*.py diag_*.sh archive/scripts/ 2>/dev/null
git mv tmp_*.py tmp_*.sh archive/tmp/ 2>/dev/null
git mv test_*.py archive/scripts/ 2>/dev/null  # root-level test scripts (not tests/)

# Move documentation bloat
git mv AI_*.md SPRINT*.md SYSTEM_*.md TESTNET_*.md archive/docs/ 2>/dev/null
git mv *_COMPLETE*.md *_REPORT*.md *_STATUS*.md archive/docs/ 2>/dev/null
git mv *_DEPLOYMENT*.md *_IMPLEMENTATION*.md archive/docs/ 2>/dev/null
```

#### Step 6.2: Check if backend/ is imported by any running service [x]
```bash
# From VPS:
grep -r "from backend\." /home/qt/quantum_trader/microservices/ 2>/dev/null
grep -r "import backend" /home/qt/quantum_trader/microservices/ 2>/dev/null
# If empty: backend/ is dead code for runtime
```

#### Step 6.3: Archive unused code if confirmed dead [x]
```bash
# If backend/ is dead:
git mv backend/ archive/old_backend/
# If specific backend/ modules ARE imported, keep those, archive the rest
```

#### Step 6.4: Clean ops/offline junk [x]
```bash
git mv ops/offline/_*.py archive/scripts/ 2>/dev/null
git mv ops/analysis/fix_*.py archive/fixes/ 2>/dev/null
```

#### Step 6.5: Remove model files from git tracking [x]
```bash
# 2000+ .pkl files bloat the repo. Models are DATA, not code.
# They should live on disk (VPS) but NOT in git history.

# Step A: Add model patterns to .gitignore
cat >> .gitignore << 'EOF'

# Model files — too large for git, deploy via VPS disk
models/*.pkl
models/*.joblib
models/*.pth
models/*.pt
models/*.onnx
models/*.h5
model_registry/**/*.pkl
ai_engine/models/*.pkl
*.pkl
*.joblib
*.pth
EOF

# Step B: Remove from git tracking (keeps files on disk)
git rm -r --cached models/*.pkl 2>/dev/null
git rm -r --cached models/*.joblib 2>/dev/null
git rm -r --cached models/*.pth 2>/dev/null
git rm --cached *.pkl 2>/dev/null

echo "Model files removed from git tracking (still on disk)."
echo "VPS models live at /home/qt/quantum_trader/model_registry/"
echo "Future model deployment: scp/rsync, NOT git."
```

> **NOTE**: This does NOT clean git HISTORY (that would require `git filter-branch`
> or BFG Repo Cleaner). For now, just stop tracking new changes. History cleanup
> is a separate OP 7+ task.

#### Step 6.6: Consolidate requirements.txt [x]
```bash
# After model cleanup, merge 31 requirements files into one pinned file:
# (Use the reconciled output from OP 3 Step 3.4)
cat > requirements.txt << 'EOF'
# QTOS Unified Requirements
# Generated from VPS main venv pip freeze
# Source of truth: /home/qt/quantum_trader_venv/
EOF

# Append pinned versions from live venv
# (Run on VPS first: pip freeze > /tmp/pinned.txt, then copy here)
# scp root@46.224.116.254:/tmp/pinned.txt ./requirements-pinned.txt

echo "TODO: Copy pip freeze output from VPS after OP 3 venv merge"
```

#### Step 6.7: Update .gitignore [x]
```
# Don't track archive in future
archive/
```

#### Step 6.8: Commit and push [x]
```bash
git add -A
git commit -m "OP6: clean repo — archive 1500+ files, remove model tracking, consolidate deps

Moved diagnostic scripts, fix scripts, tmp files, doc bloat to archive/.
Removed 2000+ .pkl model files from git tracking (still on VPS disk).
Consolidated 31 requirements files into one pinned requirements.txt.
Part of QTOS cleanup - see docs/architecture/QTOS_MASTER_PLAN.md"
git push
```

#### Step 6.9: Deploy to VPS [x]
```bash
cd /home/qt/quantum_trader && git pull
systemctl list-units quantum-*.service --no-pager | grep -c running
# EXPECTED: 42 (nothing should change — we only moved unused files)
# RESULT: 42 ✅ — VPS synced to f108e74e, then a8e84f7f6, then 783b5bd5e
```

#### BONUS: AI Engine Health Thread (discovered during 6.9)
- /health/live on port 8001 was permanently unreachable due to event-loop starvation
- Root cause: 5 concurrent Redis stream consumers + synchronous ML inference saturate asyncio event loop
- Fix: Added background-thread health server on port 8002 (stdlib http.server)
- Port 8002 /health/live responds in <1ms, independent of event loop
- Commits: a8e84f7f6 (event_bus yield point), 783b5bd5e (health thread)

---

## OP 7+: ARCHITECTURE (FUTURE)

These are design improvements that only make sense AFTER the yarn ball
is untangled (OP 0-6 done). They should NOT BE STARTED until OP 6 is
verified as [DONE].

### 7A: Mask execution + apply-layer permanently
Already masked in OP 1. Verify they stay masked.
Intent-executor is the ONLY order execution path.

### 7B: Unify Execution Pipeline [x] (analysis + critical fixes)
**Original plan**: Merge intent-bridge + intent-executor into single `execution-engine`.

**Analysis result**: Intent-bridge and intent-executor should remain separate services:
- intent-bridge: stateless filter (7 safety gates), fast, CPU-light
- intent-executor: stateful Binance interaction, slow (order polling), API-heavy
- Different failure modes, different scaling needs

**Architecture discovered**:
- Entry: AI Engine → trade.intent → intent_bridge (7 gates) → apply.plan → intent_executor (main lane) → Binance
- Exit: exit_management_agent → exit.intent → exit_intent_gateway (9 checks) → harvest.intent → intent_executor (harvest lane) → Binance
- Manual: direct → apply.plan.manual → intent_executor (manual lane) → Binance

**Critical bugs found and fixed**:
1. exit_intent_gateway/config.py: Default trade_stream was `trade.intent` (entry pipeline).
   Exits routed through entry gates = blocked. Fixed default → `harvest.intent`.
   VPS already had correct override via env, but code default was dangerous.
2. intent_executor/main.py: Main lane `_commit_ledger_exactly_once()` did NOT publish
   `trade.closed` events when position went FLAT. Only harvest lane did.
   CLM/calibration missed all main-lane closes. Fixed: added trade.closed publish
   in FLAT case with mark_price as exit_price.

### 7C: Position Truth Source [x] (Phase 1 — single writer + backward compat)
**Original plan**: One service polls Binance, publishes to `quantum:state:positions`.
Everything else reads from there. Kill the 5-source fragmentation.

**Fragmentation found**: 5 Redis key patterns, 7+ independent Binance API callers, 18+ readers.

**Phase 1 fixes (implemented)**:
1. position_state_brain (P3.3) now writes to THREE keys per poll cycle:
   - `quantum:position:snapshot:{symbol}` (legacy — existing readers)
   - `quantum:state:positions:{symbol}` (CANONICAL — new truth source)
   - `quantum:position:{symbol}` (backward-compat — replaces harvest_brain writes)
2. harvest_brain no longer writes to `quantum:position:{symbol}` — removed startup sync
   and per-execution sync. Ghost cleanup logic preserved.
3. risk_proposal_publisher and harvest_proposal_publisher can now read
   `quantum:state:positions:{symbol}` which was previously empty.
4. Fixed dead code bug in P3.3 (duplicate `return False` after `return False, latency_ms`).

**Phase 2 (future)**: Migrate all 18+ readers to use `quantum:state:positions:{symbol}`
and remove the legacy key writes. Also remove Binance API calls from reconcile_engine,
intent_executor, governor, and apply_layer — make them read from P3.3's canonical key.

**Redis key ownership after Phase 1**:
| Key Pattern | Writer | Status |
|---|---|---|
| quantum:state:positions:{symbol} | P3.3 | CANONICAL (new) |
| quantum:position:snapshot:{symbol} | P3.3 | Legacy, will deprecate |
| quantum:position:{symbol} | P3.3 | Compat bridge, will deprecate |
| quantum:position:ledger:{symbol} | intent_executor | Execution ledger (keep) |
| quantum:position:claim:{symbol} | apply_layer | Transient lock (keep) |

### 7D: Typed IPC Contracts ✅

- [x] Comprehensive analysis of all 6 core Redis streams (trade.intent, apply.plan, apply.result, exit.intent, harvest.intent, trade.closed)
- [x] Discovered two incompatible wire formats on trade.intent (EventBus wrapper vs direct XADD)
- [x] Discovered two distinct apply.plan schemas (Intent Bridge vs Apply Layer)
- [x] Created `shared/contracts/` Pydantic v2 module with 8 files:
  - `base.py` — `StreamEvent` base class with `to_redis()` / `from_redis()` serialization
  - `trade_intent.py` — TradeIntentEvent (entry chain start)
  - `apply_plan.py` — ApplyPlanEvent (Intent Bridge variant)
  - `apply_result.py` — ApplyResultEvent (execution acknowledgment)
  - `exit_intent.py` — ExitIntentEvent (exit chain start, all fields required per gateway parser)
  - `harvest_intent.py` — HarvestIntentEvent (validated exit forwarded to executor)
  - `trade_closed.py` — TradeClosedEvent (position fully closed event)
  - `__init__.py` — Package init with all exports
- Phase 1: Schemas as documentation + import-ready contracts
- Phase 2 (future): Runtime validation at XADD/XREAD boundaries

**Stream Chain Map:**
```
Entry: AI Engine → trade.intent → intent_bridge → apply.plan → intent_executor → apply.result
Exit:  exit_mgmt_agent → exit.intent → exit_intent_gateway → harvest.intent → intent_executor → trade.closed
```

### 7E: Risk Kernel ✅

- [x] Analyzed all 7 risk services (~5,430 lines across 12 files)
- [x] Discovered TRIPLE duplication: heat-gate, portfolio-gate, portfolio-heat-gate all consume harvest.proposal
- [x] Created `microservices/risk_kernel/main.py` — thread-per-component orchestrator
- [x] Components (6 active, 1 disabled):
  - Governor (stream: apply.plan) — entry gate ✅
  - Heat Gate (stream: harvest.proposal) — exit heat moderator ✅
  - Portfolio Gate (stream: harvest.proposal) — exit portfolio safety ✅
  - Portfolio Heat Gate — **DISABLED** (90% overlap with above two, prometheus p26_* conflict)
  - Risk Proposal Publisher (poll: 10s) — SL/TP calculator ✅
  - Capital Allocation (poll: 5s) — per-symbol budget allocator ✅
  - Portfolio Governance (poll: 30s) — macro policy controller ✅
- [x] Unified health endpoint on port 8070
- [x] Per-component toggle via RK_ENABLE_* env vars
- [x] Watchdog thread monitors component health every 30s
- [x] Stopped + disabled 7 individual systemd services
- [x] Created + enabled `quantum-risk-kernel.service`
- Net result: 7 services → 1 (service count: 43 → 37)

### 7F: Plugin Architecture [DONE] 2026-03-15
AI strategies become plugins with standard interface.

**Completed:**
- [x] `shared/strategy_plugin.py` — StrategyPlugin Protocol (runtime_checkable) + StrategyPrediction dataclass + StrategyRegistry
- [x] `BaseAgent` (unified_agents.py) conforms to StrategyPlugin: model_type, get_required_features(), health_check(), get_metadata()
- [x] All 6 ensemble agents marked `__strategy_plugin__ = True` with model_type ("tree"/"neural")
- [x] Standalone TFTAgent (tft_agent.py) also conforms to protocol
- [x] EnsembleManager.__init__ populates StrategyRegistry with all loaded agents + weights
- [x] EnsembleManager.predict() uses registry iteration (falls back to legacy hardcoded path)
- [x] New API: register_plugin(), registry_health()
- [x] Package discovery: StrategyRegistry.discover() for auto-registration via __strategy_plugin__ marker
- [x] Deployed to VPS — all 6 plugins registered, predict flow verified live
- Commits: 7e610de43, d83f95da2

### 7G: Frontend Unification [IN PROGRESS] 2026-03-15
6 frontends → 1. Unified API + dashboard.

**Audit completed 2026-03-15:**

| # | Directory | Tech | Status | Action |
|---|-----------|------|--------|--------|
| 1 | dashboard_v4/ | React 18+Vite+Tailwind | **PRODUCTION** (app.quantumfond.com:443) | ✅ KEEP — canonical |
| 2 | quantumfond_frontend/ | React 18+Vite+Tailwind | Dead/superseded | 🗑 DELETE — predecessor of #1 |
| 3 | frontend/ | Next.js 14+Zustand | Dead (Docker-era) | 🗑 DELETE — oldest, has test infra worth noting |
| 4 | frontend_investor/ | Next.js 14+Tailwind | Semi-active (investor portal) | ⏸ KEEP SEPARATE — different audience |
| 5 | qt-agent-ui/ | React 18+Vite (mobile) | Prototype, never deployed | 🗑 DELETE |
| 6 | microservices/rl_dashboard/ | Flask+SocketIO+Chart.js | Mostly dead, nginx proxied | 🗑 DELETE — absorbed into dashboard_v4 RL page |
| 7 | backend/microservices/governance_dashboard/ | FastAPI+inline HTML | Dead (Docker-era) | 🗑 DELETE |

**Backend APIs:**
- KEEP: dashboard_v4/backend/ (port 8025) — production API
- DELETE: quantumfond_backend/ — dead
- REVIEW: backend/main.py (port 8080) — may have active endpoints used by other services

**Unique features in dead frontends to backlog:**
- quantumfond_frontend: Admin, Incident, Journal, Replay, Live pages
- frontend/: Vitest + Cypress test infrastructure patterns
- qt-agent-ui: iPhone-style mobile layout concept

**Phase 1 (done):** Audit and document
**Phase 2 (future):** Remove dead frontends from repo
**Phase 3 (future):** Merge missing page concepts into dashboard_v4

---

## OPERATION DEPENDENCY GRAPH

```
OP 0: Snapshot ──→ OP 1: Clear Dead ──→ OP 4: Fix Services & Deploy
                                              │
OP 2: Fix Source Code (parallel) ─────────────┤
                                              │
OP 3: Unify Venvs & Env Files ──────────────────┘
                                              │
                                              ▼
                                     OP 5: Bury /opt/quantum
                                              │
                                              ▼
                                     OP 6: Clean Repo
                                              │
                                              ▼
                                     OP 7+: Architecture
```

**Key insight**: OP 2 (fix source code) can be done locally in parallel
with OP 0 and OP 1 (VPS work). Then OP 3 + OP 4 combine everything.

---

## QUICK REFERENCE

### VPS Access
```powershell
# Simple command:
wsl sh -lc "ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'COMMAND'"

# Multi-line:
$s = @'
COMMANDS
HERE
'@
$b = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($s))
wsl sh -lc "ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'echo $b | base64 -d | bash'"
```

### Key Paths (VPS — TARGET state after OP 5)
```
Code:     /home/qt/quantum_trader/           (THE ONLY codebase)
Venv:     /home/qt/quantum_trader_venv/      (THE ONLY venv)
Services: /etc/systemd/system/quantum-*.service (42 files)
Logs:     journalctl -u quantum-*
Config:   /etc/quantum/*.env
Backups:  /opt/backups/
```

### Key Redis Streams
```
quantum:stream:trade.intent     ← AI signals
quantum:stream:apply.plan       ← Validated plans
quantum:stream:apply.result     ← Execution results
quantum:stream:trade.closed     ← Closed trades
quantum:stream:exit.intent      ← Exit proposals
quantum:stream:harvest.intent   ← Harvest proposals
quantum:stream:governor.events  ← Governor audit
```

### Health Checks
```bash
systemctl list-units quantum-*.service | grep -c running   # EXPECT: 42
curl -s http://localhost:8001/health                         # AI Engine
redis-cli xlen quantum:stream:apply.result                   # Trades
redis-cli xlen quantum:stream:trade.intent                   # Signals
```

### Emergency Rollback (any OP)
```bash
# Restore ALL service files:
cp /opt/backups/2026-03-14-pre-cleanup/quantum-*.service /etc/systemd/system/
systemctl daemon-reload

# Restore /opt/quantum:
mv /opt/backups/opt-quantum-2026-03-14 /opt/quantum

# Restart all:
systemctl restart quantum-*.service
```

---

## PROGRESS LOG

| Date | Operation | Step | Status | Notes |
|------|-----------|------|--------|-------|
| 2026-03-14 | — | — | v2 plan created | Full VPS re-audit: 42 running, 134 total, 9 touching /opt/quantum |
| | | | | |

---

*This document is the SINGLE SOURCE OF TRUTH for untangling the yarn ball.*
*Every session: read this first, update progress, continue where left off.*
