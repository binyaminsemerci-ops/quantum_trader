# QTOS MASTER PLAN — Single Source of Truth
## Version: 1.0 | Created: 2026-03-14 | Last Updated: 2026-03-14

---

> **RULE #1**: This document IS the plan. Every session starts by reading this file.
> Never search from scratch. Never redo mapping. Open this, read status, continue.

> **RULE #2**: Work ONE operation at a time. Mark it [DONE] before starting next.
> No jumping. No parallel experiments. Sequential, verified, documented.

> **RULE #3**: Every change gets verified BEFORE marking done.
> Verify = running service status + Redis check + log check on VPS.

---

## TABLE OF CONTENTS

1. [Current System State](#1-current-system-state)
2. [Operation 1: Kill the Two-Headed Monster](#operation-1-kill-the-two-headed-monster)
3. [Operation 2: Remove the Zombie Army](#operation-2-remove-the-zombie-army)
4. [Operation 3: One Python, One Path](#operation-3-one-python-one-path)
5. [Operation 4: Clean the Coral Reef](#operation-4-clean-the-coral-reef)
6. [Operation 5: Unify Execution Pipeline](#operation-5-unify-execution-pipeline)
7. [Operation 6: Position Truth Source](#operation-6-position-truth-source)
8. [Operation 7: Typed IPC Contracts](#operation-7-typed-ipc-contracts)
9. [Operation 8: Risk Kernel](#operation-8-risk-kernel)
10. [Operation 9: Plugin Architecture](#operation-9-plugin-architecture)
11. [Operation 10: Shell Unification](#operation-10-shell-unification)
12. [Quick Reference](#quick-reference)

---

## 1. CURRENT SYSTEM STATE

### As of 2026-03-14

| Metric | Value | Target |
|--------|-------|--------|
| Running services | 43 | 23 |
| Systemd service files | 134 | 23 |
| Redis streams | ~70 | ~25 |
| Python interpreters | 3 | 1 |
| Virtual environments | 10 | 1 |
| Code trees | 3 (backend/, microservices/, ai_engine/) | 1 unified |
| Root-level files | ~1500 | <10 |
| Frontend projects | 6 | 1 |
| Execution paths for entries | 2 (DANGER) | 1 |
| Close paths | 2 | 1 |
| Sources of position truth | 5+ | 1 |

### Services currently RUNNING (43)

```
RING 0 (target: kernel):
  governor, heat-gate, portfolio-gate, risk-proposal,
  portfolio-heat-gate, portfolio-governance, capital-allocation

RING 1 (target: drivers):
  price-feed, market-publisher, balance-tracker,
  stream-bridge, execution-result-bridge

RING 2 (target: system services):
  intent-bridge, intent-executor, reconcile-engine,
  exit-management-agent, exit-brain-shadow, exit-intelligence,
  exit-intent-gateway, harvest-optimizer, harvest-proposal,
  harvest-metrics-exporter, portfolio-clusters,
  portfolio-state-publisher, universe-service, utf-publisher,
  performance-attribution, performance-tracker,
  metricpack-builder, p35-decision-intelligence, trade-logger,
  layer1-data-sink, layer1-historical-backfill (exited)

RING 3 (target: user space):
  ai-engine, ai-strategy-router, ensemble-predictor,
  rl-agent, rl-sizer, rl-trainer, rl-policy-publisher,
  rl-shadow-metrics-exporter, clm, marketstate

RING 4 (target: shell):
  (none running currently — backend/dashboard not active)
```

### Services currently DEAD but have .service files (19 loaded)

```
bsc, core-health, diagnostic, ess-watch, execution,
exit-owner-watch, health-gate, layer4-portfolio-optimizer,
ledger-sync, offline-evaluator, p28a3-latency-proof,
portfolio-intelligence, portfolio-risk-governor, redis (not-found),
risk-brain, rl-monitor (not-found), rl-shadow-scorecard,
strategy-brain, stream-recover, training-worker
```

### Service files that exist but are NOT EVEN LOADED (~72 more)

```
(The difference between 134 total files and 62 loaded units)
These are service files in /etc/systemd/system/ that systemd doesn't even track.
```

---

## OPERATION 1: KILL THE TWO-HEADED MONSTER
### Priority: CRITICAL | Risk: HIGH | Status: [ ] NOT STARTED

**Problem**: Two services can execute trades against Binance:
1. `intent-executor` (ACTIVE, running) — the correct one
2. `execution` (INACTIVE, dead) — the dangerous legacy one

Even though `execution` is currently dead, it can be accidentally started.
Its code subscribes to `trade.intent` via EventBus, meaning if started it
would execute trades IN PARALLEL with intent-executor.

**The Plan**:

#### Step 1.1: Verify execution service is dead [ ]
```bash
systemctl status quantum-execution.service
# EXPECTED: inactive (dead), disabled
```

#### Step 1.2: Disable and mask the execution service [ ]
```bash
# disable prevents auto-start, mask prevents ANY start (even manual)
systemctl disable quantum-execution.service
systemctl mask quantum-execution.service
# Verify:
systemctl status quantum-execution.service
# EXPECTED: masked
```

#### Step 1.3: Rename service file to .KILLED [ ]
```bash
mv /etc/systemd/system/quantum-execution.service \
   /etc/systemd/system/quantum-execution.service.KILLED
systemctl daemon-reload
```

#### Step 1.4: Also disable apply-layer (entry disabled, but still has close code) [ ]
```bash
# apply-layer is already dead and disabled, but let's mask it too
# Its close functionality will be absorbed by execution-engine later
systemctl mask quantum-apply-layer.service
mv /etc/systemd/system/quantum-apply-layer.service \
   /etc/systemd/system/quantum-apply-layer.service.KILLED
systemctl daemon-reload
```

#### Step 1.5: Verify ONLY intent-executor handles orders [ ]
```bash
# Check only one process connects to Binance for orders
grep -r "create_order\|place_order\|futures_create_order" \
  /home/qt/quantum_trader/microservices/intent_executor/ \
  /home/qt/quantum_trader/microservices/intent_bridge/
# Check no other running process has Binance order capability
for pid in $(pgrep -f "quantum_trader"); do
  cmdline=$(cat /proc/$pid/cmdline | tr '\0' ' ')
  echo "$pid: $cmdline"
done | grep -v intent_executor | grep -v intent_bridge
```

#### Step 1.6: Verify apply-layer close path is handled [ ]
```bash
# Harvest close proposals go to: quantum:stream:apply.plan
# Intent-executor reads apply.plan → so it picks up close proposals
# Also reads harvest.intent → so autonomous exits are handled
# VERIFY: intent-executor consumer groups
redis-cli xinfo groups quantum:stream:apply.plan
redis-cli xinfo groups quantum:stream:harvest.intent
# EXPECTED: intent_executor group exists in both
```

#### Rollback plan:
```bash
# If something breaks:
mv /etc/systemd/system/quantum-execution.service.KILLED \
   /etc/systemd/system/quantum-execution.service
systemctl unmask quantum-execution.service
systemctl daemon-reload
# DO NOT START IT — investigate first
```

#### Verification checklist:
- [ ] execution.service is masked and renamed .KILLED
- [ ] apply-layer.service is masked and renamed .KILLED
- [ ] intent-executor is running and handling both entries and closes
- [ ] No duplicate trade execution possible
- [ ] Trades still execute normally (check last trade in Redis)

---

## OPERATION 2: REMOVE THE ZOMBIE ARMY
### Priority: HIGH | Risk: LOW | Status: [ ] NOT STARTED

**Problem**: 134 service files exist. Only 43 are running. 91 are zombies that:
- Clutter systemd
- Can be accidentally started
- Create confusion about what's real

**The Plan**:

#### Step 2.1: Create backup of all service files [ ]
```bash
mkdir -p /opt/backups/systemd-2026-03-14
cp /etc/systemd/system/quantum-*.service /opt/backups/systemd-2026-03-14/
ls /opt/backups/systemd-2026-03-14/ | wc -l
# EXPECTED: 134
```

#### Step 2.2: List the 43 services that MUST stay [ ]
```bash
# These are the running services as of 2026-03-14:
cat > /opt/backups/keep-services.txt << 'EOF'
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
quantum-layer1-historical-backfill.service
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
EOF
```

#### Step 2.3: Move ALL non-kept services to graveyard [ ]
```bash
mkdir -p /opt/backups/systemd-graveyard
for f in /etc/systemd/system/quantum-*.service; do
  basename=$(basename "$f")
  if ! grep -q "$basename" /opt/backups/keep-services.txt; then
    echo "REMOVING: $basename"
    systemctl stop "$basename" 2>/dev/null
    systemctl disable "$basename" 2>/dev/null
    mv "$f" /opt/backups/systemd-graveyard/
  fi
done
systemctl daemon-reload
```

#### Step 2.4: Also move drop-in directories for removed services [ ]
```bash
for d in /etc/systemd/system/quantum-*.service.d; do
  service_name=$(basename "$d" .d)
  if ! grep -q "$service_name" /opt/backups/keep-services.txt; then
    echo "REMOVING DROP-IN: $d"
    mv "$d" /opt/backups/systemd-graveyard/
  fi
done
systemctl daemon-reload
```

#### Step 2.5: Verify [ ]
```bash
ls /etc/systemd/system/quantum-*.service | wc -l
# EXPECTED: 43
systemctl list-units quantum-*.service --all | grep -c "running"
# EXPECTED: 42 (layer1-historical-backfill is "exited")
```

#### Rollback plan:
```bash
cp /opt/backups/systemd-graveyard/quantum-*.service /etc/systemd/system/
systemctl daemon-reload
```

---

## OPERATION 3: ONE PYTHON, ONE PATH
### Priority: HIGH | Risk: MEDIUM | Status: [ ] NOT STARTED

**Problem**: 3 different Python interpreters running across 43 services:
- `/home/qt/quantum_trader_venv/bin/python` (main — 12 services)
- `/opt/quantum/venvs/ai-engine/bin/python3` (secondary — 7 services)
- `/usr/bin/python3` (system — 24 services)

**Target**: ALL services use `/home/qt/quantum_trader_venv/bin/python`

**The Plan**:

#### Step 3.1: Catalog packages in secondary venv [ ]
```bash
/opt/quantum/venvs/ai-engine/bin/pip freeze > /tmp/secondary-venv-packages.txt
cat /tmp/secondary-venv-packages.txt
```

#### Step 3.2: Catalog packages in main venv [ ]
```bash
/home/qt/quantum_trader_venv/bin/pip freeze > /tmp/main-venv-packages.txt
```

#### Step 3.3: Find missing packages [ ]
```bash
# Compare: what does secondary have that main doesn't?
comm -23 <(cut -d= -f1 /tmp/secondary-venv-packages.txt | sort) \
         <(cut -d= -f1 /tmp/main-venv-packages.txt | sort)
```

#### Step 3.4: Install missing packages into main venv [ ]
```bash
# Install whatever is missing (command TBD based on Step 3.3 output)
/home/qt/quantum_trader_venv/bin/pip install <missing-packages>
```

#### Step 3.5: Update the 7 services using secondary venv [ ]
Services to update:
```
quantum-ai-strategy-router.service
quantum-ensemble-predictor.service
quantum-harvest-metrics-exporter.service
quantum-harvest-proposal.service
quantum-marketstate.service
quantum-portfolio-governance.service
quantum-risk-proposal.service
```

For each: change ExecStart from `/opt/quantum/venvs/ai-engine/bin/python3` to
`/home/qt/quantum_trader_venv/bin/python` and update PATH environment.

#### Step 3.6: Update the ~24 services using system Python [ ]
Services to update:
```
quantum-balance-tracker.service
quantum-capital-allocation.service
quantum-governor.service
quantum-heat-gate.service
quantum-intent-bridge.service
quantum-intent-executor.service
quantum-p35-decision-intelligence.service
quantum-performance-attribution.service
quantum-performance-tracker.service
quantum-portfolio-clusters.service
quantum-portfolio-gate.service
quantum-portfolio-heat-gate.service
quantum-portfolio-state-publisher.service
quantum-reconcile-engine.service
quantum-rl-policy-publisher.service
quantum-rl-shadow-metrics-exporter.service
quantum-trade-logger.service
quantum-universe-service.service
quantum-utf-publisher.service
```

For each: change ExecStart from `/usr/bin/python3` to
`/home/qt/quantum_trader_venv/bin/python` and update PATH.

#### Step 3.7: Restart services in groups (5 at a time) [ ]
```bash
# Restart non-critical first, verify, then critical
# Group 1 (monitoring): performance-*, metricpack-*, p35-*
# Group 2 (portfolio): portfolio-*, capital-*
# Group 3 (risk): governor, heat-gate, risk-proposal
# Group 4 (execution): intent-bridge, intent-executor
# Group 5 (AI): ai-engine, ensemble-predictor, marketstate
# Between each group: verify no errors in journal
```

#### Step 3.8: Fix market-publisher WorkingDirectory [ ]
```bash
# market-publisher is the ONLY service still using /opt/quantum as WorkingDir
# Change WorkingDirectory=/opt/quantum/ops/market → /home/qt/quantum_trader/ops/market
# Then restart
```

#### Step 3.9: Verify [ ]
```bash
# All services should use same Python
for svc in $(systemctl list-units quantum-*.service --no-pager --plain | grep running | awk '{print $1}'); do
  pid=$(systemctl show $svc -p MainPID --value)
  exe=$(readlink /proc/$pid/exe 2>/dev/null)
  echo "$svc → $exe"
done | sort -t→ -k2
# EXPECTED: ALL show /home/qt/quantum_trader_venv/bin/python*
```

#### Step 3.10: Remove secondary venv and unused venvs [ ]
```bash
# Only after ALL services verified working on main venv
mv /opt/quantum/venvs /opt/backups/venvs-2026-03-14
# Volume venvs:
mv /mnt/HC_Volume_104287969/quantum-venvs /opt/backups/volume-venvs-2026-03-14
```

---

## OPERATION 4: CLEAN THE CORAL REEF
### Priority: MEDIUM | Risk: LOW | Status: [ ] NOT STARTED

**Problem**: The codebase has grown like a coral reef. 3 code trees overlap:
- `backend/` — original monolith (70+ services, largely unused by runtime)
- `microservices/` — where running services live (80+ dirs)
- `ai_engine/` — AI models (used by microservices/ai_engine/)

Plus ~1500 diagnostic/fix/doc files at root level.

**The Coral Reef Solution**: Don't destroy the reef. Map which parts are alive,
mark the rest as dead coral, and build new structure around the living parts.

**The Plan**:

#### Step 4.1: Create archive directory structure [ ]
```bash
mkdir -p archive/{scripts,docs,fixes,old_services,old_backend}
```

#### Step 4.2: Move root-level scripts to archive [ ]
```bash
# Diagnostic scripts (prefix _)
mv _*.py _*.sh archive/scripts/ 2>/dev/null

# Fix scripts
mv fix_*.py fix_*.sh archive/fixes/ 2>/dev/null

# Check scripts
mv check_*.py check_*.sh archive/scripts/ 2>/dev/null

# Deploy scripts
mv deploy_*.sh deploy_*.py archive/scripts/ 2>/dev/null

# Diag scripts
mv diag_*.py diag_*.sh archive/scripts/ 2>/dev/null
```

#### Step 4.3: Move AI documentation to archive [ ]
```bash
mv AI_*.md archive/docs/ 2>/dev/null
mv *_COMPLETE*.md *_REPORT*.md *_STATUS*.md archive/docs/ 2>/dev/null
mv *_IMPLEMENTATION*.md *_DEPLOYED*.md archive/docs/ 2>/dev/null
mv FORENSIC_*.md PHASE_*.md ACTION_PLAN_*.md archive/docs/ 2>/dev/null
```

#### Step 4.4: Identify which backend/ code is actually imported by runtime [ ]
```bash
# Check if ANY running service imports from backend/
grep -r "from backend\." /home/qt/quantum_trader/microservices/ 2>/dev/null | head -20
grep -r "import backend" /home/qt/quantum_trader/microservices/ 2>/dev/null | head -20
# If nothing: backend/ is dead code for runtime
```

#### Step 4.5: Move unused backend/ code to archive [ ]
```bash
# If Step 4.4 confirms backend/ is not imported:
mv backend/ archive/old_backend/
# Keep a symlink if needed for any edge case:
# ln -s archive/old_backend backend
```

**CAUTION**: Do NOT move these yet:
- `microservices/` — all running services live here
- `ai_engine/` — imported by microservices/ai_engine/
- `lib/` — may be imported by running services
- `ops/` — market-publisher runs from ops/
- `config/`, `configs/` — active configuration

#### Step 4.6: Verify runtime still works after cleanup [ ]
```bash
# Check all 43 services still running
systemctl list-units quantum-*.service | grep -c running
# Check AI engine health
curl -s http://localhost:8001/health
# Check recent trade activity
redis-cli xlen quantum:stream:apply.result
```

#### Step 4.7: Update .gitignore [ ]
```
# Add to .gitignore:
archive/
```

---

## OPERATION 5: UNIFY EXECUTION PIPELINE
### Priority: HIGH | Risk: HIGH | Status: [ ] NOT STARTED
### Depends on: Operation 1 (two heads killed)

**Problem**: Three services handle order execution:
- `intent-bridge` — transforms trade.intent → apply.plan
- `intent-executor` — reads apply.plan, checks permits, executes on Binance
- `apply-layer` — handles harvest close proposals → Binance (DEAD but code exists)

Also harvested closes go through a different path than entries.

**Target**: ONE `execution-engine` that handles ALL order lifecycle.

**The Plan**:

#### Step 5.1: Map exact data flow for entries and closes [ ]
```
ENTRY: AI Engine → trade.intent → Intent Bridge → apply.plan
       → Governor permit + P26 permit + P33 permit
       → Intent Executor → Binance → apply.result

CLOSE: Harvest Proposal → quantum:harvest:proposal:{symbol}
       → (was: Apply Layer polls and executes directly)
       → (now: needs to go through execution-engine)

EXIT:  Exit Management Agent → exit.intent
       → Intent Executor (reads harvest.intent) → Binance
```

#### Step 5.2: Create execution-engine that absorbs all three [ ]
```
NEW: microservices/execution_engine/
├── main.py           ← Single entry point
├── entry_handler.py  ← Handles trade.intent → validate → permit check → execute
├── close_handler.py  ← Handles harvest/exit → validate → execute (reduceOnly)
├── binance_client.py ← Single Binance connection (shared)
├── permit_gate.py    ← Atomic 3-permit check (from intent-executor)
└── config.py         ← Configuration
```

#### Step 5.3: Migrate entry path [ ]
- Move intent-bridge transform logic into entry_handler.py
- Move intent-executor permit check + execution into entry_handler.py
- Single stream read: quantum:stream:trade.intent

#### Step 5.4: Migrate close path [ ]
- Move apply-layer harvest polling logic into close_handler.py
- Move exit-intent handling into close_handler.py
- Single stream read: quantum:stream:exit.intent + harvest polling

#### Step 5.5: Deploy as new service, shadow mode first [ ]
```bash
# Deploy execution-engine in shadow mode (reads but doesn't execute)
# Compare its decisions with intent-executor for 24h
# If 100% match: switch over
```

#### Step 5.6: Cut over [ ]
```bash
systemctl stop quantum-intent-bridge quantum-intent-executor
systemctl start quantum-execution-engine
# Monitor for 1h
# If OK: disable old services
```

---

## OPERATION 6: POSITION TRUTH SOURCE
### Priority: HIGH | Risk: MEDIUM | Status: [ ] NOT STARTED
### Depends on: Operation 5

**Problem**: Position information exists in 5+ places:
- Binance API (the actual truth)
- `quantum:position` (Redis hash)
- `quantum:ledger:{symbol}` (per-symbol ledger)
- `quantum:portfolio` (portfolio state)
- `quantum:slots` (active trading slots)
- `quantum:reconcile:state` (reconcile engine state)

**Target**: ONE authority: Position Driver polls Binance, writes to
`quantum:state:positions`. Everything else reads from there.

#### Step 6.1: Create position-driver service [ ]
```python
# Polls Binance every 5 seconds
# Writes canonical position state to quantum:state:positions
# Publishes changes to qtos:position.truth stream
# Replaces: balance-tracker + reconcile-engine position logic
```

#### Step 6.2: Migrate consumers [ ]
- Every service that reads position data → read from quantum:state:positions
- Remove direct Binance position queries from other services

#### Step 6.3: Deprecate old position keys [ ]
- quantum:position → redirect to quantum:state:positions
- quantum:ledger → keep for accounting, but NOT for position truth
- quantum:slots → derived from quantum:state:positions

---

## OPERATION 7: TYPED IPC CONTRACTS
### Priority: MEDIUM | Risk: LOW | Status: [ ] NOT STARTED
### Depends on: Operation 5

**Problem**: Redis streams carry flat dicts with no schema validation.
Any service can publish anything. Silent failures when format changes.

**Target**: Pydantic schemas for every stream. Publish/subscribe validates.

#### Step 7.1: Create shared/schemas/ [ ]
```python
# shared/schemas/trade_intent.py
class TradeIntent(BaseModel):
    symbol: str
    action: Literal["BUY", "SELL"]
    confidence: float = Field(ge=0.0, le=1.0)
    source: str
    sizing: SizingDecision
    regime: str
    trace_id: UUID
    timestamp: datetime
```

#### Step 7.2: Create IPC Bus wrapper [ ]
```python
# shared/ipc_bus.py
class IPCBus:
    STREAM_SCHEMAS = {
        "qtos:signal.proposal": TradeIntent,
        "qtos:order.plan": OrderPlan,
        "qtos:order.result": OrderResult,
        ...
    }
    
    def publish(self, stream: str, msg: BaseModel) -> str:
        schema = self.STREAM_SCHEMAS.get(stream)
        if schema and not isinstance(msg, schema):
            raise TypeError(f"Stream {stream} expects {schema}, got {type(msg)}")
        return self.redis.xadd(stream, msg.model_dump())
```

#### Step 7.3: Migrate streams one at a time [ ]
- Start with trade.intent (highest value)
- Then apply.plan, apply.result
- Then trade.closed
- Keep old streams alive with bridge during migration

---

## OPERATION 8: RISK KERNEL
### Priority: MEDIUM | Risk: HIGH | Status: [ ] NOT STARTED
### Depends on: Operations 5, 6, 7

**Problem**: Risk control is spread across 7 services:
governor, heat-gate, portfolio-gate, risk-proposal, portfolio-heat-gate,
portfolio-governance, capital-allocation.

Three separate permit systems. No single point of authority.

**Target**: ONE risk-kernel process that owns ALL kill/permit decisions.

#### Step 8.1: Map all risk checks [ ]
- Governor: rate limit (3/hr, 2/5m), slots (4-6), fund caps, margin (65%)
- Heat-gate: portfolio heat, correlation exposure
- Portfolio-gate: position limits, sector exposure
- Risk-proposal: per-trade risk assessment
- Capital-allocation: sizing based on Kelly/portfolio optimization

#### Step 8.2: Create risk-kernel [ ]
```
kernel/risk_kernel/
├── main.py
├── rate_limiter.py     ← from governor
├── heat_monitor.py     ← from heat-gate
├── portfolio_gate.py   ← from portfolio-gate
├── risk_assessor.py    ← from risk-proposal
├── capital_allocator.py ← from capital-allocation
└── permit_manager.py   ← unified permit system
```

#### Step 8.3: Deploy in shadow mode [ ]
- Run alongside existing services
- Compare decisions for 48h
- Cut over when 100% match

---

## OPERATION 9: PLUGIN ARCHITECTURE
### Priority: LOW | Risk: MEDIUM | Status: [ ] NOT STARTED
### Depends on: Operations 5, 7, 8

**Target**: AI strategies become plugins that implement a standard interface.

```python
class Strategy(Protocol):
    def evaluate(self, market: MarketSnapshot) -> Optional[TradeIntent]: ...
    def on_feedback(self, result: TradeResult) -> None: ...
    def health_check(self) -> bool: ...
```

This operation is FUTURE — only after Operations 1-8 are complete.

---

## OPERATION 10: SHELL UNIFICATION
### Priority: LOW | Risk: LOW | Status: [ ] NOT STARTED
### Depends on: All above

**Target**: 6 frontends → 1. Unified API. Unified CLI.

This operation is FUTURE — only after Operations 1-9 are complete.

---

## QUICK REFERENCE

### VPS Access
```powershell
# Simple command:
wsl sh -lc "ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'COMMAND'"

# Multi-line command:
$s = @'
COMMANDS
HERE
'@
$b = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($s))
wsl sh -lc "ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'echo $b | base64 -d | bash'"
```

### Key Paths (VPS)
```
Code:     /home/qt/quantum_trader/
Venv:     /home/qt/quantum_trader_venv/
Services: /etc/systemd/system/quantum-*.service
Logs:     /var/log/quantum/ + journalctl -u quantum-*
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
# All services running?
systemctl list-units quantum-*.service | grep -c running

# AI Engine alive?
curl -s http://localhost:8001/health

# Recent trades?
redis-cli xlen quantum:stream:apply.result

# Recent signals?
redis-cli xlen quantum:stream:trade.intent

# Governor permits?
redis-cli keys 'quantum:permit:*' | wc -l
```

### Emergency Rollback
```bash
# Restore ALL service files:
cp /opt/backups/systemd-2026-03-14/quantum-*.service /etc/systemd/system/
systemctl daemon-reload

# Restore venvs:
mv /opt/backups/venvs-2026-03-14 /opt/quantum/venvs

# Restore backend:
mv archive/old_backend backend/
```

---

## OPERATION DEPENDENCY GRAPH

```
Op 1: Kill Two Heads ─────┐
                           ├──→ Op 5: Unify Execution ──→ Op 6: Position Truth
Op 2: Remove Zombies ─────┤                                      │
                           │                                      ▼
Op 3: One Python ──────────┤                              Op 7: Typed IPC
                           │                                      │
Op 4: Clean Coral Reef ───┘                                      ▼
                                                          Op 8: Risk Kernel
                                                                  │
                                                                  ▼
                                                          Op 9: Plugin Arch
                                                                  │
                                                                  ▼
                                                          Op 10: Shell
```

**Critical Path**: Op 1 → Op 5 → Op 6 → Op 7 → Op 8
**Parallel Track**: Op 2, Op 3, Op 4 can run alongside Op 1

---

## PROGRESS LOG

| Date | Operation | Step | Status | Notes |
|------|-----------|------|--------|-------|
| 2026-03-14 | — | — | Plan created | Full system mapped. 134 services, 70 streams, 3 code trees |
| | | | | |

---

*This document is the SINGLE SOURCE OF TRUTH for the QTOS migration.*
*Every session: read this first, update progress, continue where left off.*
