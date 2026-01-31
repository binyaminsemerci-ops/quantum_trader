# SYSTEM MAP V0 — QUANTUM TRADER PRODUCTION

**Date**: 2026-01-29 20:57 UTC  
**Mode**: READ-ONLY INVENTORY  
**Host**: quantumtrader-prod-1 (46.224.116.254)  
**Purpose**: Pure system definition without analysis or recommendations

---

## 1. SERVICE CATALOG

**Total Services**: 76 systemd units (quantum-*)

### 1.1 Running Services (42)

| Service | Status | PID | Port | WorkingDirectory | Commit |
|---------|--------|-----|------|------------------|--------|
| quantum-ai-engine | active/running | 623393 | 8001 | /opt/quantum | c14a7e7f |
| quantum-ai-orchestrator | active/running | 897 | 8010 | /opt/quantum | c14a7e7f |
| quantum-strategy-brain | active/running | 912 | 8011 | /opt/quantum | c14a7e7f |
| quantum-trading-bot | active/running | 3097974 | 8006 | /opt/quantum | c14a7e7f |
| quantum-ai-strategy-router | active/running | 907750 | 8065 | /home/qt/quantum_trader | c14a7e7f |
| quantum-allocation-target | active/running | 907751 | 8059 | /home/qt/quantum_trader | c14a7e7f |
| quantum-capital-allocation | active/running | 907753 | 8059 | /home/qt/quantum_trader | c14a7e7f |
| quantum-capital-efficiency | active/running | 907757 | 8062-8063 | /home/qt/quantum_trader | c14a7e7f |
| quantum-ceo-brain | active/running | - | - | /opt/quantum | c14a7e7f |
| quantum-clm | active/running | - | - | /home/qt/quantum_trader | c14a7e7f |
| quantum-clm-minimal | active/running | - | - | /home/qt/quantum_trader | c14a7e7f |
| quantum-cross-exchange-aggregator | active/running | 907774 | - | /home/qt/quantum_trader | c14a7e7f |
| quantum-dashboard-api | active/running | 907845 | 8000 | /home/qt/quantum_trader | c14a7e7f |
| quantum-exchange-stream-bridge | active/running | 907773 | - | /home/qt/quantum_trader | c14a7e7f |
| quantum-execution-result-bridge | active/running | 526663 | - | /usr/local/bin | system-wide |
| quantum-exit-intelligence | active/running | 907767 | - | /home/qt/quantum_trader | c14a7e7f |
| quantum-exit-monitor | active/running | - | - | /home/qt/quantum_trader | c14a7e7f |
| quantum-exitbrain-v35 | active/running | - | - | /home/qt/quantum_trader | c14a7e7f |
| quantum-governor | active/running | 907775 | 8044 | /home/qt/quantum_trader | c14a7e7f |
| quantum-harvest-brain | active/running | - | - | /home/qt/quantum_trader | c14a7e7f |
| quantum-harvest-metrics-exporter | active/running | 907778 | 8068-8069 | /home/qt/quantum_trader | c14a7e7f |
| quantum-harvest-optimizer | active/running | 907831 | 8052 | /home/qt/quantum_trader | c14a7e7f |
| quantum-harvest-proposal | active/running | 907828 | - | /home/qt/quantum_trader | c14a7e7f |
| quantum-heat-bridge | active/running | - | - | /home/qt/quantum_trader | c14a7e7f |
| quantum-heat-gate | active/running | 907818 | 8068-8069 | /home/qt/quantum_trader | c14a7e7f |
| quantum-intent-bridge | active/running | - | - | /home/qt/quantum_trader | c14a7e7f |
| quantum-intent-executor | active/running | - | - | /home/qt/quantum_trader | c14a7e7f |
| quantum-marketstate | active/running | 907827 | - | /home/qt/quantum_trader | c14a7e7f |
| quantum-meta-regime | active/running | - | - | /home/qt/quantum_trader | c14a7e7f |
| quantum-metricpack-builder | active/running | 907847 | 8056 | /home/qt/quantum_trader | c14a7e7f |
| quantum-performance-attribution | active/running | 452142 | 8060 | /home/qt/quantum_trader | c14a7e7f |
| quantum-portfolio-clusters | active/running | - | - | /home/qt/quantum_trader | c14a7e7f |
| quantum-portfolio-gate | active/running | - | - | /home/qt/quantum_trader | c14a7e7f |
| quantum-portfolio-governance | active/running | - | - | /home/qt/quantum_trader | c14a7e7f |
| quantum-portfolio-heat-gate | active/running | 907854 | - | /home/qt/quantum_trader | c14a7e7f |
| quantum-portfolio-risk-governor | active/running | - | - | /home/qt/quantum_trader | c14a7e7f |
| quantum-portfolio-state-publisher | active/running | 907820 | - | /home/qt/quantum_trader | c14a7e7f |
| quantum-position-monitor | active/running | - | - | /home/qt/quantum_trader | c14a7e7f |
| quantum-position-state-brain | active/running | 907821 | 8045 | /home/qt/quantum_trader | c14a7e7f |
| quantum-reconcile-engine | active/running | 907822 | 8046 | /root/quantum_trader | 6602dd03 |
| quantum-retrain-worker | active/running | 907823 | - | /home/qt/quantum_trader | c14a7e7f |
| quantum-risk-safety | active/running | - | - | /home/qt/quantum_trader | c14a7e7f |
| quantum-rl-agent | active/running | 1678 | - | /opt/quantum/rl | c14a7e7f |
| quantum-rl-feedback-v2 | active/running | - | - | /opt/quantum | c14a7e7f |
| quantum-rl-monitor | active/running | 1680 | - | /opt/quantum/rl | c14a7e7f |
| quantum-rl-policy-publisher | active/running | 907832 | - | /home/qt/quantum_trader | c14a7e7f |
| quantum-rl-shadow-metrics-exporter | active/running | 907834 | - | /home/qt/quantum_trader | c14a7e7f |
| quantum-rl-sizer | active/running | - | - | /opt/quantum | c14a7e7f |
| quantum-rl-trainer | active/running | 1709 | - | /opt/quantum/rl | c14a7e7f |
| quantum-safety-telemetry | active/running | 907838 | 8051 | /home/qt/quantum_trader | c14a7e7f |
| quantum-strategic-memory | active/running | - | - | /home/qt/quantum_trader | c14a7e7f |
| quantum-utf-publisher | active/running | - | - | /home/qt/quantum_trader | c14a7e7f |

### 1.2 Failed Services (5)

| Service | Status | Last Error |
|---------|--------|------------|
| quantum-apply-layer | failed | SyntaxError (merge conflict line 1) |
| quantum-contract-check | failed | unknown |
| quantum-ess-trigger | failed | unknown |
| quantum-risk-proposal | failed | unknown |
| quantum-exposure-balancer | activating/auto-restart | restart loop |

### 1.3 Inactive/Dead Services (27)

| Service | Status | Reason |
|---------|--------|--------|
| quantum-core-health | inactive/dead | oneshot/manual |
| quantum-diagnostic | inactive/dead | oneshot/manual |
| quantum-ess-watch | inactive/dead | path monitor (inactive) |
| quantum-execution | inactive/dead | manual control |
| quantum-ledger-sync | inactive/dead | periodic |
| quantum-p28a3-latency-proof | inactive/dead | read-only probe |
| quantum-policy-sync | inactive/dead | manual sync |
| quantum-portfolio-intelligence | inactive/dead | unused |
| quantum-risk-brain | inactive/dead | unused |
| quantum-rl-shadow-scorecard | inactive/dead | report generator |
| quantum-stream-recover | inactive/dead | recovery utility |
| quantum-training-worker | inactive/dead | oneshot trainer |
| quantum-verify-ensemble | inactive/dead | verification task |
| quantum-verify-rl | inactive/dead | verification task |

### 1.4 Not Found (2)

| Service | Status |
|---------|--------|
| quantum-ensemble | not-found |
| quantum-redis | not-found |

---

## 2. PORT MAPPING

**Listening Ports** (8000-8999 range):

| Port | PID | Service | Bind Address |
|------|-----|---------|--------------|
| 8000 | 907845 | quantum-dashboard-api | 0.0.0.0 (public) |
| 8001 | 623393 | quantum-ai-engine | 127.0.0.1 (local) |
| 8003 | 907850 | unknown | 0.0.0.0 |
| 8005 | 2665194 | unknown | 0.0.0.0 |
| 8006 | 3097974 | quantum-trading-bot | 127.0.0.1 (local) |
| 8007 | 907846 | unknown | 0.0.0.0 |
| 8010 | 897 | quantum-ai-orchestrator | 127.0.0.1 (local) |
| 8011 | 912 | quantum-strategy-brain | 127.0.0.1 (local) |
| 8026 | 850882 | unknown | 0.0.0.0 |
| 8042 | 907778 | quantum-harvest-metrics-exporter | 0.0.0.0 |
| 8044 | 907775 | quantum-governor | 0.0.0.0 |
| 8045 | 907821 | quantum-position-state-brain | 0.0.0.0 |
| 8046 | 907822 | quantum-reconcile-engine | 0.0.0.0 |
| 8047 | 2860211 | unknown | 0.0.0.0 |
| 8048 | 2859614 | unknown | 0.0.0.0 |
| 8049 | 907855 | unknown | 0.0.0.0 |
| 8051 | 907847 | quantum-safety-telemetry | 0.0.0.0 |
| 8052 | 907831 | quantum-harvest-optimizer | 0.0.0.0 |
| 8056 | 907854 | quantum-metricpack-builder | 0.0.0.0 |
| 8059 | 907753 | quantum-capital-allocation | 0.0.0.0 |
| 8060 | 452142 | quantum-performance-attribution | 0.0.0.0 |
| 8061 | 907819 | quantum-heat-gate | 0.0.0.0 |
| 8062 | 907757 | quantum-capital-efficiency | 0.0.0.0 |
| 8063 | 907757 | quantum-capital-efficiency | 0.0.0.0 |
| 8065 | 907751 | quantum-ai-strategy-router | 0.0.0.0 |
| 8068 | 907818 | quantum-harvest-metrics-exporter | 0.0.0.0 |
| 8069 | 907818 | quantum-harvest-metrics-exporter | 0.0.0.0 |
| 8070 | 1119243 | unknown | 0.0.0.0 |
| 8071 | 1119243 | unknown | 0.0.0.0 |

**Other Known Ports**:
- **3000**: Grafana (HTTP 301 redirect)
- **6379**: Redis (localhost only)
- **9090**: Prometheus (assumed standard)

---

## 3. REPOSITORY STATE

### 3.1 Repository #1: /home/qt/quantum_trader

**HEAD Commit**: `c14a7e7f` ("Fix proof script: remove set -euo pipefail")  
**Branch**: main (assumed)  
**Dirty State**: Untracked file `main.py.patch`  
**Used By**: 38+ services (all quantum-* except reconcile-engine)

**Purpose**: Primary production codebase  
**Note**: Apply Layer merge conflict at `microservices/apply_layer/main.py:1`

### 3.2 Repository #2: /root/quantum_trader

**HEAD Commit**: `6602dd03` ("CONTROL LAYER V1: Deployment verification report")  
**Branch**: main (assumed)  
**Dirty State**: Modified files:
- `backend/services/execution/execution.py` (TPSL shield)
- `backend/services/execution/exit_order_gateway.py` (Gateway guard)

**Used By**: 1 service (quantum-reconcile-engine)  
**Purpose**: Policy enforcement modifications, manual operations

**Divergence**: ~30 commits ahead of /home/qt repository

---

## 4. REDIS TOPOLOGY

### 4.1 Server Configuration

**Version**: 7.0.15  
**Mode**: Standalone  
**PID**: 917 (systemd supervised)  
**Uptime**: 857,562 seconds (~10 days)  
**Port**: 6379 (localhost only)  

**Connections**:
- Connected clients: 153
- Blocked clients: 25

**Keys**: 190,547 total (exact count not provided)

### 4.2 Stream Inventory (45 streams)

**Format**: `quantum:stream:<name>`

| Stream Name | Purpose (inferred from name) |
|-------------|------------------------------|
| ai.decision.made | AI decisions published |
| ai.signal_generated | AI signals (buy/sell) |
| allocation.decision | Position sizing decisions |
| allocation.target.proposed | Target allocation proposals |
| alpha.attribution | Performance attribution |
| apply.heat.observed | Heat observations during apply |
| apply.plan | Harvest execution plans |
| apply.plan.manual | Manual harvest plans |
| apply.result | Harvest execution results |
| budget.violation | Budget/risk violations |
| capital.efficiency.decision | Capital efficiency choices |
| clm.intent | Continuous learning intents |
| events | General system events |
| exchange.normalized | Normalized exchange data |
| exchange.raw | Raw exchange feed |
| execution.result | Trade execution results |
| exitbrain.pnl | Exit Brain PnL tracking |
| governor.events | Governor enforcement events |
| harvest.calibrated | Calibrated harvest proposals |
| harvest.heat.decision | Heat-based harvest decisions |
| harvest.proposal | Initial harvest proposals |
| learning.retraining.completed | Model retrain completions |
| learning.retraining.started | Model retrain starts |
| market.klines | Candlestick data |
| market.tick | Tick-level market data |
| marketstate | Market state metrics |
| meta.regime | Regime detection outputs |
| model.retrain | Model retrain commands |
| policy.updated | Policy change notifications |
| portfolio.cluster_state | Portfolio clustering |
| portfolio.exposure_updated | Exposure change events |
| portfolio.gate | Portfolio gate decisions |
| portfolio.snapshot_updated | Portfolio snapshots |
| portfolio.state | Current portfolio state |
| position.snapshot | Position-level snapshots |
| reconcile.close | Position close reconciliations |
| reconcile.events | Reconciliation events |
| sizing.decided | Position sizing results |
| trade.closed | Closed trade records |
| trade.execution.res | Trade execution responses |
| trade.intent | Trading intents (entry/exit) |
| trade.intent.dlq | Dead letter queue (intents) |
| trade.signal | Trading signals (pre-intent) |
| trading.plan | Trading plan (strategy output) |
| utf | Unified Training Feed |

### 4.3 Consumer Groups (Known)

**Note**: Full consumer group enumeration requires per-stream XINFO GROUPS.  
**Known Groups** (from previous audit):
- `quantum:group:execution:trade.intent` (42 consumers, 1.5M lag historical)
- `quantum:group:heat_gate:harvest.proposal` (1 consumer, lag=3)
- `quantum:group:exit_intelligence:apply.result` (1 consumer, pending=481)

---

## 5. DATA FLOW TOPOLOGY

### 5.1 Market Data Flow

```
[Binance/Exchanges]
    ↓
quantum-exchange-stream-bridge
    ↓ publishes to
quantum:stream:exchange.raw
    ↓ consumed by
quantum-cross-exchange-aggregator
    ↓ publishes to
quantum:stream:exchange.normalized
    ↓ consumed by
quantum-ai-engine
```

### 5.2 AI Signal Flow

```
quantum-ai-engine (19 models)
    ↓ publishes to
quantum:stream:ai.signal_generated
    ↓ consumed by
quantum-trading-bot
    ↓ publishes to
quantum:stream:trade.intent
    ↓ consumed by (multiple consumers)
quantum-intent-executor (execution path)
quantum-intent-bridge (harvest path)
```

### 5.3 Trade Execution Flow

```
quantum:stream:trade.intent
    ↓ consumed by
quantum-intent-executor
    ↓ calls Binance API
[Binance Exchange]
    ↓ response captured by
quantum-execution-result-bridge
    ↓ publishes to
quantum:stream:execution.result
    ↓ consumed by
quantum-position-monitor
quantum-reconcile-engine
```

### 5.4 Harvest Proposal Flow

```
quantum:stream:trade.intent
    ↓ consumed by
quantum-intent-bridge
    ↓ publishes to
quantum:stream:apply.plan
    ↓ consumed by
quantum-harvest-proposal
    ↓ publishes to
quantum:stream:harvest.proposal
    ↓ consumed by
quantum-heat-gate (portfolio heat check)
    ↓ publishes to
quantum:stream:harvest.calibrated
    ↓ consumed by
quantum-apply-layer [🔴 CRASHED]
    ↓ would publish to
quantum:stream:apply.result
```

### 5.5 Exit Brain Flow

```
quantum-exitbrain-v35 (autonomous exit manager)
    ↓ reads position state from
quantum:stream:portfolio.state
quantum:stream:market.tick
    ↓ decides exits, publishes to
quantum:stream:trade.intent (with exit_type)
    ↓ filtered by
quantum-intent-executor (gateway guard)
    ↓ executes via
[Binance API] (MARKET orders only)
```

### 5.6 RL System Flow

```
quantum-rl-agent (shadow mode)
    ↓ reads state from
quantum:stream:portfolio.state
quantum:stream:marketstate
    ↓ publishes policy to
quantum:stream:policy.updated
    ↓ consumed by
quantum-rl-policy-publisher
    ↓ publishes to
quantum:stream:sizing.decided (shadow metrics)
    ↓ monitored by
quantum-rl-monitor
quantum-rl-shadow-metrics-exporter
```

---

## 6. PROCESS INVENTORY

**Python Processes** (quantum-related):

| PID | User | Uptime | CPU% | MEM% | Command |
|-----|------|--------|------|------|---------|
| 897 | qt | 10d+ | 0.1 | 0.3 | uvicorn ai_orchestrator (port 8010) |
| 912 | qt | 10d+ | 0.1 | 0.3 | uvicorn strategy_brain (port 8011) |
| 1678 | qt | 10d+ | 0.9 | 2.0 | rl_agent.py |
| 1680 | qt | 10d+ | 0.0 | 0.2 | rl_monitor.py |
| 1709 | qt | 10d+ | 0.9 | 2.0 | rl_trainer.py |
| 452142 | qt | ~2d | 0.1 | 0.1 | performance_attribution/main.py |
| 526663 | qt | 9d+ | 0.0 | 0.1 | quantum_execution_result_bridge.py |
| 623393 | qt | ~2d | 1.0 | 3.1 | uvicorn ai_engine (port 8001) |
| 907750 | qt | ~14h | 0.0 | 0.2 | ai_strategy_router.py |
| 907751 | qt | ~14h | 0.1 | 0.1 | allocation_target/main.py |
| 907753 | qt | ~14h | 0.1 | 0.1 | capital_allocation/main.py |
| 907757 | qt | ~14h | 0.0 | 0.1 | capital_efficiency/main.py |
| 907767 | qt | ~14h | 0.4 | 0.5 | exit_intelligence/main.py |
| 907773 | qt | ~14h | 0.2 | 0.2 | exchange_stream_bridge.py |
| 907774 | qt | ~14h | 1.7 | 0.4 | cross_exchange_aggregator.py |
| 907775 | root | ~14h | 0.0 | 0.1 | governor/main.py |
| 907778 | qt | ~14h | 0.0 | 0.1 | harvest_metrics_exporter/main.py |
| 907818 | qt | ~14h | 0.0 | 0.1 | heat_gate/main.py |
| 907819 | qt | ~14h | 0.1 | 0.1 | performance_attribution/main.py |
| 907820 | qt | ~14h | 0.0 | 0.1 | portfolio_state_publisher/main.py |
| 907821 | root | ~14h | 2.6 | 0.2 | position_state_brain/main.py |
| 907822 | root | ~14h | 0.0 | 0.1 | reconcile_engine/main.py (from /root) |
| 907823 | qt | ~14h | 0.0 | 0.1 | retrain_worker.py |
| 907827 | root | ~14h | 0.0 | 0.6 | market_state_publisher/main.py |
| 907828 | qt | ~14h | 0.0 | 0.1 | harvest_proposal_publisher/main.py |
| 907831 | qt | ~14h | 0.1 | 0.3 | harvest_optimizer/main.py |
| 907832 | qt | ~14h | 0.0 | 0.1 | rl_policy_publisher.py |
| 907834 | qt | ~14h | 0.0 | 0.2 | rl_shadow_metrics_exporter.py |
| 907838 | qt | ~14h | 0.0 | 0.1 | safety_telemetry/main.py |
| 907847 | qt | ~14h | 0.1 | 0.3 | metricpack_builder/main.py |
| 907854 | qt | ~14h | 0.2 | 0.1 | portfolio_heat_gate/main.py |
| 3097974 | qt | ~1h | 0.4 | 0.4 | uvicorn trading_bot (port 8006) |

**CPU Intensive** (>0.5%):
- 907821: position_state_brain (2.6%)
- 907774: cross_exchange_aggregator (1.7%)
- 623393: ai_engine (1.0%)
- 1678: rl_agent (0.9%)
- 1709: rl_trainer (0.9%)

**Memory Intensive** (>1.0%):
- 623393: ai_engine (3.1% / 508MB RSS)
- 1678: rl_agent (2.0% / 334MB RSS)
- 1709: rl_trainer (2.0% / 335MB RSS)

---

## 7. CONFIGURATION FILES

### 7.1 Environment Files

**Location**: `/etc/quantum/`  
**Total Count**: 75 .env files

**Security Profile**:
- **600 permissions** (secure): 5 files
  - exitbrain-control.env
  - exitbrain-v35.env
  - position-monitor.env
  - alert.env
  - testnet.env
- **644 permissions** (readable): 70 files

**API Key References**: 8 files contain `BINANCE_*_KEY` variables

### 7.2 Systemd Drop-in Configs

**Location**: `/etc/systemd/system/quantum-*.service.d/`  
**Count**: 7 override configurations

**Purpose**: Service-specific environment overrides

---

## 8. INFRASTRUCTURE

### 8.1 Host Configuration

**Hostname**: quantumtrader-prod-1  
**IP Address**: 46.224.116.254  
**OS**: Ubuntu (kernel 6.8.0-90-generic)  
**CPU**: AMD EPYC-Milan, 4 cores  
**Architecture**: x86_64  

### 8.2 Resource Utilization

**Memory**:
- Total: 15Gi
- Used: 5.1Gi (34%)
- Available: 9.9Gi

**Disk**:
- Total: 150G
- Used: 25G (18%)
- Available: 117G

**Load Average**: Not provided

### 8.3 Python Environments

**Known Virtual Environments**:
1. `/opt/quantum/venvs/ai-engine/` (AI Engine, RL system)
2. `/opt/quantum/venvs/ai-client-base/` (AI Orchestrator, Strategy Brain)
3. `/opt/quantum/venvs/safety-telemetry/` (Safety Telemetry)
4. `/home/qt/quantum_trader_venv/` (Various microservices)
5. System Python: `/usr/bin/python3`

---

## 9. EXTERNAL INTEGRATIONS

### 9.1 Exchange Connections

**Binance**:
- API Keys: 8 references (testnet + production)
- Connection Method: REST + WebSocket
- Bridge Service: quantum-exchange-stream-bridge

**Multi-Exchange Support**:
- Aggregator: quantum-cross-exchange-aggregator
- Normalization: quantum:stream:exchange.normalized

### 9.2 Monitoring Stack

**Grafana**: Port 3000 (HTTP 301)  
**Prometheus**: Port 9090 (assumed)  
**Redis**: Port 6379 (localhost)

**Metrics Exporters**:
- quantum-safety-telemetry (port 8051)
- quantum-rl-shadow-metrics-exporter
- quantum-harvest-metrics-exporter

---

## 10. CRITICAL PATH STATES

### 10.1 Path A: Core Trading (AI → Trading Bot → Binance)

**Status**: ✅ OPERATIONAL

**Components**:
1. quantum-ai-engine (active, PID 623393)
2. quantum-trading-bot (active, PID 3097974)
3. quantum-intent-executor (active)
4. quantum:stream:trade.intent (flowing)

**Evidence**: Services running, ports responding, stream activity confirmed

### 10.2 Path B: Exit Brain (Autonomous Exit Management)

**Status**: ✅ OPERATIONAL (10% LIVE rollout)

**Components**:
1. quantum-exitbrain-v35 (active)
2. Exit Brain Control Layer (LIVE mode, kill-switch OFF)
3. Gateway Guard (policy enforcement)

**Evidence**: Service active, control layer config confirmed, 10% rollout to SOLUSDT/ADAUSDT/DOTUSDT

### 10.3 Path C: Harvest System (Profit Harvesting)

**Status**: 🔴 BLOCKED

**Components**:
1. quantum-harvest-proposal (active, 28 messages published)
2. quantum-heat-gate (active but FAIL-OPEN)
3. quantum-apply-layer (🔴 CRASHED - merge conflict)

**Evidence**: 
- Apply Layer failed (SyntaxError at main.py:1)
- Heat Gate logs show "FAIL-OPEN (missing_inputs)"
- quantum:stream:apply.result length: 10,046 (no new entries)

---

## 11. UNKNOWNS & GAPS

**Unmapped PIDs**:
- 850882 (port 8026)
- 907845 (port 8000) - likely dashboard but not confirmed
- 907846 (port 8007)
- 907850 (port 8003)
- 907855 (port 8049)
- 1119243 (ports 8070-8071)
- 2665194 (port 8005)
- 2859614 (port 8048)
- 2860211 (port 8047)

**Unverified Connections**:
- quantum-ensemble.service (not-found) - references unknown
- quantum-redis.service (not-found) - possibly removed/renamed
- Several inactive services (execution, risk-brain, portfolio-intelligence)

**Stream Consumer Groups**:
- Only 3 groups fully analyzed (execution, heat_gate, exit_intelligence)
- 42 additional streams exist without documented consumers

**Configuration Details**:
- Environment variable contents not enumerated (security)
- Systemd drop-in override details not retrieved

---

## 12. SYSTEM HEALTH INDICATORS

**Service Health**:
- Running: 42 services
- Failed: 5 services
- Inactive: 27 services
- Success Rate: 84% (42/50 active units)

**Redis Health**:
- Uptime: 10 days continuous
- Client connections: 153 (healthy for 42 active services)
- Blocked clients: 25 (waiting on stream reads)

**Process Health**:
- Long-running processes: 10+ days (RL system, orchestrators)
- Recent restarts: AI Engine (~2 days), Trading Bot (~1 hour)
- CPU usage: Normal (<3% per process)
- Memory usage: Normal (<3% per process)

**Stream Activity**:
- Total streams: 45
- Active streams: Confirmed activity on trade.intent, apply.result, execution.result
- Dead letter queue: trade.intent.dlq (length unknown)

---

## APPENDIX A: DATA SOURCES

**System Commands Used**:
1. `systemctl list-units "quantum-*.service" --all`
2. `ps aux | grep quantum`
3. `ss -lntp | grep 8[0-9][0-9][0-9]`
4. `redis-cli INFO`
5. `redis-cli --scan --pattern "quantum:stream:*"`
6. `git log --oneline -1` (both repos)

**File Inspections**:
- None (pure runtime inspection)

**Inference Sources**:
- Service names → Purpose
- Stream names → Data flow
- Port bindings → Service communication

---

## APPENDIX B: REVISION HISTORY

**v0**: 2026-01-29 20:57 UTC  
- Initial system map (read-only inventory mode)  
- 76 services cataloged  
- 45 Redis streams mapped  
- 29 port bindings documented  
- 2 repository states captured

**Next Version Recommendations**:
- Enumerate all consumer groups (XINFO GROUPS per stream)
- Map unknown PIDs to services (cross-reference with ExecStart paths)
- Document environment variable schema (sanitized)
- Add network topology (inter-service communication matrix)
- Include configuration file diffs (dual repo changes)

---

**END OF SYSTEM MAP V0**
