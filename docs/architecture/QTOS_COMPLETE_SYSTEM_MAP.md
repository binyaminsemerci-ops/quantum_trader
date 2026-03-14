# QTOS — Quantum Trading Operating System
## Complete System Map & Architecture Blueprint
### Date: 2026-03-14 | Author: Architecture Audit Session 14

---

## 1. SYSTEM INVENTORY — What Actually Exists

### 1.1 Scale Summary

| Dimension | Count |
|-----------|-------|
| Systemd service files | **134** |
| Running services | **43** |
| Inactive/dead services | **~18** |
| Redis streams | **~70** |
| Redis key namespaces | **~60+** |
| Python virtual environments | **10** |
| Python interpreters in use | **3** |
| Code trees (backend/, microservices/, ai_engine/) | **3** |
| microservices/ subdirectories | **80+** |
| backend/services/ files | **70+** |
| AI model agents | **15+** |
| Root-level files (scripts, docs, fix_*.py) | **~1500+** |
| Frontend projects | **6** |

### 1.2 The Three Code Trees

```
quantum_trader/
├── backend/          ← ORIGINAL FastAPI monolith (50+ dirs, 70+ services)
│   ├── main.py       ← FastAPI app with /trades, /stats
│   ├── core/         ← event_bus, policy_store, health, safety
│   ├── services/     ← 70+ files: ai/, risk/, clm/, execution/, governance/...
│   ├── ai_*          ← orchestrator, risk, strategy
│   ├── federation/, memory/, world_model/, meta_learning/
│   └── trading_bot/  ← yet another trading implementation
│
├── microservices/    ← NEW standalone systemd services (80+ dirs)
│   ├── ai_engine/    ← uvicorn FastAPI (port 8001) — THE brain
│   ├── apply_layer/  ← 3000+ line monolith (handles close execution)
│   ├── intent_bridge/ ← trade.intent → apply.plan transform
│   ├── intent_executor/ ← apply.plan → Binance orders (MAIN executor)
│   ├── governor/     ← rate-limit, slots, fund caps
│   ├── execution/    ← LEGACY executor (parallel with intent_executor!)
│   ├── exit_*/       ← 5 exit services
│   ├── harvest_*/    ← 4 harvest services
│   ├── portfolio_*/  ← 6 portfolio services
│   ├── rl_*/         ← 7 RL services
│   ├── layer1-6/     ← 6 pipeline layers
│   ├── dag3-8/       ← 5 DAG safety services
│   └── 40+ others...
│
├── ai_engine/        ← AI models and services
│   ├── agents/       ← 15+ model agents (xgb, lgbm, nhits, patchtst, tft, meta, arbiter...)
│   ├── services/     ← ensemble_predictor, eventbus_bridge, feature_publisher
│   ├── calibration/  ← Signal calibration system
│   └── models/       ← ML model files (.pkl)
│
├── ops/              ← Operations: deploy, monitor, market_publisher
├── lib/              ← Shared: exit_ownership.py, policy_store.py
├── config/, configs/ ← Configuration files
└── ~1500 files       ← Diagnostics, fix-scripts, AI_*.md docs, .env files
```

### 1.3 VPS Filesystem — The Dual Codepath Problem

```
VPS Filesystem:
┌─────────────────────────────────────┐
│ /home/qt/quantum_trader/            │ ← PYTHONPATH, git repo (ALL services run from here)
│   ├── microservices/                │
│   ├── backend/                      │
│   ├── ai_engine/                    │
│   └── lib/, ops/, config/           │
├─────────────────────────────────────┤
│ /opt/quantum/                       │ ← OLD location (partial ghost)
│   ├── ai_engine/models/             │ ← model files may still be read from here
│   ├── ops/market/market_publisher.py│ ← 1 service STILL runs from here
│   ├── services/                     │ ← ~15 service dirs (UNUSED by runtime)
│   ├── venvs/ai-engine/             │ ← 7 services use this venv
│   ├── orchestration/                │
│   └── risk_brain/, strategy_brain/  │ ← Archaeological remains
├─────────────────────────────────────┤
│ /mnt/HC_Volume_104287969/           │ ← Hetzner storage volume
│   └── quantum-venvs/               │ ← 8 extra venvs (many UNUSED)
│       ├── safety-telemetry/         │
│       ├── runtime/                  │
│       ├── ai-client-base/           │
│       ├── strategy-ops/             │
│       ├── rl-dashboard/             │
│       ├── ai-engine/ (duplicate!)   │
│       ├── rl-sizer/                 │
│       └── execution/                │
└─────────────────────────────────────┘
```

### 1.4 Three Python Interpreters in Production

| Interpreter | Path | Used by |
|-------------|------|---------|
| Main venv | `/home/qt/quantum_trader_venv/bin/python` | ai-engine, clm, exit-services, harvest-optimizer, layer1, market-publisher, metricpack, price-feed, rl-agent, rl-sizer |
| Secondary venv | `/opt/quantum/venvs/ai-engine/bin/python3` | ai-strategy-router, ensemble-predictor, harvest-proposal, harvest-metrics, marketstate, portfolio-governance, risk-proposal |
| System Python | `/usr/bin/python3` | balance-tracker, capital-allocation, governor, heat-gate, intent-bridge, intent-executor, portfolio-*, reconcile-engine, performance-*, trade-logger, universe + more |

---

## 2. ALL 134 SYSTEMD SERVICES

### 2.1 Running Services (43)

| Service | Status | Description |
|---------|--------|-------------|
| ai-engine | running | AI Engine (native uvicorn, port 8001) |
| ai-strategy-router | running | AI Strategy Router |
| balance-tracker | running | Binance Account Monitor |
| capital-allocation | running | Capital Allocation Brain (P2.9) |
| clm | running | Continuous Learning Manager |
| ensemble-predictor | running | Ensemble Predictor Service (SHADOW MODE) |
| execution-result-bridge | running | Execution Result Stream Bridge |
| exit-brain-shadow | running | Exit Brain v1 (SHADOW MODE) |
| exit-intelligence | running | Exit Intelligence Service |
| exit-intent-gateway | running | Exit Intent Gateway (testnet exit forwarder) |
| exit-management-agent | running | Exit Management Agent v0.2.0 (6 ML ensemble + EMA) |
| governor | running | P3.2 Governor Service |
| harvest-metrics-exporter | running | Harvest Metrics Exporter (P2.7) |
| harvest-optimizer | running | Harvest Optimizer Service |
| harvest-proposal | running | Harvest Proposal Publisher (P2.5) |
| heat-gate | running | P2.6 Portfolio Heat Gate |
| intent-bridge | running | Intent Bridge (trade.intent → apply.plan) |
| intent-executor | running | Intent Executor (→ Binance) |
| layer1-data-sink | running | OHLCV & Feature Data Sink |
| layer1-historical-backfill | exited | Historical OHLCV Backfill (one-shot) |
| market-publisher | running | Market Data Publisher (INFRA) |
| marketstate | running | MarketState Metrics Publisher (P0.5) |
| metricpack-builder | running | MetricPack Builder v1 |
| p35-decision-intelligence | running | P3.5 Decision Intelligence Service |
| performance-attribution | running | P3.0 Performance Attribution Brain |
| performance-tracker | running | Performance Tracker |
| portfolio-clusters | running | P2.7 Portfolio Clusters |
| portfolio-gate | running | P2.6 Portfolio Gate |
| portfolio-governance | running | Portfolio Governance |
| portfolio-heat-gate | running | P2.6 Portfolio Heat Gate (Hedge Fund OS) |
| portfolio-state-publisher | running | Portfolio State Publisher |
| price-feed | running | Price Feed (WebSocket → Redis) |
| reconcile-engine | running | P3.4 Position Reconciliation Engine |
| risk-proposal | running | Risk Proposal Publisher (P1.5) |
| rl-agent | running | RL Agent Daemon - Continuous Learning |
| rl-policy-publisher | running | RL Policy Publisher v0 |
| rl-shadow-metrics-exporter | running | RL Shadow Metrics Exporter |
| rl-sizer | running | RL Position Sizing Agent (Model Server) |
| rl-trainer | running | RL Trainer Consumer |
| stream-bridge | running | Stream Bridge - Execution Results |
| trade-logger | running | Trade History Logger |
| universe-service | running | Trading Universe Service |
| utf-publisher | running | UTF Publisher - Unified Training Feed |

### 2.2 Inactive/Dead Services (~18)

bsc, core-health, diagnostic, ess-watch, execution, exit-owner-watch, health-gate,
layer4-portfolio-optimizer, ledger-sync, offline-evaluator, p28a3-latency-proof,
portfolio-intelligence, portfolio-risk-governor, risk-brain, rl-shadow-scorecard,
strategy-brain, stream-recover, training-worker

### 2.3 All 134 Service Names (alphabetical)

ai-engine, ai_engine, ai-engine-proof, ai-strategy-router, ai-universe,
allocation-target, anti-churn-guard, apply-layer, autonomous-trader, backend,
balance-tracker, binance-pnl-tracker, bsc, capital-allocation, capital-efficiency,
ceo-brain, clm, clm-minimal, contract-check, core-health, cross-exchange-aggregator,
dag3-hw-stops, dag5-lockdown-guard, dag8-freeze-exit, dashboard-api, data-collector,
diagnostic, emergency-exit-worker, ensemble-predictor, ess, ess-trigger, ess-watch,
eventbus_bridge, exchange-aggregator, exchange-stream-bridge, execution,
execution-result-bridge, exit-brain-shadow, exitbrain-v35, exit-intelligence,
exit-intent-gateway, exit-management-agent, exit-monitor, exit-owner-watch,
exposure_balancer, frontend, governor, harvest-autoscaler, harvest-metrics-exporter,
harvest-optimizer, harvest-proposal, harvest-v2, health-gate, heat-bridge, heat-gate,
intent-bridge, intent-executor, layer1-data-sink, layer1-historical-backfill,
layer2-research-sandbox, layer2-signal-promoter, layer3-backtest-runner,
layer4-portfolio-optimizer, layer5-execution-monitor, layer6-post-trade, learning-api,
learning-monitor, ledger-sync, market-proof, market-publisher, marketstate, meta-regime,
metricpack-builder, model-federation, multi-source-feed, nginx-proxy, offline-evaluator,
p28a3-latency-proof, p35-decision-intelligence, paper-trade-controller,
performance-attribution, performance-tracker, pnl-reconciler, policy-refresh,
portfolio-clusters, portfolio-gate, portfolio-governance, portfolio-heat-gate,
portfolio-intelligence, portfolio_intelligence, portfolio-risk-governor,
portfolio-state-publisher, position-monitor, position_monitor, position-state-brain,
price-feed, proof, quantumfond-frontend, reconcile-engine, reconcile-hardened,
retrain-worker, risk-brain, risk-brake, risk-proposal, risk-safety, risk_safety, rl,
rl-agent, rl-dashboard, rl-feedback-v2, rl-policy-publisher, rl-reward-publisher,
rl-shadow-metrics-exporter, rl-shadow-scorecard, rl-sizer, rl-trainer, rl_training,
safety-telemetry, scaling-orchestrator, shadow-mode-controller, signal-injector,
strategic-evolution, strategic-memory, strategy-brain, strategy-ops, stream-bridge,
stream-recover, trade-logger, trading_bot, training-worker, universe, universe,
utf-publisher

---

## 3. COMPLETE SIGNAL-TO-TRADE DATA FLOW

```
Binance WebSocket
       │
       ▼
  price-feed ──→ quantum:stream:exchange.normalized
       │
       ▼
  AI Engine (service.py, port 8001)
  [Ensemble: XGB + LGBM + NHiTS + PatchTST + TFT]
  [Meta-Agent → Arbiter → RL Sizing → CEO Brain]
       │
       ├──→ quantum:stream:trade.intent (PRIMARY SIGNAL)
       └──→ quantum:stream:ai.decision.made (audit)
       
       │ XREADGROUP
       ▼
  Intent Bridge
  [Allowlist check, anti-churn gate (900s), ledger check, fee meta]
       │
       ├──→ quantum:stream:apply.plan
       └──→ quantum:permit:p33:{plan_id} (auto-bypass)
       
       │ (Parallel permit checks)
       ├──────────────────┐
       ▼                  ▼
  Governor            Portfolio Heat Gate (P2.6)
  [Rate limit          [Exposure, correlation]
   3/hr, 2/5m,         │
   slots (4-6),        └──→ quantum:permit:p26:{plan_id}
   fund caps,
   margin 65%]
       │
       └──→ quantum:permit:governor:{plan_id}
       
       │ (All 3 permits required — Lua atomic check)
       ▼
  Intent Executor
       │
       └──→ Binance Futures Testnet (market order)
             │
             └──→ quantum:stream:apply.result
                      │
                      ├──→ Governor (slot release)
                      ├──→ RL Calibration (trade.closed)
                      └──→ Trade Logger



=== PARALLEL CLOSE PIPELINE ===

  Harvest Brain → quantum:harvest:proposal:{symbol} (Redis hash)
       │
       ▼
  Apply Layer (3000+ lines, polls per symbol)
       │
       ├──→ quantum:stream:apply.plan (close proposal)
       └──→ Binance Futures (reduceOnly, DIRECT execution)
             │
             └──→ quantum:stream:trade.closed

  Exit Management Agent (6-model ML ensemble)
       │
       └──→ quantum:stream:exit.intent
             │
             ▼
       Intent Executor (also reads harvest.intent)
```

### 3.1 Service-by-Service Redis Contracts

#### AI Engine (microservices/ai_engine/service.py)
- **READS**: `quantum:stream:exchange.normalized`, `quantum:stream:trade.closed`, `quantum:harvest:proposal:{symbol}`
- **WRITES**: `quantum:stream:trade.intent`, `quantum:stream:ai.decision.made`, `quantum:stream:ai.exit.decision`

#### Intent Bridge (microservices/intent_bridge/main.py)
- **READS**: `quantum:stream:trade.intent` (group: intent_bridge)
- **WRITES**: `quantum:stream:apply.plan`, `quantum:permit:p33:{plan_id}`

#### Governor (microservices/governor/main.py)
- **READS**: `quantum:stream:apply.plan` (group: governor), `quantum:stream:apply.result`
- **WRITES**: `quantum:permit:governor:{plan_id}`, `quantum:stream:governor.events`, `quantum:stream:apply.plan` (rotation closes)

#### Intent Executor (microservices/intent_executor/main.py)
- **READS**: `quantum:stream:apply.plan` (group: intent_executor), `quantum:stream:apply.plan.manual`, `quantum:stream:harvest.intent`
- **WRITES**: `quantum:stream:apply.result`, `quantum:ledger:{symbol}`
- **EXTERNAL**: Binance Futures Testnet API

#### Apply Layer (microservices/apply_layer/main.py)
- **READS**: `quantum:harvest:proposal:{symbol}`, `quantum:stream:reconcile.close`
- **WRITES**: `quantum:stream:apply.plan`, `quantum:stream:apply.result`, `quantum:stream:trade.closed`
- **NOTE**: Entry processing DISABLED since 2026-02-25. Only handles closes.

#### Execution Service (microservices/execution/) — LEGACY
- **READS**: `quantum:stream:trade.intent` (via EventBus subscribe — PARALLEL path!)
- **WRITES**: order.placed, order.filled, trade.opened, trade.closed events
- **NOTE**: Redundant with intent_executor. Both consume trade.intent.

### 3.2 The Permit System

```
Three permits required before any order executes:

quantum:permit:governor:{plan_id}  ← Governor: rate/fund checks
quantum:permit:p33:{plan_id}       ← Intent Bridge: auto-bypass (P3.3 inactive)
quantum:permit:p26:{plan_id}       ← P2.6 Heat Gate: exposure/correlation

Intent Executor uses atomic Lua script to consume all three.
Timeout: 8 seconds polling.
```

---

## 4. REDIS — THE NERVOUS SYSTEM

### 4.1 All ~70 Streams

#### Market Domain
- `quantum:stream:exchange.normalized` — Normalized market ticks
- `quantum:stream:exchange.raw` — Raw exchange data
- `quantum:stream:market_events` — Market events
- `quantum:stream:market.klines` — Kline/candlestick data
- `quantum:stream:market.tick` — Individual ticks
- `quantum:stream:marketstate` — Market state snapshots
- `quantum:stream:features` — Computed features

#### AI Signal Domain
- `quantum:stream:ai.decision.made` — AI decisions (audit)
- `quantum:stream:ai.exit.decision` — AI exit evaluations
- `quantum:stream:ai.signal_generated` — Raw AI signals
- `quantum:stream:signal.score` — Signal scores
- `quantum:stream:trade.signal.v5` — Trade signal v5 format

#### Trade Pipeline Domain
- `quantum:stream:trade.intent` — Entry/exit intents from AI
- `quantum:stream:apply.plan` — Validated execution plans
- `quantum:stream:apply.plan.manual` — Manual operator plans
- `quantum:stream:apply.result` — Execution results
- `quantum:stream:execution.result` — Execution results (duplicate?)
- `quantum:stream:trade.execution.res` — Trade execution results (duplicate?)
- `quantum:stream:trade.closed` — Closed trade notifications

#### Exit Domain (18 streams!)
- `quantum:stream:exit.intent` — Exit intent proposals
- `quantum:stream:exit.intent.rejected` — Rejected exit intents
- `quantum:stream:exit.audit` — Exit audit trail
- `quantum:stream:exit.metrics` — Exit metrics
- `quantum:stream:exit.outcomes` — Exit outcomes
- `quantum:stream:exit.replay` — Exit replay data
- `quantum:stream:exitbrain.pnl` — Exit brain PnL tracking
- Shadow streams (12): exit.belief.shadow, exit.decision.trace.shadow, exit.eval.summary.shadow, exit.geometry.shadow, exit.hazard.shadow, exit.intent.candidate.shadow, exit.intent.validation.shadow, exit.obituary.shadow, exit.policy.shadow, exit.regime.shadow, exit.replay.eval.shadow, exit.state.shadow, exit.tuning.recommendation.shadow, exit.utility.shadow

#### Harvest Domain
- `quantum:stream:harvest.intent` — Harvest intent
- `quantum:stream:harvest.proposal` — Harvest proposals
- `quantum:stream:harvest.suggestions` — Harvest suggestions
- `quantum:stream:harvest.v2.shadow` — Harvest v2 shadow

#### Portfolio Domain
- `quantum:stream:portfolio.state` — Portfolio state
- `quantum:stream:portfolio.gate` — Portfolio gate decisions
- `quantum:stream:portfolio.exposure_updated` — Exposure updates
- `quantum:stream:portfolio.snapshot_updated` — Portfolio snapshots
- `quantum:stream:portfolio.cluster_state` — Cluster state

#### System Domain
- `quantum:stream:system.alert` — System alerts
- `quantum:stream:governor.events` — Governor audit trail
- `quantum:stream:policy.audit` — Policy audit
- `quantum:stream:policy.update` — Policy updates
- `quantum:stream:policy.updated` — Policy updated confirmations
- `quantum:stream:account.balance` — Account balance updates
- `quantum:stream:slot.alert` — Slot alerts
- `quantum:stream:reconcile.alert` — Reconcile alerts
- `quantum:stream:reconcile.close` — Reconcile close requests
- `quantum:stream:reconcile.events` — Reconcile events
- `quantum:stream:risk.events` — Risk events
- `quantum:stream:position.snapshot` — Position snapshots

#### RL Domain
- `quantum:stream:rl_rewards` — RL reward signals
- `quantum:stream:rl.stats` — RL statistics

#### Other
- `quantum:stream:allocation.decision` — Capital allocation decisions
- `quantum:stream:sizing.decided` — Position sizing decisions
- `quantum:stream:clm.intent` — CLM intents
- `quantum:stream:bsc.events` — Baseline Safety Controller events
- `quantum:stream:meta.regime` — Meta regime signals
- `quantum:stream:model.retrain` — Model retrain triggers
- `quantum:stream:apply.heat.observed` — Heat observations
- `quantum:stream:utf` — Unified Training Feed

### 4.2 Key Namespaces (~60+)

```
quantum:position          — Position state
quantum:portfolio         — Portfolio state
quantum:equity            — Equity tracking
quantum:market            — Market data cache
quantum:state             — System state
quantum:slots             — Active trading slots
quantum:ledger            — Trade ledger per symbol
quantum:governor:*        — Governor state (block, exec, last_exec)
quantum:permit:*          — Execution permits (governor, p33, p26)
quantum:rl:*              — RL state (agent, policy, calibration)
quantum:layer1-6:*        — Layer-specific state
quantum:history:ohlcv:*   — OHLCV history per symbol (30+ symbols)
quantum:exit_log:*        — Exit log per symbol
quantum:harvest:*         — Harvest state (heat, proposal, v2:state)
quantum:config            — Configuration
quantum:cfg:universe      — Universe configuration
quantum:policy            — Policy store
quantum:cooldown:*        — Cooldown timers
quantum:dag5-8:*          — DAG safety state
quantum:kelly             — Kelly criterion data
quantum:learning          — Learning state
quantum:metrics:*         — Various metrics
quantum:shadow:*          — Shadow mode state
quantum:signal            — Signal state
quantum:strategy          — Strategy state
quantum:universe          — Universe data
quantum:ticker            — Ticker data
quantum:system            — System metadata
quantum:svc:rl_trainer    — RL trainer service state
quantum:symbol:performance — Per-symbol performance
quantum:p35:*             — P3.5 decision intelligence state
quantum:paper             — Paper trading state
quantum:sandbox:*         — Sandbox/research state
quantum:feedback          — Feedback loop state
quantum:hash:*            — Dedup hashes
quantum:set:*             — Set collections
quantum:churn:*           — Churn guard state
quantum:dpo_adapter       — DPO adapter state
quantum:ops:exitbrain     — Ops exitbrain state
quantum:metricpack        — MetricPack data
quantum:test              — Test data
```

---

## 5. ARCHITECTURAL PROBLEMS

### 5.1 Critical Issues

| # | Problem | Severity | Detail |
|---|---------|----------|--------|
| 1 | **3 overlapping code trees** | CRITICAL | backend/, microservices/, ai_engine/ — no clear boundary, duplicated logic |
| 2 | **3 Python interpreters** | HIGH | System, main venv, secondary venv — dependency hell |
| 3 | **10 venvs (8 unused?)** | MEDIUM | Ghost venvs on Hetzner volume |
| 4 | **134 services, 43 running** | CRITICAL | 91 zombie service files that never run |
| 5 | **Dual execution path** | CRITICAL | execution/ (legacy) AND intent_executor/ (new) both consume trade.intent |
| 6 | **Dual close path** | HIGH | apply_layer AND intent_executor handle closes separately |
| 7 | **3 overlapping layer numberings** | HIGH | AI layers 1-5, infra P0-P3.9, pipeline layer1-6 |
| 8 | **18 exit streams** | HIGH | 12 are shadow streams — massive overengineering for one operation: CLOSE |
| 9 | **Governor writes to own input** | MEDIUM | Circular: reads apply.plan AND writes rotation-close TO apply.plan |
| 10 | **/opt/quantum ghost** | MEDIUM | 1 service runs from there, models may be read from there |
| 11 | **~1500 root files** | LOW | Fix-scripts, diagnostics, docs — tech debt museum |
| 12 | **6 frontend projects** | LOW | frontend, frontend_v3, frontend_investor, qt-agent-ui, webapp, dashboard_v4 |

### 5.2 Missing for Hedge Fund Grade

| Missing | Category | Why it matters |
|---------|----------|----------------|
| Single source of truth for positions | DATA | 5+ places to check. Reconcile is reactive. |
| Typed IPC contracts | INFRA | Redis messages are flat dicts. No validation. |
| Circuit breakers | RELIABILITY | Binance down → all services crash uncoordinated |
| Backpressure | RELIABILITY | Streams grow unbounded. No consumer lag monitoring. |
| End-to-end tracing | OBSERVABILITY | No trace_id from signal → order → fill → PnL |
| Per-trade P&L attribution | ACCOUNTING | Fund-level PnL exists, trade-level attribution incomplete |
| Multi-exchange support | CAPABILITY | Hardcoded Binance. Adding Bybit = entirely new code. |
| Config management | OPS | .env files + hardcoded Python + Redis keys + systemd env = 4 sources |
| Deployment pipeline | OPS | SSH + git pull + restart. No CI/CD, no rollback. |
| Real-time risk dashboard | UI | No live view of exposure, drawdown, VaR for investors |

---

## 6. QTOS ARCHITECTURE — The OS Model

### 6.1 Ring Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         QTOS v1.0                               │
│              Quantum Trading Operating System                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    RING 0: KERNEL                         │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐  │   │
│  │  │ Scheduler│ │  Memory  │ │ Risk     │ │ Watchdog   │  │   │
│  │  │ (Job     │ │ Manager  │ │ Kernel   │ │ (Health +  │  │   │
│  │  │  Queue)  │ │ (Redis)  │ │ (Gates)  │ │  Recovery) │  │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └────────────┘  │   │
│  │  ┌──────────────────────────────────────────────────┐    │   │
│  │  │              IPC Bus (Event Streams)              │    │   │
│  │  │         quantum:stream:* (typed contracts)       │    │   │
│  │  └──────────────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  RING 1: DRIVERS                          │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐  │   │
│  │  │ Exchange │ │ Market   │ │ Position │ │ Account    │  │   │
│  │  │ Driver   │ │ Data     │ │ Driver   │ │ Driver     │  │   │
│  │  │(Binance, │ │ Driver   │ │(Ledger,  │ │(Balance,   │  │   │
│  │  │ Bybit..) │ │(WS,REST) │ │ Recon)   │ │ Margin)    │  │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                RING 2: SYSTEM SERVICES                    │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐  │   │
│  │  │Portfolio │ │ Execution│ │ Exit     │ │ Governor   │  │   │
│  │  │ Manager  │ │ Engine   │ │ Engine   │ │ (Policy    │  │   │
│  │  │          │ │          │ │          │ │  Enforcer) │  │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └────────────┘  │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐                 │   │
│  │  │ Universe │ │ Feature  │ │ Metrics  │                 │   │
│  │  │ Manager  │ │ Store    │ │ Collector│                 │   │
│  │  └──────────┘ └──────────┘ └──────────┘                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              RING 3: USER SPACE (STRATEGIES)              │   │
│  │  ┌──────────────────────────────────────────────────┐    │   │
│  │  │              AI Brain                             │    │   │
│  │  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐  │    │   │
│  │  │  │ XGB  │ │ LGBM │ │NHiTS │ │PatchT│ │ TFT  │  │    │   │
│  │  │  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘  │    │   │
│  │  │     └────┬────┘       │        └────┬────┘      │    │   │
│  │  │          ▼            ▼             ▼           │    │   │
│  │  │     ┌─────────────────────────────────────┐     │    │   │
│  │  │     │      Meta-Agent / Arbiter           │     │    │   │
│  │  │     └─────────────────────────────────────┘     │    │   │
│  │  └──────────────────────────────────────────────────┘    │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────────────────┐     │   │
│  │  │ RL Agent │ │ CLM      │ │ Strategy Plugins     │     │   │
│  │  │(Sizing,  │ │(Continual│ │ (future strategies)  │     │   │
│  │  │ Policy)  │ │ Learning)│ │                      │     │   │
│  │  └──────────┘ └──────────┘ └──────────────────────┘     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              RING 4: SHELL (UI / API / OPS)               │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐  │   │
│  │  │ REST API │ │Dashboard │ │ CLI/Ops  │ │ Investor   │  │   │
│  │  │(FastAPI) │ │(React)   │ │ Tools    │ │ Portal     │  │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Ring Descriptions

| Ring | Responsibility | OS Analog | Current → Consolidated |
|------|---------------|-----------|------------------------|
| **Ring 0: Kernel** | Life and death. Risk, memory, IPC, scheduling, watchdog. Nothing runs without kernel. | Linux kernel | Governor + heat-gate + portfolio-gate + risk-proposal + safety-* + dag-* → **4 processes** |
| **Ring 1: Drivers** | External world communication. Exchange API, WebSocket, market data, account status. | Device drivers | price-feed + market-publisher + balance-tracker + bridges → **3 processes** |
| **Ring 2: System Services** | Core business logic. Portfolio management, order execution, exit handling, universe. | systemd services | intent-bridge + intent-executor + apply-layer + exit-* + harvest-* + portfolio-* + universe + metrics → **8 processes** |
| **Ring 3: User Space** | Strategies and AI. Can crash without taking down the system. Pluggable. | User applications | ai-engine + ensemble-predictor + rl-* + clm + marketstate → **5 processes** |
| **Ring 4: Shell** | UI, API, ops tools. Observe, command. | Terminal/GUI | backend API + dashboard + ops → **3 processes** |

**Total: 43 services → 23 processes**

### 6.3 Consolidation Table: 43 → 23

| Ring | New Process | Absorbs From |
|------|------------|--------------|
| 0 | **risk-kernel** | governor, heat-gate, portfolio-gate, risk-proposal, dag3-hw-stops, dag5-lockdown, dag8-freeze-exit |
| 0 | **scheduler** | (new — replaces cron/timer logic) |
| 0 | **watchdog** | core-health, diagnostic, stream-recover, safety-telemetry |
| 0 | **ipc-bus** | stream-bridge, execution-result-bridge, eventbus_bridge |
| 1 | **exchange-driver** | price-feed, market-publisher, execution (Binance adapter) |
| 1 | **account-driver** | balance-tracker, binance-pnl-tracker |
| 1 | **data-driver** | layer1-data-sink, layer1-historical-backfill, data-collector |
| 2 | **execution-engine** | intent-bridge, intent-executor, apply-layer |
| 2 | **exit-engine** | exit-management-agent, exit-brain-shadow, exit-intelligence, exit-intent-gateway |
| 2 | **harvest-engine** | harvest-optimizer, harvest-proposal, harvest-metrics-exporter |
| 2 | **portfolio-manager** | portfolio-clusters, portfolio-governance, portfolio-heat-gate, portfolio-state-publisher, capital-allocation |
| 2 | **universe-manager** | universe-service, utf-publisher |
| 2 | **reconcile-engine** | reconcile-engine (keep, simplify) |
| 2 | **metrics-engine** | performance-attribution, performance-tracker, metricpack-builder, p35-decision-intelligence |
| 2 | **trade-logger** | trade-logger (keep) |
| 3 | **ai-brain** | ai-engine, ai-strategy-router, ensemble-predictor |
| 3 | **rl-system** | rl-agent, rl-sizer, rl-trainer, rl-policy-publisher, rl-shadow-metrics-exporter |
| 3 | **clm** | clm (keep, as kernel scheduled job) |
| 3 | **marketstate** | marketstate, meta-regime |
| 4 | **api-gateway** | backend (FastAPI) |
| 4 | **dashboard** | 6 frontends → 1 |
| 4 | **ops-cli** | ops/ scripts → unified CLI |

### 6.4 Redis Stream Consolidation: 70 → ~25

| New Stream | Replaces | Schema |
|-----------|----------|--------|
| `qtos:market.tick` | exchange.normalized, market_events, market.klines, market.tick | `{symbol, price, volume, ts}` |
| `qtos:market.state` | marketstate, features, meta.regime | `{symbol, regime, features{}, indicators{}}` |
| `qtos:signal.proposal` | trade.intent, ai.decision.made, ai.signal_generated, signal.score, trade.signal.v5 | `{symbol, action, confidence, source, sizing, trace_id}` |
| `qtos:order.plan` | apply.plan, apply.plan.manual | `{plan_id, symbol, action, qty, permits{}, source}` |
| `qtos:order.result` | apply.result, execution.result, trade.execution.res | `{plan_id, symbol, filled_qty, price, fee, status}` |
| `qtos:trade.lifecycle` | trade.closed, exitbrain.pnl | `{symbol, open_price, close_price, pnl, duration, reason}` |
| `qtos:exit.proposal` | exit.intent, exit.intent.rejected, harvest.intent, harvest.proposal, harvest.suggestions | `{symbol, urgency, reason, source}` |
| `qtos:portfolio.state` | portfolio.state/gate/exposure/cluster/snapshot | `{positions[], equity, margin, heat, clusters{}}` |
| `qtos:risk.event` | risk.events, governor.events, system.alert, slot.alert | `{level, source, message, action}` |
| `qtos:position.truth` | position.snapshot, reconcile.* | `{positions[], source, drift_detected}` |
| `qtos:rl.feedback` | rl_rewards, rl.stats | `{trade_id, reward, features{}}` |
| `qtos:policy.change` | policy.audit/update/updated | `{name, old, new, reason}` |
| `qtos:account.state` | account.balance, allocation.decision, sizing.decided | `{balance, margin, allocated{}}` |
| `qtos:system.health` | (new — watchdog output) | `{ring, service, status, latency}` |
| `qtos:audit.log` | exit.audit, exit.metrics, exit.outcomes | Unified audit trail |
| `qtos:shadow.log` | All 12+ shadow streams | `{source, data}` |

### 6.5 Target Directory Structure

```
quantum_trader/
├── kernel/                       ← Ring 0
│   ├── scheduler/
│   ├── risk_kernel/
│   ├── watchdog/
│   ├── ipc_bus/
│   │   ├── schemas/              ← Typed message contracts (Pydantic)
│   │   └── bus.py
│   └── state/                    ← Unified state machine
│
├── drivers/                      ← Ring 1
│   ├── exchange/                 ← Binance, Bybit, etc.
│   ├── market_data/
│   ├── account/
│   └── position/
│
├── services/                     ← Ring 2
│   ├── execution/
│   ├── exit/
│   ├── portfolio/
│   ├── universe/
│   ├── reconcile/
│   └── metrics/
│
├── plugins/                      ← Ring 3
│   ├── ai_ensemble/
│   │   ├── agents/               ← XGB, LGBM, NHiTS, PatchTST, TFT
│   │   ├── meta_agent.py
│   │   └── arbiter.py
│   ├── rl_system/
│   ├── clm/
│   └── _template/                ← New strategy template
│
├── shell/                        ← Ring 4
│   ├── api/
│   ├── dashboard/
│   └── cli/
│
├── shared/                       ← Shared code
│   ├── models/                   ← Pydantic schemas
│   ├── redis_client.py
│   └── config.py
│
├── tests/
├── ops/                          ← Deploy, systemd templates
├── config/                       ← Environment configs
├── docs/architecture/            ← This document
└── archive/                      ← All legacy files (~1500)
```

---

## 7. MIGRATION PLAN — Strangler Fig Pattern

### Phase 1: STABILIZE (Week 1-3)
- Kill 91 zombie .service files
- Remove /opt/quantum (migrate last service)
- Consolidate to 1 venv + 1 Python interpreter
- Move ~1500 root files to archive/
- Result: System runs as-is, but clean

### Phase 2: KERNEL FIRST (Week 4-7)
- Build Ring 0: Scheduler + Risk Kernel + Watchdog
- IPC Bus with typed contracts (Pydantic schema validation)
- Unified state machine (quantum:state:*)
- Governor + heat-gate + portfolio-gate → Risk Kernel
- Result: Kernel controls all permits, scheduled jobs, health

### Phase 3: DRIVER LAYER (Week 8-9)
- Exchange Driver: Binance adapter with retry, rate-limit, circuit breaker
- Market Data Driver: Unified WS + REST + historical
- Position Driver: ONE source of truth for positions
- Result: External world is isolated and testable

### Phase 4: CONSOLIDATE RING 2 (Week 10-13)
- Intent Bridge + Intent Executor + Apply Layer → Execution Engine
- 5 exit services → Exit Engine
- 4 harvest services → Harvest Engine
- 6 portfolio services → Portfolio Manager
- Reconcile + trade-logger + performance → Metrics Engine
- Result: 43 → ~23 processes

### Phase 5: PLUGIN RING 3 (Week 14-16)
- AI Engine → plugin with Strategy interface
- RL Agent → plugin
- CLM → kernel scheduled job
- Result: New strategies are plug-and-play

### Phase 6: SHELL (Week 17-18)
- 6 frontends → 1 dashboard
- REST API → unified with auth
- CLI ops → standardized
- Result: One entry point to the system

---

*Document generated: 2026-03-14*
*Source: Deep system audit of VPS (46.224.116.254) + local codebase (C:\quantum_trader)*
*Repo: binyaminsemerci-ops/quantum_trader (main)*
