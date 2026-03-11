# Quantum Trader OS — Full System Overview Pack (AS-IS → TO-BE)

**Document class:** System Architecture / Forensic Map  
**Date:** 2026-03-11  
**Grounded in:** VPS_FORENSIC_AUDIT_2026-03-03.md · SYSTEM_ARCHITECTURE_COMPLETE_FEB23_2026.md · OWNERSHIP_MAP.md · live source code reads  
**Primary goal:** Provide a single factual reference before placing any new Exit Brain / advanced-math modules, so that no write-path duplication, split-brain condition, or ownership ambiguity is introduced.

---

## SECTION 1 — EXECUTIVE CONCLUSION

### 1.1 System identity

Quantum Trader OS is a microservices-based algorithmic trading platform deployed on a single Hetzner VPS (`46.224.116.254`, hostname `quantumtrader-prod-1`). It manages the full lifecycle of Binance Futures trades: market data ingestion → feature engineering → ensemble AI signal generation → multi-gate governance → order execution → position monitoring → intelligent exit → PnL feedback → RL adaptation.

The codebase lives at `/home/qt/quantum_trader` (user `qt`). 97 systemd units are loaded; ~80 are active. The system uses Redis Streams exclusively as the event bus. All inter-service communication is either a Redis Stream XADD/XREADGROUP or a direct HTTP call to a named internal port.

### 1.2 What works right now (2026-03-11)

| Component | Status |
|-----------|--------|
| Market data ingestion (Binance WS → market.klines) | ✅ WORKS |
| AI Engine 6-model ensemble (XGB+LGBM+NHiTS+PatchTST+TFT+DLinear) | ✅ WORKS — all 6 activated |
| Trade intent generation (ai-engine → trade.intent) | ✅ WORKS — 10,004+ intents queued |
| Intent bridge (trade.intent → apply.plan, lag=0) | ✅ WORKS |
| Governor / rate-limiting (P3.2) | ✅ WORKS |
| Harvest brain (shadow mode) | ✅ WORKS in shadow |
| RL services (rl-agent, rl-trainer, feedback-v2) | ✅ RUNNING (but gated out) |
| Observability stack (Prometheus/Grafana/Loki) | ✅ WORKS |
| Backend API + Frontend | ✅ WORKS |

### 1.3 What is broken right now (2026-03-11)

| Component | Broken State |
|-----------|-------------|
| Execution consumer group | 🚨 STUCK since ~Feb 16; 10,004 trade intents not consumed |
| ExitBrain v3.5 | 🚨 DISABLED — broken `shadow.conf` drop-in syntax |
| Position State Brain | 🔴 CRASH LOOP 6568× (missing Binance credentials, wrong user/python) |
| Exposure Balancer | ❌ DEAD since Feb 21 (margin guardrails not enforced) |
| BSC (Baseline Safety Controller) | ❌ DEAD since Feb 21 |
| SL/TP SELL-side math | 🚨 INVERTED for SELL orders in ai_engine/main.py |
| RL gate | ⚠️ `rl_gate_pass=false` on all signals (kill-switch active) |
| Market regime detection | ⚠️ Always "unknown" (key mismatch between writer/reader) |
| Redis stability | ⚠️ Restarted ~19h before audit — caused execution consumer cascade |

### 1.4 Critical message for new module placement

> **Before placing any new Exit Brain / ensemble modules:** The current primary exit path is broken (ExitBrain v3.5 DISABLED, execution stuck). Any new module that writes exit intents MUST route through `quantum:stream:exit.intent` → ExitIntentGateway (PATCH-5B) → `quantum:stream:trade.intent`. It must NOT write directly to `quantum:stream:apply.plan` (unconditionally forbidden by gateway write-guard). It must NOT duplicate the existing `exit_manager.py` AI eval call. The harvest path (`harvest.intent` → intent-executor) is a separate well-defined path for partial reduces. These lanes are distinct and must remain so.

---

## SECTION 2 — AS-IS COMPONENT MAP

### 2.1 Infrastructure substrate

| Resource | Value |
|----------|-------|
| VPS provider | Hetzner Cloud |
| IP | 46.224.116.254 |
| OS | Ubuntu, Linux 6.8.0-90-generic |
| VPS uptime at audit | 42 days (since ~Jan 19 2026) |
| Load average (audit) | 3.28 / 2.66 / 2.16 (HIGH) |
| Running user | `qt` (most services); `root` (position-state-brain — bug) |
| Repo path | `/home/qt/quantum_trader` |
| AI engine venv | `/home/qt/quantum_trader_venv/` (actual runtime) |
| Ops client venv | `/opt/quantum/venvs/ai-client-base/bin/python` |
| Env files | `/etc/quantum/*.env` |
| Redis | `127.0.0.1:6379` db=0 (live), db=1 (backtest) — Redis 7.0.15 |
| Total Redis keys | 298,029 |
| Service count | 137 unit files installed; 97 loaded; ~80 running |
| Reverse proxy | nginx on 80/443 → routes to internal ports |
| Observability | Prometheus (9091) + Grafana (3000) + Loki (3100) + Promtail + Node/Redis exporter |


### 2.2 Layer-by-layer service map

#### L0 — Data Ingestion

| Service unit | ExecStart | Port | Status |
|--------------|-----------|------|--------|
| `quantum-market-publisher` | `microservices/market_publisher/main.py` | — | ✅ active |
| `quantum-multi-source-feed` | `microservices/multi_source_feed/main.py` | 8048+ | ✅ active |
| `quantum-cross-exchange-aggregator` | `microservices/cross_exchange/main.py` | — | ✅ active |
| `quantum-price-feed` | `services/price_feed.py` | — | ✅ active |
| `quantum-universe-service` | `microservices/universe_service/main.py` | 8006 (not listening!) | ✅ active |
| `quantum-exchange-stream-bridge` | — | — | ✅ active |
| `quantum-feature-publisher` | `microservices/feature_publisher/main.py` | — | ✅ active |
| `quantum-utf-publisher` | — | — | ✅ active |

#### L2/L3 — AI Engine & Brains

| Service unit | ExecStart | Port | Status |
|--------------|-----------|------|--------|
| `quantum-ai-engine` | `uvicorn microservices.ai_engine.main:app --host 127.0.0.1 --port 8001` | 8001 | ✅ PID 1732052 |
| `quantum-ceo-brain` | `uvicorn backend.ai_orchestrator.service:app` | 8010 | ✅ active |
| `quantum-strategy-brain` | `uvicorn backend.strategy_brain.service:app` | 8011 | ✅ active |
| `quantum-risk-brain` | `uvicorn backend.risk_brain.service:app` | 8012 | ✅ active |
| `quantum-meta-regime` | `microservices/meta_regime/main.py` | — | ✅ active (but output not read) |
| `quantum-ensemble-predictor` | `microservices/ensemble_predictor/main.py` | — | ✅ active |
| `quantum-p35-decision-intelligence` | `microservices/p35_decision_intelligence/main.py` | — | ✅ active |
| `quantum-ai-strategy-router` | — | — | ✅ active |
| `quantum-signal-injector` | — | — | ✅ active |
| `quantum-marketstate` | — | — | ✅ active |

#### L4 — Risk & Governance

| Service unit | ExecStart | Port | Status |
|--------------|-----------|------|--------|
| `quantum-portfolio-governance` | — | — | ✅ active |
| `quantum-portfolio-intelligence` | — | 8004 | ✅ active |
| `quantum-portfolio-risk-governor` | — | — | ✅ active |
| `quantum-portfolio-state-publisher` | — | — | ✅ active |
| `quantum-portfolio-gate` | — | — | ✅ active |
| `quantum-portfolio-heat-gate` | — | — | ✅ active |
| `quantum-portfolio-clusters` | — | — | ✅ active |
| `quantum-risk-safety` | — | — | ✅ active |
| `quantum-risk-brake` | — | — | ✅ active |
| `quantum-risk-proposal` | — | — | ✅ active |
| `quantum-heat-gate` | — | — | ✅ active |
| `quantum-capital-allocation` | — | — | ✅ active |
| `quantum-anti-churn-guard` | — | — | ✅ active |
| `quantum-exposure_balancer` | — | — | ❌ DEAD since Feb 21 |
| `quantum-bsc` | — | — | ❌ DEAD since Feb 21 |
| `quantum-dag3-hw-stops` | — | — | ✅ active |
| `quantum-dag4-deadlock-guard` | — | — | ✅ active |
| `quantum-dag5-lockdown-guard` | — | — | ✅ active |
| `quantum-dag8-freeze-exit` | — | — | ✅ active |

#### L5 — Strategy & Decision

| Service unit | ExecStart | Port | Status |
|--------------|-----------|------|--------|
| `quantum-autonomous-trader` | `python -u microservices/autonomous_trader/autonomous_trader.py` | — | ✅ active |
| `quantum-paper-trade-controller` | — | — | ✅ active |
| `quantum-shadow-mode-controller` | — | — | ✅ active |
| `quantum-strategic-memory` | — | — | ✅ active |
| `quantum-scaling-orchestrator` | — | — | ✅ active |

#### L6 — Execution & Exit (CRITICAL — PARTIALLY BROKEN)

| Service unit | ExecStart | Port | Status |
|--------------|-----------|------|--------|
| `quantum-execution` | `python3 services/execution_service.py` | 8002 | ✅ running; consumer group STUCK |
| `quantum-intent-bridge` | — | — | ✅ active, lag=0 |
| `quantum-intent-executor` | `microservices/intent_executor/main.py` | — | ✅ active |
| `quantum-apply-layer` | — | — | ✅ active |
| `quantum-governor` | `microservices/governor/main.py` | 8044 | ✅ active |
| `quantum-exit-monitor` | `python3 services/exit_monitor_service_v2.py` | 8007 | ✅ active (watchdog only) |
| `quantum-exit-intelligence` | — | — | ✅ active |
| `quantum-emergency-exit-worker` | — | — | ✅ active |
| `quantum-reconcile-engine` | — | — | ✅ active |
| `quantum-execution-result-bridge` | — | — | ✅ active |
| **`quantum-exitbrain-v35`** | `python3 microservices/position_monitor/main_exitbrain.py` | — | **❌ DISABLED** |
| **`quantum-position-state-brain`** | `/usr/bin/python3 microservices/position_state_brain/main.py` | — | **🔴 CRASH LOOP 6568×** |

#### L6b — Harvest Path

| Service unit | ExecStart | Port | Status |
|--------------|-----------|------|--------|
| `quantum-harvest-brain` | `microservices/harvest_brain/harvest_brain.py` | — | ✅ active (shadow mode) |
| `quantum-harvest-brain-2` | — | — | ✅ active |
| `quantum-harvest-v2` | — | — | ✅ active (shadow mode) |
| `quantum-harvest-optimizer` | — | — | ✅ active |
| `quantum-harvest-proposal` | — | — | ✅ active |
| `quantum-harvest-metrics-exporter` | — | — | ✅ active |

#### L7 — Analytics

| Service unit | Port | Status |
|--------------|------|--------|
| `quantum-portfolio-intelligence` | 8004 | ✅ active |
| `quantum-pnl-reconciler` | — | ✅ active |
| `quantum-performance-attribution` | — | ✅ active |
| `quantum-performance-tracker` | — | ✅ active |
| `quantum-balance-tracker` | — | ✅ active |
| `quantum-trade-logger` | — | ✅ active |
| `quantum-layer6-post-trade` | — | ✅ active |
| `quantum-runtime-truth` | — | ✅ active |
| `quantum-safety-telemetry` | — | ✅ active |
| `quantum-training-worker` | — | ❌ inactive (oneshot, dead after run) |

#### L8 — Reinforcement Learning

| Service unit | Status |
|--------------|--------|
| `quantum-rl-agent` | ✅ active (gated out) |
| `quantum-rl-trainer` | ✅ active (gated out) |
| `quantum-rl-sizer` | ✅ active (gated out) |
| `quantum-rl-feedback-v2` | ✅ active (gated out) |
| `quantum-rl-monitor` | ✅ active |
| `quantum-rl-shadow-metrics-exporter` | ✅ active |

#### L9 — Data Research / Backtest

| Service unit | Status |
|--------------|--------|
| `quantum-layer1-data-sink` | ✅ active |
| `quantum-layer1-historical-backfill` | ✅ exited (oneshot complete — expected) |
| `quantum-layer2-research-sandbox` | ✅ active |
| `quantum-layer2-signal-promoter` | ✅ active |
| `quantum-layer3-backtest-runner` | ✅ active |
| `quantum-metricpack-builder` | ✅ active |
| `quantum-stream-bridge` | ✅ active |


### 2.3 Ensemble AI Engine — current model state (2026-03-11)

| Model | Weight | Params / Notes | Status |
|-------|--------|----------------|--------|
| XGBoost (XGB) | 17% | 18 features, meta.json fixed Mar 11 | ✅ ACTIVATED |
| LightGBM (LGBM) | 17% | 49 features | ✅ ACTIVATED |
| N-HiTS | 17% | 238,119 params | ✅ ACTIVATED |
| PatchTST | 17% | 272,003 params | ✅ ACTIVATED |
| TFT (Temporal Fusion Transformer) | 16% | — | ✅ ACTIVATED |
| DLinear | 16% | 1,638,915 params | ✅ ACTIVATED |
| MetaAgent V2 | override | disabled by env | ❌ disabled |
| ArbiterAgent | override | disabled by env | ❌ disabled |

Weights can be overridden by `/app/data/model_supervisor_weights.json` (refreshed every 5 min). Default = equal weights across 6 models.

---

## SECTION 3 — AS-IS OWNERSHIP MAP

### 3.1 Who owns what — per concern

| Concern | Owner (current) | Notes |
|---------|----------------|-------|
| Market data ingestion | `quantum-market-publisher` | Writes `market.klines` |
| OHLCV feature engineering | `quantum-feature-publisher` | Writes `features` stream |
| Entry signal generation | `quantum-ai-engine` | Reads `market.klines`; writes `trade.intent` |
| Entry decision (autonomous) | `quantum-autonomous-trader` / `entry_scanner.py` | Parallel to ai-engine signals |
| Market regime classification | `quantum-meta-regime` | Writes `quantum:regime:<symbol>` key (NOT BEING READ per audit) |
| Portfolio policy | `quantum-portfolio-governance` | Writes `quantum:governance:policy` |
| Trade intent queuing | Redis stream `quantum:stream:trade.intent` | Central bus |
| Intent routing (bridge) | `quantum-intent-bridge` | trade.intent → apply.plan |
| Intent filtering / rate-limit | `quantum-governor` (P3.2) | apply.plan → permit keys |
| P3.3 permit issuance | `quantum-intent-executor` | Reads apply.plan, issues `quantum:permit:p33:<id>` |
| Binance order placement (entries) | `quantum-execution` + `quantum-intent-executor` | intent-executor: testnet; execution: testnet + mainnet |
| Position state truth | `quantum-position-state-brain` | ❌ BROKEN — crash loop |
| Intelligent TP/SL decisions | `quantum-exitbrain-v35` (ExitBrainV35) | ❌ DISABLED |
| Autonomous exit (fallback) | `quantum-autonomous-trader` / `exit_manager.py` | Calls AI Engine exit evaluator at port 8001 |
| Exit watchdog / heartbeat guard | `quantum-exit-monitor` | Monitors exitbrain heartbeat; triggers panic_close |
| Harvest (partial closes) | `quantum-harvest-brain` | Shadow mode; writes `harvest.intent` |
| Harvest optimization | `quantum-harvest-optimizer` | Advisory; writes `hover.intent` |
| Emergency all-close | `quantum-emergency-exit-worker` | Reads `system:panic_close` |
| RL policy per symbol | `quantum-rl-trainer` | Writes `quantum:rl:policy:<symbol>` |
| RL signal injection | `quantum-rl-agent` | Provides sizing adjustments (gated) |
| PnL reconciliation | `quantum-pnl-reconciler` | Writes `rl_rewards` stream |
| Capital allocation | `quantum-capital-allocation` | Writes `sizing.decided` |
| DAG 8 gate decisions | C3 (FnG) + C5 (manual) | Block live trading when FnG < 20 |

### 3.2 Exit path ownership — current vs intended

| Question | AS-IS answer |
|----------|-------------|
| Who decides to close a position? | `exit_manager.py` (calls ai-engine exit eval) × `exit-monitor` watchdog. ExitBrain v3.5 is DEAD. |
| How does an exit intent reach Binance? | exit_manager → writes to `quantum:stream:trade.intent` (not confirmed) OR via `quantum:stream:exit.intent` → ExitIntentGateway (PATCH-5B) → trade.intent → execution |
| Who decides partial close amounts? | `harvest_brain.py` (R-based ladder: 0.5R→25%, 1.0R→25%, 1.5R→25%) in shadow mode |
| Who enforces reduceOnly? | `intent_executor` (flag: `reduceOnly=true`) |
| Who controls exit rate limits? | `quantum-governor` (MAX_EXEC_PER_HOUR=3, MAX_EXEC_PER_5MIN=2) |
| Who owns the final position state? | `quantum-position-state-brain` (❌ BROKEN); Redis keys `quantum:position:<symbol>` |
| What is the panic path? | `system:panic_close` → `quantum-emergency-exit-worker` — 0 events in stream ✅ |

---

## SECTION 4 — AS-IS STREAM / KEY TOPOLOGY

### 4.1 Primary event streams

| Stream key | XLEN (Mar 3) | Produced by | Consumed by | Status |
|------------|-------------|-------------|-------------|--------|
| `quantum:stream:market.klines` | active | market-publisher | ai-engine (XREAD) | ✅ |
| `quantum:stream:exchange.raw` | — | price-feed | cross-exchange | ✅ |
| `quantum:stream:exchange.normalized` | — | cross-exchange-aggregator | ai-engine, strategy | ✅ |
| `quantum:stream:market_events` | — | multi-source-feed | ai-engine | ✅ |
| `quantum:stream:features` | — | feature-publisher | ai-engine | ✅ |
| `quantum:stream:ai.signal_generated` | — | ai-engine | decision-intelligence | present |
| `quantum:stream:ai.decision.made` | — | decision-intelligence | execution | present |
| **`quantum:stream:trade.intent`** | **10,004** | ai-engine, autonomous-trader | execution (STUCK), intent-bridge (✅) | ⚠️ CRITICAL |
| `quantum:stream:trade.execution.res` | 2,154 | execution-service | autonomous-trader, analytics | ❌ last Mar Feb 9 |
| `quantum:stream:trade.closed` | 1,006 | execution-service | analytics, RL | present |
| `quantum:stream:trade.signal.v5` | 500 | chaos_load_test | strategy | test data only |
| `quantum:stream:apply.plan` | — | intent-bridge, apply-layer | intent-executor, governor | ✅ |
| `quantum:stream:apply.plan.manual` | — | ops/manual | intent-executor (manual group) | ✅ |
| `quantum:stream:apply.result` | — | intent-executor | harvest-brain, analytics | ✅ |
| `quantum:stream:apply.heat.observed` | — | heat-gate | apply-layer | ✅ |
| `quantum:stream:ai.exit.decision` | — | exitbrain-v3.5 | exit-monitor | ❌ (exitbrain DEAD) |
| `quantum:stream:exit.intent` | — | (new modules will write here) | exit-intent-gateway | vacant |
| `quantum:stream:exit.intent.rejected` | — | exit-intent-gateway | audit | — |
| `quantum:stream:harvest.proposal` | — | harvest-proposal | harvest-brain | ✅ |
| `quantum:stream:harvest.intent` | — | harvest-brain | execution/intent-executor | ✅ (shadow) |
| `quantum:stream:hover.intent` | — | harvest-brain | harvest-optimizer | ✅ |
| `quantum:stream:harvest.suggestions` | — | harvest-brain | advisory | ✅ |
| `quantum:stream:portfolio.state` | — | portfolio-intelligence | governance, UI | ✅ |
| `quantum:stream:portfolio.snapshot_updated` | — | portfolio-intelligence | multiple | ✅ |
| `quantum:stream:portfolio.exposure_updated` | — | exposure-balancer | governance | ❌ (balancer DEAD) |
| `quantum:stream:portfolio.cluster_state` | — | portfolio-clusters | governance | ✅ |
| `quantum:stream:portfolio.gate` | — | portfolio-gate | execution | ✅ |
| `quantum:stream:risk.events` | — | risk-kernel | governance, watchdog | ✅ |
| `quantum:stream:rl_rewards` | — | pnl-reconciler | rl-trainer | ✅ |
| `quantum:stream:rl.stats` | — | rl-agent | rl-monitor | ✅ |
| `quantum:stream:sizing.decided` | — | rl-sizer / capital-alloc | execution | ✅ |
| `quantum:stream:policy.update` | — | portfolio-governance | strategy-ops | ✅ |
| `quantum:stream:policy.updated` | — | portfolio-governance | multiple | ✅ |
| `quantum:stream:policy.audit` | — | portfolio-governance | audit-logger | ✅ |
| `quantum:stream:model.retrain` | — | strategic-evolution | training-worker | ✅ |
| `quantum:stream:reconcile.close` | — | reconcile-engine | execution | ✅ |
| `quantum:stream:reconcile.alert` | — | reconcile-engine | ops/monitoring | ✅ |
| `quantum:stream:bsc.events` | — | bsc | governance | ❌ (bsc DEAD) |
| `system:panic_close` | **0** | risk-kernel/exit-monitor/ops | emergency-exit-worker | ✅ (no events) |
| `system:panic_close:completed` | — | emergency-exit-worker | audit-logger | — |

### 4.2 Consumer groups on `quantum:stream:trade.intent`

| Group name | Consumers | Pending | Last Delivered ID | Lag | Status |
|-----------|-----------|---------|-------------------|-----|--------|
| `quantum:group:execution:trade.intent` | 1 | 0 | `1771160145484-0` (~Feb 16) | NULL 🚨 | STUCK |
| `quantum:group:intent_bridge` | 30 | 4 | `1772501750890-0` (Mar 3) | 0 | ✅ |

### 4.3 Key Redis state records

| Key pattern | Written by | Read by | Description |
|------------|-----------|---------|-------------|
| `quantum:governance:policy` | portfolio-governance | ai-engine, strategy-ops | Active policy document |
| `quantum:governance:last_decision` | portfolio-governance | monitoring | Last decision timestamp |
| `quantum:position:<symbol>` | position-state-brain (BROKEN) | autonomous-trader | Per-symbol position truth |
| `quantum:rl:state:<position_id>` | rl-agent | rl-trainer, autonomous-trader | Per-position RL state |
| `quantum:rl:policy:<symbol>` | rl-trainer | ai-engine, rl-sizer | Per-symbol RL policy |
| `quantum:rl:calibration:v1:<symbol>` | rl-calibrator | rl-sizer | Calibration weights |
| `quantum:regime:<symbol>` | meta-regime | ai-engine (NOT READING) | Market regime classification |
| `quantum:metrics:exit:*` | exit-intelligence | monitoring | 298k+ exit metric records |
| `quantum:permit:p33:<id>` | intent-executor | position-state-brain | Trade permits |
| `quantum:manual_lane:enabled` | ops | intent-executor | Manual lane TTL key |
| `quantum:intent_bridge:seen:<id>` | intent-bridge | intent-bridge | Dedup cache (TTL) |
| `quantum:intent_executor:done:<id>` | intent-executor | — | Completed intent dedup |
| `quantum:svc:rl_trainer:heartbeat` | rl-trainer | rl-monitor | Service heartbeat |
| `quantum:svc:rl_feedback_v2:heartbeat` | rl-feedback-v2 | rl-monitor | Service heartbeat |

### 4.4 Active service ports

| Port | Bind | Service | Note |
|------|------|---------|------|
| 80/443 | 0.0.0.0 | nginx | Reverse proxy |
| 3000 | * | Grafana | Observability |
| 3100/9095 | * | Loki | Log aggregation |
| 6379 | 127.0.0.1 | Redis | ✅ not public |
| 8000 | 0.0.0.0 | Backend (run.py) | Dashboard API |
| 8001 | 127.0.0.1 | AI Engine | ✅ not public |
| 8002 | 0.0.0.0 | Execution service | ⚠️ public |
| 8004 | 127.0.0.1 | Portfolio Intelligence | ✅ not public |
| 8007 | 0.0.0.0 | Exit Monitor | ⚠️ public |
| 8010 | 127.0.0.1 | CEO Brain | ✅ not public |
| 8011 | 127.0.0.1 | Strategy Brain | ✅ not public |
| 8012 | 127.0.0.1 | Risk Brain | ✅ not public |
| 8044 | 0.0.0.0 | Governor (Prometheus metrics) | undocumented |
| 8042–8069 | 0.0.0.0 | 12+ unidentified python3 processes | ⚠️ undocumented |
| 9091 | * | Prometheus | Observability |
| 9100 | * | Node Exporter | Observability |
| 9121 | * | Redis Exporter | Observability |

---

## SECTION 5 — AS-IS DATAFLOW / DECISION FLOW

### 5.1 Entry flow (intent generation → order)

```
[Binance WebSocket]
    │
    ▼
quantum:stream:market.klines                  (market-publisher writes)
    │
    ▼
[quantum-ai-engine :8001]                     (reads via XREAD)
    │   6-model ensemble: XGB+LGBM+NHiTS+PatchTST+TFT+DLinear
    │   Applies: risk-safety limits, governance policy
    │   Injects: rl_gate_pass (FALSE), regime (unknown), dai_leverage
    │
    ▼
quantum:stream:trade.intent                   (ai-engine writes, 10,004 queued)
    │
    ├──► [quantum-intent-bridge]              (group: quantum:group:intent_bridge)
    │        lag=0, 30 consumers
    │        │
    │        ▼
    │    quantum:stream:apply.plan            (intent-bridge writes)
    │        │
    │        ▼
    │    [quantum-governor P3.2]              (reads apply.plan)
    │        rate-limit: 3/hr, 2/5min
    │        capital checks, kill-score gates
    │        │
    │        ▼
    │    quantum:permit:p33:<plan_id>         (governor writes permit keys)
    │        │
    │        ▼
    │    [quantum-intent-executor P3.3]       (reads apply.plan + waits for permit)
    │        permit timeout: 8s
    │        source allowlist: {intent_bridge, apply_layer}
    │        │
    │        ▼
    │    Binance Testnet API (HTTPS)          (reduceOnly=false for entries)
    │
    └──► [quantum-group:execution:trade.intent]    🚨 STUCK since Feb 16
             quantum-execution service (port 8002)
             Last execution: Feb 9 2026 (REJECTED)
```

### 5.2 Parallel autonomous entry path

```
[quantum-autonomous-trader]
    │   entry_scanner.py scans for opportunities
    │   RL sizing via RLPositionSizingAgent (model: rl_sizing_agent_v3.pth)
    │   Reads open positions: SCAN quantum:position:* keys
    │
    ▼
quantum:stream:trade.intent (or quantum:stream:shadow.intent if STREAM_ENTRY_INTENT=shadow)
    │
    └──► (same downstream flow as ai-engine entries above)
```

### 5.3 Exit flow (current — BROKEN)

```
[quantum-autonomous-trader / exit_manager.py]
    │   Scans positions each cycle
    │   Priority: SL check → TP check → funding bleed → R < -1.5 emergency
    │   For R > 0.5 or -1.5 ≤ R ≤ 0.5: calls AI Engine exit evaluator
    │       POST http://127.0.0.1:8001/exit/evaluate
    │
    ▼
ExitDecision{action: CLOSE/PARTIAL_CLOSE/HOLD}
    │
    ▼ (path not fully confirmed — likely writes to trade.intent or exit.intent)
    ?

[quantum-exit-intelligence]   (status: running, activity unknown)
    │
    ▼
    ?

[quantum-exitbrain-v35]       ❌ DISABLED — NOT RUNNING
    │ Would publish ExitPlan to quantum:stream:ai.exit.decision
    │ Would use ILFv2, AdaptiveLeverageEngine, formula-based TP/SL
    ▼
    ✗ (dead)

[quantum-exit-monitor]        ✅ RUNNING as watchdog only
    │   Monitors quantum:stream:exit_brain:heartbeat
    │   If heartbeat stale → triggers system:panic_close
    │
    ▼
system:panic_close → [quantum-emergency-exit-worker]  (close all positions)
```

### 5.4 Harvest / partial close flow

```
[quantum-harvest-brain]
    │   Reads quantum:stream:apply.result (execution fills)
    │   Tracks position from fills (independently of position-state-brain)
    │   Applies R-based ladder: 0.5R→close 25%, 1.0R→close 25%, 1.5R→close 25%
    │   Mode: SHADOW (does not execute live unless HARVEST_MODE=live)
    │
    ├──► quantum:stream:harvest.intent     (shadow: advisory only, live: executed)
    │       consumer: intent-executor (HARVEST_GROUP)
    │
    └──► quantum:stream:harvest.suggestions (advisory, no execution)
```

### 5.5 Exit intent gateway path (PATCH-5B)

```
[Any exit signal producer]
    │
    ▼
quantum:stream:exit.intent
    │
    ▼
[quantum-exit-intent-gateway PATCH-5B]
    9 validation checks:
    1. testnet_mode must be "true"
    2. EXIT_GATEWAY_ENABLED must be true
    3. Not in lockdown
    4. Message not stale (< stale_sec)
    5. Not a duplicate (dedup_ttl)
    6. Not in cooldown for symbol
    7. Action is valid (CLOSE/PARTIAL)
    8. Source is allowlisted
    9. Rate limit not exceeded
    │
    ├──► PASS → quantum:stream:trade.intent
    │           (then flows to execution/intent-executor)
    │
    └──► FAIL → quantum:stream:exit.intent.rejected
                 (audit log only)

NOTE: write to quantum:stream:apply.plan is UNCONDITIONALLY FORBIDDEN in redis_io
```

### 5.6 RL / regime data flows (gated)

```
[quantum-rl-trainer] → quantum:rl:policy:<symbol>
[quantum-rl-agent]   → annotates trade intents with rl_* fields
                        BUT: rl_gate_pass=false (kill-switch in ai-engine.env)
                        Effect: rl_weight_effective=0.0 on all signals

[quantum-meta-regime] → writes quantum:regime:<symbol>
                         BUT: key mismatch — ai-engine reads different key
                         Effect: regime="unknown" on all signals
```

### 5.7 DAG 8 gate status

| Gate | Check | Status (Mar 3) |
|------|-------|---------------|
| C1 | Max Drawdown < 28% | ✅ 5.03% |
| C2 | Backtest quality | ✅ |
| C3 | Fear & Greed Index > 20 | ❌ FnG=9 (blocks live trading) |
| C4 | Win streak | ✅ |
| C5 | Manual approval | ✅ |

> C3 (FnG < 20) is actively blocking live trade execution. This is a macro condition, not a code bug.

---

## SECTION 6 — CURRENT PAIN POINTS AND FAILURE MODES

### 6.1 Breakpoint matrix (ordered by severity)

| # | Severity | Layer | Component | Description |
|---|----------|-------|-----------|-------------|
| 1 | 🚨 CRITICAL | L3 AI Engine | `ai_engine/main.py` | **SL/TP SELL-side math inverted.** For SELL orders: SL calculates as `entry × (1 − pct)` (BUY formula). Should be `entry × (1 + pct)`. Result: stop_loss=$96M, take_profit=$-192M on BTCUSDT SELL. Every SELL signal has wrong TP/SL. |
| 2 | 🚨 CRITICAL | L6 Execution | `quantum-execution` | **Execution consumer group stuck ~15 days.** `quantum:group:execution:trade.intent` last-delivered-id = Feb 16. 10,004 trade intents unprocessed. Root cause: Redis restart on Mar 2 dropped blocking XREADGROUP, consumer did not reconnect. |
| 3 | 🚨 CRITICAL | L6 Exit | `quantum-exitbrain-v35` | **ExitBrain v3.5 DISABLED.** Broken drop-in at `/etc/systemd/system/quantum-exitbrain-v35.service.d/shadow.conf`: duplicate `[Service]` header + backslash `\EXIT_SHADOW_MODE=true\`. Zero intelligent exit decisions being made. |
| 4 | 🔴 HIGH | L6 Exit | `quantum-position-state-brain` | **Crash loop 6568×.** Error: "Missing Binance credentials". Unit bugs: running as `root`, using `/usr/bin/python3` (system Python). Wasting ~6 restarts/min of CPU. |
| 5 | 🔴 HIGH | L0 Infra | Redis | **Redis restarted ~19 hours before audit** (Mar 2 06:37 UTC vs VPS uptime 42 days). Caused execution consumer cascade. Unknown root cause (OOM? operator action?). |
| 6 | 🔴 HIGH | L4 Risk | `quantum-exposure_balancer` + `quantum-bsc` | **Both DEAD since Feb 21** (exact same second: 06:34:26). Margin guardrails (85% util, 15% max symbol) not enforced. Safety controller not running. |
| 7 | 🔴 HIGH | L3 AI | SL/TP correctness | **Consequence of #1:** All SELL-side trade intents in the 10,004-entry backlog have invalid TP/SL values. If execution is unstuck, SELL orders would fire with catastrophically wrong stops. |
| 8 | ⚠️ MEDIUM | L8 RL | All RL services | **RL fully gated.** `rl_gate_pass=false`, `rl_weight_effective=0.0` on every signal. Services running but kill-switch (`RL_ENABLED=false` in ai-engine.env) prevents any influence. |
| 9 | ⚠️ MEDIUM | L3/L5 | `quantum-meta-regime` | **Market regime always "unknown".** Meta-regime service writes to one key; ai-engine reads a different key. No regime-adaptive strategy active. |
| 10 | ⚠️ MEDIUM | L0 NGINX | nginx sites-enabled | **Config pollution.** 6 files all declare `server_name app.quantumfond.com`. 10 warnings per `nginx -t`. No functional impact but increases fragility. |
| 11 | ⚠️ MEDIUM | L0 Infra | Process sprawl | **20+ python3 processes on public ports** (0.0.0.0 bind) — 8002, 8007, 8042–8069, 9092. None proxied through nginx/auth. Undocumented. System load 3.28 (HIGH). |
| 12 | ⚠️ LOW | L6 Exit | Exit path ambiguity | **Who writes exit intents?** `exit_manager.py` calls AI Engine evaluator but the write path back to a stream is not explicit in code. ExitIntentGateway (PATCH-5B) expects writes to `quantum:stream:exit.intent` but unclear if any producer is writing there. |

### 6.2 Structural / architectural issues

1. **Split position state:** Position truth should come from `quantum-position-state-brain` (broken), but `harvest_brain` and `autonomous_trader` independently derive position state from fills/scans. Three sources of truth.

2. **Duplicate exit paths:** ExitBrain v3.5 writes `ai.exit.decision` stream. exit_manager.py calls AI Engine REST. ExitIntentGateway routes `exit.intent`. These are three separate exit mechanisms with no explicit handoff protocol — whichever is active "wins."

3. **Harvest mode mismatch:** `harvest_brain.py` defaults to `HARVEST_MODE=shadow`. The harvest path exists but is advisory-only. Live partial closes require ENV override and are not flowing through the standard exit governance path.

4. **Consensus degradation:** The live signal sample (Mar 3) shows `consensus_count=1` out of 4 models. A `fallback` rule fires due to `consensus_not_met` triggered by a `testnet_hash_pattern` override. This suggests the consensus system is working correctly, but the signal is genuinely low quality (1/4 agreement).

5. **Position-state-brain as P3.3 dependency:** Intent-executor reads P3.3 permits and position-state-brain writes them — but PSB is crash-looping. All permit logic may be degraded.

---

## SECTION 7 — TO-BE TARGET ARCHITECTURE

### 7.1 Philosophy and objectives

The TO-BE architecture must:
1. **Fix all 10 breakpoints** in Section 6 before adding any new modules
2. **Establish a single authoritative position state** (one writer, many readers)
3. **Provide a clean, guarded exit lane** for the new Exit Brain math stack
4. **Enforce write-path ownership** — each stream has exactly one class of producer per use-case
5. **Route all exit decisions through ExitIntentGateway** (PATCH-5B already implements this correctly)
6. **Keep RL, ensemble, and governance paths strictly separate** from the exit math stack

### 7.2 Required pre-conditions before any TO-BE work

These seven items MUST be resolved in order. They are not improvements — they are prerequisites.

| Order | Fix | Target file / command |
|-------|-----|----------------------|
| 1 | Fix SL/TP SELL-side math | `microservices/ai_engine/main.py` — SELL: `sl = entry * (1 + sl_pct/100)`, `tp = entry * (1 - tp_pct/100)` |
| 2 | Fix execution consumer group offset | `redis-cli XGROUP SETID quantum:stream:trade.intent quantum:group:execution:trade.intent '$'` → restart quantum-execution |
| 3 | Fix ExitBrain v3.5 drop-in | Remove duplicate `[Service]`, fix `Environment=EXIT_SHADOW_MODE=true` (no backslash) → `systemctl enable --now quantum-exitbrain-v35` |
| 4 | Fix position-state-brain | Add Binance creds to `/etc/quantum/position-state-brain.env`; fix `User=qt`; fix ExecStart to venv python |
| 5 | Fix regime key mismatch | Align key written by meta-regime with key read by ai-engine |
| 6 | Restart exposure_balancer + bsc | Debug crash cause (likely import error); fix; restart |
| 7 | Clean nginx stale backup files | Archive `.backup*` from sites-enabled; `nginx -t && systemctl reload nginx` |

### 7.3 TO-BE architecture diagram (target state)

```
DATA LAYER
══════════
[Binance WS] → market.klines → [AI Engine :8001]
                                      │
                   ┌──────────────────┴──────────────────┐
                   │                                      │
            trade.intent (ENTRY)               ai.signal_generated
                   │                                      │
                   │                              [Decision Intelligence]
                   │                                      │
                    ──────────────────────────────────────
                                    │
                          trade.intent stream
                                    │
              ┌─────────────────────┴────────────────┐
              │                                       │
    [intent-bridge]                     [execution consumer group]
    apply.plan                          (UNSTUCK, reconnect-aware)
              │                                       │
         [Governor P3.2]                    [Binance REST API]
         permit keys
              │
      [Intent-Executor P3.3]
              │
    [Binance Testnet REST]


EXIT LAYER (TO-BE — after fix + expansion)
══════════════════════════════════════════
[Position State Brain] ← single authoritative source
        │ writes: quantum:position:<symbol>
        │
[ExitBrain v3.5]  ←──────────────────────── TO-BE: receives signal from
    │                                        Exit Brain Math Stack (Section 8)
    │ exit decisions
    ▼
quantum:stream:ai.exit.decision
    │
    ▼
quantum:stream:exit.intent
    │
    ▼
[ExitIntentGateway PATCH-5B]  ← 9 validation checks
    │                              FORBIDDEN: direct write to apply.plan
    │
    ▼
quantum:stream:trade.intent  → execution


HARVEST LAYER (TO-BE — live mode)
══════════════════════════════════
[HarvestBrain] → reads apply.result + position keys
    │  R-based ladder: 0.5R→25%, 1.0R→25%, 1.5R→25%
    ▼
quantum:stream:harvest.intent
    │
    ▼
[Intent-Executor (harvest group)]
    │
    ▼
Binance (reduceOnly=true)


EMERGENCY PATH (unchanged)
══════════════════════════
system:panic_close → [EmergencyExitWorker] → close all (no validation)


RL ADAPTATION (TO-BE — after gate is lifted)
═════════════════════════════════════════════
[RL Trainer] → quantum:rl:policy:<symbol>
    ↑                     ↓
[PnL Reconciler]   [AI Engine + RL Sizer]
quantum:stream:rl_rewards   (rl_weight enabled after RL_ENABLED=true)
```

---

## SECTION 8 — TO-BE INTEGRATION MAP FOR EXIT BRAIN EXPANSION

### 8.1 New module inventory (TO-BE)

The following 13 modules form the Exit Brain Math Stack. Their roles, inputs, outputs, and Redis stream/key ownership are defined here.

| # | Module | Role | Inputs | Outputs | Owns |
|---|--------|------|--------|---------|------|
| 1 | **Position State Engine** | Single authoritative position record | Binance REST, apply.result stream | `quantum:position:<symbol>` | Write-only owner of position keys |
| 2 | **Geometry Engine** | Calculates P&L geometry: entry/current/extreme prices, R-multiple, drawdown, break-even | position key, market price | `quantum:exit:geometry:<symbol>` | Writes geometry snapshot |
| 3 | **Regime Drift Engine** | Detects when market regime has shifted since entry; measures regime distance | `quantum:regime:<symbol>`, OHLCV features | `quantum:exit:regime_drift:<symbol>` | Reads regime key (written by meta-regime) |
| 4 | **Belief Engine** | Runs ensemble sub-models in exit mode; aggregates exit-specific probability vector | `quantum:stream:market.klines`, position context | `quantum:exit:belief:<symbol>` | Calls ai-engine `/exit/evaluate` OR own inference |
| 5 | **Hazard Engine** | Computes time-to-liquidation, funding hazard, correlation hazard, drawdown hazard | geometry output, funding rates, portfolio state | `quantum:exit:hazard:<symbol>` | Reads portfolio.state stream |
| 6 | **Action Utility Engine** | Scores {HOLD, PARTIAL_CLOSE, FULL_CLOSE, ADD} by expected utility; outputs ranked action vector | belief + hazard + geometry | `quantum:exit:utility:<symbol>` | Pure computation — no stream writes |
| 7 | **Ensemble Exit Adapter** | Translates ensemble model outputs into exit-specific signals (XGB/LGBM direction change, NHiTS time-series reversal, etc.) | ensemble agent outputs | `quantum:exit:ensemble_signal:<symbol>` | Reads from ai-engine /predict endpoint |
| 8 | **Ensemble Aggregator** | Combines Ensemble Exit Adapter signal with Belief Engine to produce single exit conviction score | ensemble_signal + belief | `quantum:exit:conviction:<symbol>` | Aggregation layer — no new streams |
| 9 | **Exit Policy Engine** | Applies risk policy rules, size limits, rate limits, governance checks; translates utility + conviction into ExitPlan | utility + conviction + `quantum:governance:policy` | `quantum:exit:plan:<plan_id>` | Enforces policy — reads governance key |
| 10 | **LLM Orchestrator** | Optional: escalates ambiguous decisions to LLM for reasoning; returns structured override or confirmation | utility + conviction + context | `quantum:exit:llm_override:<symbol>` (TTL) | Async, non-blocking |
| 11 | **Gateway Validator** | Wraps ExitIntentGateway PATCH-5B — enforces 9 checks; the only path from exit.intent → trade.intent | `quantum:stream:exit.intent` | `quantum:stream:trade.intent` (pass) / `exit.intent.rejected` (fail) | **apply.plan write is UNCONDITIONALLY FORBIDDEN** |
| 12 | **Replay Obituary Writer** | After each closed position, writes structured post-mortem to replay store | apply.result + final position snapshot | `quantum:stream:exit.obituary` + persistent store | Append-only writer |
| 13 | **Offline Evaluator** | Reads obituary stream; computes strategy quality metrics; triggers model retraining if drift detected | `quantum:stream:exit.obituary` | `quantum:stream:model.retrain` + eval metrics | Downstream of Obituary Writer |

### 8.2 Module placement in the existing stream topology

```
EXISTING STREAMS (read-only by new modules)
────────────────────────────────────────────
quantum:stream:market.klines          ← Belief Engine, Ensemble Exit Adapter
quantum:stream:apply.result           ← Position State Engine, Replay Obituary Writer
quantum:stream:portfolio.state        ← Hazard Engine
quantum:stream:trade.closed           ← Replay Obituary Writer
quantum:governance:policy             ← Exit Policy Engine

NEW KEYS (written by new modules)
───────────────────────────────────
quantum:exit:geometry:<symbol>        ← Geometry Engine
quantum:exit:regime_drift:<symbol>    ← Regime Drift Engine
quantum:exit:belief:<symbol>          ← Belief Engine
quantum:exit:hazard:<symbol>          ← Hazard Engine
quantum:exit:utility:<symbol>         ← Action Utility Engine
quantum:exit:ensemble_signal:<symbol> ← Ensemble Exit Adapter
quantum:exit:conviction:<symbol>      ← Ensemble Aggregator
quantum:exit:plan:<plan_id>           ← Exit Policy Engine
quantum:exit:llm_override:<symbol>    ← LLM Orchestrator (TTL-based)

NEW STREAMS (written by new modules)
──────────────────────────────────────
quantum:stream:exit.intent            ← Exit Policy Engine (writes exit intents)
quantum:stream:exit.obituary          ← Replay Obituary Writer
quantum:stream:model.retrain          ← Offline Evaluator (appends to existing)

EXISTING STREAMS (written by gateways — unchanged)
────────────────────────────────────────────────────
quantum:stream:trade.intent           ← Gateway Validator (PASS path only)
quantum:stream:exit.intent.rejected   ← Gateway Validator (FAIL path)
```

### 8.3 Execution pipeline for Exit Brain Math Stack decisions

```
Position triggers evaluation cycle (time-based or event-based):
    │
    ├─[1]─► Position State Engine       → quantum:position:<symbol>
    │                                      (authoritative: replaces PSB crash loop)
    │
    ├─[2]─► Geometry Engine             → quantum:exit:geometry:<symbol>
    │           reads: position key, market price
    │
    ├─[3]─► Regime Drift Engine         → quantum:exit:regime_drift:<symbol>
    │           reads: quantum:regime:<symbol>
    │
    ├─[4]─► Ensemble Exit Adapter       → quantum:exit:ensemble_signal:<symbol>
    │           reads: market.klines, calls ai-engine /predict
    │
    ├─[5]─► Belief Engine               → quantum:exit:belief:<symbol>
    │           reads: ensemble_signal + market context
    │
    ├─[6]─► Hazard Engine               → quantum:exit:hazard:<symbol>
    │           reads: geometry, portfolio.state, funding rates
    │
    ├─[7]─► Action Utility Engine       → quantum:exit:utility:<symbol>
    │           reads: belief + hazard + geometry
    │
    ├─[8]─► Ensemble Aggregator         → quantum:exit:conviction:<symbol>
    │           reads: ensemble_signal + belief
    │
    ├─[9]─► Exit Policy Engine          → quantum:exit:plan:<plan_id>
    │           reads: utility + conviction + governance policy
    │           applies: size limits, rate limits, governance checks
    │
    ├─[10]─► LLM Orchestrator (async)   → quantum:exit:llm_override (TTL)
    │            escalates ambiguous signals only
    │
    └─[11]─► Exit Policy Engine (final) → quantum:stream:exit.intent
                 formats: IntentMessage per PATCH-5B spec
                 │
                 ▼
        [Gateway Validator PATCH-5B]    validated → quantum:stream:trade.intent
                 │                      rejected → exit.intent.rejected
                 ▼
        [Execution / Intent-Executor]
                 │
                 ▼
        Binance REST API (reduceOnly=true)

Post-execution:
    apply.result → [Replay Obituary Writer] → exit.obituary
    exit.obituary → [Offline Evaluator] → model.retrain | eval metrics
```

### 8.4 Integration with existing ExitBrain v3.5

ExitBrain v3.5 (`microservices/exitbrain_v3_5/exit_brain.py`) provides:
- `build_exit_plan()` method with ILFv2 + AdaptiveLeverageEngine
- Formula-based dynamic stop (`compute_dynamic_stop`)
- ATR-based trailing callback
- TP1/TP2/TP3 levels and harvest_scheme

**TO-BE integration:** The Exit Policy Engine (module 9) should **call** `ExitBrainV35.build_exit_plan()` to derive the final TP/SL levels and leverage, using the Ensemble Aggregator conviction and Hazard Engine inputs to build the `SignalContext`. ExitBrain v3.5 is a **calculation engine**, not a standalone service — route through it rather than duplicating its math.

### 8.5 Integration with existing harvest_brain

HarvestBrain already implements R-based partial closing. Do NOT re-implement:
- The R-multiple tracking
- The 0.5R/1.0R/1.5R ladder
- The dedup logic

**TO-BE integration:** The Ensemble Aggregator's conviction score should be available as an optional input to HarvestBrain's harvest threshold (`HARVEST_MIN_R`). When conviction is low, raise the R threshold (hold longer). When conviction is high (strong exit signal), lower the threshold (harvest sooner). This is an additive signal, not a replacement.

---

## SECTION 9 — TO-BE MODEL / ENSEMBLE PLACEMENT

### 9.1 Current ensemble usage (entry signals only)

The 6-model ensemble currently generates **entry** signals only. Exit evaluation is handled separately by a call to `ai-engine /exit/evaluate`. The exit evaluator is a different code path inside ai-engine; it is not the same as the ensemble.

### 9.2 Ensemble for exit signals — recommended placement

| Model | Entry use | Proposed exit use | Exit signal type |
|-------|-----------|-------------------|-----------------|
| XGBoost | Direction (BUY/SELL/HOLD) | Exit probability (0-1) | Feature-based exit score |
| LightGBM | Direction + confidence | Exit probability (0-1) | Feature-based exit score |
| NHiTS | Temporal pattern (multi-rate) | Next-N-bar direction reversal probability | Time-series regime change |
| PatchTST | Long-range dependency | Long-term price path prediction | Trend continuation/break |
| TFT | Multi-scale temporal fusion | Quantile forecast (P10/P50/P90) | Downside scenario probability |
| DLinear | Trend decomposition | Linear trend to next target | Momentum trajectory |

The **Ensemble Exit Adapter** (module 7) is responsible for translating each model's general prediction into an exit-specific signal. Each model's output schema differs — the adapter normalizes them into a common `ExitSignal{symbol, side, exit_conviction: float[0-1], reversal_probability: float, time_horizon_bars: int}` format.

### 9.3 Model weight for exit vs entry (recommended)

Do NOT use the same weights for exit as for entry. Proposed starting weights:

| Model | Entry weight | Exit weight | Reason |
|-------|-------------|-------------|--------|
| XGB | 17% | 10% | Less powerful on exit timing |
| LGBM | 17% | 10% | Less powerful on exit timing |
| NHiTS | 17% | 30% | Best at temporal pattern → regime change detection |
| PatchTST | 17% | 25% | Strong at long-range dependency → trend break |
| TFT | 16% | 20% | Quantile forecasting → downside scenario |
| DLinear | 16% | 5% | Linear trend too simple for exit |

These weights are initial recommendations. The **Offline Evaluator** (module 13) should tune them via performance attribution on the `exit.obituary` stream.

### 9.4 MetaAgent V2 / Arbiter placement

MetaAgent V2 and ArbiterAgent are currently disabled (`META_AGENT_ENABLED=false`, `ARBITER_ENABLED=false`). In TO-BE:
- **MetaAgent V2** should be the primary override escalation path for the Belief Engine — when model disagreement exceeds threshold (>2 models disagree on exit), MetaAgent arbitrates.
- **ArbiterAgent** should be the final macro-context check before the Exit Policy Engine commits an exit plan.

Both are already implemented in `ai_engine/agents/` — they require ENV activation, not new code.

---

## SECTION 10 — MIGRATION PLAN (AS-IS → TO-BE)

### Phase 0 — Stability (no new modules, just fix breakpoints)

Estimated: 1–2 sessions. Do in this exact order (dependencies exist).

```
Step 1. Fix SL/TP SELL-side math
    File: microservices/ai_engine/main.py
    Fix: SELL stop_loss = entry * (1 + sl_pct/100)
         SELL take_profit = entry * (1 - tp_pct/100)
    Test: Write unit test against BTCUSDT SELL case
    Merge: BEFORE unsticking execution consumer

Step 2. Fix ExitBrain v3.5 drop-in
    File: /etc/systemd/system/quantum-exitbrain-v35.service.d/shadow.conf
    Fix: Remove duplicate [Service] header, remove backslash from env var
    Then: systemctl daemon-reload && systemctl enable --now quantum-exitbrain-v35
    Verify: systemctl status quantum-exitbrain-v35; journalctl -u quantum-exitbrain-v35 -f

Step 3. Fix position-state-brain unit
    File: /etc/systemd/system/quantum-position-state-brain.service
    Fix: User=qt; ExecStart=/opt/quantum/venvs/ai-engine/bin/python3 ...
    File: /etc/quantum/position-state-brain.env
    Fix: Add BINANCE_API_KEY, BINANCE_API_SECRET
    Then: systemctl daemon-reload && systemctl restart quantum-position-state-brain

Step 4. Unsync execution consumer group (ONLY AFTER step 1 is deployed and verified)
    redis-cli XGROUP SETID quantum:stream:trade.intent quantum:group:execution:trade.intent '$'
    systemctl restart quantum-execution

Step 5. Fix regime key mismatch
    grep quantum:regime microservices/meta_regime/main.py    (what key does it write?)
    grep quantum:regime microservices/ai_engine/main.py      (what key does it read?)
    Align both to the same key: quantum:regime:<symbol>

Step 6. Debug exposure_balancer + bsc crash
    journalctl -u quantum-exposure_balancer --no-pager | tail -30
    Fix the crash cause (likely import error in code changed Feb 21)
    Restart both services

Step 7. Clean nginx
    cd /etc/nginx/sites-enabled/
    mkdir -p /etc/nginx/archive
    mv *.backup *.bak *.backup2 *.backup3 *.backup_jan16 /etc/nginx/archive/
    nginx -t && systemctl reload nginx
```

### Phase 1 — Exit Brain Foundation (modules 1–3)

Pre-condition: Phase 0 complete; ExitBrain v3.5 running; execution not stuck.

```
Module 1: Position State Engine
    - Replaces (or wraps) quantum-position-state-brain
    - Single source of truth for quantum:position:<symbol>
    - Reads: Binance REST positionRisk + apply.result stream
    - Writes: quantum:position:<symbol> (HASH)
    - Deploy as: quantum-position-state-v2 service (keep PSB running during transition)

Module 2: Geometry Engine
    - New service or library; can be a function called by ExitBrain v3.5
    - Reads: quantum:position:<symbol> + market price
    - Writes: quantum:exit:geometry:<symbol> (TTL=60s)
    - No stream ownership change required

Module 3: Regime Drift Engine
    - Reads: quantum:regime:<symbol> (fixed in Phase 0 step 5)
    - Writes: quantum:exit:regime_drift:<symbol> (TTL=30s)
    - Can be integrated into ExitBrain v3.5's build_exit_plan() as an input parameter
```

### Phase 2 — Belief + Hazard Engines (modules 4–6)

Pre-condition: Phase 1 complete; Position State Engine stable.

```
Module 4: Ensemble Exit Adapter
    - Calls ai-engine /predict (reuses existing endpoint with exit_mode=true flag)
    - Translates 6-model output to ExitSignal format
    - Writes: quantum:exit:ensemble_signal:<symbol> (TTL=30s)

Module 5: Belief Engine
    - Aggregates ensemble_signal + internal scoring
    - Writes: quantum:exit:belief:<symbol> (TTL=30s)

Module 6: Hazard Engine
    - Reads: geometry key + portfolio.state + funding rates (via multi-source-feed)
    - Writes: quantum:exit:hazard:<symbol> (TTL=30s)
```

### Phase 3 — Policy + Gateway (modules 7–11)

Pre-condition: Phase 2 complete; belief/hazard keys reliably populated.

```
Module 7: Action Utility Engine (can be a function within Exit Policy Engine)
Module 8: Ensemble Aggregator (can be a function within Belief Engine)
Module 9: Exit Policy Engine
    - The core orchestrator — reads all `quantum:exit:*` keys
    - Calls ExitBrainV35.build_exit_plan() for final TP/SL/leverage
    - Formats IntentMessage per PATCH-5B spec
    - Writes to quantum:stream:exit.intent
    - Writes plan record to quantum:exit:plan:<plan_id>

Module 10: LLM Orchestrator (optional / later phase)
    - Async escalation only; non-blocking
    - Enable after module 9 is validated in shadow mode

Module 11: Gateway Validator
    - Already implemented as ExitIntentGateway PATCH-5B
    - No new code required; confirm it is running with EXIT_GATEWAY_ENABLED=true
```

### Phase 4 — Replay + Offline Evaluation (modules 12–13)

Pre-condition: Phase 3 complete; exits are being executed live.

```
Module 12: Replay Obituary Writer
    - Consumes apply.result + trade.closed streams
    - Writes structured post-mortem to quantum:stream:exit.obituary
    - Persistent store: append to file or PostgreSQL

Module 13: Offline Evaluator
    - Consumes exit.obituary stream
    - Computes: win rate, R-multiple distribution, Sharpe by model, by regime
    - Writes quantum:stream:model.retrain if drift threshold exceeded
    - Outputs: eval metrics to Prometheus (for Grafana dashboard)
```

### Phase 5 — RL Reactivation + Regime Adaptive Strategy

Pre-condition: Phase 4 complete; obituary validates positive P&L contribution from new exit stack.

```
Step 1: Enable RL gate
    /etc/quantum/ai-engine.env → RL_ENABLED=true
    Monitor: rl_gate_pass in trade.intent events (should become true)
    Observe: rl_weight_effective > 0 on signals

Step 2: Connect Ensemble Aggregator → HarvestBrain threshold adjustment
    When conviction > 0.7 → HARVEST_MIN_R = 0.3 (harvest sooner)
    When conviction < 0.3 → HARVEST_MIN_R = 0.8 (hold longer)

Step 3: Enable MetaAgent V2 for exit signal arbitration
    META_AGENT_ENABLED=true
    META_OVERRIDE_THRESHOLD=0.65

Step 4: Enable Arbiter Agent
    ARBITER_ENABLED=true
    ARBITER_THRESHOLD=0.70

Step 5: Connect Offline Evaluator → model weight update
    Offline Evaluator writes updated exit weights to supervisor_weights_file
    EnsembleManager picks up on 5-minute refresh cycle
```

---

## SECTION 11 — VERIFICATION CHECKLIST

### 11.1 Phase 0 verification (run after each fix)

```bash
# After SL/TP fix (step 1):
redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 5 | grep -A3 '"side": "SELL"'
# Expected: stop_loss > entry_price, take_profit < entry_price for SELL orders

# After ExitBrain v3.5 fix (step 2):
systemctl status quantum-exitbrain-v35
journalctl -u quantum-exitbrain-v35 --since "5 minutes ago"
redis-cli XREVRANGE quantum:stream:exit_brain:heartbeat + - COUNT 1
# Expected: active (running), heartbeat within last 2 seconds

# After position-state-brain fix (step 3):
systemctl status quantum-position-state-brain
journalctl -u quantum-position-state-brain --since "2 minutes ago"
# Expected: active (running), no "Missing Binance credentials" errors

# After execution consumer unsync (step 4):
redis-cli XINFO GROUPS quantum:stream:trade.intent
# Expected: quantum:group:execution:trade.intent lag = 0 (or small)
redis-cli XREVRANGE quantum:stream:trade.execution.res + - COUNT 1
# Expected: recent timestamp (within last 10 minutes)

# After regime fix (step 5):
redis-cli KEYS "quantum:regime:*" | head -5
redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 2 | grep regime
# Expected: regime != "unknown"

# After exposure balancer restart (step 6):
systemctl status quantum-exposure_balancer quantum-bsc
# Expected: both active (running)

# After nginx fix (step 7):
nginx -t
# Expected: no conflicting server_name warnings

# System health check (all phases):
curl -s http://127.0.0.1:8001/health | python3 -m json.tool
redis-cli XLEN quantum:stream:trade.intent
redis-cli XINFO GROUPS quantum:stream:trade.intent
```

### 11.2 Phase 1 verification

```bash
# Position State Engine:
redis-cli HGETALL quantum:position:BTCUSDT
# Expected: qty, entry_price, side, leverage, unrealized_pnl fields present

# Geometry Engine:
redis-cli HGETALL quantum:exit:geometry:BTCUSDT
# Expected: R_multiple, drawdown_pct, distance_to_sl, distance_to_tp fields

# Regime Drift Engine:
redis-cli HGETALL quantum:exit:regime_drift:BTCUSDT
# Expected: regime_at_entry, regime_now, drift_magnitude fields
```

### 11.3 Phase 2–3 verification

```bash
# Ensemble Exit Adapter + Belief Engine:
redis-cli HGETALL quantum:exit:belief:BTCUSDT
# Expected: exit_conviction, reversal_probability, model_breakdown fields

# Hazard Engine:
redis-cli HGETALL quantum:exit:hazard:BTCUSDT
# Expected: time_to_liq, funding_hazard, correlation_hazard fields

# Exit Policy Engine — shadow mode test:
redis-cli XREVRANGE quantum:stream:exit.intent + - COUNT 5
# Expected: WellFormed IntentMessage with source=exit_policy_engine

# Gateway Validator:
redis-cli XREVRANGE quantum:stream:exit.intent.rejected + - COUNT 5
# Expected: 0 rejects for valid intents (monitor for false rejects)
```

### 11.4 Phase 4 verification

```bash
# Obituary Writer:
redis-cli XREVRANGE quantum:stream:exit.obituary + - COUNT 3
# Expected: structured post-mortem per closed position

# Offline Evaluator:
redis-cli XREVRANGE quantum:stream:model.retrain + - COUNT 3
# Expected: retrain events only if drift threshold exceeded (not on every close)
```

### 11.5 Regression tests (run before each phase merge)

| Test | Pass condition |
|------|----------------|
| SELL SL/TP correctness | stop_loss > entry_price, take_profit < entry_price for all SELL positions |
| Execution consumer alive | lag < 100 within 60 seconds of restart |
| ExitBrain heartbeat | `exit_brain:heartbeat` key updated within last 2 seconds |
| Trade intent schema | All required fields present: symbol, side, stop_loss, take_profit, position_size_usd |
| Harvest dedup | Same position not harvested twice within HARVEST_DEDUP_TTL_SEC (900s) |
| Gateway apply.plan guard | Gateway never writes to `quantum:stream:apply.plan` (check redis_io.py write guard) |
| RL gate state | `rl_gate_pass` field matches `RL_ENABLED` env var |
| Governance policy present | `redis-cli GET quantum:governance:policy` returns valid JSON |
| Exit conviction schema | `quantum:exit:conviction:<symbol>` includes `conviction_score`, `model_votes`, `timestamp` |

---

## SECTION 12 — APPENDIX: FACTS / INFERENCES / UNKNOWNS

### 12.1 Confirmed facts

| # | Fact | Source |
|---|------|--------|
| F1 | VPS uptime 42 days; Redis restarted 2026-03-02 06:37 UTC (19h mismatch) | VPS_FORENSIC_AUDIT |
| F2 | ExitBrain v3.5: Python file = `microservices/position_monitor/main_exitbrain.py` on VPS, not `exitbrain_v3_5/exit_brain.py` in repo | VPS audit (ExecStart) |
| F3 | Execution consumer stuck since ~2026-02-16 (last-delivered-id timestamp) | redis XINFO GROUPS |
| F4 | Last actual execution result: 2026-02-09, status: "rejected" | XREVRANGE trade.execution.res |
| F5 | Position-state-brain running as root, using /usr/bin/python3 (wrong user + wrong python) | VPS audit Unit table |
| F6 | `quantum-exposure_balancer` and `quantum-bsc` both died at exactly 2026-02-21 06:34:26 UTC | VPS audit |
| F7 | All RL services running but `rl_gate_pass=false` due to kill-switch in ai-engine.env | VPS audit + trade.intent XREVRANGE |
| F8 | 10,004 trade intents queued in `quantum:stream:trade.intent` as of 2026-03-03 | redis XLEN |
| F9 | `consensus_count=1, total_models=4` on live signals — low-conviction signals being generated | XREVRANGE |
| F10 | ExitIntentGateway PATCH-5B forbids direct write to apply.plan (enforced in redis_io.py) | Source code |
| F11 | HarvestBrain defaults to shadow mode (`HARVEST_MODE=shadow`) — no live partial closes executing | Source code |
| F12 | Governor rate limit: MAX_EXEC_PER_HOUR=3, MAX_EXEC_PER_5MIN=2 | Source code |
| F13 | 6-model ensemble with equal weights (17/17/17/17/16/16) confirmed active as of 2026-03-11 | Previous session |
| F14 | SL/TP SELL bug confirmed: BTCUSDT SELL stop_loss=$96M, take_profit=$-192M in actual stream entries | VPS audit XREVRANGE |
| F15 | Market regime always "unknown" in trade.intent events — key mismatch between writer/reader | VPS audit |
| F16 | 134 connected Redis clients, 30 in blocking XREAD (normal) as of audit time | redis INFO server |
| F17 | `quantum:metrics:exit:*` — 298,000+ records (evidence exit intelligence has been running for a long time) | VPS audit |
| F18 | DAG 8 C3 FnG=9 (< 20) blocking live trading — macro condition not a code bug | VPS audit |

### 12.2 Reasonable inferences (not directly confirmed)

| # | Inference | Basis |
|---|-----------|-------|
| I1 | The execution consumer group stuck because Redis restart dropped the blocking XREADGROUP TCP connection and the service did not reconnect | Redis uptime mismatch + consumer group last-delivered-id timestamp correlation |
| I2 | ExitBrain v3.5 was previously working (in shadow mode) and was accidentally disabled during a maintenance operation that created the broken drop-in | The drop-in exists with content — implies someone created it intentionally |
| I3 | `quantum-exposure_balancer` dying at same second as `quantum-bsc` (06:34:26 Feb 21) suggests a shared dependency (Redis restart? Python package update? Key collision?) | Simultaneous exact timestamp |
| I4 | The `testnet_hash_pattern` fallback trigger in the live signal sample suggests testnet-specific override logic is being applied on what may now be live execution | XREVRANGE evidence |
| I5 | HarvestBrain's independent position tracking from `apply.result` is more reliable than `position-state-brain` for determining exit R-multiple, given PSB is crash-looping | Code analysis |
| I6 | The 20+ undocumented processes on public ports are individual quantum-* services that each bind their own Prometheus metrics endpoint to 0.0.0.0 by default | Governor source code (METRICS_PORT=8044, binds 0.0.0.0) |
| I7 | The current exit path in production is: autonomous_trader `exit_manager.py` evaluates via AI Engine; if exit triggered, writes to `trade.intent` (not `exit.intent`) bypassing ExitIntentGateway | ExitIntentGateway PATCH-5B description notes "AutonomousTrader is NOT modified" |

### 12.3 Known unknowns (items requiring SSH verification)

| # | Unknown | Where to look |
|---|---------|--------------|
| U1 | What does `quantum-exit-intelligence.service` actually do? | `systemctl show quantum-exit-intelligence` + `journalctl` |
| U2 | What stream does `exit_manager.py` actually write to when it determines CLOSE? | `microservices/autonomous_trader/exit_manager.py` — the write path is not in the first 150 lines read |
| U3 | What is writing to `quantum:stream:ai.exit.decision` now that exitbrain is dead? | `redis-cli XREVRANGE quantum:stream:ai.exit.decision + - COUNT 3` |
| U4 | Is ExitIntentGateway running and consuming `quantum:stream:exit.intent`? | `systemctl status quantum-exit-intent-gateway` |
| U5 | What key exactly does `quantum-meta-regime` write? | `microservices/meta_regime/main.py` — key name in writer |
| U6 | What is causing Redis to restart? (appendonly? maxmemory? OOM killer?) | `/var/log/redis/redis-server.log` + `dmesg | grep -i oom` |
| U7 | What are the 12+ undocumented processes on ports 8042–8069? | `ss -lntp | grep python3` then `ps aux | grep <pid>` |
| U8 | Is `quantum:stream:exit.intent` stream even being read by anything right now? | `redis-cli XINFO GROUPS quantum:stream:exit.intent` |
| U9 | What is `services/execution_service.py` vs `microservices/execution/main.py`? Two different implementations? | File comparison between the two |
| U10 | What happens to `quantum-autonomous-trader` exit decisions — does HarvestBrain pick them up or run independently? | `harvest_brain.py` + `autonomous_trader.py` integration points |
| U11 | Is the 10,004-entry trade.intent backlog safe to discard (skip to '$')? Are there any unfilled SELL orders in there that would leave open positions unhedged? | XRANGE + cross-reference with Binance open positions via REST |

### 12.4 Deferred design decisions

| # | Decision | Options |
|----|----------|---------|
| D1 | Should Exit Brain Math Stack run as a single service or as separate microservices per module? | Single service (simpler, fewer connections) vs. separate (independently deployable, separate metrics) |
| D2 | Should LLM Orchestrator (module 10) be synchronous (blocks exit decision) or async (suggests override, doesn't block)? | Async strongly preferred — exit latency is critical |
| D3 | Where should the Offline Evaluator store its post-mortem corpus? PostgreSQL vs. Redis Streams vs. local file? | Redis Streams for real-time; file/DB for offline analysis |
| D4 | Should HarvestBrain remain a separate service or be merged into the Exit Policy Engine? | Keep separate — it has its own dedup, rate limiting, and R-tracking state that would complicate the policy engine |
| D5 | Should Ensemble Aggregator use the same 6-model weights as entry, or maintain a separate exit weight registry? | Separate exit weights — entry and exit have different model strengths (see Section 9.3) |

---

*Document generated from verified sources: VPS_FORENSIC_AUDIT_2026-03-03.md, OWNERSHIP_MAP.md, SYSTEM_ARCHITECTURE_COMPLETE_FEB23_2026.md, source code reads of exitbrain_v3_5/exit_brain.py, autonomous_trader/exit_manager.py, autonomous_trader/autonomous_trader.py, intent_executor/main.py, harvest_brain/harvest_brain.py, exit_intent_gateway/main.py, governor/main.py, ai_engine/ensemble_manager.py.*

*All facts are sourced. Inferences are explicitly labeled. Unknowns are enumerated. Do not act on unknowns without SSH verification first.*
