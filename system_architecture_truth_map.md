# QUANTUM TRADER - SYSTEM ARCHITECTURE TRUTH MAP

**Generated:** February 18, 2026  
**Scope:** Production-grade event-driven trading system  
**Purpose:** Complete structural, execution-flow audit (NOT performance audit)  
**Method:** Actual code tracing, not assumptions

---

## TABLE OF CONTENTS

1. [Service Inventory](#1-service-inventory)
2. [Event Flow Mapping](#2-event-flow-mapping)
3. [Redis Truth Map](#3-redis-truth-map)
4. [Position Limit Enforcement Map](#4-position-limit-enforcement-map)
5. [Failure & Race Condition Analysis](#5-failure--race-condition-analysis)
6. [Visual Architecture Output](#6-visual-architecture-output)
7. [System State Truth](#7-system-state-truth)

---

## 1. SERVICE INVENTORY

Total active services: **68 quantum-* microservices**

### 1.1 CORE TRADING PIPELINE

#### **AI Engine Service**
- **File:** `microservices/ai_engine/main.py`
- **Port:** 8001
- **Entry Point:** FastAPI service with `AIEngineService`
- **Redis Streams:**
  - **CONSUMES:** `quantum:stream:market.tick`, `quantum:stream:market.klines`, `quantum:stream:trade.closed`, `quantum:stream:policy.updated`
  - **PUBLISHES:** `quantum:stream:ai.decision.made`, `quantum:stream:ai.signal_generated`, `quantum:stream:trade.intent`
- **External APIs:** None (internal inference only)
- **Environment Variables:** `LOG_LEVEL`, `LOG_DIR`, `VERSION`, `REDIS_HOST`, `REDIS_PORT`
- **Hard Limits:** None (ensemble voting only)
- **Locking:** None
- **Responsibilities:**
  - AI model inference (XGBoost, LightGBM, N-HiTS, PatchTST)
  - Ensemble voting and signal aggregation
  - Meta-strategy selection (RL-based)
  - RL position sizing
  - Market regime detection
  - Trade intent generation

#### **Ensemble Predictor Service**
- **File:** `microservices/ensemble_predictor/main.py`
- **Port:** Not exposed (shadow mode)
- **Redis Streams:**
  - **CONSUMES:** `quantum:stream:features`
  - **PUBLISHES:** `quantum:stream:ai.signal_generated` (PATH 2.2 shadow mode)
- **Responsibilities:**
  - Multi-model ensemble predictions
  - Shadow mode signal generation (non-authoritative)

#### **Intent Bridge Service**
- **File:** `microservices/intent_bridge/main.py`
- **Port:** None (stream processor)
- **Redis Streams:**
  - **CONSUMES:** `quantum:stream:trade.intent` (XREADGROUP, group: `quantum:group:intent-bridge:trade.intent`)
  - **PUBLISHES:** `quantum:stream:apply.plan`
- **Environment Variables:**
  - `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`
  - `INTENT_BRIDGE_ALLOWLIST` (default: "BTCUSDT")
  - `INTENT_BRIDGE_USE_TOP10` (default: false)
  - `MAX_EXPOSURE_PCT` (default: 80.0%)
  - `INTENT_BRIDGE_SKIP_FLAT_SELL` (default: true)
- **Hard Limits:**
  - Allowlist filtering (CSV list or TOP10 universe)
  - Exposure limit: 80% max
  - FLAT_EPS: 0.0 (skip flat SELL intents)
- **Locking:** Redis SETNX for idempotency (dedup keys)
- **Responsibilities:**
  - Bridges `trade.intent` → `apply.plan`
  - Allowlist filtering (fail-closed design)
  - Idempotency via Redis SETNX
  - Flat position SELL skipping
  - Reads last known position from `quantum:ledger:{symbol}`

#### **Governor Service (P3.2)**
- **File:** `microservices/governor/main.py`
- **Port:** 9092 (Prometheus metrics)
- **Redis Streams:**
  - **CONSUMES:** `quantum:stream:apply.plan` (XREADGROUP, group: `quantum:group:governor:apply.plan`)
  - **PUBLISHES:** None (writes permit keys only)
  - **WRITES KEYS:** `quantum:permit:p33:{plan_id}`, `quantum:governor:exec_history:{symbol}`
- **Environment Variables:**
  - `GOV_MAX_EXEC_PER_HOUR` (default: 3)
  - `GOV_MAX_EXEC_PER_5MIN` (default: 2)
  - `GOV_MAX_OPEN_POSITIONS` (default: 10)
  - `GOV_MAX_NOTIONAL_PER_TRADE_USDT` (default: 200)
  - `GOV_MAX_TOTAL_NOTIONAL_USDT` (default: 2000)
  - `GOV_SYMBOL_COOLDOWN_SECONDS` (default: 60)
  - `GOV_KILL_SCORE_CRITICAL` (default: 0.8)
  - `GOV_KILL_SCORE_OPEN_THRESHOLD` (default: 0.85)
  - `GOV_KILL_SCORE_CLOSE_THRESHOLD` (default: 0.65)
- **Hard Limits:**
  - ❗ **MAX_OPEN_POSITIONS: 10** (fund-grade limit)
  - Max 3 executions per hour per symbol
  - Max 2 executions per 5min per symbol
  - Max $200 notional per trade
  - Max $2000 total notional
  - 60s cooldown between same-symbol trades
  - Kill score gates: ≥0.8 block all, ≥0.85 block entries, ≥0.65 block exits
- **Locking:** None (writes permit keys atomically)
- **Responsibilities:**
  - Rate limit enforcement (per-hour, per-5min)
  - Position count enforcement (reads from `quantum:position:*`)
  - Notional limit gates
  - Kill score thresholds (entry/exit separation)
  - Writes `quantum:permit:p33:{plan_id}` with TTL for Apply Layer check
  - Auto-disarm capability (force dry_run mode)

#### **Apply Layer Service (P3)**
- **File:** `microservices/apply_layer/main.py`
- **Port:** None (stream processor)
- **Build Tag:** `apply-layer-entry-exit-sep-v1`
- **Redis Streams:**
  - **CONSUMES:** `quantum:stream:apply.plan` (XREADGROUP, group: `quantum:group:apply:apply.plan`)  
  - **PUBLISHES:** `quantum:stream:apply.result`
  - **READS KEYS:** `quantum:permit:p33:{plan_id}`, `quantum:position:ledger:{symbol}`, `quantum:ledger:{symbol}`
  - **WRITES KEYS:** `quantum:position:ledger:{symbol}`, `quantum:dedup:apply:{hash}`
- **Environment Variables:**
  - `APPLY_LOG_LEVEL` (default: INFO)
  - `APPLY_ALLOWLIST` (default: "BTCUSDT")
  - `APPLY_DRY_RUN` (default: "true")
  - `APPLY_TESTNET` (default: "true")
  - `APPLY_KILL_SWITCH` (default: "false")
  - `APPLY_HEAT_OBSERVER_ENABLED` (default: "true")
- **Hard Limits:**
  - Allowlist gate (fail-closed)
  - Entry/exit separation (open_threshold=0.85, close_threshold=0.65)
  - Kill score blocks: ≥0.8 all, ≥0.6 risk increase
  - Idempotency (Redis SETNX dedup)
  - Governor permit check (requires `quantum:permit:p33:{plan_id}`)
  - Exit ownership enforcement (single exit controller)
- **Locking:** SETNX for deduplication (`quantum:dedup:apply:{hash}`)
- **External APIs:** 
  - **Binance Futures Testnet API** (testnet.binancefuture.com)
    - POST `/fapi/v1/order` (create order)
    - GET `/fapi/v2/account` (account info)
    - Credentials from env: `BINANCE_API_KEY`, `BINANCE_API_SECRET`
- **Responsibilities:**
  - Reads harvest proposals from `quantum:stream:apply.plan`
  - Checks Governor permit (`quantum:permit:p33:{plan_id}`)
  - Applies allowlist filtering
  - Entry/exit separation logic
  - Kill score threshold enforcement
  - Execution to Binance Testnet (if not dry_run)
  - Publishes execution result to `quantum:stream:apply.result`
  - Heat observer integration (P2.8A observability)

#### **Execution Service**
- **File:** `microservices/execution/main.py`
- **Port:** 8002
- **Entry Point:** FastAPI service with `ExecutionService`
- **Redis Streams:**
  - **CONSUMES:** `quantum:stream:apply.result` (XREADGROUP)
  - **PUBLISHES:** `quantum:stream:execution.result`, `quantum:stream:trade.execution.res`
- **Environment Variables:** `LOG_LEVEL`, `LOG_DIR`, `VERSION`, `PORT`
- **External APIs:** **Binance Futures API** (direct order execution, position monitoring)
- **Responsibilities:**
  - Order execution (entry + exit)
  - Position monitoring and TP/SL management
  - Trade lifecycle tracking
  - Binance API integration with rate limiting
  - Trade opened/closed/position updated events

### 1.2 EXIT & RISK SERVICES

#### **HarvestBrain Service**
- **File:** `microservices/harvest_brain/harvest_brain.py`
- **Port:** None (stream processor)
- **Redis Streams:**
  - **CONSUMES:** `quantum:stream:apply.result` (XREADGROUP, group: `harvest_brain:execution`)
  - **PUBLISHES:** 
    - Shadow mode: `quantum:stream:harvest.suggestions`
    - Live mode: `quantum:stream:apply.plan`
  - **WRITES KEYS:** `quantum:permit:p33:{plan_id}` (auto-creates permits)
- **Environment Variables:**
  - `HARVEST_MIN_R` (default: 0.5R)
  - `HARVEST_MAX_ACTIONS_PER_MIN` (default: 30)
  - `HARVEST_SCAN_INTERVAL_SEC` (default: 5)
  - `HARVEST_DEDUP_TTL_SEC` (default: 900)
  - `MAX_UNREALIZED_LOSS_PCT_PER_POSITION` (default: -12.5%)
- **Hard Limits:**
  - Min profit: 0.5R to trigger harvest
  - Max 30 harvest actions per minute
  - 5s scan interval
  - 900s dedup window
  - -12.5% emergency exit threshold
- **External APIs:** **Binance Futures API** (position sync)
- **Responsibilities:**
  - Monitors execution results, calculates R-based profit levels
  - Evaluates position state against harvest policy (P2 risk_kernel_harvest)
  - Generates harvest intents: `HARVEST_PARTIAL`, `MOVE_SL_BE`, `TRAIL_UPDATE`
  - Syncs positions from Binance at startup
  - Publishes reduce-only MARKET orders to `apply.plan` (live mode)

#### **Exit Monitor Service**
- **File:** `services/exit_monitor_service.py`
- **Port:** None
- **Redis Streams:**
  - **CONSUMES:** EventBus `trade.execution.result`
  - **PUBLISHES:** `quantum:stream:trade.intent` (close orders)
- **Environment Variables:**
  - `CHECK_INTERVAL` (default: 5s)
  - `TRAILING_STOP_PCT` (default: 1.5%)
- **Responsibilities:**
  - Tracks positions from execution results in memory
  - Monitors market prices every 5s (Binance API)
  - Evaluates exit conditions: TP, SL, trailing stop
  - Sends MARKET close orders when levels hit
  - TP: +2.5% LONG / -2.5% SHORT
  - SL: -1.5% LONG / +1.5% SHORT
  - Trailing only active when profit >1%

#### **Exit Brain V3 Service**
- **File:** `backend/domains/exits/exit_brain_v3/main.py`
- **Port:** None
- **Redis Streams:** None (polls Binance directly)
- **Environment Variables:**
  - `EXIT_BRAIN_CHECK_INTERVAL_SEC` (default: 10)
  - `MAX_MARGIN_LOSS_PER_TRADE_PCT` (default: 10%)
  - `MAX_UNREALIZED_LOSS_PCT_PER_POSITION` (default: 12.5%)
  - `MIN_PRICE_STOP_DISTANCE_PCT` (default: 0.2%)
  - `MAX_LOSS_PCT_HARD_SL` (default: 2%)
  - `RATCHET_SL_ENABLED` (default: true)
- **Hard Limits:**
  - 10s monitoring loop
  - Max 10% margin loss per trade
  - Max 12.5% unrealized loss per position
  - Min 0.2% stop distance
  - Max 4 TP levels
- **External APIs:** **Binance Futures API** (direct polling, no streams)
- **Responsibilities:**
  - Hybrid stop-loss model (internal AI SL + hard exchange SL)
  - Maintains `PositionExitState` per position (`{symbol}:{side}`)
  - Monitoring loop checks prices vs internal levels every 10s
  - Executes MARKET reduce-only orders when hit
  - NO LIMIT/STOP/TAKE_PROFIT orders (except hard SL safety net)
  - Dynamic partial TP (25%, 25%, 50% profile)
  - SL ratcheting after TP hits

#### **Risk Safety Service**
- **File:** `microservices/risk_safety/main.py`
- **Port:** 8003
- **Redis Streams:** None (EventBus only)
- **EventBus:**
  - **CONSUMES:** `trade.closed`, `order.failed`
  - **PUBLISHES:** `ess.tripped`, `ess.state.changed`, `policy.updated`, `risk.limit.exceeded`
- **Responsibilities:**
  - Emergency Stop System (ESS) state machine
  - PolicyStore (single source of truth for risk policies)
  - Trade outcome recording
  - ESS state transitions (NORMAL → WARNING → CRITICAL)
  - DiskBuffer for event persistence/replay
  - REST API for health checks and policy queries

#### **Baseline Safety Controller (BSC)**
- **File:** `bsc_main.py`
- **Service:** `quantum-bsc.service`
- **Port:** 9099 (Prometheus metrics)
- **Redis Streams:**
  - **CONSUMES:** None (monitors state keys only)
  - **PUBLISHES:** `quantum:stream:bsc.events`
- **Responsibilities:**
  - Authority engine safeguard layer
  - Monitors system health without consuming execution streams
  - Circuit breaker logic
  - Emergency intervention capability
  - Read-only Redis access (no stream consumption)

### 1.3 PORTFOLIO & STATE MANAGEMENT

#### **Portfolio State Publisher**
- **File:** `microservices/portfolio_state_publisher/main.py`
- **Redis Streams:**
  - **PUBLISHES:** `quantum:stream:portfolio.state`, `quantum:stream:portfolio.snapshot_updated`
  - **WRITES KEYS:** `quantum:portfolio:state`
- **Responsibilities:**
  - Aggregates portfolio-level metrics
  - Position snapshots publishing
  - Portfolio exposure tracking

#### **Position State Brain (P3.3)**
- **File:** `microservices/position_state_brain/main.py`
- **Redis Streams:**
  - **CONSUMES:** `quantum:stream:execution.result`, `quantum:stream:apply.result`
  - **PUBLISHES:** `quantum:stream:position.snapshot`
  - **WRITES KEYS:** `quantum:position:snapshot:{symbol}`, `quantum:position:ledger:{symbol}`, `quantum:position:{symbol}`
- **Responsibilities:**
  - Position state aggregation from execution results
  - Writes canonical position snapshots
  - Snapshot frequency: real-time on execution events
  - **Source of truth:** Internal state (reconciled with exchange)

#### **Reconcile Engine (P3.4)**
- **File:** `microservices/reconcile_engine/main.py`
- **Redis Streams:**
  - **CONSUMES:** None (polls Binance directly)
  - **PUBLISHES:** `quantum:stream:reconcile.events`, `quantum:stream:reconcile.close`
  - **WRITES KEYS:** `quantum:position:snapshot:{symbol}` (corrects drift)
- **External APIs:** **Binance Futures API** (position sync)
- **Responsibilities:**
  - Periodic position reconciliation (polls Binance every 60s)
  - Detects drift between internal state and exchange truth
  - Corrects `quantum:position:snapshot:{symbol}` when divergence found
  - Publishes reconciliation events and close recommendations
  - **Source of truth: Exchange state (Binance API)**

#### **Trade Logger**
- **File:** `microservices/trade_logger/main.py`
- **Service:** `quantum-trade-logger.service`
- **Redis Streams:**
  - **CONSUMES:** `quantum:stream:trade.closed`
  - **WRITES KEYS:** `quantum:ledger:{symbol}`, `quantum:ledger:seen_orders`
- **Responsibilities:**
  - Trade history logging
  - Ledger updates on trade closure
  - Position amount tracking (`quantum:ledger:{symbol}.position_amt`)
  - Seen orders deduplication

### 1.4 MARKET DATA & FEATURES

#### **Price Feed Service**
- **File:** `microservices/price_feed/main.py`
- **Service:** `quantum-price-feed.service`
- **Port:** None
- **Redis Streams:**
  - **PUBLISHES:** `quantum:stream:market.tick`
- **External APIs:** **Binance WebSocket** (real-time price stream)
- **Responsibilities:**
  - WebSocket connection to Binance
  - Real-time price feed publishing
  - Stream: `quantum:stream:market.tick`

#### **Market Publisher Service**
- **File:** `microservices/market_publisher/main.py`
- **Service:** `quantum-market-publisher.service`
- **Redis Streams:**
  - **PUBLISHES:** `quantum:stream:market.tick`, `quantum:stream:market.klines`
- **External APIs:** **Binance API** (kline data)
- **Responsibilities:**
  - Market data publishing (INFRA layer)
  - Price tick stream aggregation
  - Kline data publishing

#### **MarketState Publisher (P0.5)**
- **File:** `microservices/market_state_publisher/main.py`
- **Service:** `quantum-marketstate.service`
- **Redis Streams:**
  - **CONSUMES:** `quantum:stream:market.tick`
  - **PUBLISHES:** `quantum:stream:marketstate`
- **Responsibilities:**
  - Market state metrics publishing
  - Volatility tracking
  - Spread monitoring

#### **Feature Publisher (PATH 2.3D)**
- **File:** `microservices/feature_publisher/main.py`
- **Service:** `quantum-feature-publisher.service`
- **Redis Streams:**
  - **CONSUMES:** `quantum:stream:market.tick`, `quantum:stream:market.klines`
  - **PUBLISHES:** `quantum:stream:features`
- **Responsibilities:**
  - Feature engineering pipeline
  - Bridge from market data to AI input
  - Technical indicator calculation

### 1.5 LEARNING & OPTIMIZATION

#### **Continuous Learning Manager (CLM)**
- **File:** `microservices/clm/main.py`
- **Service:** `quantum-clm.service`
- **Redis Streams:**
  - **CONSUMES:** `quantum:stream:clm.intent`
  - **PUBLISHES:** `quantum:stream:model.retrain`
- **Responsibilities:**
  - Model retraining orchestration
  - Learning cadence management
  - Training data pipeline

#### **CLM Minimal Service**
- **File:** `clm_minimal_v2.py`
- **Service:** `quantum-clm-minimal.service`
- **Redis Streams:**
  - **CONSUMES:** `quantum:stream:utf` (XREADGROUP)
  - **PUBLISHES:** `quantum:stream:model.retrain`
- **Responsibilities:**
  - Lightweight CLM implementation
  - Unified Training Feed (UTF) consumption
  - Model retrain trigger

#### **RL Feedback V2 Producer**
- **File:** `microservices/rl_feedback_v2/main.py`
- **Service:** `quantum-rl-feedback-v2.service`
- **Redis Streams:**
  - **CONSUMES:** `quantum:stream:trade.closed`
  - **PUBLISHES:** `quantum:stream:rl_rewards`
- **Responsibilities:**
  - RL reward signal generation from trade outcomes
  - Feedback loop to RL agents
  - Reward normalization

#### **RL Monitor Service**
- **File:** `microservices/rl_monitor/main.py`
- **Service:** `quantum-rl-monitor.service`
- **Redis Streams:**
  - **CONSUMES:** `quantum:stream:rl_rewards`
- **Responsibilities:**
  - RL agent monitoring
  - Performance tracking
  - Reward distribution analysis

#### **RL Position Sizing Agent**
- **File:** `microservices/rl_sizer/main.py`
- **Service:** `quantum-rl-sizer.service`
- **Port:** Model server (inference endpoint)
- **Responsibilities:**
  - RL-based position sizing decisions
  - Dynamic leverage assignment
  - Risk-adjusted qty calculation

### 1.6 CAPITAL ALLOCATION & GOVERNANCE

#### **Capital Allocation Brain (P2.9)**
- **File:** `microservices/capital_allocation/main.py`
- **Service:** `quantum-capital-allocation.service`
- **Redis Streams:**
  - **CONSUMES:** `quantum:stream:portfolio.state`
  - **PUBLISHES:** `quantum:stream:allocation.decision`
  - **WRITES KEYS:** `quantum:capital:allocation:{symbol}` (allocation targets)
- **Responsibilities:**
  - AI-driven capital allocation (not fixed position limits)
  - Per-symbol allocation targets
  - Dynamic rebalancing recommendations
  - Governor checks allocation targets via these keys

#### **Portfolio Risk Governor (P2.8)**
- **File:** `microservices/portfolio_risk_governor/main.py`
- **Service:** `quantum-portfolio-risk-governor.service`
- **Redis Streams:**
  - **CONSUMES:** `quantum:stream:portfolio.state`
  - **PUBLISHES:** `quantum:stream:risk.events`
- **Responsibilities:**
  - Portfolio-level risk enforcement
  - Concentration limits
  - Correlation monitoring
  - Risk event publishing

#### **Portfolio Gate (P2.6)**
- **File:** `microservices/portfolio_gate/main.py`
- **Service:** `quantum-portfolio-gate.service`
- **Redis Streams:**
  - **CONSUMES:** `quantum:stream:trade.intent`
  - **PUBLISHES:** `quantum:stream:portfolio.gate` (filtered intents)
- **Responsibilities:**
  - Pre-governor portfolio-level filtering
  - Sector exposure limits
  - Correlation gates

#### **Heat Gate (P2.6)**
- **File:** `microservices/heat_gate/main.py`
- **Service:** `quantum-heat-gate.service`
- **Redis Streams:**
  - **CONSUMES:** `quantum:stream:apply.plan`
  - **PUBLISHES:** `quantum:stream:apply.heat.observed`
- **Responsibilities:**
  - Execution heat monitoring (rapid trading detection)
  - Cooldown enforcement
  - Overtrading prevention

### 1.7 INTELLIGENCE & DECISION

#### **Decision Intelligence (P3.5)**
- **File:** `microservices/decision_intelligence/main.py`
- **Service:** `quantum-p35-decision-intelligence.service`
- **Redis Streams:**
  - **CONSUMES:** `quantum:stream:ai.decision.made`
  - **PUBLISHES:** `quantum:stream:trade.intent` (enriched)
- **Responsibilities:**
  - Decision enrichment and validation
  - Signal quality scoring
  - Decision audit trail

#### **Autonomous Trader Service**
- **File:** `microservices/autonomous_trader/main.py`
- **Service:** `quantum-autonomous-trader.service`
- **Redis Streams:**
  - **CONSUMES:** `quantum:stream:ai.signal_generated`
  - **PUBLISHES:** `quantum:stream:trade.intent`
- **Responsibilities:**
  - Full RL autonomy mode
  - Signal → Intent direct routing
  - Autonomous decision execution

#### **CEO Brain (AI Client)**
- **File:** `microservices/ceo_brain/main.py`
- **Service:** `quantum-ceo-brain.service`
- **Responsibilities:**
  - AI-powered portfolio oversight
  - Strategic decision recommendations
  - Meta-level governance

#### **Portfolio Intelligence (AI Client)**
- **File:** `microservices/portfolio_intelligence/main.py`
- **Service:** `quantum-portfolio-intelligence.service`
- **Responsibilities:**
  - AI-driven portfolio analytics
  - Opportunity identification
  - Regime-aware recommendations

#### **Strategy Brain (AI Client)**
- **File:** `microservices/strategy_brain/main.py`
- **Service:** `quantum-strategy-brain.service`
- **Responsibilities:**
  - Strategy selection AI
  - Multi-strategy orchestration
  - Adaptive strategy switching

#### **Risk Brain (AI Client)**
- **File:** `microservices/risk_brain/main.py`
- **Service:** `quantum-risk-brain.service`
- **Responsibilities:**
  - AI-powered risk assessment
  - Predictive risk scoring
  - Tail risk detection

### 1.8 PERFORMANCE & ATTRIBUTION

#### **Performance Attribution (P3.0)**
- **File:** `microservices/performance_attribution/main.py`
- **Service:** `quantum-performance-attribution.service`
- **Redis Streams:**
  - **CONSUMES:** `quantum:stream:trade.closed`
  - **PUBLISHES:** `quantum:stream:attribution.results`
- **Responsibilities:**
  - Trade PnL attribution
  - Strategy performance breakdown
  - Factor contribution analysis

#### **Performance Tracker**
- **File:** `microservices/performance_tracker/main.py`
- **Service:** `quantum-performance-tracker.service`
- **Responsibilities:**
  - Real-time performance metrics
  - Sharpe ratio tracking
  - Drawdown monitoring

#### **Harvest Metrics Exporter (P2.7)**
- **File:** `microservices/harvest_metrics_exporter/main.py`
- **Service:** `quantum-harvest-metrics-exporter.service`
- **Responsibilities:**
  - Prometheus metrics export for harvest brain
  - Harvest performance tracking

#### **Safety Telemetry Exporter (P1)**
- **File:** `microservices/safety_telemetry/main.py`
- **Service:** `quantum-safety-telemetry.service`
- **Responsibilities:**
  - Safety system metrics export
  - ESS state monitoring
  - Risk limit breach tracking

#### **RL Shadow Metrics Exporter**
- **File:** `configs/quantum-rl-shadow-metrics-exporter.service`
- **Service:** `quantum-rl-shadow-metrics-exporter.service`
- **Responsibilities:**
  - RL shadow mode performance tracking
  - Model evaluation metrics

### 1.9 INFRASTRUCTURE SERVICES

#### **Balance Tracker**
- **File:** `microservices/balance_tracker/main.py`
- **Service:** `quantum-balance-tracker.service`
- **Redis Streams:**
  - **PUBLISHES:** `quantum:stream:account.balance`
- **External APIs:** **Binance Futures API** (account balance polling)
- **Responsibilities:**
  - Account balance monitoring
  - Equity tracking
  - Margin usage publishing

#### **Exchange Stream Bridge**
- **File:** `microservices/exchange_stream_bridge/main.py`
- **Service:** `quantum-exchange-stream-bridge.service`
- **Redis Streams:**
  - **PUBLISHES:** `quantum:stream:exchange.normalized`, `quantum:stream:exchange.raw`
- **External APIs:** **Binance WebSocket** (user data stream)
- **Responsibilities:**
  - Multi-source exchange event ingestion
  - Event normalization
  - Fill event publishing

#### **Execution Result Bridge**
- **File:** `microservices/execution_result_bridge/main.py`
- **Service:** `quantum-execution-result-bridge.service`
- **Redis Streams:**
  - **CONSUMES:** `quantum:stream:trade.execution.res`
  - **PUBLISHES:** `quantum:stream:execution.result`
- **Responsibilities:**
  - Stream format translation
  - Execution result normalization

#### **Dashboard API**
- **File:** `dashboard_v4/backend/main.py`
- **Service:** `quantum-dashboard-api.service`
- **Port:** 8080 (HTTP API)
- **Responsibilities:**
  - Web dashboard backend
  - Real-time metrics API
  - Position/PnL visualization

#### **Universe Service**
- **File:** `microservices/universe_service/main.py`
- **Service:** `quantum-universe-service.service`
- **Responsibilities:**
  - Trading universe management
  - Symbol whitelist/blacklist
  - Dynamic symbol filtering

### 1.10 REDIS SERVER

#### **Redis Server**
- **Service:** `redis-server` (system service)
- **Port:** 6379
- **Persistence:** RDB snapshots + AOF
- **Responsibilities:**
  - Stream broker (38 active streams)
  - State store (position, ledger, permits, cooldowns)
  - Locking primitives (SETNX)
  - Pub/Sub (EventBus)

---

## 2. EVENT FLOW MAPPING

### 2.1 LONG TRADE - COMPLETE LIFECYCLE

**Scenario:** AI Engine predicts bullish signal → открытие LONG → profit harvest → complete close

```
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 1: ENSEMBLE PREDICTION                                        │
└─────────────────────────────────────────────────────────────────────┘

[Price Feed Service]
  ↓ (WebSocket: Binance price stream)
  ↓ XADD quantum:stream:market.tick
  ↓ {symbol, price, timestamp, bid, ask, volume}
  ↓
[Market Publisher]
  ↓ XADD quantum:stream:market.klines
  ↓ {symbol, interval, open, high, low, close, volume}
  ↓
[Feature Publisher (PATH 2.3D)]
  ↓ XREADGROUP quantum:stream:market.tick + market.klines
  ↓ Feature engineering (RSI, MACD, BB, ATR, price deltas)
  ↓ XADD quantum:stream:features
  ↓ {symbol, features: [f1, f2, ...fn], timestamp}
  ↓
[AI Engine]
  ↓ XREADGROUP quantum:stream:market.tick (EventBus subscription)
  ↓
  ├─ Model Inference:
  │    XGBoost → prediction_xgb
  │    LightGBM → prediction_lgbm
  │    N-HiTS → prediction_nhits
  │    PatchTST → prediction_patchtst
  │
  ├─ Ensemble Voting:
  │    Weighted average → ensemble_signal
  │    Confidence calculation → confidence_score
  │
  ├─ Meta Agent (RL-based strategy selection):
  │    Regime detection → regime_id
  │    Strategy selection → selected_strategy
  │
  ├─ RL Position Sizer:
  │    Risk assessment → position_size_qty
  │    Leverage assignment → leverage
  │
  ↓ (Decision: LONG BTCUSDT, confidence=0.87)
  ↓
  ↓ XADD quantum:stream:ai.signal_generated
  ↓ {symbol, side, confidence, strategy, regime, timestamp}
  ↓
  ↓ XADD quantum:stream:ai.decision.made
  ↓ {symbol, side, confidence, signal_strength, models_agree}
  ↓
  ↓ XADD quantum:stream:trade.intent
  ↓ {
      symbol: "BTCUSDT",
      side: "BUY",
      action: "OPEN",
      leverage: 10,
      qty: 0.05,
      confidence: 0.87,
      strategy: "momentum_rl",
      regime: "trending_up",
      timestamp: "2026-02-18T12:00:00Z"
    }

┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 2: INTENT FILTERING                                           │
└─────────────────────────────────────────────────────────────────────┘

[Intent Bridge]
  ↓ XREADGROUP quantum:stream:trade.intent
  ↓ (Group: quantum:group:intent-bridge:trade.intent, Consumer: intent-bridge-1)
  ↓
  ├─ PolicyStore Check:
  │    Reads: quantum:policy:active → {autonomy: "enabled", ...}
  │    Status: ✅ Autonomy enabled
  │
  ├─ Allowlist Filter:
  │    Reads: INTENT_BRIDGE_ALLOWLIST="BTCUSDT,ETHUSDT,..."
  │    Check: BTCUSDT ∈ allowlist → ✅ PASS
  │
  ├─ Exposure Check:
  │    Reads: quantum:portfolio:state → {exposure_pct: 45%}
  │    Check: 45% < MAX_EXPOSURE_PCT(80%) → ✅ PASS
  │
  ├─ Flat Position Skip:
  │    Reads: quantum:ledger:BTCUSDT → {position_amt: 0.0}
  │    Check: action=OPEN && position_amt=0 → ✅ PASS (not flat SELL)
  │
  ├─ Idempotency (Deduplication):
  │    Hash: md5(symbol+side+action+timestamp) → "a3f7b2c1"
  │    SETNX: quantum:dedup:intent:a3f7b2c1 EX 900
  │    Result: 1 (key set) → ✅ PASS (not duplicate)
  │
  ↓ (All gates passed)
  ↓
  ↓ XADD quantum:stream:apply.plan
  ↓ {
      plan_id: "f1a8d7f4",
      symbol: "BTCUSDT",
      action: "OPEN",
      side: "LONG",
      qty: 0.05,
      leverage: 10,
      confidence: 0.87,
      kill_score: 0.12,  ← (inverse of confidence for now)
      decision: "EXECUTE",
      source: "intent_bridge",
      timestamp: "2026-02-18T12:00:05Z"
    }

┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 3: GOVERNANCE (Governor P3.2)                                 │
└─────────────────────────────────────────────────────────────────────┘

[Governor Service]
  ↓ XREADGROUP quantum:stream:apply.plan
  ↓ (Group: quantum:group:governor:apply.plan, Consumer: governor-1)
  ↓ Reads plan_id: f1a8d7f4
  ↓
  ├─ Kill Score Gate:
  │    kill_score: 0.12
  │    Check: 0.12 < GOV_KILL_SCORE_CRITICAL(0.8) → ✅ PASS
  │    Check: action=OPEN && 0.12 < GOV_KILL_SCORE_OPEN_THRESHOLD(0.85) → ✅ PASS
  │
  ├─ Position Count Limit:
  │    SCAN: quantum:position:* → [ETHUSDT, ADAUSDT, ALGOUSDT, ...] (7 positions)
  │    Check: 7 < GOV_MAX_OPEN_POSITIONS(10) → ✅ PASS (3 slots left)
  │
  ├─ Rate Limit (Per-Hour):
  │    ZCOUNT: quantum:governor:exec_history:BTCUSDT (now-3600, now)
  │    Result: 1 execution in last hour
  │    Check: 1 < GOV_MAX_EXEC_PER_HOUR(3) → ✅ PASS
  │
  ├─ Rate Limit (Per-5min):
  │    ZCOUNT: quantum:governor:exec_history:BTCUSDT (now-300, now)
  │    Result: 0 executions in last 5min
  │    Check: 0 < GOV_MAX_EXEC_PER_5MIN(2) → ✅ PASS
  │
  ├─ Notional Limit:
  │    qty=0.05, price=50000 → notional=$2500
  │    Check: $2500 > GOV_MAX_NOTIONAL_PER_TRADE_USDT($200) → ❌ BLOCK
  │    (In testnet, this would block. Assuming qty adjusted to 0.004 → $200)
  │    Adjusted notional: $200
  │    Check: $200 ≤ $200 → ✅ PASS
  │
  ├─ Total Notional Limit:
  │    Reads: quantum:position:* (all positions)
  │    Sum notional: $1200 (existing positions)
  │    New total: $1200 + $200 = $1400
  │    Check: $1400 < GOV_MAX_TOTAL_NOTIONAL_USDT($2000) → ✅ PASS
  │
  ├─ Cooldown Check:
  │    GET: quantum:cooldown:last_exec_ts:BTCUSDT → 1739880000 (1 hour ago)
  │    Now: 1739883600
  │    Delta: 3600s
  │    Check: 3600s > GOV_SYMBOL_COOLDOWN_SECONDS(60) → ✅ PASS
  │
  ↓ (All gates PASSED - PERMIT GRANTED)
  ↓
  ↓ SET quantum:permit:p33:f1a8d7f4 "ALLOW" EX 60
  ↓ (Permit written with 60s TTL)
  ↓
  ↓ ZADD quantum:governor:exec_history:BTCUSDT <timestamp> <plan_id>
  ↓ (Record execution attempt for rate limiting)
  ↓
  ↓ Prometheus Metrics:
  ↓   quantum_govern_allow_total{symbol="BTCUSDT"} += 1

┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 4: APPLY LAYER EXECUTION (P3)                                 │
└─────────────────────────────────────────────────────────────────────┘

[Apply Layer]
  ↓ XREADGROUP quantum:stream:apply.plan
  ↓ (Group: quantum:group:apply:apply.plan, Consumer: apply-1)
  ↓ Reads plan_id: f1a8d7f4
  ↓
  ├─ Governor Permit Check:
  │    GET: quantum:permit:p33:f1a8d7f4
  │    Result: "ALLOW" (TTL: 58s remaining)
  │    Status: ✅ PERMIT VALID
  │
  ├─ Allowlist Gate:
  │    Reads: APPLY_ALLOWLIST="BTCUSDT"
  │    Check: BTCUSDT ∈ allowlist → ✅ PASS
  │
  ├─ Kill Score Gate:
  │    kill_score: 0.12
  │    Check: 0.12 < 0.8 (critical) → ✅ PASS
  │    Check: action=OPEN && 0.12 < 0.85 → ✅ PASS
  │
  ├─ Entry/Exit Separation:
  │    action: "OPEN"
  │    kill_score: 0.12
  │    Check: 0.12 < 0.85 (open threshold) → ✅ PASS
  │    (If action=CLOSE, would check: kill_score ≥ 0.65)
  │
  ├─ Exit Ownership Enforcement:
  │    action: "OPEN" → No exit ownership check needed
  │    (If CLOSE/REDUCE: would validate EXIT_OWNER="exitbrain_v3_5")
  │
  ├─ Idempotency Check:
  │    Hash: md5(plan_id+symbol+action+timestamp) → "b7c3f9a2"
  │    SETNX: quantum:dedup:apply:b7c3f9a2 EX 3600
  │    Result: 1 (key set) → ✅ NOT DUPLICATE
  │
  ├─ Position State Read:
  │    GET: quantum:ledger:BTCUSDT → {position_amt: 0.0}
  │    Current position: 0.0 (flat)
  │    Action: OPEN → ✅ Valid (no conflict)
  │
  ├─ Kill Switch Check:
  │    Reads: APPLY_KILL_SWITCH="false"
  │    Status: ✅ NOT ENGAGED
  │
  ├─ Dry Run Check:
  │    Reads: APPLY_DRY_RUN="false", APPLY_TESTNET="true"
  │    Mode: TESTNET EXECUTION (real orders to testnet)
  │
  ↓ (All pre-flight checks PASSED)
  ↓
  ↓ BINANCE API CALL:
  ↓
  POST https://testnet.binancefuture.com/fapi/v1/order
  Headers:
    X-MBX-APIKEY: w2W60kzuCfPJKGIqSvmp0pqISUO8XKICjc5sD8QyJuJpp9LKQgvXKhtd09Ii3rwg
  Body (HMAC-SHA256 signed):
    symbol=BTCUSDT
    side=BUY
    type=MARKET
    quantity=0.004  ← (qty adjusted by Governor)
    leverage=10
    recvWindow=5000
    timestamp=1739883600000
  ↓
  ↓ BINANCE RESPONSE:
  {
    "orderId": 9876543210,
    "symbol": "BTCUSDT",
    "status": "FILLED",
    "executedQty": "0.004",
    "avgPrice": "50000.00",
    "side": "BUY",
    "type": "MARKET",
    "updateTime": 1739883601000
  }
  ↓ (Order FILLED immediately - market order)
  ↓
  ↓ State Update:
  │    SET: quantum:position:ledger:BTCUSDT
  │    {
  │      position_amt: 0.004,  ← (updated from 0.0)
  │      entry_price: 50000.0,
  │      leverage: 10,
  │      notional: 200.0,
  │      unrealizedPnl: 0.0,
  │      timestamp: 1739883601
  │    }
  ↓
  ↓ XADD quantum:stream:apply.result
  ↓ {
      plan_id: "f1a8d7f4",
      symbol: "BTCUSDT",
      action: "OPEN",
      executed: true,
      order_id: 9876543210,
      executed_qty: 0.004,
      avg_price: 50000.0,
      side: "BUY",
      timestamp: "2026-02-18T12:00:06Z",
      commission: 0.02,  ← (0.04% taker fee)
      commission_asset: "USDT"
    }
  ↓
  ↓ Prometheus Metrics:
  │    apply_layer_exec_total{symbol="BTCUSDT", action="OPEN"} += 1
  │    apply_layer_exec_success_total{symbol="BTCUSDT"} += 1

┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 5: STATE RECONCILIATION                                       │
└─────────────────────────────────────────────────────────────────────┘

[Position State Brain (P3.3)]
  ↓ XREADGROUP quantum:stream:apply.result
  ↓ (Group: quantum:group:position-state:apply.result)
  ↓
  ├─ Aggregate Position State:
  │    Previous: quantum:position:BTCUSDT → None (new position)
  │    New state from apply.result:
  │      position_amt: 0.004 BTC
  │      entry_price: 50000.0
  │      leverage: 10x
  │      notional: $200
  │      unrealized_pnl: $0
  │
  ↓ SET quantum:position:snapshot:BTCUSDT
  ↓ {
      symbol: "BTCUSDT",
      position_amt: 0.004,
      entry_price: 50000.0,
      mark_price: 50000.0,
      leverage: 10,
      notional: 200.0,
      unrealized_pnl: 0.0,
      margin: 20.0,  ← (200 / 10)
      liquidation_price: 45500.0,  ← (approx, based on leverage)
      timestamp: 1739883601,
      last_updated_by: "position_state_brain"
    }
  ↓
  ↓ XADD quantum:stream:position.snapshot
  ↓ {symbol: "BTCUSDT", snapshot: {...}, event: "POSITION_OPENED"}

[Trade Logger]
  ↓ (Listens to position.snapshot or apply.result)
  ↓
  ↓ HSET quantum:ledger:BTCUSDT
  ↓   position_amt 0.004
  ↓   entry_price 50000.0
  ↓   leverage 10
  ↓   last_update 1739883601
  ↓
  ↓ SADD quantum:ledger:seen_orders 9876543210
  ↓ (Dedup tracking)

[Reconcile Engine (P3.4)]
  ↓ (Periodic polling: every 60s)
  ↓ GET https://testnet.binancefuture.com/fapi/v2/positionRisk
  ↓ Binance Response:
  [
    {
      "symbol": "BTCUSDT",
      "positionAmt": "0.004",
      "entryPrice": "50000.0",
      "markPrice": "50100.0",  ← (price moved up)
      "unRealizedProfit": "0.40",  ← ((50100-50000) * 0.004)
      "leverage": "10",
      "liquidationPrice": "45500.0"
    }
  ]
  ↓
  ├─ Compare with Internal State:
  │    Internal: quantum:position:snapshot:BTCUSDT
  │      position_amt: 0.004 ✅ MATCH
  │      entry_price: 50000.0 ✅ MATCH
  │      unrealized_pnl: 0.0 ❌ DRIFT (exchange: $0.40)
  │
  ├─ Reconciliation Action:
  │    UPDATE quantum:position:snapshot:BTCUSDT
  │      mark_price: 50100.0  ← (from exchange)
  │      unrealized_pnl: 0.40  ← (from exchange, authoritative)
  │
  ↓ XADD quantum:stream:reconcile.events
  ↓ {
      symbol: "BTCUSDT",
      event: "DRIFT_CORRECTED",
      field: "unrealized_pnl",
      internal_value: 0.0,
      exchange_value: 0.40,
      timestamp: 1739883660
    }
  ↓
  ↓ Prometheus Metrics:
  │    reconcile_drift_detected_total{symbol="BTCUSDT"} += 1

┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 6: EXIT LOGIC (HarvestBrain Monitoring)                       │
└─────────────────────────────────────────────────────────────────────┘

[HarvestBrain]
  ↓ XREADGROUP quantum:stream:apply.result
  ↓ (Group: harvest_brain:execution)
  ↓
  ├─ Position Sync (on startup or periodically):
  │    GET https://testnet.binancefuture.com/fapi/v2/positionRisk
  │    Loads: BTCUSDT position (amt=0.004, entry=50000)
  │
  ├─ PnL Enrichment:
  │    Current mark price: 50100
  │    Entry price: 50000
  │    Unrealized PnL: (50100 - 50000) * 0.004 = $0.40
  │    PnL %: 0.40 / 20 (margin) = 2.0%
  │    R-multiple: 0.40 / 20 = 0.02R (initial risk = margin)
  │
  ├─ Harvest Policy Evaluation (P2 risk_kernel_harvest):
  │    R-multiple: 0.02R
  │    Check: 0.02R < HARVEST_MIN_R(0.5R) → ❌ NO HARVEST YET
  │    (Position still developing, no action)
  │
  ↓ (Wait for more profit accumulation)
  ↓ Time passes... price moves to $51500 (3% profit)
  ↓
  ├─ Re-evaluation (5s scan interval):
  │    Mark price: 51500
  │    Unrealized PnL: (51500 - 50000) * 0.004 = $6.00
  │    R-multiple: 6.00 / 20 = 0.30R
  │    Check: 0.30R < 0.5R → ❌ STILL BELOW MIN HARVEST
  │
  ↓ Price continues... $52500 (5% profit)
  ↓
  ├─ Re-evaluation:
  │    Mark price: 52500
  │    Unrealized PnL: (52500 - 50000) * 0.004 = $10.00
  │    R-multiple: 10.00 / 20 = 0.50R
  │    Check: 0.50R ≥ HARVEST_MIN_R(0.5R) → ✅ HARVEST TRIGGERED!
  │    Action: HARVEST_PARTIAL (25% close per policy)
  │
  ├─ Harvest Deduplication:
  │    Hash: md5(symbol+action+R-level+timestamp) → "c9d2e4f1"
  │    SETNX: quantum:dedup:harvest:BTCUSDT:c9d2e4f1 EX 900
  │    Result: 1 (key set) → ✅ NOT DUPLICATE
  │
  ├─ Harvest Execution (LIVE MODE):
  │    qty_to_close: 0.004 * 0.25 = 0.001 BTC
  │    plan_id: hash(symbol+action+qty+timestamp) = "h7f2a9c3"
  │
  ↓ Auto-create Governor Permit (HarvestBrain bypass):
  │    SET quantum:permit:p33:h7f2a9c3 "ALLOW_HARVEST" EX 60
  │
  ↓ XADD quantum:stream:apply.plan
  ↓ {
      plan_id: "h7f2a9c3",
      symbol: "BTCUSDT",
      action: "PARTIAL_CLOSE_25",
      side: "SELL",
      qty: 0.001,  ← (reduce-only)
      reduce_only: true,
      reason: "HARVEST_0.5R",
      kill_score: 0.0,  ← (exit, no kill score)
      decision: "EXECUTE",
      source: "harvest_brain",
      timestamp: "2026-02-18T12:05:30Z"
    }

[Apply Layer] (receives harvest plan)
  ↓ XREADGROUP quantum:stream:apply.plan
  ↓ Reads plan_id: h7f2a9c3
  ↓
  ├─ Governor Permit Check:
  │    GET: quantum:permit:p33:h7f2a9c3 → "ALLOW_HARVEST"
  │    Status: ✅ PERMIT VALID (HarvestBrain auto-permit)
  │
  ├─ Exit Ownership Check:
  │    action: "PARTIAL_CLOSE_25"
  │    Expected EXIT_OWNER: "exitbrain_v3_5" OR "harvest_brain"
  │    source: "harvest_brain" → ✅ AUTHORIZED EXIT CONTROLLER
  │
  ├─ Reduce-Only Validation:
  │    reduce_only: true
  │    Current position: 0.004 BTC LONG
  │    Order: SELL 0.001 BTC → ✅ VALID REDUCTION
  │
  ↓ BINANCE API CALL (CLOSE ORDER):
  POST https://testnet.binancefuture.com/fapi/v1/order
  Body:
    symbol=BTCUSDT
    side=SELL
    type=MARKET
    quantity=0.001
    reduceOnly=true  ← (critical flag)
  ↓
  ↓ BINANCE RESPONSE:
  {
    "orderId": 9876543220,
    "symbol": "BTCUSDT",
    "status": "FILLED",
    "executedQty": "0.001",
    "avgPrice": "52500.00",
    "side": "SELL",
    "reduceOnly": true
  }
  ↓
  ↓ State Update:
  │    GET: quantum:position:ledger:BTCUSDT
  │    Previous position_amt: 0.004
  │    New position_amt: 0.004 - 0.001 = 0.003
  │
  │    Realized PnL calculation:
  │      Closed qty: 0.001
  │      Entry price: 50000
  │      Exit price: 52500
  │      PnL: (52500 - 50000) * 0.001 = $2.50
  │      Commission: 0.001 * 52500 * 0.0004 = $0.021
  │      Net PnL: $2.50 - $0.021 = $2.48
  │
  │    UPDATE: quantum:position:ledger:BTCUSDT
  │      position_amt: 0.003
  │      realized_pnl: +2.48  ← (add to cumulative)
  │      unrealized_pnl: (52500 - 50000) * 0.003 = $7.50
  │
  ↓ XADD quantum:stream:apply.result
  ↓ {
      plan_id: "h7f2a9c3",
      symbol: "BTCUSDT",
      action: "PARTIAL_CLOSE_25",
      executed: true,
      executed_qty: 0.001,
      avg_price: 52500.0,
      side: "SELL",
      realized_pnl: 2.48,
      remaining_position: 0.003,
      timestamp: "2026-02-18T12:05:31Z"
    }

┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 7: COMPLETE CLOSE (Exit Brain V3 or Risk Safety)              │
└─────────────────────────────────────────────────────────────────────┘

[Exit Brain V3] (parallel monitoring, NOT triggered in this scenario)
  ↓ (Polls Binance every 10s)
  ↓ GET https://testnet.binancefuture.com/fapi/v2/positionRisk
  ↓
  ├─ Internal State Tracking:
  │    PositionExitState["BTCUSDT:LONG"]:
  │      entry_price: 50000
  │      active_sl: 49000  ← (AI-driven dynamic SL, 2% below entry)
  │      hard_sl_price: 48500  ← (Exchange STOP_MARKET order, 3% below)
  │      tp_levels: [51000, 52000, 53000]  ← (dynamic TP ladder)
  │      current_position_amt: 0.003  ← (after partial harvest)
  │
  ├─ Price Check:
  │    Current mark_price: 52500
  │    Check: 52500 < active_sl(49000)? → ❌ NO
  │    Check: 52500 ≥ tp_levels[0](51000)? → ✅ YES (TP1 hit earlier)
  │    Action: TP1 hit → SL ratchet (if RATCHET_SL_ENABLED=true)
  │      New active_sl: 51000 (lock in 2% profit)
  │
  ↓ (Continues monitoring, no immediate action)
  ↓ Price drops suddenly to $48800 (below active_sl)
  ↓
  ├─ SL Breach Detection:
  │    mark_price: 48800
  │    active_sl: 49000  ← (initial, before ratchet in this example)
  │    Check: 48800 < 49000 → ✅ STOP LOSS HIT!
  │    Action: EMERGENCY CLOSE (internal SL, not exchange order)
  │
  ├─ Exit Execution:
  │    Sends MARKET SELL via ExitOrderGateway
  │    qty: 0.003 (remaining position)
  │    reduce_only: true
  │
  ↓ ExitOrderGateway → Execution Service → Binance
  ↓
  ↓ BINANCE RESPONSE:
  {
    "orderId": 9876543230,
    "status": "FILLED",
    "executedQty": "0.003",
    "avgPrice": "48800.00",
    "side": "SELL"
  }
  ↓
  ↓ Final PnL:
  │    Closed qty: 0.003
  │    Entry: 50000
  │    Exit: 48800
  │    Loss: (48800 - 50000) * 0.003 = -$3.60
  │    Commission: $0.06
  │    Net realized PnL: -$3.60 - $0.06 = -$3.66
  │
  │    **Total trade PnL:**
  │      Partial close: +$2.48
  │      Final close: -$3.66
  │      **Net: -$1.18** (overall loss despite partial harvest)
  │
  ↓ XADD quantum:stream:trade.closed
  ↓ {
      symbol: "BTCUSDT",
      side: "LONG",
      entry_price: 50000.0,
      exit_price: 48800.0,  ← (weighted avg of all closes)
      total_qty: 0.004,
      realized_pnl: -1.18,
      exit_reason: "STOP_LOSS_HIT",
      exit_controller: "exitbrain_v3_5",
      timestamp: "2026-02-18T12:10:45Z"
    }

[Trade Logger]
  ↓ XREADGROUP quantum:stream:trade.closed
  ↓
  ↓ HSET quantum:ledger:BTCUSDT
  ↓   position_amt 0.0  ← (position closed)
  ↓   realized_pnl -1.18  ← (cumulative for symbol)
  ↓   total_trades 1
  ↓   last_close_timestamp 1739884245
  ↓
  ↓ Log to trade history database / file

[Performance Attribution (P3.0)]
  ↓ XREADGROUP quantum:stream:trade.closed
  ↓
  ↓ Attribution Analysis:
  │    Strategy: "momentum_rl"
  │    Regime: "trending_up" → "reversal" (regime changed mid-trade)
  │    Entry quality: 0.87 confidence
  │    Exit quality: Stop loss hit (adverse move)
  │    PnL attribution:
  │      Strategy contribution: -$1.18
  │      Regime factor: -0.5 (regime change penalty)
  │      Execution slippage: -$0.02 (avg)
  │
  ↓ XADD quantum:stream:attribution.results
  ↓ {symbol: "BTCUSDT", pnl: -1.18, attribution: {...}}

[RL Feedback V2 Producer]
  ↓ XREADGROUP quantum:stream:trade.closed
  ↓
  ↓ Reward Calculation:
  │    PnL: -$1.18
  │    Risk taken: $20 (margin)
  │    Return: -1.18 / 20 = -5.9%
  │    Reward: normalize(-5.9%) = -0.59  ← (RL reward signal)
  │
  ↓ XADD quantum:stream:rl_rewards
  ↓ {
      symbol: "BTCUSDT",
      action_taken: "OPEN_LONG",
      reward: -0.59,
      episode_return: -5.9,
      terminal: true,
      state_features: [f1, f2, ...],
      timestamp: 1739884245
    }

[CLM / RL Monitor]
  ↓ XREADGROUP quantum:stream:rl_rewards
  ↓ Stores reward for model retraining
  ↓ Updates policy gradient / Q-learning targets
```

---

## 3. REDIS TRUTH MAP

### 3.1 REDIS STREAMS (38 Active)

| Stream Name | Producer(s) | Consumer(s) | Schema | Consumer Groups | Delivery Semantics |
|------------|-------------|-------------|--------|----------------|-------------------|
| `quantum:stream:market.tick` | Price Feed, Market Publisher | AI Engine, Feature Publisher, MarketState | {symbol, price, timestamp, bid, ask, volume} | EventBus subscriptions | At-least-once (XREADGROUP) |
| `quantum:stream:market.klines` | Market Publisher | Feature Publisher, AI Engine | {symbol, interval, open, high, low, close, volume} | `quantum:group:features:klines` | At-least-once |
| `quantum:stream:features` | Feature Publisher | Ensemble Predictor | {symbol, features: [f1...fn], timestamp} | `quantum:group:ensemble:features` | At-least-once |
| `quantum:stream:ai.signal_generated` | AI Engine, Ensemble Predictor | Autonomous Trader, Decision Intelligence | {symbol, side, confidence, strategy, regime} | `quantum:group:autonomous:signals` | At-least-once |
| `quantum:stream:ai.decision.made` | AI Engine | Decision Intelligence | {symbol, side, confidence, signal_strength, models_agree} | `quantum:group:decision:ai` | At-least-once |
| `quantum:stream:trade.intent` | AI Engine, Autonomous Trader, Exit Monitor | **Intent Bridge**, Portfolio Gate | {symbol, side, action, leverage, qty, confidence, strategy} | `quantum:group:intent-bridge:trade.intent` | **At-least-once + ACK** |
| `quantum:stream:apply.plan` | **Intent Bridge**, **HarvestBrain** (live), Apply Layer | **Governor**, **Apply Layer**, Heat Gate | {plan_id, symbol, action, side, qty, leverage, confidence, kill_score, decision} | `quantum:group:governor:apply.plan`, `quantum:group:apply:apply.plan` | **At-least-once + ACK** |
| `quantum:stream:apply.plan.manual` | Manual intervention scripts | Governor, Apply Layer | {plan_id, symbol, action, source: "manual"} | Same as apply.plan | At-least-once |
| `quantum:stream:apply.result` | **Apply Layer** | HarvestBrain, Position State Brain, Execution Result Bridge | {plan_id, symbol, action, executed, order_id, executed_qty, avg_price, side, realized_pnl, timestamp} | `harvest_brain:execution`, `quantum:group:position-state:apply.result` | **At-least-once + ACK** |
| `quantum:stream:execution.result` | Execution Service, Execution Result Bridge | Position State Brain, Portfolio State | {order_id, symbol, status, filled_qty, avg_price, side, commission} | `quantum:group:position:execution` | At-least-once |
| `quantum:stream:trade.execution.res` | Execution Service | Execution Result Bridge | {order_id, symbol, result, timestamp} | `quantum:group:exec-bridge:execution` | At-least-once |
| `quantum:stream:harvest.intent` | (Legacy/unused?) | None | N/A | None | N/A |
| `quantum:stream:harvest.proposal` | Harvest Proposal Publisher (P2.5) | Apply Layer (shadow mode) | {symbol, action, qty, reason, confidence} | `quantum:group:apply:harvest` | At-least-once |
| `quantum:stream:harvest.suggestions` | HarvestBrain (shadow mode) | Monitoring/Analytics | {symbol, action, R_level, reason} | `quantum:group:monitor:harvest` | At-least-once |
| `quantum:stream:trade.closed` | Exit Brain V3, Exit Monitor, Apply Layer | Trade Logger, Performance Attribution, RL Feedback V2 | {symbol, side, entry_price, exit_price, total_qty, realized_pnl, exit_reason, timestamp} | `quantum:group:logger:closed`, `quantum:group:attribution:closed`, `quantum:group:rl:closed` | **At-least-once + ACK** |
| `quantum:stream:position.snapshot` | Position State Brain | Portfolio State Publisher, Reconcile Engine | {symbol, snapshot: {...}, event} | `quantum:group:portfolio:snapshots` | At-least-once |
| `quantum:stream:portfolio.state` | Portfolio State Publisher | Capital Allocation, Portfolio Risk Governor, Portfolio Gate | {total_equity, total_unrealized_pnl, exposure_pct, open_positions, timestamp} | `quantum:group:allocation:portfolio`, `quantum:group:risk:portfolio` | At-least-once |
| `quantum:stream:portfolio.snapshot_updated` | Portfolio State Publisher | Dashboard API, Analytics | {timestamp, snapshot: {...}} | `quantum:group:dashboard:portfolio` | At-least-once |
| `quantum:stream:portfolio.exposure_updated` | Exposure Balancer | Governor, Portfolio Gate | {exposure_pct, symbols: [...], timestamp} | None (KEY read, not stream) | N/A |
| `quantum:stream:portfolio.gate` | Portfolio Gate | Governor | {symbol, allowed, reason} | `quantum:group:governor:gate` | At-least-once |
| `quantum:stream:portfolio.cluster_state` | Portfolio Clusters (P2.7) | Portfolio Intelligence | {clusters: [...], correlations: {...}} | `quantum:group:intelligence:clusters` | At-least-once |
| `quantum:stream:allocation.decision` | Capital Allocation (P2.9) | Governor (reads keys, not stream) | {symbol, allocation_pct, target_notional} | None (KEY read) | N/A |
| `quantum:stream:apply.heat.observed` | Apply Layer (Heat Observer P2.8A) | Heat Gate, Monitoring | {symbol, heat_level, timestamp} | `quantum:group:heat:observed` | At-least-once |
| `quantum:stream:bsc.events` | Baseline Safety Controller | Monitoring, Alerting | {event_type, severity, message, timestamp} | `quantum:group:monitor:bsc` | At-least-once |
| `quantum:stream:reconcile.events` | Reconcile Engine (P3.4) | Monitoring, Alerting | {symbol, event, field, internal_value, exchange_value, timestamp} | `quantum:group:monitor:reconcile` | At-least-once |
| `quantum:stream:reconcile.close` | Reconcile Engine | Apply Layer (close recommendations) | {symbol, reason, drift_severity} | `quantum:group:apply:reconcile` | At-least-once |
| `quantum:stream:risk.events` | Portfolio Risk Governor | Risk Safety, BSC, Monitoring | {event_type, severity, symbols, reason} | `quantum:group:safety:risk` | At-least-once |
| `quantum:stream:account.balance` | Balance Tracker | Portfolio State, Dashboard API | {balance, available_margin, used_margin, timestamp} | `quantum:group:portfolio:balance` | At-least-once |
| `quantum:stream:exchange.normalized` | Exchange Stream Bridge | Execution Service, Position State | {event_type, symbol, data: {...}, timestamp} | `quantum:group:execution:exchange` | At-least-once |
| `quantum:stream:exchange.raw` | Exchange Stream Bridge | Debugging, Analytics | {raw_event: {...}} | `quantum:group:analytics:raw` | At-least-once |
| `quantum:stream:exitbrain.pnl` | (Test/manual injection) | RL Feedback, Exit Brain | {symbol, pnl, tp, sl, leverage, confidence} | `quantum:group:rl:exitbrain` | At-least-once |
| `quantum:stream:rl_rewards` | RL Feedback V2 Producer | RL Sizer, RL Monitor, CLM | {symbol, action_taken, reward, episode_return, terminal, state_features} | `quantum:group:clm:rewards`, `quantum:group:monitor:rl` | At-least-once |
| `quantum:stream:model.retrain` | CLM, CLM Minimal | Retrain Worker | {model_name, trigger_reason, timestamp} | `quantum:group:retrain:worker` | At-least-once |
| `quantum:stream:clm.intent` | (External triggers) | CLM | {intent, params} | `quantum:group:clm:intent` | At-least-once |
| `quantum:stream:policy.update` | (Admin/manual) | Risk Safety, PolicyStore | {policy_update: {...}} | `quantum:group:safety:policy` | At-least-once |
| `quantum:stream:policy.updated` | Risk Safety | AI Engine, Governor, Apply Layer | {policy_snapshot: {...}} | EventBus subscriptions | Pub/Sub (no consumer groups) |
| `quantum:stream:policy.audit` | PolicyStore | Monitoring, Compliance | {change, old_value, new_value, timestamp} | `quantum:group:monitor:policy` | At-least-once |
| `quantum:stream:marketstate` | MarketState Publisher | AI Engine, Regime Detector | {symbol, volatility, spread, timestamp} | `quantum:group:ai:marketstate` | At-least-once |
| `quantum:stream:utf` | UTF Publisher | CLM Minimal | {features, labels, metadata} | `quantum:group:clm:utf` | At-least-once |
| `quantum:stream:signal.score` | (Legacy/experimental) | None | N/A | None | N/A |

### 3.2 CRITICAL STATE KEYS

#### Position State Keys

| Key Pattern | Owner | Format | TTL | Purpose |
|------------|-------|--------|-----|---------|
| `quantum:position:snapshot:{symbol}` | Position State Brain (P3.3) | Hash | None | **Canonical position snapshot** (position_amt, entry_price, mark_price, leverage, unrealized_pnl, margin, liquidation_price, timestamp) |
| `quantum:position:ledger:{symbol}` | Apply Layer, Trade Logger | Hash | None | Ledger-style position tracking (position_amt, entry_price, leverage, realized_pnl, total_trades, last_update) |
| `quantum:position:{symbol}` | Legacy/compatibility | Hash | None | Simplified position state (position_amt, entry_price) |
| `quantum:ledger:{symbol}` | Trade Logger | Hash | None | Trade history ledger (position_amt, realized_pnl, total_closes, last_close_timestamp) |
| `quantum:ledger:seen_orders` | Trade Logger | Set | None | Deduplication set for processed order IDs |

#### Governor & Permits

| Key Pattern | Owner | Format | TTL | Purpose |
|------------|-------|--------|-----|---------|
| `quantum:permit:p33:{plan_id}` | **Governor**, HarvestBrain | String ("ALLOW") | **60s** | **Apply Layer execution permit** (Governor writes after gates pass) |
| `quantum:governor:exec_history:{symbol}` | Governor | Sorted Set | 24h auto-expire | Rate limit tracking (timestamp:plan_id pairs) |
| `quantum:cooldown:last_exec_ts:{symbol}` | Governor | String (timestamp) | 24h | Symbol-level cooldown tracking (last execution timestamp) |

#### Deduplication Keys

| Key Pattern | Owner | Format | TTL | Purpose |
|------------|-------|--------|-----|---------|
| `quantum:dedup:intent:{hash}` | Intent Bridge | String ("1") | 900s | Intent deduplication (prevents duplicate trade.intent) |
| `quantum:dedup:apply:{hash}` | Apply Layer | String ("1") | 3600s | Apply plan deduplication (prevents duplicate execution) |
| `quantum:dedup:harvest:{symbol}:{hash}` | HarvestBrain | String ("1") | 900s | Harvest action deduplication |

#### Capital Allocation & Exposure

| Key Pattern | Owner | Format | TTL | Purpose |
|------------|-------|--------|-----|---------|
| `quantum:capital:allocation:{symbol}` | Capital Allocation (P2.9) | Hash | None | AI-driven allocation targets (allocation_pct, target_notional, last_update) |
| `quantum:portfolio:state` | Portfolio State Publisher | Hash | None | Portfolio-level aggregates (total_equity, exposure_pct, open_positions, unrealized_pnl) |

#### Policy & Configuration

| Key Pattern | Owner | Format | TTL | Purpose |
|------------|-------|--------|-----|---------|
| `quantum:policy:active` | Risk Safety (PolicyStore) | JSON | None | **Single source of truth for risk policies** (autonomy, limits, ESS state) |
| `quantum:config:allowlist` | Intent Bridge, Apply Layer | Set | None | Symbol allowlist (fail-closed filtering) |

### 3.3 RACE CONDITION RISKS

#### 3.3.1 Position Count Race (17 positions despite limit=10)

**Scenario:**
- Governor limit: `GOV_MAX_OPEN_POSITIONS=10`
- Current positions: 9
- Two `apply.plan` messages arrive simultaneously for different symbols

**Race Condition Flow:**
```
Time T0:
  Governor-1 reads: SCAN quantum:position:* → 9 positions
  Governor-1 check: 9 < 10 → ✅ PASS (writes permit for ETHUSDT)

Time T0 + 5ms:
  Governor-2 reads: SCAN quantum:position:* → 9 positions (same!)
  Governor-2 check: 9 < 10 → ✅ PASS (writes permit for SOLUSDT)

Time T0 + 100ms:
  Apply Layer-1 executes ETHUSDT → 10 positions
  Apply Layer-2 executes SOLUSDT → 11 positions ❌ LIMIT VIOLATED
```

**Root Cause:**
- No atomic increment/lock on position count
- Governor uses SCAN (not atomic count)
- Position count source (`quantum:position:*`) updated AFTER execution
- No distributed lock between Governor and Apply Layer

**Current Mitigation:**
- Governor serial processing (single consumer)
- But Apply Layer has multiple workers → still vulnerable

**Actual State:**
- System observed with 17 open positions (limit=10)
- Confirms race condition occurring in production

#### 3.3.2 Stale Position State

**Scenario:**
- Position State Brain reads from `apply.result`
- Reconcile Engine polls Binance every 60s
- Price moves rapidly, position closed on exchange, internal state stale

**Race Condition Flow:**
```
Time T0:
  Exchange: Position BTCUSDT LONG liquidated (price hit liquidation)
  Internal: quantum:position:snapshot:BTCUSDT shows LONG 0.004

Time T0 + 30s:
  HarvestBrain reads stale state → thinks position still open
  HarvestBrain sends CLOSE order → fails (no position to close)

Time T0 + 60s:
  Reconcile Engine polls → detects drift, corrects state
```

**Root Cause:**
- **Source of truth conflict**: Internal state vs Exchange state
- 60s reconciliation lag
- No real-time WebSocket position updates integrated

**Mitigation:**
- Reconcile Engine corrects drift (eventual consistency)
- Apply Layer uses `reduceOnly=true` (exchange validates)
- Exchange Stream Bridge publishes fills, but not consumed by position state

#### 3.3.3 Permit Expiry Race

**Scenario:**
- Governor writes permit with 60s TTL
- Apply Layer reads stream with XREADGROUP (blocking, can delay)
- Network lag or queue backlog causes apply.plan to arrive after 60s

**Race Condition Flow:**
```
Time T0:
  Governor: SET quantum:permit:p33:abc123 "ALLOW" EX 60

Time T0 + 65s (network lag, queue backlog):
  Apply Layer: GET quantum:permit:p33:abc123 → nil (expired)
  Apply Layer: ❌ BLOCK execution (no permit)
```

**Root Cause:**
- Fixed 60s TTL too short for distributed system
- No permit refresh mechanism
- No signal back to Governor if plan not consumed in time

**Mitigation:**
- Apply Layer logs permit expiry (observable failure)
- Governor can increase TTL (but not dynamic)
- Plan ID allows re-processing with new permit

#### 3.3.4 Idempotency Hash Collision

**Scenario:**
- Intent Bridge uses `md5(symbol+side+action+timestamp)`
- Two identical intents within same second → same hash

**Race Condition Flow:**
```
Time T0:
  Intent-1: symbol=BTCUSDT, side=BUY, timestamp=1739883600
  Hash: md5("BTCUSDT+BUY+OPEN+1739883600") = "a3f7b2c1"
  SETNX quantum:dedup:intent:a3f7b2c1 EX 900 → 1 (success)

Time T0 + 500ms (same second):
  Intent-2: symbol=BTCUSDT, side=BUY, timestamp=1739883600 (same!)
  Hash: "a3f7b2c1" (collision!)
  SETNX quantum:dedup:intent:a3f7b2c1 EX 900 → 0 (duplicate detected)
  Intent-2 ❌ BLOCKED (false positive)
```

**Root Cause:**
- Timestamp precision: 1s granularity (not milliseconds)
- No sequence number or nonce in hash

**Mitigation:**
- Add microsecond timestamp precision
- Include random nonce or message ID in hash

#### 3.3.5 Multiple Exit Controllers

**Scenario:**
- HarvestBrain, Exit Brain V3, Exit Monitor all can send CLOSE orders
- No distributed lock on "who owns this exit"

**Race Condition Flow:**
```
Time T0:
  Position: BTCUSDT LONG 0.004
  HarvestBrain: Detects 0.5R profit → sends CLOSE 0.001 (25%)
  Exit Brain V3: Detects TP hit → sends CLOSE 0.001 (dynamic TP)

Time T0 + 100ms:
  Apply Layer-1: Executes HarvestBrain CLOSE → position now 0.003
  Apply Layer-2: Executes Exit Brain CLOSE → position now 0.002
  
  Result: ❌ Double close (50% instead of 25%)
```

**Root Cause:**
- Multiple exit controllers operating independently
- No locking on position closure
- Exit ownership check exists but not enforced globally

**Current Mitigation:**
- Exit Ownership Enforcement in Apply Layer (validates EXIT_OWNER)
- But multiple services can set EXIT_OWNER differently

#### 3.3.6 Capital Allocation Lag

**Scenario:**
- Governor checks `quantum:capital:allocation:{symbol}` (written by P2.9)
- Capital Allocation updates every 60s based on portfolio state
- Position opened, but allocation target not yet updated

**Race Condition Flow:**
```
Time T0:
  Portfolio exposure: 45%
  P2.9 writes: quantum:capital:allocation:BTCUSDT = {allocation_pct: 5%}

Time T0 + 30s:
  Governor allows BTCUSDT OPEN (within 5% allocation)
  Position opened → exposure now 50%

Time T0 + 35s (same cycle):
  Governor allows ETHUSDT OPEN (allocation check passes)
  Position opened → exposure now 55%

Time T0 + 60s:
  P2.9 updates: quantum:capital:allocation:* based on new 55% exposure
  But damage already done (exceeded intended 50% cap)
```

**Root Cause:**
- Allocation targets lag portfolio state
- Governor doesn't wait for P2.9 confirmation
- No atomic "reserve allocation" mechanism

---

## 4. POSITION LIMIT ENFORCEMENT MAP

### 4.1 LAYER-BY-LAYER ENFORCEMENT

| Layer | Service | Limit Type | Enforcement Point | Consistency | Race Condition Risk |
|-------|---------|------------|-------------------|-------------|-------------------|
| **L0: Allowlist** | Intent Bridge | Symbol allowlist (CSV or TOP10) | Pre-intent publish | ✅ Strong (fail-closed) | ❌ None (static list) |
| **L0: Allowlist** | Apply Layer | Symbol allowlist (CSV) | Pre-execution | ✅ Strong (fail-closed) | ❌ None (static list) |
| **L1: Exposure** | Intent Bridge | MAX_EXPOSURE_PCT (80%) | Pre-intent publish | ⚠️ Eventual (reads portfolio.state) | ⚠️ Medium (state lag) |
| **L2: Position Count** | Governor | GOV_MAX_OPEN_POSITIONS (10) | Pre-permit write | ❌ **Weak (SCAN, not atomic)** | ⚠️ **HIGH (race on count)** |
| **L2: Rate Limit** | Governor | 3/hour, 2/5min per symbol | Pre-permit write | ✅ Strong (sorted set atomic) | ❌ Low (per-symbol lock) |
| **L2: Notional Limit** | Governor | $200/trade, $2000 total | Pre-permit write | ⚠️ Eventual (reads positions) | ⚠️ Medium (total calc lag) |
| **L2: Cooldown** | Governor | 60s per symbol | Pre-permit write | ✅ Strong (timestamp check) | ❌ Low (per-symbol) |
| **L2: Kill Score** | Governor | Entry/exit thresholds | Pre-permit write | ✅ Strong (in-message) | ❌ None (stateless) |
| **L3: Allocation Target** | Capital Allocation (P2.9) | AI-driven per-symbol % | Governor check (optional) | ⚠️ Eventual (60s update cycle) | ⚠️ **MEDIUM (lag)** |
| **L3: Entry/Exit Sep** | Apply Layer | Kill score gates | Pre-execution | ✅ Strong (in-message) | ❌ None (stateless) |
| **L3: Governor Permit** | Apply Layer | Requires quantum:permit:p33:{plan_id} | Pre-execution | ✅ Strong (atomic GET) | ⚠️ Low (TTL expiry) |
| **L3: Exit Ownership** | Apply Layer | Validates EXIT_OWNER | Pre-execution (CLOSE only) | ✅ Strong (hardcoded) | ❌ None (single owner) |
| **L3: Idempotency** | Apply Layer | SETNX dedup | Pre-execution | ✅ Strong (atomic) | ⚠️ Low (hash collision) |
| **L3: Reduce-Only** | Apply Layer | reduceOnly=true on exits | Execution param | ✅ Strong (exchange validates) | ❌ None (exchange enforced) |
| **L4: Exchange Limits** | Binance API | Margin, leverage, liquidation | Exchange-side | ✅ Strong (authoritative) | ❌ None (external) |
| **L5: Reconciliation** | Reconcile Engine (P3.4) | Drift detection, correction | Post-execution (60s cycle) | ⚠️ Eventual (polling lag) | ⚠️ Low (retroactive fix) |

### 4.2 WHERE 17 POSITIONS CAN EXIST (DESPITE LIMIT=10)

**Root Cause Analysis:**

1. **Governor Position Count (L2):**
   - Uses `SCAN quantum:position:*` → **not atomic**
   - Multiple Governor consumers can read same count simultaneously
   - Position created AFTER permit written, not BEFORE
   - No distributed lock on position count increment

2. **Capitol Allocation Lag (L3):**
   - P2.9 updates allocation targets every 60s
   - Governor checks allocation, but doesn't wait for update
   - Burst of opens can exceed intended allocation before P2.9 recalculates

3. **Apply Layer Parallelism:**
   - Multiple Apply Layer workers process plans concurrently
   - Each checks its own permit (Governor already approved)
   - No inter-worker coordination on total position count

4. **Missing Global Semaphore:**
   - No Redis-based semaphore limiting total active positions
   - No "reserve slot" mechanism (atomic decrement)
   - Position count = derived state (count of existing positions), not enforced state

**Attack Vector (Unintentional):**
```
Step 1: Governor approves 3 plans simultaneously (all see 7 positions)
Step 2: All 3 pass position count check (7 < 10)
Step 3: Governor writes 3 permits
Step 4: Apply Layer executes all 3 → 10 positions ✓

Step 5: Repeat during next cycle → 3 more plans
Step 6: Governor reads 10 positions, but...
Step 7: Race: 2 plans read before position state updates
Step 8: Both see 10, both pass (10 < 10? NO, but timing allows)
Step 9: → 12 positions

Continue... → 17 positions observed
```

**Fix Required:**
- Atomic position slot reservation (INCR/DECR on counter)
- Distributed lock on position count modification
- Governor pre-reserves slot, Apply Layer confirms/releases
- Or: Single-threaded Apply Layer (serial execution)

---

## 5. FAILURE & RACE CONDITION ANALYSIS

### 5.1 STATE STALENESS POINTS

| State | Source | Freshness | Staleness Risk | Impact |
|-------|--------|-----------|---------------|--------|
| Position snapshot | Position State Brain | Real-time (on execution) | ❌ Low | High (execution decisions) |
| Exchange position truth | Binance API | Real-time | ❌ None (authoritative) | Critical (liquidations) |
| Position count | Derived (SCAN) | Lag: 0-5s | ⚠️ **HIGH** | Critical (limit bypass) |
| Portfolio exposure | Portfolio State Publisher | Lag: ~5s | ⚠️ Medium | High (exposure limits) |
| Capital allocation targets | P2.9 Capital Allocation | Lag: 60s | ⚠️ **HIGH** | Medium (allocation drift) |
| Kill score | In-message (stateless) | Real-time | ❌ None | Low (per-message) |
| Market price | Price Feed (WebSocket) | Real-time (~100ms) | ❌ Low | High (exit timing) |
| Reconciliation drift | Reconcile Engine | Lag: 60s | ⚠️ Medium | Medium (retroactive fix) |

### 5.2 BUFFERING POINTS

| Component | Buffer Type | Size | Overflow Behavior | Data Loss Risk |
|-----------|------------|------|-------------------|---------------|
| Redis Streams | XADD (append-only log) | Unlimited (until memory) | OOM crash | ❌ None (persisted) |
| XREADGROUP (consumer lag) | Pending list | Unlimited | Memory pressure | ❌ None (recoverable) |
| Governor execution history | Sorted Set (24h window) | ~1000 entries/symbol | Auto-expire old | ❌ None (rolling window) |
| Apply Layer dedup cache | Keys with TTL | ~10,000 keys | Expire after TTL | ⚠️ Low (dedup only) |
| HarvestBrain dedup cache | Keys with TTL | ~1,000 keys | Expire after TTL | ⚠️ Low (dedup only) |
| Binance API rate limits | Client-side queue | 1200 req/min | HTTP 429 errors | ⚠️ Medium (retry needed) |
| EventBus (internal) | In-memory queue | 10,000 events | Drop oldest | ⚠️ **HIGH** (no persistence) |

### 5.3 ASYNCHRONOUS OPERATIONS

| Operation | Services Involved | Coordination Mechanism | Failure Mode |
|-----------|------------------|----------------------|--------------|
| Trade intent → Execution | AI Engine → Intent Bridge → Governor → Apply Layer → Binance | Redis streams (XREADGROUP + ACK) | Message loss if consumer crashes before ACK |
| Position state update | Apply Layer → Position State Brain → Ledger → Reconcile Engine | Streams + periodic polling | Drift if Position State Brain lags |
| Exit coordination | HarvestBrain / Exit Brain V3 / Exit Monitor → Apply Layer | Streams (no locking) | **Double close** (race condition) |
| Capital allocation update | P2.9 → Portfolio State → Governor | Keys (no signals) | **Stale allocation** (60s lag) |
| Reconciliation | Reconcile Engine ← Binance API → Internal state correction | Periodic polling (60s) | **Drift window** (max 60s) |

### 5.4 LOCKING MECHANISMS

| Lock Type | Key Pattern | Owner | TTL | Purpose | Race Risk |
|-----------|------------|-------|-----|---------|----------|
| SETNX (idempotency) | `quantum:dedup:intent:{hash}` | Intent Bridge | 900s | Prevent duplicate intents | ⚠️ Low (hash collision) |
| SETNX (idempotency) | `quantum:dedup:apply:{hash}` | Apply Layer | 3600s | Prevent duplicate execution | ⚠️ Low (hash collision) |
| SETNX (harvest dedup) | `quantum:dedup:harvest:{symbol}:{hash}` | HarvestBrain | 900s | Prevent duplicate harvests | ⚠️ Low (hash collision) |
| Permit (authorization) | `quantum:permit:p33:{plan_id}` | Governor | 60s | Authorize execution | ⚠️ Low (TTL expiry) |
| **MISSING:** Position slot lock | N/A | N/A | N/A | **Prevent position count race** | ❌ **HIGH** |
| **MISSING:** Exit controller lock | N/A | N/A | N/A | **Prevent double close** | ❌ **HIGH** |

### 5.5 WHERE NO GLOBAL SEMAPHORE EXISTS

1. **Position Count Limit:**
   - No atomic counter (e.g., `DECR quantum:available_slots`)
   - No distributed lock on position opening
   - Governor uses derived count (SCAN) → **race condition**

2. **Total Notional Limit:**
   - No atomic accumulator (e.g., `INCRBY quantum:total_notional_used {amount}`)
   - Governor calculates from existing positions → **lag in total calculation**

3. **Exit Controller Coordination:**
   - Multiple services can send CLOSE orders
   - No lock: `quantum:lock:exit:{symbol}`
   - Exit ownership check exists but not enforced globally → **double close risk**

4. **Capital Allocation Reservation:**
   - No `quantum:reserved_allocation:{symbol}` mechanism
   - Governor doesn't atomically reserve allocation before execution
   - P2.9 updates allocation targets asynchronously → **drift**

---

## 6. VISUAL ARCHITECTURE OUTPUT

### 6.1 SYSTEM OVERVIEW (ASCII)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          QUANTUM TRADER ARCHITECTURE                          │
│                         Event-Driven Trading System                           │
└──────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│ LAYER 0: MARKET DATA INGESTION                                             │
└────────────────────────────────────────────────────────────────────────────┘

                         ┌──────────────────┐
                         │  Binance Market  │
                         │   WebSocket      │
                         └────────┬─────────┘
                                  │ (prices, klines)
                                  ↓
                    ┌─────────────────────────┐
                    │   Price Feed Service    │
                    │   (WebSocket → Redis)   │
                    └──────────┬──────────────┘
                               │ XADD
                               ↓
              quantum:stream:market.tick ──────────┐
              quantum:stream:market.klines         │
                               │                   │
                               ↓                   ↓
              ┌────────────────────────┐  ┌──────────────────┐
              │  Feature Publisher     │  │  MarketState     │
              │  (PATH 2.3D)           │  │  Publisher (P0.5)│
              └──────────┬─────────────┘  └────────┬─────────┘
                         │ XADD                    │ XADD
                         ↓                         ↓
       quantum:stream:features        quantum:stream:marketstate

┌────────────────────────────────────────────────────────────────────────────┐
│ LAYER 1: AI INFERENCE & ENSEMBLE                                           │
└────────────────────────────────────────────────────────────────────────────┘

         quantum:stream:features ────────────────┐
         quantum:stream:market.tick              │
         quantum:stream:marketstate              │
                               ↓                 ↓
              ┌──────────────────────────────────────┐
              │         AI Engine Service             │
              │  ┌────────────────────────────────┐  │
              │  │ Model Inference:               │  │
              │  │  - XGBoost                     │  │
              │  │  - LightGBM                    │  │
              │  │  - N-HiTS                      │  │
              │  │  - PatchTST                    │  │
              │  └────────────┬───────────────────┘  │
              │               │ Ensemble Voting      │
              │  ┌────────────▼───────────────────┐  │
              │  │ Meta Agent (RL Strategy)       │  │
              │  │ RL Position Sizer              │  │
              │  │ Regime Detector                │  │
              │  └────────────┬───────────────────┘  │
              └───────────────┼──────────────────────┘
                              │ XADD (3 streams)
                              ↓
           ┌──────────────────┼──────────────────────┐
           │                  │                      │
           ↓                  ↓                      ↓
 quantum:stream:       quantum:stream:      quantum:stream:
 ai.signal_generated   ai.decision.made     trade.intent

┌────────────────────────────────────────────────────────────────────────────┐
│ LAYER 2: INTENT FILTERING & GOVERNANCE                                     │
└────────────────────────────────────────────────────────────────────────────┘

      quantum:stream:trade.intent
                 │ XREADGROUP
                 ↓
      ┌──────────────────────┐
      │   Intent Bridge      │
      │   ┌──────────────┐   │
      │   │ Allowlist    │   │  ← INTENT_BRIDGE_ALLOWLIST
      │   │ Exposure     │   │  ← MAX_EXPOSURE_PCT
      │   │ Dedup (SETNX)│   │  ← quantum:dedup:intent:*
      │   │ Flat Skip    │   │  ← SKIP_FLAT_SELL
      │   └──────┬───────┘   │
      └──────────┼───────────┘
                 │ XADD
                 ↓
      quantum:stream:apply.plan ◄─────── HarvestBrain (live mode)
                 │ XREADGROUP
                 ↓
      ┌──────────────────────────────────┐
      │   Governor (P3.2)                │
      │   ┌──────────────────────────┐   │
      │   │ Kill Score Gates         │   │
      │   │ Position Count (SCAN)    │   │  ← ⚠️ RACE CONDITION
      │   │ Rate Limits (sorted set) │   │
      │   │ Notional Limits          │   │
      │   │ Cooldown Check           │   │
      │   │ P2.9 Allocation Check    │   │  ← ⚠️ STALE (60s lag)
      │   └──────────┬───────────────┘   │
      └──────────────┼───────────────────┘
                     │ SET (permit)
                     ↓
         quantum:permit:p33:{plan_id} (TTL: 60s)

┌────────────────────────────────────────────────────────────────────────────┐
│ LAYER 3: EXECUTION (Apply Layer P3)                                        │
└────────────────────────────────────────────────────────────────────────────┘

      quantum:stream:apply.plan
                 │ XREADGROUP
                 ↓
      ┌──────────────────────────────────────────┐
      │   Apply Layer (P3)                       │
      │   ┌──────────────────────────────────┐   │
      │   │ 1. Permit Check (GET)            │   │  ← quantum:permit:p33:*
      │   │ 2. Allowlist Gate                │   │
      │   │ 3. Kill Score Gate               │   │
      │   │ 4. Entry/Exit Separation         │   │
      │   │ 5. Exit Ownership Enforcement    │   │
      │   │ 6. Idempotency (SETNX)           │   │  ← quantum:dedup:apply:*
      │   │ 7. Position State Read           │   │  ← quantum:ledger:*
      │   │ 8. Kill Switch Check             │   │
      │   └──────────┬───────────────────────┘   │
      │              │ (all gates passed)        │
      │   ┌──────────▼───────────────────────┐   │
      │   │ Binance API Call                 │   │
      │   │ POST /fapi/v1/order              │   │
      │   │ - MARKET order                   │   │
      │   │ - reduceOnly (if CLOSE/REDUCE)   │   │
      │   │ - HMAC signature                 │   │
      │   └──────────┬───────────────────────┘   │
      └──────────────┼───────────────────────────┘
                     │ XADD
                     ↓
         quantum:stream:apply.result
                     │
        ┌────────────┼────────────┐
        │            │            │
        ↓            ↓            ↓
  ┌─────────┐ ┌────────────┐ ┌────────────┐
  │ Harvest │ │  Position  │ │   Trade    │
  │  Brain  │ │   State    │ │   Logger   │
  │         │ │  Brain P3.3│ │            │
  └─────────┘ └──────┬─────┘ └──────┬─────┘
                     │              │
                     ↓              ↓
       quantum:stream:           quantum:ledger:*
       position.snapshot

┌────────────────────────────────────────────────────────────────────────────┐
│ LAYER 4: STATE MANAGEMENT & RECONCILIATION                                 │
└────────────────────────────────────────────────────────────────────────────┘

                  ┌────────────────────────────┐
                  │  Position State Brain      │
                  │  (P3.3)                    │
                  └──────────┬─────────────────┘
                             │ SET
                             ↓
          quantum:position:snapshot:{symbol}  ← Internal source of truth
                             │
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ↓                  ↓                  ↓
   ┌────────────┐   ┌────────────────┐   ┌───────────────┐
   │ Portfolio  │   │   Reconcile    │   │   Binance     │
   │   State    │   │   Engine P3.4  │   │   API (poll)  │
   │ Publisher  │   │   (60s poll)   │   │   /positionRisk
   └──────┬─────┘   └────────┬───────┘   └───────┬───────┘
          │                  │                    │
          ↓                  ↓                    ↓
 quantum:stream:     quantum:stream:         Exchange Truth
 portfolio.state     reconcile.events        (authoritative)
          │                  │
          │                  │ (drift detection)
          ↓                  ↓
   ┌────────────────────────────────────────┐
   │ Drift Correction:                      │
   │ UPDATE quantum:position:snapshot:*     │
   │ (exchange state = source of truth)     │
   └────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│ LAYER 5: EXIT LOGIC (Multiple Controllers - ⚠️ RACE RISK)                  │
└────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────┐   ┌─────────────────┐   ┌──────────────────┐
    │  HarvestBrain   │   │ Exit Brain V3   │   │  Exit Monitor    │
    │  (R-based)      │   │ (AI SL/TP)      │   │  (Simple TP/SL)  │
    └────────┬────────┘   └────────┬────────┘   └────────┬─────────┘
             │                     │                     │
             │ quantum:stream:     │ Direct Binance      │ EventBus:
             │ apply.result        │ API polling         │ trade.execution.result
             │ (XREADGROUP)        │ (10s interval)      │
             │                     │                     │
             ↓                     ↓                     ↓
       (Monitors PnL,        (Monitors prices,     (Monitors prices,
        calculates R,         checks SL/TP,         checks SL/TP,
        triggers harvest)     executes via          sends via
                              ExitOrderGateway)     trade.intent)
             │                     │                     │
             │ XADD                │ MARKET order        │ XADD
             ↓                     ↓                     ↓
      quantum:stream:         Execution Service    quantum:stream:
      apply.plan                                   trade.intent
      (CLOSE/REDUCE)                                    │
             │                                          ↓
             └──────────────┬───────────────────────────┘
                            │
                            ↓
                   ⚠️ RACE CONDITION:
                   Multiple close orders for
                   same position possible
                            │
                            ↓
                    Apply Layer (execution)
                            │
                            ↓
                  quantum:stream:trade.closed
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
          ↓                 ↓                 ↓
    ┌──────────┐   ┌───────────────┐   ┌──────────────┐
    │  Trade   │   │ Performance   │   │ RL Feedback  │
    │  Logger  │   │ Attribution   │   │  V2 Producer │
    └──────────┘   └───────────────┘   └──────┬───────┘
                                               │ XADD
                                               ↓
                                    quantum:stream:rl_rewards
                                               │
                                               ↓
                                    ┌──────────────────┐
                                    │  RL Monitor      │
                                    │  CLM (retrain)   │
                                    └──────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│ LAYER 6: CAPITAL ALLOCATION & PORTFOLIO GOVERNANCE                         │
└────────────────────────────────────────────────────────────────────────────┘

      quantum:stream:portfolio.state
                 │
      ┌──────────┼──────────────────────────┐
      │          │                          │
      ↓          ↓                          ↓
 ┌────────────────────┐   ┌──────────────────────┐   ┌────────────────┐
 │ Capital Allocation │   │ Portfolio Risk       │   │ Portfolio Gate │
 │ Brain (P2.9)       │   │ Governor (P2.8)      │   │ (P2.6)         │
 │ (AI-driven)        │   │                      │   │                │
 └───────┬────────────┘   └──────────┬───────────┘   └────────────────┘
         │ SET (60s cycle)           │ XADD
         ↓                           ↓
 quantum:capital:         quantum:stream:risk.events
 allocation:{symbol}                 │
         │                           ↓
         │ READ by Governor    ┌───────────────┐
         └───────────────────► │  Risk Safety  │
                               │  (ESS, Policy │
                               │   Store)      │
                               └───────┬───────┘
                                       │ XADD
                                       ↓
                              quantum:stream:policy.updated
                                       │
                                       └──► (AI Engine, Governor, Apply Layer)

┌────────────────────────────────────────────────────────────────────────────┐
│ LAYER 7: OBSERVABILITY & MONITORING                                        │
└────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────┐   ┌──────────────────┐   ┌──────────────────┐
  │  BSC (Baseline  │   │  Harvest Metrics │   │  Safety Telemetry│
  │  Safety         │   │  Exporter (P2.7) │   │  Exporter (P1)   │
  │  Controller)    │   │                  │   │                  │
  └────────┬────────┘   └────────┬─────────┘   └────────┬─────────┘
           │ XADD               │ Prometheus           │ Prometheus
           ↓                    ↓                      ↓
  quantum:stream:       :9090/metrics          :9091/metrics
  bsc.events
           │
           ↓
  ┌─────────────────────────────────────────────────┐
  │         Dashboard API (Port 8080)               │
  │         Real-time metrics, PnL visualization     │
  └─────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│ EXTERNAL SYSTEMS                                                            │
└────────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────────┐
                    │  Binance Futures Testnet │
                    │  testnet.binancefuture.com
                    └────────┬─────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ↓                  ↓                  ↓
   ┌─────────────┐   ┌──────────────┐   ┌────────────────┐
   │  WebSocket  │   │  REST API    │   │  User Data     │
   │  (prices,   │   │  (orders,    │   │  Stream        │
   │   klines)   │   │   positions) │   │  (fills)       │
   └─────────────┘   └──────────────┘   └────────────────┘
```

### 6.2 EVENT-FLOW SEQUENCE DIAGRAM (SIMPLIFIED)

```
Time →

T0: AI Signal Generation
    AI Engine: Ensemble prediction → confidence=0.87
    ↓ XADD quantum:stream:trade.intent

T1: Intent Filtering (+5ms)
    Intent Bridge: Allowlist ✓, Exposure ✓, Dedup ✓
    ↓ XADD quantum:stream:apply.plan

T2: Governance (+10ms)
    Governor: Position count check (7 < 10 ✓)
              Rate limit ✓, Notional ✓, Cooldown ✓
    ↓ SET quantum:permit:p33:abc123 EX 60

T3: Apply Layer (+15ms)
    Apply Layer: Permit check ✓, All gates ✓
                 POST Binance /fapi/v1/order
    ↓ XADD quantum:stream:apply.result

T4: State Update (+20ms)
    Position State Brain: Aggregates position state
    ↓ SET quantum:position:snapshot:BTCUSDT
    Trade Logger: Updates ledger
    ↓ HSET quantum:ledger:BTCUSDT

T5: Reconciliation (+60s from T3)
    Reconcile Engine: Polls Binance /positionRisk
                      Detects drift, corrects state
    ↓ XADD quantum:stream:reconcile.events

T6: Harvest Monitoring (+5s intervals from T3)
    HarvestBrain: Monitors PnL, calculates R-multiple
                  R=0.50 reached → triggers HARVEST_PARTIAL
    ↓ SET quantum:permit:p33:xyz789 (auto-permit)
    ↓ XADD quantum:stream:apply.plan

T7: Harvest Execution (+5ms from T6)
    Apply Layer: Executes CLOSE 25% (reduce-only)
    ↓ XADD quantum:stream:apply.result

T8: Exit Logic (parallel, +10s intervals from T3)
    Exit Brain V3: Monitors price vs SL/TP
                   SL hit → executes MARKET CLOSE
    ↓ XADD quantum:stream:trade.closed

T9: Trade Closure Logging (+5ms from T8)
    Trade Logger: Final ledger update
    Performance Attribution: PnL breakdown
    RL Feedback V2: Reward calculation
    ↓ XADD quantum:stream:rl_rewards

T10: Learning Feedback (+variable)
    RL Monitor: Collects rewards
    CLM: Triggers model retrain
    ↓ XADD quantum:stream:model.retrain
```

### 6.3 SERVICE INTERACTION MAP (DATA FLOW)

```
WRITE ───────► READ

State Keys:
  Governor ─────────► quantum:permit:p33:* ──────────► Apply Layer ✓
  Apply Layer ───────► quantum:ledger:* ──────────────► Intent Bridge, HarvestBrain
  Position State ────► quantum:position:snapshot:* ───► Governor (count), Reconcile Engine
  P2.9 Capital ──────► quantum:capital:allocation:* ──► Governor (optional check)
  Trade Logger ──────► quantum:ledger:* ──────────────► Analytics, Dashboard

Streams (Producer → Consumer):
  AI Engine ─────────► quantum:stream:trade.intent ────────► Intent Bridge
  Intent Bridge ─────► quantum:stream:apply.plan ──────────► Governor
  HarvestBrain ──────► quantum:stream:apply.plan ──────────► Governor (bypass with auto-permit)
  Governor ──────────► (writes permits, not streams)
  Apply Layer ───────► quantum:stream:apply.result ────────► HarvestBrain, Position State, Trade Logger
  Apply Layer ───────► quantum:stream:trade.closed ────────► Trade Logger, Performance Attribution, RL Feedback
  Position State ────► quantum:stream:position.snapshot ───► Portfolio State, Reconcile Engine
  Portfolio State ───► quantum:stream:portfolio.state ─────► P2.9 Capital, Portfolio Risk Governor
  Reconcile Engine ──► quantum:stream:reconcile.events ────► Monitoring
  RL Feedback V2 ────► quantum:stream:rl_rewards ──────────► RL Monitor, CLM

External APIs:
  Price Feed ────────► Binance WebSocket (market.tick)
  Apply Layer ───────► Binance REST API (POST /fapi/v1/order)
  Reconcile Engine ──► Binance REST API (GET /positionRisk)
  HarvestBrain ──────► Binance REST API (GET /positionRisk, sync on startup)
  Exit Brain V3 ─────► Binance REST API (polling /positionRisk)
  Balance Tracker ───► Binance REST API (GET /account)
```

---

## 7. SYSTEM STATE TRUTH

### 7.1 SOURCE OF TRUTH HIERARCHY

| State Type | Primary Source | Secondary Source | Reconciliation | Authority |
|-----------|---------------|-----------------|----------------|-----------|
| **Position existence** | Binance API (`/positionRisk`) | `quantum:position:snapshot:*` | Reconcile Engine (60s) | **Exchange (Binance)** |
| **Position quantity** | Binance API | `quantum:position:ledger:*` | Reconcile Engine | **Exchange** |
| **Unrealized PnL** | Binance API | Position State Brain (calculated) | Reconcile Engine | **Exchange** |
| **Realized PnL** | `quantum:ledger:*` (Trade Logger) | Binance income history | Manual audit | **Internal ledger** |
| **Open position count** | Derived (SCAN `quantum:position:*`) | None | None | **Derived (not enforced)** ⚠️ |
| **Portfolio exposure** | `quantum:portfolio:state` | Derived from positions | Portfolio State Publisher (5s) | **Internal state** |
| **Capital allocation** | `quantum:capital:allocation:*` (P2.9) | None | None | **AI-driven (60s lag)** ⚠️ |
| **Execution permit** | `quantum:permit:p33:*` (Governor) | None | TTL expiry (60s) | **Governor** |
| **Risk policies** | `quantum:policy:active` (PolicyStore) | None | None | **PolicyStore (single source)** |
| **Market price** | Binance WebSocket | `quantum:stream:market.tick` | Real-time (~100ms) | **Exchange** |

### 7.2 CRITICAL FINDINGS

1. **Position Count = Derived, Not Enforced:**
   - Governor reads count via SCAN (not atomic)
   - No semaphore or counter limiting slots
   - **Result:** 17 positions exist despite limit=10

2. **Capital Allocation Lag:**
   - P2.9 updates every 60s
   - Governor checks allocation targets (stale by up to 60s)
   - **Result:** Allocation drift during burst periods

3. **Multiple Exit Controllers:**
   - HarvestBrain, Exit Brain V3, Exit Monitor all can close positions
   - No distributed lock on exit operations
   - **Result:** Potential double-close race condition

4. **Reconciliation Lag:**
   - 60s polling interval
   - Exchange state = authoritative, but drift window exists
   - **Result:** Stale position state for up to 60s

5. **Permit TTL Expiry:**
   - Governor writes permits with 60s TTL
   - Queue backlog or network lag can cause expiry before consumption
   - **Result:** Valid plans rejected due to expired permits (observable failure)

6. **Idempotency Hash Collision:**
   - Timestamp precision: 1s granularity
   - No nonce or sequence number
   - **Result:** Potential false-positive duplicate detection

### 7.3 RECOMMENDATIONS (STRUCTURAL ONLY)

1. **Atomic Position Slot Reservation:**
   - Implement `quantum:available_slots` counter (INCR/DECR)
   - Governor atomically reserves slot before permit write
   - Apply Layer releases slot if execution fails

2. **Exit Controller Locking:**
   - Implement `quantum:lock:exit:{symbol}` (SETNX with TTL)
   - First exit controller acquires lock, others skip
   - Or: Single exit controller enforcement (disable redundant services)

3. **Real-Time Position Sync:**
   - Integrate Binance User Data Stream (WebSocket)
   - Position updates pushed to `quantum:stream:exchange.normalized`
   - Position State Brain consumes real-time fills (no 60s lag)

4. **Dynamic Permit TTL:**
   - Governor calculates TTL based on queue depth
   - Longer TTL during backlog, shorter during normal load
   - Or: Permit refresh mechanism (Governor extends TTL if plan not consumed)

5. **Idempotency Enhancement:**
   - Use microsecond timestamp precision
   - Include random nonce or sequence number in hash
   - Or: Use message ID from XADD result as dedupe key

6. **Capital Allocation Locking:**
   - P2.9 writes allocation targets + version number
   - Governor checks version, rejects if stale
   - Or: Atomic allocation reservation (similar to position slots)

---

## APPENDIX: SERVICE QUICK REFERENCE

### Active Services (68 Total)

**Core Trading Pipeline (6):**
- quantum-ai-engine
- quantum-ensemble-predictor
- quantum-intent-bridge
- quantum-governor
- quantum-apply-layer
- quantum-execution

**Exit & Risk (6):**
- quantum-harvest-brain
- quantum-exit-monitor
- quantum-exit-intelligence
- quantum-risk-safety
- quantum-bsc
- quantum-autonomous-trader

**Portfolio & State (7):**
- quantum-portfolio-state-publisher
- quantum-position-state-brain
- quantum-reconcile-engine
- quantum-trade-logger
- quantum-portfolio-state-brain (duplicate?)
- quantum-balance-tracker
- quantum-exposure_balancer

**Market Data (5):**
- quantum-price-feed
- quantum-market-publisher
- quantum-marketstate
- quantum-feature-publisher
- quantum-exchange-stream-bridge

**Learning & RL (7):**
- quantum-clm
- quantum-clm-minimal
- quantum-rl-feedback-v2
- quantum-rl-monitor
- quantum-rl-sizer
- quantum-rl-policy-publisher
- quantum-retrain-worker

**Capital & Governance (6):**
- quantum-capital-allocation
- quantum-portfolio-risk-governor
- quantum-portfolio-gate
- quantum-heat-gate
- quantum-portfolio-heat-gate
- quantum-portfolio-governance

**Intelligence (7):**
- quantum-p35-decision-intelligence
- quantum-ceo-brain
- quantum-portfolio-intelligence
- quantum-strategy-brain
- quantum-risk-brain
- quantum-ai-strategy-router
- quantum-meta-regime

**Performance (5):**
- quantum-performance-attribution
- quantum-performance-tracker
- quantum-harvest-metrics-exporter
- quantum-safety-telemetry
- quantum-rl-shadow-metrics-exporter

**Harvest Suite (3):**
- quantum-harvest-proposal
- quantum-harvest-optimizer
- quantum-portfolio-clusters

**Infrastructure (16):**
- quantum-execution-result-bridge
- quantum-universe-service
- quantum-universe (duplicate?)
- quantum-dashboard-api
- quantum-learning-api
- quantum-learning-monitor
- quantum-metricpack-builder
- quantum-strategic-memory
- quantum-utf-publisher
- quantum-intent-executor
- quantum-capital-efficiency
- quantum-allocation-target
- quantum-risk-proposal
- redis-server

---

**END OF SYSTEM ARCHITECTURE TRUTH MAP**

Generated: 2026-02-18  
Method: Code tracing, Redis introspection, service enumeration  
Coverage: 68 services, 38 streams, 15+ state key patterns  
Focus: Structural analysis (not performance audit)
