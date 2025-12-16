# STEP 1 - GLOBAL DISCOVERY & HEALTH SNAPSHOT
## Quantum Trader v2.0 - System Architecture Discovery
**Date**: December 5, 2025  
**Environment**: Binance Testnet (STAGING)  
**Status**: Discovery Phase

---

## 🏗️ MICROSERVICES ARCHITECTURE

### Core Backend (Port 8000)
**Location**: `backend/main.py`  
**Container**: `quantum_backend`  
**Responsibility**: Main trading engine, API gateway, orchestration  
**Health Endpoints**:
- `/health` - Basic liveness
- `/api/v2/health` - Comprehensive v2 health with dependencies
- `/health/system` - System Health Monitor aggregate
- `/health/ai` - AI subsystems health
- `/health/risk` - Risk Guard status
- `/health/scheduler` - Scheduler heartbeat
- `/health/monitor` - Health Monitor with auto-fix recommendations
- `/api/aios_status` - AI-OS module health dashboard

### Portfolio Intelligence Service (Port 8004)
**Location**: `services/analytics_os_service/`  
**Container**: `portfolio_service`  
**Responsibility**: Portfolio analytics, position tracking, PnL calculation  
**Health**: `/health` (presumed)

### Execution & Risk Service (Port 8003)
**Location**: `services/exec_risk_service/`  
**Container**: `exec_risk_service`  
**Responsibility**: Order execution, risk checks, exchange adapters  
**Health**: `/health` (confirmed in code)

### AI Service (Port 8001)
**Location**: `services/ai_service/`  
**Container**: `ai_service`  
**Responsibility**: AI model serving, inference, ensemble management  
**Health**: `/health` (presumed)

---

## 🤖 AI MODULES & COMPONENTS

### 1. **AI Ensemble System**
**Location**: `ai_engine/ensemble_manager.py`  
**Models**:
- XGBoost (fast gradient boosting)
- LightGBM (light gradient boosting)
- N-HiTS (neural hierarchical interpolation for time series)
- PatchTST (patched time series transformer)

**Current Config**: 
- AI_MODEL=hybrid (TFT 60% + XGBoost 40%)
- 2-model ensemble active (XGB+LGBM)

**Files**:
- `backend/utils/scheduler.py` (line 834): EnsembleManager import
- `backend/tests/test_shadow_model_manager.py`: XGBoost/LightGBM model promotion tests

### 2. **Regime Detector V2**
**Location**: `backend/utils/regime_detector_v2.py`  
**Class**: `RegimeDetectorV2`  
**Responsibility**: Market regime classification (trending/ranging/volatile)

### 3. **World Model**
**Location**: `backend/world_model/world_model.py`  
**Class**: `WorldModel`  
**Enum**: `MarketRegime`  
**Responsibility**: Market state representation and prediction

### 4. **RL v3 Position Sizing**
**Config**:
- RL_SIZING_EPSILON=0.50 (50% exploration)
- RL_SIZING_ALPHA=0.15 (learning rate)
- RL_SIZING_DISCOUNT=0.95 (discount factor)

**Location**: TBD (likely in `backend/ai_risk/` or `backend/rl/`)

### 5. **Model Supervisor**
**Config**:
- QT_MODEL_SUPERVISOR_MODE=ENFORCED (blocks biased trades)
- QT_MODEL_SUPERVISOR_BIAS_THRESHOLD=0.70
- QT_MODEL_SUPERVISOR_MIN_SAMPLES=20

**Responsibility**: Bias detection, model performance tracking

### 6. **Portfolio Balancer AI**
**Location**: `backend/tests/test_portfolio_balancer.py` (TestPortfolioBalancerAI)  
**Config**:
- QT_PORTFOLIO_BALANCER_INTERVAL=60 (every 1 min)
- QT_MAX_CORRELATION=0.7

### 7. **Strategy Generator / Signal Generator**
**Location**: `backend/routes/live_ai_signals.py`  
**Endpoint**: `/signals/recent`

### 8. **Continuous Learning Manager (CLM)**
**Location**: `backend/routes/clm_routes.py`, `backend/routes/clm.py`  
**Config**:
- QT_CONTINUOUS_LEARNING=true
- QT_MIN_SAMPLES_FOR_RETRAIN=50
- QT_RETRAIN_INTERVAL_HOURS=24
- QT_AUTO_BACKTEST_AFTER_TRAIN=true

---

## 🛡️ RISK & SAFETY MODULES

### 1. **Risk v3 (Risk Guard)**
**Location**: `backend/risk/` (presumed)  
**Class**: `RiskGuard` (Protocol in `backend/services/trade_replay_engine/trade_replay_engine.py`)  
**Health**: `/health/risk`  
**Config**:
- RM_MAX_POSITION_USD=2000
- RM_MIN_POSITION_USD=100
- RM_MAX_LEVERAGE=30.0
- RM_RISK_PER_TRADE_PCT=0.10
- RM_MAX_EXPOSURE_PCT=2.00
- RM_MAX_CONCURRENT_TRADES=20

### 2. **ESS (Emergency Stop System)**
**Status**: Referenced in configs but implementation TBD  
**Expected Behavior**:
- Monitor equity drawdown
- Switch to ACTIVE on threshold breach
- Block new orders when ACTIVE

### 3. **Dynamic TP/SL**
**Config**: QT_USE_AI_DYNAMIC_TPSL=true  
**Responsibility**: Volatility and confidence-based exit management

---

## 🌐 CORE DOMAINS

### 1. **OrderService**
**Location**: `backend/domains/orders/service.py`  
**Responsibility**: Order history queries from TradeLog  
**Tests**: `backend/tests/test_order_service.py` (24 tests)

### 2. **SignalService**
**Location**: `backend/domains/signals/service.py`  
**Responsibility**: AI signal retrieval from /signals/recent  
**Tests**: `backend/tests/test_signal_service.py`

### 3. **StrategyService**
**Location**: `backend/domains/strategies/service.py`  
**Responsibility**: Strategy info from PolicyStore  
**Tests**: `backend/tests/test_strategy_service.py`

### 4. **PolicyStore**
**Location**: `backend/core/policy_store.py` (presumed)  
**Routes**: `backend/routes/policy.py`  
**Responsibility**: Global trading policies (Redis-backed)

---

## 📡 EVENT BUS & MESSAGING

**Type**: Redis Streams (presumed)  
**Config**: QT_EVENT_DRIVEN_MODE=true  
**Evidence**: Event-driven execution references in config

**Files to investigate**:
- `backend/events/`
- `backend/core/service_rpc.py`

---

## 🔍 OBSERVABILITY

### Logging
**Config**: 
- LOG_LEVEL=DEBUG
- QT_LOG_LEVEL=DEBUG
- JSON structured logs (presumed)

**Location**: `backend/logging_config.py`

### Metrics
**Endpoints**:
- `/api/metrics/system` - System metrics for dashboard
- Prometheus/StatsD integration (TBD)

### Tracing
**Status**: TBD (OpenTelemetry/Jaeger?)

### Health Monitoring
**System Health Monitor**: `/health/system/*`  
**Health Monitor**: `/health/monitor` (auto-fix recommendations)  
**Module Health**: `/health/system/module/{module_name}`  
**History**: `/health/system/history`

---

## 📊 DASHBOARD BFF

**Location**: `backend/api/dashboard/`  
**Endpoints**:
- `/api/dashboard/overview` - System overview
- `/api/dashboard/trading` - Positions, orders, signals, strategies
- `/api/metrics/system` - System metrics
- `/positions` - Current positions
- `/trades` - Recent trades

**Frontend**: Next.js at http://localhost:3000  
**Tests**: `tests/api/test_dashboard_*.py`, `frontend/__tests__/TradingTab.test.tsx`

---

## 🔌 EXCHANGE ADAPTERS

### Primary: Binance Futures (TESTNET)
**Config**:
- QT_EXECUTION_EXCHANGE=binance-futures
- QT_MARKET_TYPE=usdm_perp
- QT_PAPER_TRADING=false (LIVE TESTNET)

**Location**: `backend/integrations/binance/` (presumed)  
**Tests**: `backend/tests/test_binance_futures_adapter.py`

### Other Exchanges
- Bybit
- OKX
- KuCoin
- Kraken
- Firi

**Status**: Available but testnet focus on Binance

---

## 🧪 TEST SUITES

### Backend Tests
**Location**: `backend/tests/`  
**Categories**:
- Unit tests (`test_*.py`)
- Integration tests (`tests/integrations/`)
- API tests (`tests/api/`)
- E2E tests (`tests/e2e_test_suite.py`)

**Key Test Files**:
- `test_order_service.py` (24 passing)
- `test_signal_service.py`
- `test_strategy_service.py`
- `test_portfolio_balancer.py`
- `test_shadow_model_manager.py`
- `test_binance_futures_adapter.py`
- `test_covariate_shift_handler.py`
- `test_risk_guard_service.py`

### Frontend Tests
**Location**: `frontend/__tests__/`  
**Files**:
- `TradingTab.test.tsx` (45 tests)
- `OverviewTab.test.tsx`
- `RiskTab.test.tsx`
- `SystemTab.test.tsx`

### Integration Test Harness
**Location**: `tests/integration_test_harness.py`  
**Methods**:
- `test_service_health()`
- `test_all_services_health()`
- `test_load_health_checks()`

### E2E Test Suite
**Location**: `tests/e2e_test_suite.py`  
**Methods**:
- `check_service_health()`
- `test_all_services_healthy()`
- `test_health_monitoring()`

---

## ⚙️ CONFIGURATION

### Environment Variables (50 USDT pairs configured)
**Symbols**: BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, etc. (50 total)  
**Universe**: QT_UNIVERSE=l1l2-top (Layer1+Layer2 by volume)  
**Max Symbols**: QT_MAX_SYMBOLS=50

### Trading Configuration
- Confidence threshold: 0.45 (testnet lowered)
- Check interval: 10 seconds
- Cooldown: 120 seconds
- Min confidence: 0.65 (AI re-evaluation)
- Max positions: 20 concurrent

### Risk Configuration
- Max position: $2,000
- Min position: $100
- Max leverage: 30x
- Risk per trade: 10%
- Max exposure: 200%

---

## 🚨 MISSING / TBD COMPONENTS

1. **ESS Implementation**: Config exists, implementation unclear
2. **RL v3 Service**: Config exists, service location TBD
3. **Training/CLM Service**: Routes exist, microservice unclear
4. **Monitoring/Gateway Service**: Mentioned but not found
5. **EventBus Implementation**: Redis Streams presumed, needs verification
6. **Tracing System**: Not discovered yet

---

## 📋 NEXT STEPS (STEP 2)

1. Test all health endpoints:
   - Backend: `/health`, `/api/v2/health`, `/health/system`, `/health/ai`, `/health/risk`
   - Portfolio Service: Port 8004 `/health`
   - Exec/Risk Service: Port 8003 `/health`
   - AI Service: Port 8001 `/health`

2. Add missing health endpoints where needed

3. Create comprehensive health check test suite

4. Fix any unreachable services or broken dependencies

---

**Discovery Status**: ✅ COMPLETE  
**Health Check Status**: ⏳ PENDING (STEP 2)  
**AI Module Status**: ⏳ PENDING (STEP 3)
