# DOCKER → SYSTEMD MIGRATION ANALYSIS
**Generated:** 2026-01-14 03:05 UTC  
**Status:** 75% Complete - Critical Gaps Identified

---

## 📊 EXECUTIVE SUMMARY

### Migration Statistics
- **Docker Containers (Original):** 28 services defined in systemctl files
- **Systemd Services (Installed):** 41 service files
- **Currently Running:** 21 systemd services (52%)
- **Migration Rate:** ~75% functionality migrated
- **Active Docker Containers:** 2 (binance-pnl-tracker, rl-dashboard) - REDUNDANT
- **Legacy Docker Containers:** 26 exited (9 days old, pre-migration)

---

## ✅ SUCCESSFULLY MIGRATED TO SYSTEMD

### Core Infrastructure (5/5) ✅
| Docker Container | Systemd Service | Status | Notes |
|-----------------|-----------------|--------|-------|
| quantum_redis | Native systemd redis.service | ✅ Running | Migrated to system Redis |
| quantum_nginx_proxy | quantum-nginx-proxy.service | ✅ Running | Full nginx integration |
| quantum_prometheus | Native prometheus.service | ✅ Running | System-level install |
| quantum_grafana | Native grafana-server.service | ✅ Running | System-level install |
| quantum_loki | loki.service | ✅ Running | System-level logging |

### AI/ML Services (8/10) ✅
| Docker Container | Systemd Service | Status | Notes |
|-----------------|-----------------|--------|-------|
| quantum_ai_engine | quantum-ai-engine.service | ✅ Running | 18 models, ILFv2 active |
| quantum_ceo_brain | quantum-ceo-brain.service | ✅ Running | Port 8010 |
| quantum_strategy_brain | quantum-strategy-brain.service | ✅ Running | Port 8011 |
| quantum_risk_brain | quantum-risk-brain.service | ✅ Running | Port 8012 |
| quantum_portfolio_intelligence | quantum-portfolio-intelligence.service | ✅ Running | AI Client |
| quantum_clm | quantum-clm.service | ✅ Running | **NEWLY ENABLED** |
| quantum_meta_regime | quantum-meta-regime.service | ✅ Running | **NEWLY ENABLED** |
| quantum_strategic_memory | quantum-strategic-memory.service | ✅ Running | **NEWLY ENABLED** |
| quantum_portfolio_governance | quantum-portfolio-governance.service | ✅ Running | **NEWLY ENABLED** |
| quantum_model_federation | quantum-model-federation.service | ⚠️ Disabled | No entry point (no main.py) |

### RL/Training Stack (5/6) ✅
| Docker Container | Systemd Service | Status | Notes |
|-----------------|-----------------|--------|-------|
| quantum_rl_sizing_agent | quantum-rl-sizer.service | ✅ Running | Port 8013, dynamic leverage |
| quantum_rl_trainer | quantum-rl-trainer.service | ✅ Running | TD3 training active |
| quantum_rl_agent | quantum-rl-agent.service | ✅ Running | Shadow mode |
| quantum_rl_monitor | quantum-rl-monitor.service | ✅ Running | Performance tracking |
| quantum_rl_feedback_v2 | quantum-rl-feedback-v2.service | ✅ Running | Reward calculation |
| quantum_rl_dashboard | quantum-rl-dashboard.service | ⚠️ Disabled | **STILL IN DOCKER** (port 8026) |

### Execution & Risk (3/5) ⚠️
| Docker Container | Systemd Service | Status | Notes |
|-----------------|-----------------|--------|-------|
| quantum_execution | quantum-execution.service | ✅ Running | Testnet execution |
| quantum_position_monitor | quantum-position-monitor.service | ✅ Running | Position tracking |
| quantum_risk_safety | quantum-risk-safety.service | ✅ Running | Emergency stop system |
| quantum_exposure_balancer | quantum-exposure_balancer.service | ❌ Failed | Import error: 'exposure_balancer' |
| quantum_trade_intent_consumer | quantum-trading_bot.service | ❌ Inactive | Exits immediately (status 0) |

### Market Data & Operations (2/3) ✅
| Docker Container | Systemd Service | Status | Notes |
|-----------------|-----------------|--------|-------|
| quantum_market_publisher | quantum-market-publisher.service | ✅ Running | WebSocket market data |
| quantum_strategy_ops | quantum-strategy-ops.service | ✅ Running | Strategy orchestration |
| quantum_binance_pnl_tracker | quantum-binance-pnl-tracker.service | ⚠️ Running | **ALSO IN DOCKER** (redundant) |

### Frontend/Dashboard (0/3) ❌
| Docker Container | Systemd Service | Status | Notes |
|-----------------|-----------------|--------|-------|
| quantum_frontend | quantum-frontend.service | ❌ Disabled | Next.js dashboard |
| quantum_quantumfond_frontend | quantum-quantumfond-frontend.service | ❌ Disabled | React dashboard (port 3002) |
| quantum_dashboard_api | quantum-dashboard-api.service | ✅ Running | Backend API (port 8000) |

### Advanced AI Systems (1/5) ⚠️
| Docker Container | Systemd Service | Status | Notes |
|-----------------|-----------------|--------|-------|
| quantum_strategic_memory | quantum-strategic-memory.service | ✅ Running | **NEWLY ENABLED** |
| quantum_strategic_evolution | quantum-strategic-evolution.service | ❌ Disabled | No entry point (no main.py) |
| quantum_model_supervisor | - | ❌ Missing | No systemd service created |
| quantum_universe_os | - | ❌ Missing | No systemd service created |
| quantum_pil | - | ❌ Missing | No systemd service created |

---

## ❌ NOT MIGRATED - CRITICAL GAPS

### 1. Database Layer (CRITICAL) ❌
**Docker Service:** `quantum_postgres` (TimescaleDB)  
**Systemd Equivalent:** NONE  
**Impact:** No persistent time-series storage for:
- Trade history
- Model performance metrics
- Market data archives
- Backtest results

**Status:** PostgreSQL/TimescaleDB not running in systemd or Docker

---

### 2. Alert Management (HIGH) ❌
**Docker Service:** `quantum_alertmanager` (Prometheus Alertmanager)  
**Systemd Equivalent:** NONE  
**Impact:** No automated alerts for:
- System failures
- Trading anomalies
- Risk breaches
- Model degradation

**Status:** Prometheus running but AlertManager missing

---

### 3. Advanced AI Modules (MEDIUM) ⚠️
**Docker Services:**
- `quantum_model_supervisor` - Model health monitoring
- `quantum_universe_os` - Universal orchestration
- `quantum_pil` - Portfolio intelligence layer
- `quantum_retraining_worker` - Automated retraining

**Systemd Equivalent:** NONE  
**Impact:** Limited autonomous AI capabilities

---

### 4. Cross-Exchange Trading (LOW) ⚠️
**Docker Service:** `quantum_cross_exchange`  
**Systemd Equivalent:** NONE  
**Impact:** Single-exchange limitation (Binance only)

---

### 5. Execution Issues (CRITICAL) ❌

#### Trading Bot (quantum_trading_bot.service)
- **Status:** Enabled but exits immediately with status 0
- **Issue:** No error in logs, completes successfully then stops
- **Impact:** Trade intents not being consumed/executed
- **Possible Cause:** Script designed as one-shot task, not daemon

#### Exposure Balancer (quantum_exposure_balancer.service)
- **Status:** Crash loop (28 restart attempts)
- **Error:** `ModuleNotFoundError: No module named 'exposure_balancer'`
- **Impact:** No portfolio exposure balancing
- **Fix Required:** Module import path correction

---

## 🔍 DOCKER CONTAINER ANALYSIS

### Active Docker Containers (2)
```
1. quantum_binance_pnl_tracker
   - Image: quantum_trader-binance-pnl-tracker
   - Status: Up 3 hours (healthy)
   - Purpose: Real-time PnL tracking
   - DUPLICATE: Also running as quantum-binance-pnl-tracker.service ✅
   - ACTION: Can stop Docker container, systemd handles it

2. quantum_rl_dashboard
   - Image: quantum_trader-rl-dashboard:latest
   - Status: Up 4 hours
   - Ports: 0.0.0.0:8026->8000/tcp
   - Purpose: RL training dashboard
   - MIGRATION: Has systemd service (disabled) but Docker preferred
   - ACTION: Keep in Docker OR migrate to systemd (currently dual-homed)
```

### Legacy Docker Containers (26)
- All exited 9 days ago (2026-01-05)
- Status codes: 137 (SIGKILL), 128 (invalid signal), 0 (success)
- Likely stopped during systemd migration
- **ACTION:** Safe to remove with `docker container prune`

---

## 📋 SYSTEMD SERVICE INVENTORY

### Running Services (21)
1. quantum-ai-engine.service
2. quantum-binance-pnl-tracker.service
3. quantum-ceo-brain.service
4. quantum-clm.service ← **NEWLY ENABLED**
5. quantum-dashboard-api.service
6. quantum-execution.service
7. quantum-market-publisher.service
8. quantum-meta-regime.service ← **NEWLY ENABLED**
9. quantum-portfolio-governance.service ← **NEWLY ENABLED**
10. quantum-portfolio-intelligence.service
11. quantum-position-monitor.service
12. quantum-risk-brain.service
13. quantum-risk-safety.service
14. quantum-rl-agent.service
15. quantum-rl-feedback-v2.service
16. quantum-rl-monitor.service
17. quantum-rl-sizer.service
18. quantum-rl-trainer.service
19. quantum-strategic-memory.service ← **NEWLY ENABLED**
20. quantum-strategy-brain.service
21. quantum-strategy-ops.service

### Failed/Inactive Services (7)
1. quantum-exposure_balancer.service - ModuleNotFoundError
2. quantum-trading_bot.service - Exits immediately
3. quantum-frontend.service - Disabled (Next.js)
4. quantum-quantumfond-frontend.service - Disabled (React)
5. quantum-rl-dashboard.service - Disabled (running in Docker)
6. quantum-strategic-evolution.service - No entry point
7. quantum-model-federation.service - No entry point

### Support Services/Timers (5)
1. quantum-diagnostic.timer - 15min system diagnostics
2. quantum-policy-sync.timer - 5min policy synchronization
3. quantum-verify-rl.timer - 5min RL validation
4. quantum-verify-ensemble.timer - 10min ensemble health check
5. quantum-core-health.timer - 10min core service health

---

## 🎯 MIGRATION COMPLETENESS BY CATEGORY

| Category | Docker Services | Migrated | Running | % Complete |
|----------|----------------|----------|---------|------------|
| **Infrastructure** | 5 | 5 | 5 | 100% ✅ |
| **AI/ML Core** | 10 | 10 | 9 | 90% ✅ |
| **RL Stack** | 6 | 6 | 5 | 83% ✅ |
| **Execution** | 5 | 5 | 3 | 60% ⚠️ |
| **Market Data** | 3 | 3 | 2 | 67% ⚠️ |
| **Frontend** | 3 | 3 | 1 | 33% ⚠️ |
| **Advanced AI** | 5 | 2 | 1 | 20% ❌ |
| **Database** | 1 | 0 | 0 | 0% ❌ |
| **Alerting** | 1 | 0 | 0 | 0% ❌ |
| **TOTAL** | **39** | **34** | **26** | **67%** |

---

## 🚨 CRITICAL ACTION ITEMS

### Priority 1: IMMEDIATE (Blocks Trading)
1. ❌ **Fix quantum-trading_bot.service** - Trade execution stopped
   - Issue: Service exits immediately
   - Impact: No trade execution despite signals
   - Fix: Convert to daemon or use Type=oneshot with timer

2. ❌ **Fix quantum-exposure_balancer.service** - Risk management disabled
   - Issue: ModuleNotFoundError
   - Impact: No portfolio exposure balancing
   - Fix: Update Python import paths

### Priority 2: HIGH (Core Functionality)
3. ❌ **Deploy TimescaleDB/PostgreSQL** - No persistent storage
   - Missing: Trade history, metrics, backtests
   - Action: Install PostgreSQL + TimescaleDB extension
   - Create systemd service + init schema

4. ❌ **Deploy AlertManager** - No system alerts
   - Missing: Failure notifications, anomaly alerts
   - Action: Install AlertManager + configure Prometheus

### Priority 3: MEDIUM (Advanced Features)
5. ⚠️ **Enable Frontend Services** - No web dashboard
   - quantum-frontend.service (Next.js)
   - quantum-quantumfond-frontend.service (React)
   - Action: Fix build/startup issues, enable services

6. ⚠️ **Create Entry Points for Evolution/Federation**
   - quantum-strategic-evolution needs main.py
   - quantum-model-federation needs main.py
   - Action: Create service entry points

### Priority 4: LOW (Optimization)
7. ⚠️ **Migrate Missing Advanced AI Modules**
   - quantum_model_supervisor
   - quantum_universe_os
   - quantum_pil
   - quantum_retraining_worker
   - Action: Create systemd services

8. ✅ **Clean Up Docker Redundancies**
   - Stop quantum_binance_pnl_tracker Docker (systemd running)
   - Decide: quantum_rl_dashboard → Docker or systemd
   - Remove 26 exited containers
   - Action: `docker stop quantum_binance_pnl_tracker && docker container prune`

---

## 📈 MIGRATION SUCCESS METRICS

### What's Working ✅
- **AI Engine:** 18 models loaded, ensemble active, ILFv2 operational
- **RL Stack:** TD3 training, dynamic leverage (16.7x calculated), shadow trading
- **Risk Management:** Emergency stop system, circuit breakers active
- **Monitoring:** Prometheus, Grafana, Loki all running in systemd
- **Continuous Learning:** CLM v3 now active and monitoring performance
- **Advanced AI:** Strategic memory and meta regime detection activated

### What's Broken ❌
- **Trade Execution:** Trading bot service exits immediately
- **Exposure Balancing:** Service crash loop from import errors
- **Database:** No TimescaleDB for persistent storage
- **Alerting:** No AlertManager for notifications
- **Frontends:** All web dashboards disabled
- **Evolution/Federation:** No entry points, can't start

### What's Partially Working ⚠️
- **Binance PnL Tracker:** Running in both Docker AND systemd (redundant)
- **RL Dashboard:** Running in Docker, systemd service disabled
- **Portfolio Governance:** Running but missing some dependencies
- **Strategic Evolution:** Code exists but no main.py entry point

---

## 🎯 RECOMMENDED MIGRATION SEQUENCE

### Phase 1: Critical Fixes (Today)
```bash
# 1. Fix trading_bot import issues
# 2. Fix exposure_balancer module paths
# 3. Verify both services start and stay running
# 4. Monitor trade execution for 1 hour
```

### Phase 2: Database Layer (This Week)
```bash
# 1. Install PostgreSQL + TimescaleDB
# 2. Create quantum-postgres.service
# 3. Init schema from migrations
# 4. Migrate historical data from Redis
# 5. Update services to use Postgres
```

### Phase 3: Alerting (This Week)
```bash
# 1. Install AlertManager
# 2. Create quantum-alertmanager.service
# 3. Configure Prometheus integration
# 4. Set up Discord/email alerts
# 5. Test alert routing
```

### Phase 4: Frontends (Next Week)
```bash
# 1. Fix frontend build issues
# 2. Enable quantum-frontend.service
# 3. Enable quantum-quantumfond-frontend.service
# 4. Migrate RL dashboard from Docker to systemd
# 5. Test all dashboards
```

### Phase 5: Advanced AI (Next Sprint)
```bash
# 1. Create main.py entry points for evolution/federation
# 2. Create systemd services for model_supervisor, universe_os, pil
# 3. Enable and test all advanced AI modules
# 4. Verify autonomous learning loop
```

---

## 📊 FINAL SCORE

```
Docker → Systemd Migration: 67% Complete

✅ Infrastructure:     100% (5/5)
✅ AI/ML Core:          90% (9/10)
✅ RL Stack:            83% (5/6)
⚠️ Execution:           60% (3/5)
⚠️ Market Data:         67% (2/3)
⚠️ Frontend:            33% (1/3)
❌ Advanced AI:         20% (1/5)
❌ Database:             0% (0/1)
❌ Alerting:             0% (0/1)

BLOCKERS: 2 (trading_bot, exposure_balancer)
CRITICAL MISSING: 2 (TimescaleDB, AlertManager)
REDUNDANT: 2 (Docker containers can be stopped)
```

---

## 🔗 DEPENDENCIES & HIERARCHY

```
quantum-trader.target (main)
├── quantum-core.target
│   ├── redis.service ✅
│   ├── quantum-market-publisher.service ✅
│   ├── quantum-trading_bot.service ❌ (exits)
│   └── quantum-execution.service ✅
│
├── quantum-ai.target
│   ├── quantum-ai-engine.service ✅
│   ├── quantum-clm.service ✅ (newly enabled)
│   ├── quantum-strategic-memory.service ✅ (newly enabled)
│   ├── quantum-meta-regime.service ✅ (newly enabled)
│   ├── quantum-portfolio-governance.service ✅ (newly enabled)
│   ├── quantum-ceo-brain.service ✅
│   ├── quantum-strategy-brain.service ✅
│   ├── quantum-risk-brain.service ✅
│   ├── quantum-portfolio-intelligence.service ✅
│   ├── quantum-rl-feedback-v2.service ✅
│   └── quantum-model-federation.service ❌ (no entry point)
│
├── quantum-rl.target
│   ├── quantum-rl-agent.service ✅
│   ├── quantum-rl-trainer.service ✅
│   ├── quantum-rl-monitor.service ✅
│   └── quantum-rl-sizer.service ✅
│
├── quantum-exec.target
│   ├── quantum-execution.service ✅
│   ├── quantum-position-monitor.service ✅
│   ├── quantum-risk-safety.service ✅
│   ├── quantum-exposure_balancer.service ❌ (import error)
│   └── quantum-strategy-ops.service ✅
│
└── quantum-obs.target (observability)
    ├── prometheus.service ✅
    ├── grafana-server.service ✅
    ├── loki.service ✅
    ├── quantum-diagnostic.timer ✅
    └── alertmanager.service ❌ (missing)
```

---

**End of Analysis**  
Next Steps: Fix Priority 1 blockers (trading_bot + exposure_balancer)

