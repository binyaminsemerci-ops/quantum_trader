# QUANTUM TRADER HEDGE FUND OS — SYSTEM OVERVIEW
## VPS DEPLOYMENT AUDIT — December 24, 2025

**Status**: PRODUCTION DEPLOYMENT — ACTIVE TRADING
**Audit Type**: FACTUAL REVERSE ENGINEERING (NOT DESIGN)

---

## 1. INFRASTRUCTURE BASELINE

### VPS Configuration
- **Hostname**: `quantumtrader-prod-1`
- **OS**: Ubuntu 22.04 LTS (Linux 6.8.0-88-generic x86_64)
- **Uptime**: 6 days, 12 hours
- **CPU**: 4 cores
- **Memory**: 16 GB (12 GB used, 2.8 GB available)
- **Disk**: 150 GB total, 106 GB used (74%), 39 GB free
- **Load Average**: 0.29, 0.46, 0.79 (normal for 4-core system)

### Runtime Environment
- **Container Engine**: Docker (latest)
- **Orchestration**: Docker Compose (NOT Kubernetes)
- **Deployment Path**: `/opt/quantum_trader` (NON-GIT DEPLOYMENT)
- **Redis Version**: 7.4.7
- **Redis Memory**: 91.04 MB
- **Redis Uptime**: < 1 day (recent restart)

### Repository State
**CRITICAL**: The VPS deployment is NOT a git repository
- Code is deployed as static files in `/opt/quantum_trader`
- No `.git` directory present
- No version tracking on production
- Updates require manual file copy or Docker image rebuild
- **Gap**: No deployment versioning or rollback capability

---

## 2. ACTIVE SERVICES (32 Containers)

### CORE TRADING SERVICES (8)
| Service | Status | Uptime | Purpose |
|---------|--------|--------|---------|
| quantum_backend | ACTIVE | 33 min | Main FastAPI backend (port 8000) |
| quantum_ai_engine | ACTIVE | 22 min | AI prediction service (port 8001) |
| quantum_trading_bot | ACTIVE | 8 min | Signal generator (port 8003) |
| quantum_risk_safety | ACTIVE | 2 hours | Risk validation service (port 8005) |
| quantum_redis | ACTIVE | 2 hours | Event bus storage (port 6379) |
| quantum_market_publisher | ACTIVE | 9 sec | Market data collector |
| quantum_position_monitor | ACTIVE | 2 hours | Position tracking |
| quantum_cross_exchange | ACTIVE | 2 hours | Cross-exchange data |

### GOVERNANCE LAYER (6)
| Service | Status | Uptime | Purpose |
|---------|--------|--------|---------|
| quantum_ceo_brain | ACTIVE | 2 hours | Top-level governance (port 8010) |
| quantum_risk_brain | ACTIVE | 2 hours | Risk governance (port 8012) |
| quantum_strategy_brain | ACTIVE | 2 hours | Strategy governance (port 8011) |
| quantum_universe_os | ACTIVE | 2 hours | Orchestration layer (port 8006) |
| quantum_governance_dashboard | ACTIVE | 2 hours | Streamlit UI (port 8501) |
| quantum_governance_alerts | ACTIVE | 2 hours | Alert system |

### PORTFOLIO INTELLIGENCE (4)
| Service | Status | Uptime | Purpose |
|---------|--------|--------|---------|
| quantum_portfolio_intelligence | ACTIVE | 2 hours | Portfolio analytics (port 8004) |
| quantum_pil | ACTIVE | 2 hours | Position Intelligence Layer (port 8013) |
| quantum_model_supervisor | ACTIVE | 2 hours | Model monitoring (port 8007) |
| quantum_model_federation | ACTIVE | 2 hours | Model ensemble |

### LEARNING & OPTIMIZATION (7)
| Service | Status | Uptime | Purpose |
|---------|--------|--------|---------|
| quantum_clm | ACTIVE | 2 hours | Continuous Learning Module |
| quantum_rl_optimizer | ACTIVE | 2 hours | RL optimization |
| quantum_strategy_evolution | ACTIVE | 2 hours | Strategy evolution |
| quantum_strategy_evaluator | ACTIVE | 2 hours | Strategy evaluation |
| quantum_strategic_evolution | ACTIVE | 2 hours | Strategic evolution |
| quantum_policy_memory | ACTIVE | 2 hours | Policy storage |
| quantum_trade_journal | ACTIVE | 2 hours | Trade history |

### INFRASTRUCTURE (6)
| Service | Status | Uptime | Purpose |
|---------|--------|--------|---------|
| quantum_postgres | ACTIVE | 2 hours | PostgreSQL DB (port 5432) |
| quantum_prometheus | ACTIVE | 2 hours | Metrics (port 9090) |
| quantum_grafana | ACTIVE | 2 hours | Visualization (port 3001) |
| quantum_alertmanager | ACTIVE | 2 hours | Alert routing (port 9093) |
| quantum_nginx | UNHEALTHY | 2 hours | Reverse proxy (port 80/443) |
| quantum_dashboard | ACTIVE | 2 hours | Main dashboard (port 8080) |

### STUBS & BRIDGES (2)
| Service | Status | Uptime | Purpose |
|---------|--------|--------|---------|
| quantum_federation_stub | ACTIVE | 2 hours | Federation placeholder |
| quantum_eventbus_bridge | ACTIVE | 2 hours | Event bus bridge |

---

## 3. ENTRY POINTS & CODE PATHS

### Core Services Entry Points
```
quantum_backend:
  Command: uvicorn backend.main:app --host 0.0.0.0 --port 8000
  File: /app/backend/main.py
  Working Dir: /app
  Mode: PRODUCTION (MODE=prod)

quantum_ai_engine:
  Command: python -m uvicorn microservices.ai_engine.main:app --host 0.0.0.0 --port 8001
  File: /app/microservices/ai_engine/main.py
  Working Dir: /app

quantum_trading_bot:
  Command: uvicorn microservices.trading_bot.main:app --host 0.0.0.0 --port 8003
  File: /app/microservices/trading_bot/main.py
  Working Dir: /app

quantum_risk_safety:
  Command: python3 /app/microservices/risk_safety/stub_main.py
  File: /app/microservices/risk_safety/stub_main.py
  Working Dir: /app
```

---

## 4. DATA FLOW INFRASTRUCTURE

### Redis Streams (21 Active)
```
MARKET DATA:
- quantum:stream:market_data       (raw market feed)
- quantum:stream:market.tick       (tick updates)
- quantum:stream:market.klines     (candlestick data)
- quantum:stream:exchange.raw      (exchange raw data)
- quantum:stream:exchange.normalized (normalized data)

AI & SIGNALS:
- quantum:stream:ai.signal_generated (AI predictions)
- quantum:stream:ai.decision.made    (AI decisions)
- quantum:stream:sizing.decided      (position sizing)

TRADING:
- quantum:stream:trade.intent       (trade intentions)
- quantum:stream:execution.result   (execution results)
- quantum:stream:trade.closed       (closed trades)

PORTFOLIO:
- quantum:stream:portfolio.snapshot_updated  (portfolio state)
- quantum:stream:portfolio.exposure_updated  (exposure tracking)

LEARNING:
- quantum:stream:rl_v3.training.started     (RL training start)
- quantum:stream:rl_v3.training.completed   (RL training done)
- quantum:stream:learning.retraining.started
- quantum:stream:learning.retraining.completed
- quantum:stream:learning.retraining.failed
- quantum:stream:model.retrain              (model retraining)

GOVERNANCE:
- quantum:stream:policy.updated     (policy changes)
- quantum:stream:meta.regime        (regime detection)
```

---

## 5. SYSTEM ARCHITECTURE TYPE

**Classification**: **ADVANCED ALGORITHMIC TRADING SYSTEM** with **PARTIAL HEDGE FUND OS COMPONENTS**

### What This System IS:
- ✅ Multi-microservice architecture
- ✅ Event-driven (Redis Streams)
- ✅ AI-powered (ML models, RL optimization)
- ✅ Risk management layer
- ✅ Governance framework (CEO/Risk/Strategy brains)
- ✅ Continuous learning infrastructure
- ✅ Portfolio intelligence
- ✅ Real-time monitoring (Prometheus/Grafana)

### What This System IS NOT (Yet):
- ❌ NOT using advanced ILF metadata for leverage calculation (gap discovered)
- ❌ NOT git-versioned on production (deployment risk)
- ❌ NOT using Kubernetes (scalability limit)
- ❌ NO automated rollback mechanism
- ❌ NO blue-green deployment
- ❌ Nginx is UNHEALTHY (reverse proxy broken)

---

## 6. HEALTH STATUS SUMMARY

### CRITICAL Services (Must Be Up):
- ✅ quantum_backend (HEALTHY)
- ✅ quantum_ai_engine (HEALTHY) 
- ✅ quantum_trading_bot (HEALTHY)
- ✅ quantum_redis (HEALTHY)
- ❌ quantum_nginx (UNHEALTHY) ← **NEEDS ATTENTION**

### Recent Restarts (< 1 hour):
- quantum_trading_bot (8 minutes) — Recent deployment
- quantum_market_publisher (9 seconds) — Just restarted
- quantum_backend (33 minutes) — Recent update
- quantum_ai_engine (22 minutes) — Recent update

### Long-Running Services (2+ hours):
- All governance services ✅
- All learning services ✅
- All infrastructure services ✅ (except Nginx)
- Portfolio intelligence ✅

---

## 7. RESOURCE UTILIZATION

### Memory Status
- **Total**: 16 GB
- **Used**: 12 GB (75%)
- **Available**: 2.8 GB
- **Assessment**: **APPROACHING LIMIT** — Monitor for OOM kills

### Disk Status
- **Total**: 150 GB
- **Used**: 106 GB (74%)
- **Free**: 39 GB
- **Assessment**: **HIGH USAGE** — Need cleanup/expansion plan

### CPU Load
- **Cores**: 4
- **Load**: 0.29, 0.46, 0.79
- **Assessment**: NORMAL (well below core count)

### Redis Memory
- **Used**: 91 MB
- **Assessment**: LOW (Redis is efficient)

---

## 8. DEPLOYMENT GAPS & RISKS

### Critical Infrastructure Risks
1. **NO GIT VERSIONING on production** → No rollback capability
2. **Nginx UNHEALTHY** → Reverse proxy broken
3. **High memory usage (75%)** → Risk of OOM kills
4. **High disk usage (74%)** → Need cleanup policy
5. **Recent service restarts** → Recent deployments, stability unknown

### Architectural Gaps
1. **ILF metadata NOT consumed** (discovered in audit)
2. **Trade Intent Subscriber doesn't use AI leverage** (hardcoded leverage=1)
3. **ExitBrain v3.5 NOT receiving ILF inputs** (integration gap)
4. **No deployment pipeline** (manual Docker operations)
5. **No automated health recovery** (manual intervention required)

---

## 9. NEXT STEPS (Post-Audit)

### Immediate Actions Required
1. Fix Nginx UNHEALTHY status
2. Integrate ILF metadata in Trade Intent Subscriber
3. Monitor memory usage closely (approaching limit)
4. Implement git-based deployments
5. Set up disk cleanup automation

### System Understanding Required
- Complete service catalog (detailed in SERVICE_CATALOG.md)
- Event flow mapping (detailed in EVENT_FLOW_MAP.md)
- Order lifecycle (detailed in ORDER_LIFECYCLE.md)
- Exit mechanics (detailed in TP_SL_EXIT_AUDIT.md)
- AI modules inventory (detailed in AI_MODULES_STATUS.md)

---

**End of System Overview**
