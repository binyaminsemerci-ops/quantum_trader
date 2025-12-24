# SERVICE CATALOG — VPS DEPLOYMENT AUDIT
**Date**: December 24, 2025 05:00 UTC  
**VPS**: quantumtrader-prod-1 (Ubuntu 24.04.3 LTS)  
**Docker**: 29.1.3 | Docker Compose: v5.0.0

## SYSTEM RESOURCES
- CPU: 4 cores (load: 0.28, 0.36, 0.52)
- Memory: 15GB total, 12GB used (80%), 2.8GB available
- Disk: 150GB total, 106GB used (74%), 39GB free

---

## CORE TRADING SERVICES (8 containers)

### quantum_backend
- **Image**: quantum_trader-backend
- **Status**: Up 5 minutes (healthy)
- **Ports**: 0.0.0.0:8000->8000/tcp
- **Purpose**: Main FastAPI backend, API gateway
- **Health**: http://localhost:8000/health → {\ status\:\ok\,\phases\:{\phase4_aprl\:{...}}}
- **Category**: CORE

### quantum_ai_engine
- **Image**: quantum_trader-ai-engine
- **Status**: Up 39 minutes (healthy)
- **Ports**: 0.0.0.0:8001->8001/tcp
- **Purpose**: AI prediction service (ML ensemble)
- **Category**: CORE

### quantum_trading_bot
- **Image**: quantum_trading_bot:latest
- **Status**: Up 26 minutes (healthy)
- **Ports**: 0.0.0.0:8003->8003/tcp
- **Purpose**: Signal generator, publishes trade.intent with ILF metadata
- **Category**: CORE

### quantum_redis
- **Image**: redis:7-alpine
- **Status**: Up 2 hours (healthy)
- **Ports**: 0.0.0.0:6379->6379/tcp
- **Memory**: 91MB
- **Streams**: 21 active event streams
- **Purpose**: Event bus backbone
- **Category**: CORE

### quantum_risk_safety
- **Image**: quantum_trader-risk-safety
- **Status**: Up 2 hours (healthy)
- **Ports**: 0.0.0.0:8005->8005/tcp
- **Purpose**: Risk validation (STUB implementation)
- **Category**: CORE (STUB)

### quantum_position_monitor
- **Image**: quantum_position_monitor
- **Status**: Up 2 hours
- **Purpose**: Position tracking
- **Category**: CORE

### quantum_market_publisher
- **Image**: quantum_trader-market-publisher
- **Status**: Up 6 minutes (healthy)
- **Purpose**: Market data collector
- **Category**: CORE

### quantum_cross_exchange
- **Image**: quantum_cross_exchange:latest
- **Status**: Up 2 hours
- **Purpose**: Cross-exchange data aggregation
- **Category**: CORE

---

## GOVERNANCE LAYER (6 containers)

### quantum_ceo_brain
- **Image**: quantum_trader-ceo-brain
- **Status**: Up 2 hours (healthy)
- **Ports**: 0.0.0.0:8010->8010/tcp
- **Purpose**: Top-level governance orchestration
- **Category**: GOVERNANCE

### quantum_risk_brain
- **Image**: quantum_trader-risk-brain
- **Status**: Up 2 hours (healthy)
- **Ports**: 0.0.0.0:8012->8012/tcp
- **Purpose**: Risk governance and policy
- **Category**: GOVERNANCE

### quantum_strategy_brain
- **Image**: quantum_trader-strategy-brain
- **Status**: Up 2 hours (healthy)
- **Ports**: 0.0.0.0:8011->8011/tcp
- **Purpose**: Strategy governance
- **Category**: GOVERNANCE

### quantum_universe_os
- **Image**: quantum_trader-universe-os
- **Status**: Up 2 hours (healthy)
- **Ports**: 0.0.0.0:8006->8006/tcp
- **Purpose**: Orchestration layer coordinator
- **Category**: GOVERNANCE

### quantum_governance_dashboard
- **Image**: quantum_trader-governance-dashboard
- **Status**: Up 2 hours (healthy)
- **Ports**: 0.0.0.0:8501->8501/tcp
- **Purpose**: Streamlit governance UI
- **Category**: GOVERNANCE

### quantum_governance_alerts
- **Image**: quantum_trader-governance-alerts
- **Status**: Up 2 hours (healthy)
- **Purpose**: Alert routing
- **Category**: GOVERNANCE

---

## PORTFOLIO & INTELLIGENCE (3 containers)

### quantum_portfolio_intelligence
- **Image**: quantum_trader-portfolio-intelligence
- **Status**: Up 2 hours (healthy)
- **Ports**: 0.0.0.0:8004->8004/tcp
- **Purpose**: Portfolio analytics
- **Category**: PORTFOLIO

### quantum_pil
- **Image**: quantum_trader-pil
- **Status**: Up 2 hours (healthy)
- **Ports**: 0.0.0.0:8013->8013/tcp
- **Purpose**: Position Intelligence Layer
- **Category**: PORTFOLIO

### quantum_model_supervisor
- **Image**: quantum_trader-model-supervisor
- **Status**: Up 2 hours (healthy)
- **Ports**: 0.0.0.0:8007->8007/tcp
- **Purpose**: Model monitoring
- **Category**: PORTFOLIO

---

## LEARNING & OPTIMIZATION (7 containers)

### quantum_clm
- **Image**: quantum_trader-clm
- **Status**: Up 2 hours
- **Purpose**: Continuous Learning Module
- **Category**: LEARNING

### quantum_rl_optimizer
- **Image**: quantum_trader-rl-optimizer
- **Status**: Up 2 hours (healthy)
- **Purpose**: RL optimization
- **Category**: LEARNING

### quantum_strategy_evolution
- **Image**: quantum_trader-strategy-evolution
- **Status**: Up 2 hours (healthy)
- **Purpose**: Strategy evolution
- **Category**: LEARNING

### quantum_strategy_evaluator
- **Image**: quantum_trader-strategy-evaluator
- **Status**: Up 2 hours (healthy)
- **Purpose**: Strategy evaluation
- **Category**: LEARNING

### quantum_strategic_evolution
- **Image**: quantum_strategic_evolution
- **Status**: Up 2 hours
- **Purpose**: Strategic evolution
- **Category**: LEARNING

### quantum_policy_memory
- **Image**: quantum_trader-quantum-policy-memory
- **Status**: Up 2 hours (healthy)
- **Purpose**: Policy storage
- **Category**: LEARNING

### quantum_trade_journal
- **Image**: quantum_trader-trade-journal
- **Status**: Up 2 hours (healthy)
- **Purpose**: Trade history logging
- **Category**: LEARNING

### quantum_model_federation
- **Image**: quantum_trader-model-federation
- **Status**: Up 2 hours
- **Purpose**: Model ensemble coordination
- **Category**: LEARNING

---

## INFRASTRUCTURE (6 containers)

### quantum_postgres
- **Image**: postgres:15-alpine
- **Status**: Up 2 hours (healthy)
- **Ports**: 127.0.0.1:5432->5432/tcp (internal only)
- **Purpose**: PostgreSQL database
- **Category**: INFRA

### quantum_prometheus
- **Image**: prom/prometheus:v2.48.1
- **Status**: Up 2 hours (healthy)
- **Ports**: 127.0.0.1:9090->9090/tcp
- **Purpose**: Metrics collection
- **Category**: INFRA

### quantum_grafana
- **Image**: grafana/grafana:10.2.3
- **Status**: Up 2 hours (healthy)
- **Ports**: 127.0.0.1:3001->3000/tcp
- **Purpose**: Visualization
- **Category**: INFRA

### quantum_alertmanager
- **Image**: prom/alertmanager:v0.26.0
- **Status**: Up 2 hours
- **Ports**: 127.0.0.1:9093->9093/tcp
- **Purpose**: Alert routing
- **Category**: INFRA

### quantum_nginx
- **Image**: nginx:alpine
- **Status**: Up 2 hours (UNHEALTHY) ⚠️
- **Ports**: 127.0.0.1:80->80/tcp, 127.0.0.1:443->443/tcp
- **Purpose**: Reverse proxy
- **Category**: INFRA
- **ISSUE**: Marked unhealthy by Docker

### quantum_dashboard
- **Image**: quantum_dashboard:latest
- **Status**: Up 2 hours
- **Ports**: 0.0.0.0:8080->8080/tcp
- **Purpose**: Main dashboard UI
- **Category**: INFRA

---

## STUBS & BRIDGES (2 containers)

### quantum_federation_stub
- **Image**: quantum_trader-federation-stub
- **Status**: Up 2 hours (healthy)
- **Purpose**: Federation placeholder
- **Category**: STUB

### quantum_eventbus_bridge
- **Image**: quantum_eventbus_bridge
- **Status**: Up 2 hours
- **Purpose**: Event bus bridge
- **Category**: BRIDGE

---

## SUMMARY

**Total Containers**: 32 running
**Healthy**: 31
**Unhealthy**: 1 (quantum_nginx)
**Stub Implementations**: 2 (risk_safety, federation_stub)

**Deployment Method**: Docker images (NOT git repo)
**Orchestration**: Manual docker run (NO docker-compose detected)
**Version Control**: NONE on VPS (images built elsewhere)
