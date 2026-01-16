# SPRINT 2 - MICROSERVICES IMPLEMENTATION

## 📋 TODO LIST

### Phase 1: Boilerplate & Infrastructure ✅
- [x] Create folder structure for all 7 services
- [x] **risk-safety-service:** COMPLETE implementation
  - [x] main.py (FastAPI + lifespan + graceful shutdown)
  - [x] service.py (ESS + PolicyStore integration)
  - [x] api.py (REST endpoints)
  - [x] config.py (Settings)
  - [x] Dockerfile
  - [x] requirements.txt
  - [x] tests/test_service.py
  - [x] README.md

### Phase 2: Service #2 - AI Engine (NEXT) 🔄
- [ ] Create boilerplate files (main.py, service.py, api.py, etc.)
- [ ] Migrate RLPositionSizingAgent from `backend/ai/`
- [ ] Migrate MetaStrategySelector from `backend/ai/`
- [ ] Migrate AITradingEngine from `backend/ai/`
- [ ] Migrate PAL from `backend/ai_system/`
- [ ] Implement event handlers (trade.closed, market.conditions.updated)
- [ ] Implement API endpoints (/ai/decision, /ai/position-size, /ai/strategy)
- [ ] Write unit tests
- [ ] Test inter-service communication with risk-safety-service

### Phase 3: Service #3 - Execution 🔶
- [ ] Create boilerplate files
- [ ] Migrate EventDrivenExecutor
- [ ] Migrate BinanceFuturesExecutionAdapter
- [ ] Migrate HybridTPSL
- [ ] Migrate ExecutionSafetyGuard (D7)
- [ ] Migrate SafeOrderExecutor (D7)
- [ ] Migrate GlobalRateLimiter (D6)
- [ ] Migrate TradeStore (D5)
- [ ] Implement event handlers (ai.decision.made, ess.tripped)
- [ ] Implement API endpoints (/execution/order, /execution/trades)
- [ ] Write unit tests

### Phase 4: Service #4 - Portfolio Intelligence 🔶
- [ ] Create boilerplate files
- [ ] Migrate PBA from `backend/ai_system/`
- [ ] Implement portfolio tracking logic
- [ ] Implement event handlers (trade.opened, trade.closed)
- [ ] Implement API endpoints (/portfolio/summary, /portfolio/rebalance)
- [ ] Write unit tests

### Phase 5: Service #5 - RL Training 🔶
- [ ] Create boilerplate files
- [ ] Migrate RL training pipeline from `backend/ai/rl_training/`
- [ ] Migrate retraining scheduler
- [ ] Implement offline training job management
- [ ] Implement API endpoints (/training/start, /training/models)
- [ ] Write unit tests

### Phase 6: Service #6 - Monitoring & Health 🔶
- [ ] Create boilerplate files
- [ ] Implement health check aggregator
- [ ] Implement metrics collection
- [ ] Implement alert triggers
- [ ] Implement API endpoints (/health/all, /metrics/system)
- [ ] Write unit tests

### Phase 7: Service #7 - Market Data (Optional) 🔶
- [ ] Create boilerplate files
- [ ] Implement Binance WebSocket client
- [ ] Implement candle cache
- [ ] Implement event publishers (market.price.updated)
- [ ] Implement API endpoints (/market/price, /market/candles)
- [ ] Write unit tests

### Phase 8: Integration & Testing 🔶
- [ ] Create Docker Compose file for all services
- [ ] Implement API Gateway (FastAPI proxy)
- [ ] Replace in-memory EventBus with Redis Streams
- [ ] Test full event flow (ai.decision → execution → trade.closed)
- [ ] Test ESS trip flow (trade.closed → ess.tripped → order blocked)
- [ ] Test policy update flow (policy.updated → all services refresh)
- [ ] Load testing (1000 concurrent requests)
- [ ] Failover testing (kill service, verify recovery)

### Phase 9: Documentation & Deployment 🔶
- [ ] Create EVENT_SCHEMA.md (all event definitions)
- [ ] Create API_REFERENCE.md (all endpoints)
- [ ] Create DEPLOYMENT_GUIDE.md (Docker Compose + Kubernetes)
- [ ] Create MONITORING_GUIDE.md (metrics, alerts, dashboards)
- [ ] Update Master Plan v1.0 with microservices architecture
- [ ] Production deployment checklist

---

## 🎯 Current Focus

**Service #1 (risk-safety-service):** ✅ COMPLETE
- 9 files created
- Full ESS + PolicyStore implementation
- Event-driven architecture
- Graceful shutdown
- Health checks
- Docker ready

**Next Action:** Start Phase 2 - AI Engine Service

---

## 📊 Progress Tracking

**Overall Sprint 2 Progress:** 14% (1/7 services complete)

| Service | Boilerplate | Migration | Events | API | Tests | Status |
|---------|-------------|-----------|--------|-----|-------|--------|
| risk-safety | ✅ | ✅ | ✅ | ✅ | 🔶 | **COMPLETE** |
| ai-engine | 🔶 | 🔶 | 🔶 | 🔶 | 🔶 | NOT STARTED |
| execution | 🔶 | 🔶 | 🔶 | 🔶 | 🔶 | NOT STARTED |
| portfolio-intel | 🔶 | 🔶 | 🔶 | 🔶 | 🔶 | NOT STARTED |
| rl-training | 🔶 | 🔶 | 🔶 | 🔶 | 🔶 | NOT STARTED |
| monitoring-health | 🔶 | 🔶 | 🔶 | 🔶 | 🔶 | NOT STARTED |
| marketdata | 🔶 | 🔶 | 🔶 | 🔶 | 🔶 | NOT STARTED |

**Legend:**
- ✅ Complete
- 🔶 Not started
- 🔄 In progress
- ❌ Blocked

---

## 🚀 Quick Start (Service #1)

```bash
cd microservices/risk_safety
python -m pip install -r requirements.txt
python main.py
```

Test endpoints:
```bash
# Health check
curl http://localhost:8003/health

# ESS status
curl http://localhost:8003/api/risk/ess/status

# Get policy
curl http://localhost:8003/api/policy/execution.max_slippage_pct

# Update policy
curl -X POST http://localhost:8003/api/policy/update \
  -H "Content-Type: application/json" \
  -d '{"key": "execution.max_slippage_pct", "value": 0.01}'
```

