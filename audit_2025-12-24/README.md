# VPS DEPLOYMENT AUDIT — December 24, 2025

## AUDIT OVERVIEW

**Purpose**: Complete reverse-engineering audit of quantum_trader production deployment on VPS  
**Date**: December 24, 2025 05:00-05:15 UTC  
**Methodology**: Read-only evidence collection, NO changes made  
**Target**: quantumtrader-prod-1 (46.224.116.254)  

---

## AUDIT STRUCTURE

### Evidence Files (raw/)
17+ raw evidence files collected:
- System context (OS, hardware, Docker versions)
- Container inventory (32 services)
- Docker inspect JSONs (5 core services)
- Container logs (4 services, 200 lines each)
- Redis metrics (memory, stats, clients, slowlog)
- Redis stream inventory (21 streams)
- Stream samples (trade.intent with ILF metadata)
- Consumer group status (CRITICAL: 10K lag discovered)
- Health endpoints (backend, ai_engine, trading_bot)

### Audit Reports (md/)
7 comprehensive markdown reports:

1. **SERVICE_CATALOG.md** (7.1KB)
   - Complete inventory of 32 containers
   - Categorized: CORE, GOVERNANCE, PORTFOLIO, LEARNING, INFRA, STUBS
   - Status: 31 healthy, 1 unhealthy (nginx)

2. **EVENT_FLOW_MAP.md** (6.3KB)
   - 21 Redis streams documented
   - Producer/consumer mapping
   - Event flow diagram (market data → exit)
   - CRITICAL: Consumer lag 10,014 events

3. **ORDER_LIFECYCLE.md** (9.8KB)
   - Complete trade flow: Market data → AI → Decision → Sizing → Intent → Execution → Exit → Learning
   - Phase-by-phase breakdown
   - CRITICAL GAP: Execution layer disconnected (Phase 5)

4. **TP_SL_EXIT_AUDIT.md** (8.5KB)
   - ExitBrain v3 status: ACTIVE, managing 15 positions
   - ExitBrain v3.5 status: Code exists, never invoked
   - ILF integration: Ready but not used (consumer gap)

5. **LEVERAGE_SIZING_AUDIT.md** (15KB)
   - ILF metadata generation: COMPLETE (5 fields)
   - Adaptive leverage calculation: NEVER EXECUTED
   - Financial impact estimate: 400K theoretical opportunity cost

6. **AI_MODULES_STATUS.md** (13KB)
   - 11 AI/ML services: ALL HEALTHY
   - 8 learning streams: ACTIVE
   - Model coverage: Some 404s (fallback to simple strategy)

7. **GAPS_AND_FIXES_BACKLOG.md** (736B)
   - 11 gaps identified (P0-P3)
   - P0: Consumer lag, nginx unhealthy
   - P1: ILF integration blocked, regime detection, risk safety stub
   - P2: No git, manual orchestration, high resource usage
   - P3: Model coverage, funding/divergence data

---

## KEY FINDINGS

### CRITICAL (P0)
🚨 **Consumer Group Lag**: 10,014 unprocessed trade.intent events
- 34 consumers registered but NOT processing
- System processed 45K+ events historically (WAS working)
- Root cause: Consumers crashed/stopped (NOT deployment issue)
- Impact: Trading effectively offline, 400K theoretical opportunity loss

⚠️ **Nginx Unhealthy**: Reverse proxy failing health checks
- Impact: External access, dashboards may be affected

### HIGH (P1)
❌ **ILF Integration Gap**: Adaptive leverage never calculated
- Metadata generated: ✅ (atr_value, volatility_factor, etc.)
- Code deployed: ✅ (Session 3 hot-copy)
- Execution: ❌ (blocked by P0 consumer lag)
- Impact: Positions use leverage=1 instead of adaptive 5-80x

🟡 **Regime Detection**: regime= unknown for all events
- Impact: Regime-based leverage adjustment disabled

🟡 **Risk Safety Stub**: Using placeholder implementation
- Impact: Risk checks may be incomplete

### MEDIUM (P2)
📦 **No Git Version Control**: Pre-built images, no source code
- Impact: Cannot rollback, no audit trail

🔧 **Manual Orchestration**: 32 containers without docker-compose
- Impact: Hard to reproduce, error-prone restarts

💾 **High Resource Usage**: 80% RAM, 74% disk
- Impact: Risk of OOMKill (may explain P0 crash)

### LOW (P3)
🤖 **AI Engine 404s**: Some symbols use fallback strategy
📊 **Funding/Divergence=0**: May be real or data gap

---

## SYSTEM HEALTH

**VPS**: quantumtrader-prod-1
- OS: Ubuntu 24.04.3 LTS
- CPU: 4 cores, Load: 0.28, 0.36, 0.52 (normal)
- Memory: 15GB total, 12GB used (80% ⚠️)
- Disk: 150GB total, 106GB used (74% ⚠️)
- Uptime: 6 days

**Docker**: 29.1.3, Compose: v5.0.0
**Containers**: 32 running (31 healthy, 1 unhealthy)
**Redis**: 7-alpine, 91MB memory, 21 active streams

---

## EVIDENCE LOCATIONS

`
/opt/quantum_trader/audit_2025-12-24/
├── README.md (this file)
├── raw/ (17+ evidence files)
│   ├── system_context.txt
│   ├── docker_ps.txt
│   ├── compose_files.txt
│   ├── inspect_quantum_backend.json
│   ├── inspect_quantum_ai_engine.json
│   ├── inspect_quantum_trading_bot.json
│   ├── inspect_quantum_redis.json
│   ├── inspect_quantum_risk_safety.json
│   ├── logs_tail_quantum_backend.txt
│   ├── logs_tail_quantum_ai_engine.txt
│   ├── logs_tail_quantum_trading_bot.txt
│   ├── logs_tail_quantum_redis.txt
│   ├── redis_info_memory.txt
│   ├── redis_info_stats.txt
│   ├── redis_client_list.txt
│   ├── redis_slowlog.txt
│   ├── redis_stream_keys.txt
│   ├── redis_streams_lengths.txt
│   ├── redis_sample_trade_intent.txt
│   ├── redis_groups_trade_intent.txt ← CRITICAL EVIDENCE
│   ├── http_health_backend.txt
│   ├── http_health_ai_engine.txt
│   └── http_health_trading_bot.txt
└── md/ (7 audit reports)
    ├── SERVICE_CATALOG.md (7.1KB)
    ├── EVENT_FLOW_MAP.md (6.3KB)
    ├── ORDER_LIFECYCLE.md (9.8KB)
    ├── TP_SL_EXIT_AUDIT.md (8.5KB)
    ├── LEVERAGE_SIZING_AUDIT.md (15KB)
    ├── AI_MODULES_STATUS.md (13KB)
    └── GAPS_AND_FIXES_BACKLOG.md (736B)
`

---

## NEXT STEPS

### IMMEDIATE (P0 - Day 1):
1. Investigate consumer crash (check logs for OOMKill, exceptions)
2. Identify which service contains crashed consumers
3. Restart Trade Intent Subscriber
4. Monitor consumer lag decrease (10K → 0)
5. Fix nginx unhealthy status

### HIGH PRIORITY (P1 - Week 1):
6. Verify ILF integration works (after P0 fixed)
7. Connect regime detection to Trading Bot
8. Review Risk Safety stub implementation
9. Address resource usage (prevent OOMKill)

### DEPLOYMENT HYGIENE (P2 - Weeks 2-3):
10. Setup git version control
11. Create docker-compose.yml for all 32 services
12. Implement log rotation, disk cleanup

### OPTIMIZATION (P3 - Month 2):
13. Expand AI Engine model coverage (reduce 404s)
14. Investigate funding_rate and exchange_divergence data

---

## AUDIT METHODOLOGY

**Phase 0**: Setup audit directory, capture system baseline  
**Phase 1**: Container inventory, docker inspect, logs  
**Phase 2**: Redis audit (streams, consumers, samples) ← CRITICAL DISCOVERY  
**Phase 3**: Health endpoint checks  
**Phase 4**: Create markdown reports  

**Critical Discovery**: Phase 2 revealed consumer group with 10K lag, fundamentally changing understanding of the issue from missing startup to consumer crash/stop.

---

## CONCLUSION

**As-Is State**: 90% operational system with critical execution layer gap. Infrastructure is robust (32 services, all healthy except nginx), AI/ML pipeline active, learning systems operational. However, 10K event backlog in execution layer prevents ANY trades from executing, blocking ILF integration and adaptive leverage calculation.

**Root Issue**: NOT a code issue, NOT a deployment issue. Consumers were working (45K+ events processed), then crashed/stopped. Need to investigate crash cause before adding new functionality.

**Estimated Effort**: 2-3 weeks to resolve P0-P1, 4-5 weeks for complete backlog.

**Audit Status**: ✅ COMPLETE (Read-only, no changes made)
