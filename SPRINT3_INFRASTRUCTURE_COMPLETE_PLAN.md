# 🏗️ SPRINT 3 — INFRASTRUCTURE HARDENING
## Complete Implementation Plan

**Status**: ✅ SKELETON COMPLETE  
**Priority**: P0 (Production Readiness)  
**Sprint Goal**: "Robusthet, failover og driftssikkerhet for hele microservice-plattformen"

---

## 📋 TABLE OF CONTENTS

1. [Overview](#overview)
2. [Problem Analysis](#problem-analysis)
3. [Implementation Modules](#implementation-modules)
4. [File Structure](#file-structure)
5. [Deployment Roadmap](#deployment-roadmap)
6. [Testing Strategy](#testing-strategy)
7. [Acceptance Criteria](#acceptance-criteria)

---

## 1. OVERVIEW

### **Objective**

Transform the current microservice architecture from development/prototype state to production-ready infrastructure with:
- **High Availability** (no single points of failure)
- **Automatic Failover** (Redis Sentinel, Postgres replication)
- **API Gateway** (centralized routing, rate limiting, security)
- **Unified Logging** (correlation IDs, structured JSON logs)
- **Metrics + Monitoring** (Prometheus + Grafana dashboards)
- **Graceful Shutdown** (no data loss during restarts)
- **Daily Maintenance** (automated restarts, backups)

### **Microservices in Scope**

| Service | Port | Status | Critical Dependencies |
|---------|------|--------|----------------------|
| ai-engine-service | 8001 | ✅ Active | Redis, EventBus, HTTP |
| execution-service | 8002 | ✅ Active | Redis, Binance API, EventBus |
| risk-safety-service | 8003 | ✅ Active | Redis, Postgres, EventBus |
| portfolio-intelligence-service | 8004 | 🔶 Skeleton | Redis, Postgres, HTTP |
| monitoring-health-service | 8080 | ✅ Sprint 2 | Redis, HTTP, EventBus |
| rl-training-service | TBD | 🔶 Planned | Redis, Postgres |
| marketdata-service | TBD | 🔶 Planned | Redis, Binance API |

---

## 2. PROBLEM ANALYSIS

### **P0 Issues (Must Fix)**

| # | Problem | Impact | Priority |
|---|---------|--------|----------|
| 1 | **Redis Single-Node** | Total system outage if Redis fails | 🔴 CRITICAL |
| 2 | **Postgres Single-Node** | Data loss risk, no backup strategy | 🔴 CRITICAL |
| 3 | **No API Gateway** | Direct service exposure, no rate limiting | 🟠 HIGH |
| 4 | **Inconsistent Logging** | Hard to trace cross-service requests | 🟠 HIGH |
| 5 | **No Graceful Shutdown** | Potential data corruption on restart | 🟠 HIGH |
| 6 | **TradeStore SQLite** | No replication, single file failure point | 🟡 MEDIUM |
| 7 | **EventBus Reconnect** | Limited retry strategy despite DiskBuffer | 🟡 MEDIUM |

### **Existing Strengths**

✅ **EventBus DiskBuffer**: Fallback to disk if Redis unavailable  
✅ **Bulletproof API Client**: Retry logic, circuit breaker  
✅ **Emergency Stop System (ESS)**: Hard stop on critical failures  
✅ **Docker Health Checks**: Service monitoring at container level  
✅ **FastAPI Framework**: Modern, async, production-ready  

---

## 3. IMPLEMENTATION MODULES

### **MODULE A: REDIS HIGH AVAILABILITY** 🔴 P0

**Problem**: Single Redis instance = single point of failure for EventBus, PolicyStore, TradeStore

**Solution**: Redis Sentinel (1 master + 2 replicas + 3 sentinels)

**Files Created**:
- ✅ `infra/redis/redis-sentinel-example.yml` - Docker Compose config for 3-node HA setup
- ✅ `infra/redis/sentinel.conf` - Sentinel monitoring configuration (quorum=2)
- ✅ `infra/health/redis_health.py` - Health checker with Sentinel discovery support

**Configuration**:
- Quorum: 2 sentinels required to elect new master
- Failover timeout: 10 seconds
- Down detection: 5 seconds
- Password authentication enabled
- Volume persistence for all nodes

**Integration Steps**:
1. Replace single Redis in docker-compose.yml with Sentinel setup
2. Update all services to use Sentinel connection string:
   ```python
   REDIS_SENTINEL_HOSTS = ["redis-sentinel-1:26379", "redis-sentinel-2:26380", "redis-sentinel-3:26381"]
   REDIS_MASTER_NAME = "mymaster"
   ```
3. Test failover by killing master node

**Acceptance Criteria**:
- [ ] All 3 Redis nodes running and synchronized
- [ ] Sentinel detects master failure within 5 seconds
- [ ] Automatic failover completes within 10 seconds
- [ ] EventBus reconnects to new master automatically
- [ ] No message loss during failover

---

### **MODULE B: POSTGRES HIGH AVAILABILITY** 🔴 P0

**Problem**: Single Postgres instance, no backups, SQLite TradeStore not replicated

**Solution**:
- **Tier 1 (Sprint 3)**: Primary + Automated Backups + PgBouncer
- **Tier 2 (Sprint 4+)**: Primary + Read Replica + Auto-failover

**Files Created**:
- ✅ `infra/postgres/postgres-ha-plan.md` - Complete HA strategy documentation
- ✅ `infra/postgres/backup.sh` - Automated daily backup script (pg_dump → S3/Azure)
- ✅ `infra/postgres/restore.sh` - Restore script with verification
- ✅ `infra/postgres/postgres_helper.py` - Connection pool with retry logic

**Tier 1 Implementation**:
1. **PgBouncer**: Connection pooling (port 6432)
   - Max 1000 client connections
   - Pool size: 25 connections
   - Transaction pooling mode

2. **Automated Backups**:
   - Daily full backup at 02:00 UTC
   - Upload to S3/Azure Blob
   - Retention: 7 days
   - Incremental WAL backups every 6 hours

3. **Connection Retry**:
   - 3 retry attempts with exponential backoff
   - Max retry wait: 30 seconds
   - Health checks before returning connection

**TradeStore Migration** (Sprint 4):
- Migrate SQLite → Postgres table
- Create schema with indexes
- Dual-write for 48h validation period
- Switch reads to Postgres

**Acceptance Criteria**:
- [ ] PgBouncer deployed and routing connections
- [ ] Daily backups running and uploading to cloud
- [ ] Restore tested (RTO < 15 minutes)
- [ ] Connection pool reduces DB load by 80%
- [ ] All services use postgres_helper.py

---

### **MODULE C: NGINX API GATEWAY** 🟠 P1

**Problem**: Direct service exposure, no rate limiting, no centralized routing

**Solution**: NGINX reverse proxy with load balancing, rate limiting, security headers

**Files Created**:
- ✅ `infra/nginx/nginx.conf.example` - Complete NGINX configuration
- ✅ `infra/nginx/docker-compose-nginx.yml` - Gateway deployment config

**Features**:
1. **Reverse Proxy Routing**:
   - `/api/ai/` → ai-engine:8001
   - `/api/execution/` → execution:8002
   - `/api/risk/` → risk-safety:8003
   - `/api/portfolio/` → portfolio-intelligence:8004
   - `/api/health/` → monitoring-health:8080

2. **Rate Limiting**:
   - General API: 100 req/s
   - Execution: 20 req/s
   - Trading operations: 10 req/s
   - Burst capacity configured

3. **Load Balancing**:
   - Algorithm: least_conn
   - Health checks: max_fails=3, fail_timeout=30s
   - Keepalive: 32 connections

4. **Security**:
   - X-Frame-Options, X-Content-Type-Options headers
   - CORS configuration
   - Custom error pages (JSON format)

5. **Logging**:
   - Detailed access logs with request timing
   - Upstream connection time tracking

**Acceptance Criteria**:
- [ ] NGINX gateway accepts traffic on port 80
- [ ] All microservices routed correctly
- [ ] Rate limiting triggers at configured thresholds
- [ ] Health checks detect failed services
- [ ] Access logs include timing metrics

---

### **MODULE D: AUTO-RESTART & SELF-HEALING** 🟠 P1

**Problem**: No graceful shutdown, docker restart policy exists but no hooks

**Solution**: Graceful shutdown endpoints + health-based restart

**Implementation**:
1. **Graceful Shutdown Endpoint** (`/admin/shutdown`):
   ```python
   @app.post("/admin/shutdown")
   async def shutdown():
       # 1. Stop accepting new requests
       # 2. Finish in-flight requests (5s timeout)
       # 3. Save state to Redis
       # 4. Close connections (Redis, HTTP)
       # 5. Signal SIGTERM
   ```

2. **Health-Based Restart**:
   - Docker health checks every 30s
   - Restart if 3 consecutive failures
   - Max restart attempts: 5

3. **State Preservation**:
   - Save to Redis before shutdown
   - Restore from Redis on startup
   - EventBus DiskBuffer for queued messages

**Acceptance Criteria**:
- [ ] All services respond to `/admin/shutdown`
- [ ] No data loss during graceful shutdown
- [ ] Services auto-restart on health check failure
- [ ] State successfully restored after restart

---

### **MODULE E: UNIFIED LOGGING** 🟠 P1

**Problem**: Mixed JSON/text logs, no correlation IDs, hard to trace cross-service requests

**Solution**: Structured JSON logging with correlation IDs and sensitive data masking

**Files Created**:
- ✅ `infra/logging/logging_config.yml` - Unified logging configuration
- ✅ `infra/logging/filters.py` - Custom filters (correlation ID, service name, sensitive data masking)
- ✅ `infra/logging/middleware.py` - FastAPI middleware for request logging

**Features**:
1. **JSON Format**:
   ```json
   {
     "asctime": "2025-12-04T12:34:56.789Z",
     "name": "ai_engine",
     "levelname": "INFO",
     "message": "Processing signal",
     "correlation_id": "abc-123",
     "service": "ai-engine-service",
     "module": "ensemble",
     "funcName": "generate_signal",
     "lineno": 142
   }
   ```

2. **Correlation IDs**:
   - Auto-generated for each request
   - Propagated via `X-Correlation-ID` header
   - Included in all log records

3. **Sensitive Data Masking**:
   - API keys, passwords, tokens → `***REDACTED***`
   - Configurable field list

4. **Log Rotation**:
   - Max file size: 100MB
   - Backup count: 10 files
   - Error-only log: 50MB

**Integration**:
```python
# Load config
import logging.config
import yaml

with open("infra/logging/logging_config.yml") as f:
    logging.config.dictConfig(yaml.safe_load(f))

# Add middleware
from infra.logging.middleware import LoggingMiddleware
app.add_middleware(LoggingMiddleware)
```

**Acceptance Criteria**:
- [ ] All services log in JSON format
- [ ] Correlation IDs present in all logs
- [ ] Sensitive data masked in logs
- [ ] Cross-service requests traceable by correlation ID

---

### **MODULE F: BASIC METRICS + GRAFANA** 🟡 P2

**Problem**: No operational visibility, no performance metrics, hard to diagnose issues

**Solution**: Prometheus metrics + Grafana dashboards

**Files Created**:
- ✅ `infra/metrics/metrics.py` - Prometheus metrics instrumentation
- ✅ `infra/metrics/grafana-guide.md` - Dashboard configuration guide

**Metrics Collected**:
1. **HTTP Metrics**:
   - `http_requests_total` (counter)
   - `http_request_duration_seconds` (histogram)
   - `http_requests_in_progress` (gauge)

2. **EventBus Metrics**:
   - `eventbus_messages_published_total` (counter)
   - `eventbus_messages_consumed_total` (counter)
   - `eventbus_message_processing_duration_seconds` (histogram)
   - `eventbus_queue_lag` (gauge)

3. **Trading Metrics**:
   - `trades_executed_total` (counter)
   - `trade_execution_duration_seconds` (histogram)
   - `positions_open` (gauge)
   - `portfolio_value_usd` (gauge)

4. **Risk Metrics**:
   - `risk_checks_total` (counter)
   - `emergency_stop_triggered_total` (counter)
   - `policy_violations_total` (counter)

5. **Infrastructure Metrics**:
   - `redis_operations_total` (counter)
   - `redis_operation_duration_seconds` (histogram)
   - `database_operations_total` (counter)
   - `service_health_status` (gauge)

**Grafana Dashboards** (Sprint 4):
1. Service Health Overview
2. Trading Performance
3. EventBus Monitoring
4. Infrastructure Metrics

**Integration**:
```python
from infra.metrics.metrics import metrics_endpoint

@app.get("/metrics")
async def metrics():
    return metrics_endpoint()
```

**Acceptance Criteria**:
- [ ] All services expose `/metrics` endpoint
- [ ] Prometheus scraping all services
- [ ] Key metrics visible in Grafana
- [ ] Alert rules configured for critical metrics

---

### **MODULE G: DAILY RESTART PLAN** 🟡 P2

**Problem**: No scheduled maintenance, potential memory leaks, stale connections

**Solution**: Daily 04:00 UTC restart with graceful shutdown sequence

**Files Created**:
- ✅ `infra/restart/daily-restart-plan.md` - Complete restart strategy

**Restart Sequence**:
1. **Pre-restart (03:55 UTC)**:
   - Pause signal generation
   - Flush EventBus queues
   - Save state to Redis

2. **Graceful Shutdown (04:00 UTC)**:
   - ai-engine → portfolio-intelligence → execution → risk-safety → monitoring-health
   - 30s timeout per service

3. **Infrastructure Check (04:03 UTC)**:
   - Redis health check
   - Postgres health check
   - Optional backup trigger

4. **Restart (04:05 UTC)**:
   - risk-safety → execution → ai-engine → portfolio-intelligence → monitoring-health
   - Dependency-first order

5. **Health Verification (04:08 UTC)**:
   - All services respond to `/health`
   - Resume trading

**Implementation Options**:
- Cron + Docker restart (simple)
- Kubernetes CronJob (scalable)
- Python orchestrator (recommended)

**Acceptance Criteria**:
- [ ] Restart completes in < 5 minutes
- [ ] No data loss during restart
- [ ] No active orders during restart window
- [ ] All services pass health checks post-restart

---

## 4. FILE STRUCTURE

```
infra/
├── redis/
│   ├── redis-sentinel-example.yml   ← Docker Compose for 3-node HA
│   ├── sentinel.conf                 ← Sentinel configuration
│   └── README.md                     ← Setup instructions
│
├── postgres/
│   ├── postgres-ha-plan.md           ← HA strategy document
│   ├── backup.sh                     ← Automated backup script
│   ├── restore.sh                    ← Restore script
│   ├── postgres_helper.py            ← Connection pool helper
│   └── migration_sqlite_to_pg.sql    ← TradeStore migration (TODO)
│
├── nginx/
│   ├── nginx.conf.example            ← NGINX configuration
│   ├── docker-compose-nginx.yml      ← Gateway deployment
│   └── README.md                     ← Configuration guide (TODO)
│
├── health/
│   └── redis_health.py               ← Redis health checker with Sentinel
│
├── logging/
│   ├── logging_config.yml            ← Unified logging config
│   ├── filters.py                    ← Custom filters (correlation ID, masking)
│   └── middleware.py                 ← FastAPI logging middleware
│
├── metrics/
│   ├── metrics.py                    ← Prometheus metrics
│   ├── grafana-guide.md              ← Dashboard setup guide
│   ├── prometheus.yml                ← Prometheus config (TODO)
│   └── docker-compose-monitoring.yml ← Prometheus + Grafana (TODO)
│
└── restart/
    └── daily-restart-plan.md         ← Restart strategy
    └── restart_orchestrator.py       ← Python restart script (TODO)
```

**Files Created**: 16 files  
**TODO Files**: 5 files (Sprint 4)

---

## 5. DEPLOYMENT ROADMAP

### **Sprint 3 - Part 1: Skeleton Complete** ✅ DONE

- [x] Problem analysis and prioritization
- [x] Redis HA configuration files
- [x] Postgres HA plan and helpers
- [x] NGINX gateway configuration
- [x] Unified logging configuration
- [x] Metrics instrumentation
- [x] Daily restart plan
- [x] Status report document

### **Sprint 3 - Part 2: Core Implementation** (Next Phase)

**Week 1: Redis + Postgres HA**
- [ ] Deploy Redis Sentinel cluster
- [ ] Test Redis failover scenarios
- [ ] Deploy PgBouncer
- [ ] Setup automated Postgres backups
- [ ] Integrate postgres_helper.py into all services

**Week 2: Gateway + Logging**
- [ ] Deploy NGINX gateway
- [ ] Configure rate limiting
- [ ] Integrate unified logging into all services
- [ ] Test correlation ID propagation

**Week 3: Metrics + Testing**
- [ ] Add `/metrics` to all services
- [ ] Deploy Prometheus + Grafana
- [ ] Create Grafana dashboards
- [ ] End-to-end failover testing

### **Sprint 4: Advanced Features**

- [ ] Migrate TradeStore to Postgres
- [ ] Implement daily restart orchestrator
- [ ] Postgres read replicas
- [ ] AlertManager integration
- [ ] Long-term metrics storage (Thanos)

---

## 6. TESTING STRATEGY

### **Chaos Testing**

1. **Redis Failover Test**:
   ```bash
   # Kill Redis master
   docker stop quantum_redis_master
   
   # Expected: Sentinel promotes replica within 10s
   # Expected: EventBus reconnects automatically
   # Expected: No message loss
   ```

2. **Postgres Connection Loss**:
   ```bash
   # Simulate network partition
   iptables -A INPUT -p tcp --dport 5432 -j DROP
   
   # Expected: Services retry with exponential backoff
   # Expected: Connection restored after network recovery
   ```

3. **Service Crash**:
   ```bash
   # Kill execution service
   docker kill quantum_execution
   
   # Expected: Docker restarts service within 30s
   # Expected: State restored from Redis
   # Expected: In-flight orders recovered
   ```

### **Load Testing**

1. **Rate Limiting**:
   ```bash
   # Send 200 req/s to /api/ai/
   ab -n 10000 -c 200 http://localhost/api/ai/health
   
   # Expected: 100 req/s accepted, rest rate-limited (429)
   ```

2. **EventBus Throughput**:
   ```bash
   # Publish 1000 messages/s
   python test_eventbus_load.py --rate 1000
   
   # Expected: All messages processed
   # Expected: Queue lag < 100
   ```

### **Recovery Testing**

1. **Backup/Restore**:
   ```bash
   # Perform backup
   ./infra/postgres/backup.sh
   
   # Simulate data loss
   docker exec quantum_postgres psql -U postgres -c "DROP DATABASE quantum_trader;"
   
   # Restore
   ./infra/postgres/restore.sh /backups/quantum_trader_20251204_020000.sql.gz
   
   # Expected: All data restored
   # Expected: RTO < 15 minutes
   ```

---

## 7. ACCEPTANCE CRITERIA

### **Sprint 3 Complete Checklist**

#### **Infrastructure**
- [ ] Redis Sentinel running with 3 nodes
- [ ] Redis failover tested (< 10s recovery)
- [ ] PgBouncer deployed and handling connections
- [ ] Daily Postgres backups to S3/Azure
- [ ] Backup restore tested successfully
- [ ] NGINX gateway accepting traffic on port 80

#### **Services**
- [ ] All services use unified logging (JSON + correlation IDs)
- [ ] All services expose `/metrics` endpoint
- [ ] All services have graceful shutdown endpoint
- [ ] All services use postgres_helper.py for DB connections
- [ ] All services reconnect to Redis Sentinel on failover

#### **Monitoring**
- [ ] Prometheus scraping all services
- [ ] Grafana dashboards created (Service Health, Trading, EventBus, Infrastructure)
- [ ] Alert rules configured for critical failures
- [ ] Correlation IDs traceable across services

#### **Testing**
- [ ] Chaos testing passed (Redis failover, service crash, network partition)
- [ ] Load testing passed (rate limiting, EventBus throughput)
- [ ] Recovery testing passed (backup/restore, state recovery)
- [ ] No data loss in any failure scenario

#### **Documentation**
- [ ] README files for each infra component
- [ ] Runbook for common operations (backup, restore, failover)
- [ ] Alert response procedures documented
- [ ] Architecture diagrams updated

---

## 🎯 SUCCESS METRICS

| Metric | Current | Target (Sprint 3) | Target (Sprint 4) |
|--------|---------|-------------------|-------------------|
| System Uptime | ~95% | 99.5% | 99.9% |
| Redis Failover Time | N/A (no HA) | < 10s | < 5s |
| RTO (Recovery Time) | ~30 min | < 15 min | < 5 min |
| RPO (Recovery Point) | Unknown | < 24h | < 1h |
| Request Latency P95 | Unknown | < 500ms | < 200ms |
| Error Rate | Unknown | < 1% | < 0.1% |

---

## 📞 NEXT STEPS

1. **Review this plan** with team/stakeholders
2. **Prioritize** which modules to implement first (recommend: A → B → C → E → D → F → G)
3. **Create detailed TODO list** for Sprint 3 Part 2
4. **Setup staging environment** for testing
5. **Begin implementation** starting with Redis HA

---

**Document Version**: 1.0  
**Last Updated**: December 4, 2025  
**Status**: ✅ PLANNING COMPLETE - Ready for Implementation
