# Daily Restart Strategy
# SPRINT 3 - Module G: Scheduled Maintenance

**Status**: Planning / Strategy Document  
**Priority**: P2 (implement after core infrastructure)

---

## 🎯 OBJECTIVE

Perform daily restarts of all microservices to:
- Clear memory leaks
- Reset connection pools
- Refresh authentication tokens
- Apply pending configuration updates
- Minimize accumulated state drift

---

## ⏰ SCHEDULE

**Daily restart window**: 04:00 UTC (12:00 AM EST)

**Rationale**:
- Lowest trading volume (between US market close and Asia market open)
- Binance maintenance typically 02:00-04:00 UTC (avoid conflict)
- Minimal impact on active positions

---

## 🔄 RESTART SEQUENCE

### **Phase 1: Pre-Restart (03:55 UTC)**

1. **Stop accepting new signals** (ai-engine pause)
2. **Flush EventBus queues** (process all pending messages)
3. **Save all state to Redis** (positions, policies, RL agent state)
4. **Notify monitoring** (send restart.scheduled event)

### **Phase 2: Graceful Shutdown (04:00 UTC)**

Shutdown order (reverse dependency):
1. **ai-engine-service** (stops generating signals)
2. **portfolio-intelligence-service** (stops analytics)
3. **execution-service** (waits for in-flight orders to complete)
4. **risk-safety-service** (saves ESS state)
5. **monitoring-health-service** (last to shut down)

Each service gets 30 seconds for graceful shutdown.

### **Phase 3: Infrastructure Check (04:03 UTC)**

1. **Check Redis health** (ping, memory usage)
2. **Check Postgres health** (connection test, disk space)
3. **Backup key data** (if backup time coincides)

### **Phase 4: Restart (04:05 UTC)**

Startup order (dependency-first):
1. **risk-safety-service** (load ESS state)
2. **execution-service** (reconnect to Binance)
3. **ai-engine-service** (load models)
4. **portfolio-intelligence-service** (load portfolio state)
5. **monitoring-health-service** (resume monitoring)

### **Phase 5: Health Verification (04:08 UTC)**

1. **Health checks** (all services respond to /health)
2. **Dependency checks** (Redis, Postgres, Binance reachable)
3. **Resume trading** (ai-engine resumes signal generation)
4. **Notify monitoring** (send restart.completed event)

---

## 🛠️ IMPLEMENTATION OPTIONS

### **Option 1: Cron + Docker Restart**

```bash
# /etc/crontab or cron job

# Daily restart at 04:00 UTC
0 4 * * * /app/infra/restart/daily_restart.sh >> /var/log/daily_restart.log 2>&1
```

**Script**: `infra/restart/daily_restart.sh`

```bash
#!/bin/bash
# Daily restart script

set -e

echo "$(date): Starting daily restart sequence"

# Phase 1: Pre-restart
echo "Phase 1: Preparing for restart..."
curl -X POST http://localhost:8001/admin/pause  # Pause AI engine
sleep 5

# Phase 2: Graceful shutdown
echo "Phase 2: Graceful shutdown..."
docker-compose stop ai-engine-service execution-service risk-safety-service

# Phase 3: Infrastructure check
echo "Phase 3: Checking infrastructure..."
docker exec quantum_redis redis-cli PING
docker exec quantum_postgres pg_isready

# Phase 4: Restart
echo "Phase 4: Starting services..."
docker-compose up -d risk-safety-service execution-service ai-engine-service

# Phase 5: Health verification
echo "Phase 5: Verifying health..."
sleep 10
curl http://localhost:8003/health  # risk-safety
curl http://localhost:8002/health  # execution
curl http://localhost:8001/health  # ai-engine

echo "$(date): Daily restart complete"
```

### **Option 2: Kubernetes CronJob**

```yaml
# infra/restart/daily-restart-cronjob.yaml

apiVersion: batch/v1
kind: CronJob
metadata:
  name: daily-restart
spec:
  schedule: "0 4 * * *"  # 04:00 UTC daily
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: restart-controller
            image: bitnami/kubectl:latest
            command:
            - /bin/sh
            - -c
            - |
              # Graceful restart
              kubectl rollout restart deployment/ai-engine-service
              kubectl rollout restart deployment/execution-service
              kubectl rollout restart deployment/risk-safety-service
              
              # Wait for rollout
              kubectl rollout status deployment/ai-engine-service
              kubectl rollout status deployment/execution-service
              kubectl rollout status deployment/risk-safety-service
          restartPolicy: OnFailure
```

### **Option 3: Python Orchestrator** (Recommended)

```python
# infra/restart/restart_orchestrator.py

import asyncio
import httpx
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


async def daily_restart():
    """Orchestrate daily microservice restart."""
    
    logger.info("=== DAILY RESTART SEQUENCE ===")
    logger.info(f"Start time: {datetime.utcnow()}")
    
    # Phase 1: Pre-restart
    logger.info("Phase 1: Pre-restart preparation")
    await pause_trading()
    await flush_eventbus_queues()
    await save_state_to_redis()
    
    # Phase 2: Graceful shutdown
    logger.info("Phase 2: Graceful shutdown")
    await shutdown_service("ai-engine-service", 8001)
    await shutdown_service("portfolio-intelligence-service", 8004)
    await shutdown_service("execution-service", 8002)
    await shutdown_service("risk-safety-service", 8003)
    
    # Phase 3: Infrastructure check
    logger.info("Phase 3: Infrastructure check")
    await check_redis_health()
    await check_postgres_health()
    
    # Phase 4: Restart
    logger.info("Phase 4: Restart services")
    await start_service("risk-safety-service")
    await start_service("execution-service")
    await start_service("ai-engine-service")
    await start_service("portfolio-intelligence-service")
    
    # Phase 5: Verification
    logger.info("Phase 5: Health verification")
    await verify_all_services()
    await resume_trading()
    
    logger.info(f"Restart complete: {datetime.utcnow()}")


async def shutdown_service(name: str, port: int):
    """Gracefully shutdown a service."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://localhost:{port}/admin/shutdown",
                timeout=30.0
            )
            logger.info(f"{name} shutdown: {response.status_code}")
    except Exception as e:
        logger.error(f"Failed to shutdown {name}: {e}")
```

---

## 🔒 GRACEFUL SHUTDOWN HOOKS

Each service needs `/admin/shutdown` endpoint:

```python
# backend/services/ai_engine/app.py

from fastapi import FastAPI
import signal
import sys

app = FastAPI()

@app.post("/admin/shutdown")
async def graceful_shutdown():
    """Gracefully shut down service."""
    
    # 1. Stop accepting new requests
    logger.info("Shutdown requested - stopping new requests")
    
    # 2. Finish processing in-flight requests
    logger.info("Waiting for in-flight requests to complete")
    await asyncio.sleep(5)  # Give requests time to finish
    
    # 3. Save state
    logger.info("Saving state to Redis")
    await save_state()
    
    # 4. Close connections
    logger.info("Closing connections")
    await redis_client.close()
    await http_client.aclose()
    
    # 5. Exit
    logger.info("Shutdown complete")
    os.kill(os.getpid(), signal.SIGTERM)
    
    return {"status": "shutting down"}
```

---

## ⚠️ RISK MITIGATION

### **1. Prevent Restart During Active Trades**

```python
async def can_restart() -> bool:
    """Check if restart is safe."""
    
    # Check for active orders
    active_orders = await get_active_orders()
    if active_orders:
        logger.warning(f"Cannot restart: {len(active_orders)} active orders")
        return False
    
    # Check for pending fills
    pending_fills = await get_pending_fills()
    if pending_fills:
        logger.warning(f"Cannot restart: {len(pending_fills)} pending fills")
        return False
    
    # Check EventBus queue lag
    queue_lag = await get_queue_lag()
    if queue_lag > 100:
        logger.warning(f"Cannot restart: Queue lag = {queue_lag}")
        return False
    
    return True
```

### **2. Rollback on Failure**

```python
async def restart_with_rollback():
    """Restart with automatic rollback on failure."""
    
    # Save current state
    snapshot = await create_snapshot()
    
    try:
        await daily_restart()
        await verify_health()
    except Exception as e:
        logger.error(f"Restart failed: {e}")
        logger.info("Rolling back to previous state")
        await restore_snapshot(snapshot)
        raise
```

---

## 📊 MONITORING

**Metrics to track**:
- Restart duration (should be < 5 minutes)
- Services that failed to restart
- Orders lost during restart (should be 0)
- Health check failures post-restart

**Alerts**:
- Restart takes > 10 minutes
- Any service fails health check after restart
- Active orders detected during restart window

---

## 📋 TODO (Sprint 4)

- [ ] Implement graceful shutdown endpoints in all services
- [ ] Create restart orchestrator script
- [ ] Setup cron job or Kubernetes CronJob
- [ ] Add restart monitoring to Grafana
- [ ] Test restart procedure in staging
- [ ] Document emergency manual restart procedure

---

**Next Steps**: Complete core infrastructure (Redis HA, Postgres HA, NGINX), then implement daily restart in Sprint 4
