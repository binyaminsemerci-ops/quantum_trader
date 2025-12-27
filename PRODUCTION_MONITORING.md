# ============================================================================
# Quantum Trader - Production Monitoring Dashboard
# ============================================================================
# Quick reference for monitoring production system
# ============================================================================

## HEALTH CHECKS

### Backend Health
curl http://localhost:8000/health

### AI OS Status  
curl http://localhost:8000/api/aios_status

### Container Status
docker compose ps

### Metrics Endpoint
curl http://localhost:9090/metrics

## CRITICAL FEATURE MONITORING

### 1. Flash Crash Detection
# Monitor for flash crash events (should detect <10s)
docker compose logs backend | grep -i "flash_crash_detected"

# Check drawdown monitor status
docker compose logs backend | grep -i "check_flash_crash"

# Expected: Lines showing equity monitoring every 10s
# Alert if: No activity for >60s or detection lag >15s

### 2. Dynamic Stop Loss Widening
# Monitor SL adjustments by regime
docker compose logs backend | grep -i "dynamic.*sl.*regime"

# Check for regime transitions
docker compose logs backend | grep -i "regime.*HIGH_VOL\|EXTREME_VOL"

# Expected: SL multipliers (1.5x HIGH_VOL, 2.5x EXTREME_VOL)
# Alert if: Static 1.0x multiplier during volatility spikes

### 3. Hybrid Order Strategy
# Monitor LIMIT vs MARKET execution
docker compose logs backend | grep -i "order.*STOP.*LIMIT\|STOP_MARKET"

# Check slippage metrics
docker compose logs backend | grep -i "slippage"

# Expected: Majority LIMIT fills, <1% average slippage
# Alert if: >5% MARKET orders or >3% slippage

### 4. Atomic Model Promotion
# Monitor promotion lock acquisition
docker compose logs backend | grep -i "acquire_promotion_lock"

# Check for ACK confirmations
docker compose logs backend | grep -i "ack_promotion"

# Verify lock release
docker compose logs backend | grep -i "release_promotion_lock"

# Expected: acquire → ACKs (3) → release sequence
# Alert if: Missing ACKs, timeout errors, or lock held >60s

### 5. Federation v2 Event Bridge
# Monitor event bridging
docker compose logs backend | grep -i "federation_v2_event_bridge"

# Check v2 node synchronization
docker compose logs backend | grep -i "broadcast_to_v2_nodes"

# Expected: Events bridged on model lifecycle changes
# Alert if: No bridge activity during promotions

### 6. Event Priority Sequencing
# Monitor event processing order
docker compose logs backend | grep -i "subscribe_with_priority"

# Check for race conditions
docker compose logs backend | grep -i "NoneType.*model\|KeyError.*ensemble"

# Expected: Ensemble → SESA/Meta → Federation ordering
# Alert if: NoneType errors or out-of-order processing

## PERFORMANCE METRICS

### Active Positions
curl -s http://localhost:8000/api/positions | jq '.data | length'

### Model Performance
curl -s http://localhost:8000/api/ai/model/performance | jq

### Recent Trades
curl -s http://localhost:8000/api/closed_positions?limit=10 | jq

### Redis Stats
docker exec quantum_redis redis-cli INFO stats

## ERROR MONITORING

### Backend Errors (Last 100 lines)
docker compose logs --tail=100 backend | grep -i "error\|exception\|traceback"

### Critical Issues Only
docker compose logs backend | grep -i "CRITICAL"

### Redis Connection Issues
docker compose logs backend | grep -i "redis.*connection\|redis.*timeout"

## ALERTING THRESHOLDS

### Flash Crash Detection
# ⚠️  ALERT: No flash_crash_detected events AND equity drop >3% in 10s
# 🚨 CRITICAL: Detection lag >15s during crash

### Stop Loss Slippage  
# ⚠️  ALERT: Average slippage >2% over 1 hour
# 🚨 CRITICAL: Single trade slippage >5%

### Model Promotion
# ⚠️  ALERT: Missing ACK from handler (timeout)
# 🚨 CRITICAL: Promotion lock held >120s

### Federation Sync
# ⚠️  ALERT: No v2 bridge activity during promotion
# 🚨 CRITICAL: v2 nodes using stale models >5min

### Event Ordering
# ⚠️  ALERT: NoneType errors in Meta-Strategy RL
# 🚨 CRITICAL: Repeated race conditions (>3 in 1 hour)

## QUICK DIAGNOSTICS

### Restart Backend (Zero Downtime)
docker compose restart backend

### Check Redis Persistence
docker exec quantum_redis redis-cli LASTSAVE

### View Real-Time Logs
docker compose logs -f backend

### Tail Specific Module
docker compose logs -f backend | grep -i "position_monitor\|execution\|continuous_learning"

### Export Metrics for Analysis
curl -s http://localhost:9090/metrics > metrics_$(date +%Y%m%d_%H%M%S).txt

## ROLLBACK PROCEDURE

### Step 1: Stop production
docker compose down

### Step 2: Restore backup
cp backups/pre-deploy-YYYYMMDD-HHMMSS/.env.backup .env

### Step 3: Restart previous version
docker compose up -d

### Step 4: Verify health
curl http://localhost:8000/health

## MAINTENANCE COMMANDS

### View Container Resources
docker stats quantum_backend quantum_redis

### Check Disk Usage
docker system df

### Prune Old Data (Careful!)
docker system prune -a --volumes

### Export Database Backup
docker exec quantum_backend tar czf /tmp/db_backup.tar.gz /app/database
docker cp quantum_backend:/tmp/db_backup.tar.gz ./backups/

## LOG ROTATION

### Backend Logs
docker compose logs --since 24h backend > logs/backend_$(date +%Y%m%d).log

### Compress Old Logs
gzip logs/backend_*.log

### Clean Logs Older Than 7 Days
find logs/ -name "*.log.gz" -mtime +7 -delete

## GRAFANA DASHBOARD (Optional)

### Access Grafana
# URL: http://localhost:3001 (if configured)
# Default: admin / change_me_in_production

### Key Dashboards
# - Flash Crash Detection Rate
# - Stop Loss Slippage Distribution
# - Model Promotion Success Rate
# - Event Processing Latency
# - Position Performance Heatmap

## CONTACT & ESCALATION

### P0 Issues (Immediate)
# - Trading halted unexpectedly
# - Flash crash undetected >30s
# - Promotion lock deadlock
# - Multiple NoneType errors

### P1 Issues (Within 4 hours)  
# - High slippage (>3% average)
# - Missing event ACKs
# - Federation v2 desync
# - Degraded performance

### P2 Issues (Next business day)
# - Single missed ACK
# - Minor log noise
# - Non-critical warnings
