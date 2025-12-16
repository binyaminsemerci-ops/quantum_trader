# 📚 Operations Guide - Quantum Trader v3.0 Microservices

## Complete Operations Manual for Production

**Version:** 3.0.0  
**Date:** December 2, 2025  
**Audience:** DevOps Engineers, SREs, System Administrators

---

## 📋 Table of Contents

1. [Daily Operations](#daily-operations)
2. [Service Management](#service-management)
3. [Monitoring & Alerting](#monitoring--alerting)
4. [Backup & Recovery](#backup--recovery)
5. [Performance Tuning](#performance-tuning)
6. [Scaling Operations](#scaling-operations)
7. [Incident Response](#incident-response)
8. [Maintenance Procedures](#maintenance-procedures)

---

## 📅 Daily Operations

### Morning Checks (Start of Day)

**Duration:** 10-15 minutes

```powershell
# 1. Check system health
.\scripts\daily_health_check.ps1

# Or manually:
# Check all services
docker-compose ps

# Verify all healthy
curl http://localhost:8001/health | ConvertFrom-Json
curl http://localhost:8002/health | ConvertFrom-Json
curl http://localhost:8003/health | ConvertFrom-Json

# 2. Review overnight activity
docker-compose logs --since 24h | Select-String -Pattern "ERROR|CRITICAL|WARN"

# 3. Check account status
curl http://localhost:8002/health | ConvertFrom-Json | Select-Object open_positions, daily_pnl

# 4. Verify Binance connection
docker-compose logs exec-risk-service --since 1h | Select-String -Pattern "Binance"

# 5. Check disk space
docker system df

# 6. Review Grafana dashboards
Start-Process http://localhost:3000
```

### Evening Checks (End of Day)

```powershell
# 1. Daily PnL summary
python analyze_all_closed_positions.py --date $(Get-Date -Format 'yyyy-MM-dd')

# 2. Model performance review
python analyze_model_performance.py --window 1d

# 3. System resource usage
docker stats --no-stream

# 4. Backup today's data
.\scripts\backup_daily.ps1

# 5. Review any alerts
curl http://localhost:8003/health | ConvertFrom-Json | Select-Object alerts
```

### Weekly Checks (Every Monday)

```powershell
# 1. Full system health report
python scripts/generate_health_report.py --period week

# 2. Performance benchmarks
python tests/performance_benchmark.py

# 3. Database maintenance
docker exec quantum_trader_postgres vacuumdb -U quantum_user -d quantum_trader

# 4. Log rotation
docker-compose logs --since 7d > logs/archive/week_$(Get-Date -Format 'yyyyMMdd').log

# 5. Model retraining check
curl http://localhost:8003/health | ConvertFrom-Json | Select-Object learning_state

# 6. Update dependencies (if needed)
pip list --outdated
```

---

## 🔧 Service Management

### Starting Services

**Standard Startup:**
```powershell
# Start all services
docker-compose up -d

# Wait for services to be ready (90 seconds)
Start-Sleep -Seconds 90

# Verify startup
.\scripts\verify_startup.ps1
```

**Startup with Logs:**
```powershell
# Start with logs visible
docker-compose up

# Ctrl+C to stop, or in another terminal:
docker-compose logs -f
```

**Selective Startup:**
```powershell
# Start only specific services
docker-compose up -d redis postgres
Start-Sleep -Seconds 10

docker-compose up -d ai-service
Start-Sleep -Seconds 60

docker-compose up -d exec-risk-service analytics-os-service
```

### Stopping Services

**Graceful Shutdown:**
```powershell
# Stop all services gracefully
docker-compose stop

# Verify all stopped
docker-compose ps

# Stop specific service
docker-compose stop ai-service
```

**Force Stop:**
```powershell
# Force stop (if graceful fails)
docker-compose kill

# Clean up
docker-compose down
```

**Stop for Maintenance:**
```powershell
# 1. Enable drain mode (stop accepting new requests)
docker-compose exec ai-service touch /tmp/drain_mode
docker-compose exec exec-risk-service touch /tmp/drain_mode

# 2. Wait for in-flight requests (2 minutes)
Start-Sleep -Seconds 120

# 3. Stop services
docker-compose stop
```

### Restarting Services

**Standard Restart:**
```powershell
# Restart all services
docker-compose restart

# Restart specific service
docker-compose restart ai-service

# Restart with rebuild
docker-compose up -d --build ai-service
```

**Rolling Restart (Zero Downtime):**
```powershell
# Restart services one by one
docker-compose restart analytics-os-service
Start-Sleep -Seconds 30

docker-compose restart ai-service
Start-Sleep -Seconds 60

docker-compose restart exec-risk-service
Start-Sleep -Seconds 30
```

### Viewing Logs

**Real-time Logs:**
```powershell
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f ai-service

# Multiple services
docker-compose logs -f ai-service exec-risk-service

# With timestamps
docker-compose logs -f --timestamps
```

**Historical Logs:**
```powershell
# Last 100 lines
docker-compose logs --tail=100 ai-service

# Last hour
docker-compose logs --since 1h

# Specific time range
docker-compose logs --since 2025-12-02T10:00:00 --until 2025-12-02T11:00:00
```

**Filtered Logs:**
```powershell
# Errors only
docker-compose logs | Select-String -Pattern "ERROR|CRITICAL"

# Specific pattern
docker-compose logs ai-service | Select-String -Pattern "signal.*generated"

# Export to file
docker-compose logs --since 24h > logs/today_$(Get-Date -Format 'yyyyMMdd').log
```

---

## 📊 Monitoring & Alerting

### Health Endpoints

**Service Health:**
```powershell
# AI Service
$aiHealth = Invoke-RestMethod http://localhost:8001/health
Write-Host "AI Service: $($aiHealth.status)"
Write-Host "  Models Loaded: $($aiHealth.models_loaded)"
Write-Host "  Uptime: $($aiHealth.uptime_seconds)s"

# Exec-Risk Service
$execHealth = Invoke-RestMethod http://localhost:8002/health
Write-Host "Exec-Risk Service: $($execHealth.status)"
Write-Host "  Binance Connected: $($execHealth.binance_connected)"
Write-Host "  Open Positions: $($execHealth.open_positions)"
Write-Host "  Daily PnL: $$($execHealth.daily_pnl)"

# Analytics-OS Service
$analyticsHealth = Invoke-RestMethod http://localhost:8003/health
Write-Host "Analytics-OS Service: $($analyticsHealth.status)"
Write-Host "  HFOS Enabled: $($analyticsHealth.ai_hfos_enabled)"
Write-Host "  Portfolio Value: $$($analyticsHealth.portfolio_state.total_value_usd)"
```

**Readiness Probes:**
```powershell
# Check if services are ready to accept traffic
$services = @(8001, 8002, 8003)
foreach ($port in $services) {
    try {
        $response = Invoke-RestMethod "http://localhost:$port/ready"
        if ($response.status -eq "ready") {
            Write-Host "✓ Service on port $port is ready" -ForegroundColor Green
        } else {
            Write-Host "✗ Service on port $port is NOT ready" -ForegroundColor Red
        }
    } catch {
        Write-Host "✗ Service on port $port is unreachable" -ForegroundColor Red
    }
}
```

### Prometheus Metrics

**Access Prometheus:**
```powershell
Start-Process http://localhost:9090
```

**Key Queries:**

```promql
# Service uptime
ai_service_uptime_seconds
exec_risk_service_uptime_seconds
analytics_os_service_uptime_seconds

# Trading metrics
rate(exec_risk_service_orders_executed_total[5m])
rate(exec_risk_service_positions_closed_total[5m])
exec_risk_service_daily_pnl_usd

# Performance metrics
histogram_quantile(0.95, rate(ai_service_signal_latency_seconds_bucket[5m]))
rate(exec_risk_service_execution_errors_total[5m])

# System health
up{job="ai-service"}
up{job="exec-risk-service"}
up{job="analytics-os-service"}

# Error rates
rate(ai_service_errors_total[5m])
rate(exec_risk_service_execution_errors_total[5m])
```

### Grafana Dashboards

**Access Grafana:**
```powershell
Start-Process http://localhost:3000
# Login: admin / quantum_admin_2025
```

**Essential Dashboards:**

1. **System Overview Dashboard**
   - Service health status
   - Uptime metrics
   - Request rates
   - Error rates

2. **Trading Performance Dashboard**
   - Daily PnL
   - Win rate
   - Average profit per trade
   - Position count

3. **AI Performance Dashboard**
   - Model predictions
   - Signal quality
   - Confidence distribution
   - Ensemble weights

4. **Risk Metrics Dashboard**
   - Current exposure
   - Max drawdown
   - Risk alerts
   - Leverage usage

5. **System Resources Dashboard**
   - CPU usage
   - Memory usage
   - Disk I/O
   - Network traffic

### Alert Rules

**Critical Alerts (Immediate Action):**

```yaml
# config/prometheus_alerts.yml

groups:
  - name: critical
    rules:
      # Service down
      - alert: ServiceDown
        expr: up{job=~"ai-service|exec-risk-service|analytics-os-service"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service {{ $labels.job }} is down"
          
      # High error rate
      - alert: HighErrorRate
        expr: rate(errors_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate on {{ $labels.service }}"
          
      # Large loss
      - alert: LargeLoss
        expr: exec_risk_service_daily_pnl_usd < -1000
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Daily loss exceeds $1000"
```

**Warning Alerts (Monitor):**

```yaml
  - name: warnings
    rules:
      # Degraded performance
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(signal_latency_seconds_bucket[5m])) > 2
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Signal generation latency > 2s"
          
      # High memory usage
      - alert: HighMemoryUsage
        expr: container_memory_usage_bytes / container_spec_memory_limit_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "{{ $labels.container_name }} memory usage > 90%"
```

### Alert Notification Channels

**Email Alerts:**
```yaml
# config/alertmanager.yml
receivers:
  - name: 'email'
    email_configs:
      - to: 'ops-team@your-company.com'
        from: 'quantum-trader@your-company.com'
        smarthost: 'smtp.gmail.com:587'
        auth_username: 'your-email@gmail.com'
        auth_password: 'your-app-password'
```

**Slack Alerts:**
```yaml
  - name: 'slack'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
        channel: '#quantum-trader-alerts'
        title: 'Quantum Trader Alert'
```

---

## 💾 Backup & Recovery

### Automated Backups

**Daily Backup Script:**
```powershell
# scripts/backup_daily.ps1

$BackupDir = "backups/daily"
$Date = Get-Date -Format 'yyyyMMdd_HHmmss'

# Create backup directory
New-Item -Path "$BackupDir/$Date" -ItemType Directory -Force

# 1. Backup Redis
docker exec quantum_trader_redis redis-cli SAVE
docker cp quantum_trader_redis:/data/dump.rdb "$BackupDir/$Date/redis_dump.rdb"
Write-Host "✓ Redis backed up"

# 2. Backup PostgreSQL
docker exec quantum_trader_postgres pg_dump -U quantum_user quantum_trader > "$BackupDir/$Date/postgres_dump.sql"
Write-Host "✓ PostgreSQL backed up"

# 3. Backup configuration
Copy-Item .env "$BackupDir/$Date/.env.backup"
Copy-Item docker-compose.yml "$BackupDir/$Date/docker-compose.backup.yml"
Write-Host "✓ Configuration backed up"

# 4. Backup model checkpoints
Copy-Item ai_engine/models/ "$BackupDir/$Date/models/" -Recurse
Write-Host "✓ Models backed up"

# 5. Compress backup
Compress-Archive -Path "$BackupDir/$Date" -DestinationPath "$BackupDir/quantum_trader_$Date.zip"
Remove-Item "$BackupDir/$Date" -Recurse
Write-Host "✓ Backup compressed"

# 6. Cleanup old backups (keep last 7 days)
Get-ChildItem $BackupDir -Filter "*.zip" | 
    Where-Object {$_.LastWriteTime -lt (Get-Date).AddDays(-7)} | 
    Remove-Item
Write-Host "✓ Old backups cleaned up"

Write-Host "Backup complete: quantum_trader_$Date.zip"
```

**Schedule Daily Backups:**
```powershell
# Create scheduled task (Windows)
$Action = New-ScheduledTaskAction -Execute "pwsh.exe" -Argument "-File C:\quantum_trader\scripts\backup_daily.ps1"
$Trigger = New-ScheduledTaskTrigger -Daily -At 3am
Register-ScheduledTask -TaskName "QuantumTraderBackup" -Action $Action -Trigger $Trigger
```

### Recovery Procedures

**Restore from Backup:**
```powershell
# scripts/restore_backup.ps1

param(
    [Parameter(Mandatory=$true)]
    [string]$BackupFile
)

# 1. Extract backup
$TempDir = "backups/temp_restore"
Expand-Archive -Path $BackupFile -DestinationPath $TempDir -Force

# 2. Stop services
docker-compose stop

# 3. Restore Redis
docker cp "$TempDir/redis_dump.rdb" quantum_trader_redis:/data/dump.rdb
docker-compose restart redis
Write-Host "✓ Redis restored"

# 4. Restore PostgreSQL
docker exec -i quantum_trader_postgres psql -U quantum_user quantum_trader < "$TempDir/postgres_dump.sql"
Write-Host "✓ PostgreSQL restored"

# 5. Restore configuration
Copy-Item "$TempDir/.env.backup" .env
Write-Host "✓ Configuration restored"

# 6. Restore models
Copy-Item "$TempDir/models/*" ai_engine/models/ -Recurse -Force
Write-Host "✓ Models restored"

# 7. Start services
docker-compose up -d

# 8. Cleanup
Remove-Item $TempDir -Recurse

Write-Host "Restore complete from: $BackupFile"
```

**Disaster Recovery:**
```powershell
# Complete system recovery from backup

# 1. Fresh installation
git clone https://github.com/your-org/quantum_trader.git
cd quantum_trader

# 2. Restore backup
.\scripts\restore_backup.ps1 -BackupFile "backups/quantum_trader_YYYYMMDD_HHMMSS.zip"

# 3. Verify system
.\scripts\verify_startup.ps1

# 4. Test trading (testnet first)
# Update .env: BINANCE_TESTNET=true
docker-compose restart

# 5. Monitor for 1 hour before enabling live trading
```

---

## ⚡ Performance Tuning

### Redis Optimization

**Configuration:**
```conf
# config/redis.conf

# Memory management
maxmemory 4gb
maxmemory-policy allkeys-lru

# Persistence
save 900 1
save 300 10
save 60 10000
appendonly yes
appendfsync everysec

# Streams optimization
stream-node-max-entries 100
stream-node-max-bytes 4096

# Network
tcp-keepalive 300
timeout 0

# Performance
maxclients 10000
```

**Apply Configuration:**
```powershell
# Restart Redis with new config
docker-compose restart redis

# Verify configuration
docker exec quantum_trader_redis redis-cli CONFIG GET maxmemory
```

### PostgreSQL Optimization

**Configuration:**
```conf
# config/postgresql.conf

# Memory
shared_buffers = 2GB
effective_cache_size = 6GB
maintenance_work_mem = 512MB
work_mem = 32MB

# Checkpoint
checkpoint_completion_target = 0.9
wal_buffers = 16MB

# Query planner
random_page_cost = 1.1
effective_io_concurrency = 200

# Autovacuum
autovacuum = on
autovacuum_max_workers = 4
```

**Apply Configuration:**
```powershell
# Copy config
docker cp config/postgresql.conf quantum_trader_postgres:/var/lib/postgresql/data/postgresql.conf

# Restart PostgreSQL
docker-compose restart postgres
```

### Service Resource Limits

**Optimize Docker Resources:**
```yaml
# docker-compose.prod.yml

services:
  ai-service:
    deploy:
      resources:
        limits:
          cpus: '8.0'      # Increase for better performance
          memory: 16G
        reservations:
          cpus: '4.0'      # Guaranteed minimum
          memory: 8G
  
  exec-risk-service:
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 4G
  
  analytics-os-service:
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 4G
```

### Application-Level Tuning

**Environment Variables:**
```env
# .env

# AI Service performance
AI_BATCH_SIZE=32              # Larger batches = better throughput
AI_NUM_WORKERS=4              # Parallel processing
AI_CACHE_SIZE=1000            # Cache recent predictions

# Exec-Risk Service performance
EXEC_MAX_CONCURRENT_ORDERS=10  # Parallel execution
EXEC_TIMEOUT_SECONDS=5         # Order timeout
EXEC_RETRY_ATTEMPTS=3          # Retry failed orders

# Analytics-OS Service performance
ANALYTICS_BATCH_SIZE=100       # Batch event processing
ANALYTICS_FLUSH_INTERVAL=10    # Seconds between flushes
HEALTH_CHECK_INTERVAL=5        # Health check frequency

# EventBus tuning
EVENT_BUS_BATCH_SIZE=100       # Read events in batches
EVENT_BUS_BLOCK_MS=1000        # Block time for new events
```

---

## 📈 Scaling Operations

### Horizontal Scaling

**Scale AI Service:**
```powershell
# Scale to 3 instances
docker-compose up -d --scale ai-service=3

# Verify scaling
docker-compose ps ai-service

# Load balancing (requires nginx or similar)
# Requests will be distributed across instances
```

**Load Balancer Configuration:**
```nginx
# nginx.conf

upstream ai_service {
    least_conn;  # Load balancing method
    server ai-service-1:8001;
    server ai-service-2:8001;
    server ai-service-3:8001;
}

server {
    listen 80;
    
    location /ai/ {
        proxy_pass http://ai_service/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Vertical Scaling

**Increase Resources:**
```yaml
# docker-compose.override.yml

services:
  ai-service:
    deploy:
      resources:
        limits:
          cpus: '16.0'      # Double CPU
          memory: 32G        # Double memory
```

```powershell
# Apply changes
docker-compose up -d --force-recreate ai-service
```

### Auto-Scaling (Kubernetes)

**Horizontal Pod Autoscaler:**
```yaml
# k8s/ai-service-hpa.yml

apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ai-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ai-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Pods
      pods:
        metric:
          name: signal_generation_latency
        target:
          type: AverageValue
          averageValue: 500ms
```

---

## 🚨 Incident Response

### Incident Severity Levels

**SEV-1 (Critical):**
- System completely down
- Data loss occurring
- Financial loss > $1000
- Security breach

**Response Time:** < 15 minutes  
**Resolution Time:** < 2 hours

**SEV-2 (High):**
- One service down
- Degraded performance (latency > 5s)
- Error rate > 10%

**Response Time:** < 30 minutes  
**Resolution Time:** < 4 hours

**SEV-3 (Medium):**
- Non-critical feature broken
- Performance degradation (latency > 2s)
- Error rate 5-10%

**Response Time:** < 2 hours  
**Resolution Time:** < 1 day

**SEV-4 (Low):**
- Minor issues
- Cosmetic problems
- Documentation updates

**Response Time:** < 1 day  
**Resolution Time:** < 1 week

### Incident Response Playbooks

**Playbook 1: Service Down**

```powershell
# 1. Verify service is actually down
curl http://localhost:8001/health  # AI Service
curl http://localhost:8002/health  # Exec-Risk
curl http://localhost:8003/health  # Analytics-OS

# 2. Check container status
docker-compose ps

# 3. Check logs for errors
docker-compose logs <service-name> --tail=100 | Select-String -Pattern "ERROR|CRITICAL|FATAL"

# 4. Check resource usage
docker stats --no-stream <service-name>

# 5. Restart service
docker-compose restart <service-name>

# 6. Verify recovery
Start-Sleep -Seconds 30
curl http://localhost:<port>/health

# 7. Document incident
# - What failed
# - Error messages
# - Resolution steps
# - Root cause
```

**Playbook 2: High Latency**

```powershell
# 1. Measure current latency
# Via Prometheus
Start-Process http://localhost:9090/graph?g0.expr=histogram_quantile(0.95%2C%20rate(ai_service_signal_latency_seconds_bucket%5B5m%5D))

# 2. Check system resources
docker stats --no-stream

# 3. Check for slow queries (if using PostgreSQL)
docker exec quantum_trader_postgres psql -U quantum_user -d quantum_trader -c "SELECT * FROM pg_stat_activity WHERE state = 'active' AND query_start < NOW() - INTERVAL '10 seconds'"

# 4. Check Redis performance
docker exec quantum_trader_redis redis-cli --latency

# 5. Identify bottleneck
# - CPU saturated? → Scale up/out
# - Memory full? → Increase limits
# - Network slow? → Check connectivity
# - Disk I/O high? → Optimize queries

# 6. Apply fix based on bottleneck

# 7. Verify improvement
```

**Playbook 3: High Error Rate**

```powershell
# 1. Identify error source
docker-compose logs --since 10m | Select-String -Pattern "ERROR" | Group-Object | Sort-Object Count -Descending

# 2. Check specific error types
docker-compose logs <service-name> | Select-String -Pattern "ERROR.*<error-pattern>"

# 3. Check external dependencies
# - Binance API status
curl https://api.binance.com/api/v3/ping

# - Redis connectivity
docker exec quantum_trader_redis redis-cli ping

# - PostgreSQL connectivity
docker exec quantum_trader_postgres pg_isready -U quantum_user

# 4. Check rate limits
docker-compose logs exec-risk-service | Select-String -Pattern "429|rate.limit"

# 5. Apply temporary fix
# - Enable circuit breaker
# - Reduce request rate
# - Disable non-critical features

# 6. Permanent fix
# - Fix code bug
# - Increase rate limits
# - Optimize requests
```

**Playbook 4: Data Inconsistency**

```powershell
# 1. Stop trading immediately
# Set TRADING_ENABLED=false in .env
docker-compose restart exec-risk-service

# 2. Verify inconsistency
# Compare:
# - Binance actual positions
python -c "from binance.client import Client; c = Client(api_key, secret); print(c.futures_position_information())"

# - System tracked positions
curl http://localhost:8002/health | ConvertFrom-Json | Select-Object open_positions

# 3. Identify root cause
# - Event loss?
docker exec quantum_trader_redis redis-cli XINFO STREAM quantum:events:position.opened

# - Database corruption?
docker exec quantum_trader_postgres psql -U quantum_user -d quantum_trader -c "SELECT COUNT(*) FROM positions WHERE status = 'open'"

# 4. Reconcile data
python scripts/reconcile_positions.py

# 5. Verify fix
# Compare again

# 6. Resume trading
# Set TRADING_ENABLED=true
docker-compose restart exec-risk-service
```

---

## 🛠️ Maintenance Procedures

### Planned Maintenance

**Pre-Maintenance Checklist:**
- [ ] Schedule announced (24-48 hours notice)
- [ ] Backup completed
- [ ] Rollback plan prepared
- [ ] Team on standby
- [ ] Monitoring active

**Maintenance Window:**
```powershell
# 1. Notify users (if applicable)
# - Email
# - Slack/Discord
# - Status page

# 2. Enable maintenance mode
docker-compose exec ai-service touch /tmp/maintenance_mode
docker-compose exec exec-risk-service touch /tmp/maintenance_mode

# 3. Wait for in-flight requests (5 minutes)
Start-Sleep -Seconds 300

# 4. Stop services
docker-compose stop ai-service exec-risk-service analytics-os-service

# 5. Perform maintenance
# - Update code
# - Upgrade dependencies
# - Database migration
# - Configuration changes

# 6. Test changes
docker-compose up -d
.\scripts\verify_startup.ps1
python tests/integration_test_harness.py

# 7. Resume normal operations
docker-compose exec ai-service rm /tmp/maintenance_mode
docker-compose exec exec-risk-service rm /tmp/maintenance_mode

# 8. Notify users of completion
```

### Database Maintenance

**Weekly Maintenance:**
```powershell
# 1. Analyze tables
docker exec quantum_trader_postgres psql -U quantum_user -d quantum_trader -c "ANALYZE"

# 2. Vacuum database
docker exec quantum_trader_postgres vacuumdb -U quantum_user -d quantum_trader --analyze

# 3. Reindex if needed
docker exec quantum_trader_postgres reindexdb -U quantum_user -d quantum_trader

# 4. Check database size
docker exec quantum_trader_postgres psql -U quantum_user -d quantum_trader -c "SELECT pg_size_pretty(pg_database_size('quantum_trader'))"

# 5. Archive old data (optional)
# Archive events older than 30 days
docker exec quantum_trader_postgres psql -U quantum_user -d quantum_trader -c "
  DELETE FROM events WHERE timestamp < NOW() - INTERVAL '30 days'
"
```

### Model Updates

**Deploy New Models:**
```powershell
# 1. Train new models
python ai_engine/training/train_all_models.py

# 2. Backup old models
Copy-Item ai_engine/models/ backups/models_$(Get-Date -Format 'yyyyMMdd')/ -Recurse

# 3. Copy new models
Copy-Item ai_engine/training/output/* ai_engine/models/ -Force

# 4. Restart AI Service
docker-compose restart ai-service

# 5. Wait for model loading
Start-Sleep -Seconds 60

# 6. Verify models loaded
curl http://localhost:8001/health | ConvertFrom-Json | Select-Object models_loaded

# 7. Monitor performance for 24 hours
# Compare old vs new model performance
```

---

**Version:** 3.0.0  
**Last Updated:** December 2, 2025  
**Operations Support:** Available 24/7

**Questions? Contact:** ops-team@your-company.com
