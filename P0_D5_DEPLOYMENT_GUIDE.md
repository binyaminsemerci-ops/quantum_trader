# P0.D.5 BACKLOG HARDENING - Deployment Guide

**Status**: ✅ All code changes complete, ready for deployment  
**Safety Level**: PROD SAFE (all features behind env flags, conservative defaults)  
**Date**: 2026-01-21

---

## Pre-Deployment Checklist

### 1. Schema Validation Test
```bash
# Run schema gate tests (must pass before deploy)
python tests/test_trade_intent_schema.py
```

Expected output: `✅ All schema validation tests PASSED - safe to deploy`

---

## Deployment Steps

### 2. Create Environment Configuration
```bash
# SSH to VPS
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254

# Create environment file
cat > /etc/quantum/ai-engine.env << 'EOF'
# P0.D.5 BACKLOG HARDENING Configuration
# Conservative defaults - increase gradually as system stabilizes

# Stale Intent TTL (seconds) - drop intents older than this
INTENT_MAX_AGE_SEC=600

# XREADGROUP batch size - controls throughput
# Start: 10, Increase gradually: 20 → 30 → 50
XREADGROUP_COUNT=10

# Execution concurrency - parallel execution workers
# Start: 1, Increase gradually: 2 → 4 → 8
EXEC_CONCURRENCY=1

# Diagnostic logging (P0.D.4 cleanup)
PIPELINE_DIAG=false
EOF

# Verify file created
cat /etc/quantum/ai-engine.env
```

### 3. Update Systemd Service to Load Env File
```bash
# Edit execution service
nano /etc/systemd/system/quantum-execution.service

# Add this line under [Service] section:
EnvironmentFile=/etc/quantum/ai-engine.env

# Edit AI engine service
nano /etc/systemd/system/quantum-ai-engine.service

# Add same line:
EnvironmentFile=/etc/quantum/ai-engine.env

# Reload systemd
systemctl daemon-reload
```

### 4. Deploy Code Changes
```bash
# From local machine (WSL)
cd /mnt/c/quantum_trader

# Deploy execution_service.py
wsl scp -i ~/.ssh/hetzner_fresh \
  services/execution_service.py \
  root@46.224.116.254:/opt/quantum/services/execution_service.py

# Deploy eventbus_bridge.py
wsl scp -i ~/.ssh/hetzner_fresh \
  ai_engine/services/eventbus_bridge.py \
  root@46.224.116.254:/opt/quantum/ai_engine/services/eventbus_bridge.py

# Deploy proof_pipeline.sh
wsl scp -i ~/.ssh/hetzner_fresh \
  ops/proof_pipeline.sh \
  root@46.224.116.254:/opt/quantum/ops/proof_pipeline.sh

# Make script executable
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 \
  'chmod +x /opt/quantum/ops/proof_pipeline.sh'
```

### 5. Restart Services
```bash
# SSH to VPS
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254

# Restart services in order
systemctl restart quantum-ai-engine
sleep 3
systemctl restart quantum-execution

# Verify services started
systemctl status quantum-execution --no-pager -l
systemctl status quantum-ai-engine --no-pager -l
```

---

## Post-Deployment Verification

### 6. Run Proof Pipeline Snapshot
```bash
# SSH to VPS
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254

# Run proof pipeline
/opt/quantum/ops/proof_pipeline.sh --once
```

**Expected Output:**
- ✅ execution.result stream advancing (last-id incrementing)
- ✅ Rate Limiter Status section showing quantum:rate:* keys
- ✅ Services status: active (running)

### 7. Verify TTL Logic (Stale Intent Drops)
```bash
# Monitor execution logs for TTL drops
tail -f /var/log/quantum/execution.log | grep STALE_INTENT_DROP

# Expected (if backlog has old messages):
# 2026-01-21 00:05:23 [WARN] STALE_INTENT_DROP intent_id=... age_sec=1204 (exceeds 600s TTL)
```

### 8. Check Stream Statistics
```bash
# Check execution.result stream health
redis-cli XINFO STREAM quantum:stream:execution.result | \
  egrep 'last-generated-id|entries-added' -A1

# Check trade.intent stream lag
redis-cli XINFO GROUPS quantum:stream:trade.intent | \
  egrep 'name|lag|pending' -A1
```

### 9. Verify Configuration Applied
```bash
# Check execution service logs for config startup
journalctl -u quantum-execution -n 50 | grep "P0.D.5 Config"

# Expected:
# P0.D.5 Config: INTENT_MAX_AGE_SEC=600, XREADGROUP_COUNT=10, EXEC_CONCURRENCY=1
```

---

## Gradual Throughput Increase Plan

**Current State**: XREADGROUP_COUNT=10, EXEC_CONCURRENCY=1  
**Target State**: XREADGROUP_COUNT=50, EXEC_CONCURRENCY=4

### Phase 1 (Day 1): Baseline
- COUNT=10, CONCURRENCY=1
- Monitor: CPU, memory, Redis lag, execution success rate
- Goal: Establish stable baseline metrics

### Phase 2 (Day 2): Increase Batch Size
```bash
# Edit /etc/quantum/ai-engine.env
XREADGROUP_COUNT=20

# Restart services
systemctl restart quantum-execution
```
- Monitor for 24 hours
- Verify no increased error rate or resource exhaustion

### Phase 3 (Day 3): Continue Batch Increase
```bash
XREADGROUP_COUNT=30
```

### Phase 4 (Day 4): Add Concurrency
```bash
XREADGROUP_COUNT=30
EXEC_CONCURRENCY=2
```

### Phase 5 (Day 5-7): Final Tuning
```bash
XREADGROUP_COUNT=50
EXEC_CONCURRENCY=4
```

**Safety Rules:**
- If lag increases → reduce COUNT
- If CPU >80% → reduce CONCURRENCY
- If error rate >5% → rollback to previous config
- Always restart services after config change

---

## Monitoring Commands

### Stream Health Check
```bash
/opt/quantum/ops/proof_pipeline.sh --once
```

### Rate Limiter Status
```bash
# Check all rate limiters
redis-cli --scan --pattern 'quantum:rate:*' | while read key; do
  echo "$key:"
  redis-cli GET $key
  redis-cli TTL $key
  echo
done
```

### Backlog Metrics
```bash
# Lag and pending messages
redis-cli XINFO GROUPS quantum:stream:trade.intent | \
  egrep 'lag|pending' -A1
```

### Stale Intent Count
```bash
# Count stale intent drops in last hour
journalctl -u quantum-execution --since '1 hour ago' | \
  grep STALE_INTENT_DROP | wc -l
```

---

## Rollback Plan

If deployment causes issues:

```bash
# 1. Restore original files from backup
cd /opt/quantum
git checkout HEAD services/execution_service.py
git checkout HEAD ai_engine/services/eventbus_bridge.py
git checkout HEAD ops/proof_pipeline.sh

# 2. Remove environment file
rm /etc/quantum/ai-engine.env

# 3. Restore systemd services
nano /etc/systemd/system/quantum-execution.service
# Remove: EnvironmentFile=/etc/quantum/ai-engine.env

nano /etc/systemd/system/quantum-ai-engine.service
# Remove: EnvironmentFile=/etc/quantum/ai-engine.env

# 4. Reload and restart
systemctl daemon-reload
systemctl restart quantum-ai-engine
systemctl restart quantum-execution

# 5. Verify rollback
/opt/quantum/ops/proof_pipeline.sh --once
```

---

## P0.D.5 Feature Summary

### ✅ Stale Intent TTL
- **Purpose**: Drop old intents from 1.4M message backlog
- **Config**: `INTENT_MAX_AGE_SEC=600` (10 minutes)
- **Behavior**: Parse intent timestamp → if age > TTL → XACK + log drop
- **Logging**: Rate-limited to max 1 per 30s (prevents spam)

### ✅ Controlled Throughput
- **Purpose**: Gradual backlog processing, prevent system overload
- **Config**: `XREADGROUP_COUNT=10` (start conservative, increase to 50)
- **Behavior**: XREADGROUP reads N messages per batch
- **Safety**: Start low, increase gradually while monitoring

### ✅ Bounded Concurrency
- **Purpose**: Limit parallel executions, prevent resource exhaustion
- **Config**: `EXEC_CONCURRENCY=1` (start sequential, increase to 4)
- **Behavior**: Semaphore limits concurrent trade executions
- **Safety**: Async worker pool with configurable bound

### ✅ Rate Limiter Visibility
- **Purpose**: Operational visibility into rate limiting state
- **Implementation**: proof_pipeline.sh queries quantum:rate:* keys
- **Output**: Shows COUNT, TTL, and next reset ETA for each limiter
- **Usage**: `/opt/quantum/ops/proof_pipeline.sh --once`

### ✅ Schema Gate
- **Purpose**: Pre-deployment validation of schema contract
- **Implementation**: Unit tests in `tests/test_trade_intent_schema.py`
- **Coverage**: Required fields, validation rules, type safety, mock publish/consume
- **Usage**: `python tests/test_trade_intent_schema.py` (must pass before deploy)

---

## Success Criteria

- ✅ Schema tests pass
- ✅ Services restart cleanly
- ✅ execution.result stream advances
- ✅ Rate limiter visibility in proof_pipeline
- ✅ No P0.D.4d/e logging spam
- ✅ TTL drops logged (if old intents processed)
- ✅ Backlog lag decreases over time

---

**End of P0.D.5 Deployment Guide**
