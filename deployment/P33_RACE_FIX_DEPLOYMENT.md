# P3.3 Race Fix Deployment Guide

## Overview

This deployment eliminates the race condition between Apply Layer and P3.3 Position State Brain by:

1. **Event-driven P3.3**: Switches from polling to Redis Streams consumer group for instant permit issuance
2. **Controlled wait in Apply**: 1200ms window for both Governor and P3.3 permits to arrive
3. **No testnet bypass**: Testnet now requires both Governor (P3.2) and P3.3 permits, just like production

## Architecture Changes

### Before (Polling, Race-Prone)
```
Apply Layer publishes plan → (immediate execution attempt)
    ↓ (fails - no permit yet)
P3.3 polls every 1-5s → issues permit
    ↓ (too late)
Apply Layer has already moved on
```

### After (Event-Driven, Race-Free)
```
Apply Layer publishes plan → waits 1200ms for permits
    ↓
P3.3 consumes from stream → issues permit within ~10-100ms
    ↓
Governor issues permit (manual/auto)
    ↓
Apply Layer finds both permits → executes atomically
```

## Code Changes

### 1. Apply Layer (`microservices/apply_layer/main.py`)

**Removed**:
- Testnet Governor permit bypass (lines 636-642)

**Added**:
- Controlled permit wait loop (1200ms max, 100ms intervals)
- Both permits required: `quantum:permit:<plan_id>` (Governor) + `quantum:permit:p33:<plan_id>` (P3.3)
- Clear error: `permit_timeout` if window expires

**Key Logic**:
```python
for attempt in range(12):  # 12 x 100ms = 1200ms max
    gov_exists = self.redis.exists(permit_key)
    p33_exists = self.redis.exists(p33_permit_key)
    
    if gov_exists and p33_exists:
        permits_ready = True
        break
    
    time.sleep(0.1)
```

### 2. P3.3 Position State Brain (`microservices/position_state_brain/main.py`)

**Removed**:
- `process_apply_plans(symbol)` - polling-based plan scanner

**Added**:
- `process_apply_plans_stream()` - Redis Streams consumer
- Consumer group: `p33` on `quantum:stream:apply.plan`
- XREADGROUP with BLOCK 1000ms (non-busy wait)
- Message ACK after permit issuance

**Key Logic**:
```python
messages = self.redis.xreadgroup(
    groupname='p33',
    consumername=f'p33-{os.getpid()}',
    streams={'quantum:stream:apply.plan': '>'},
    count=10,
    block=1000
)

for msg_id, fields in messages:
    if fields['decision'] == 'EXECUTE':
        self.evaluate_plan(plan_id, symbol, fields)
        self.redis.xack(plan_stream, 'p33', msg_id)
```

## Deployment Steps

### 1. Backup Current State

```bash
ssh root@<VPS_IP> << 'EOF'
# Backup services
systemctl stop quantum-apply-layer quantum-position-state-brain
cp -r /home/qt/quantum_trader /home/qt/quantum_trader.backup.$(date +%Y%m%d_%H%M%S)

# Backup Redis data
redis-cli SAVE
cp /var/lib/redis/dump.rdb /var/lib/redis/dump.rdb.backup.$(date +%Y%m%d_%H%M%S)
EOF
```

### 2. Deploy Updated Code

```bash
# From local machine
cd c:\quantum_trader

# Deploy Apply Layer
cat microservices/apply_layer/main.py | wsl ssh -i ~/.ssh/hetzner_fresh root@<VPS_IP> "cat > /home/qt/quantum_trader/microservices/apply_layer/main.py"

# Deploy P3.3
cat microservices/position_state_brain/main.py | wsl ssh -i ~/.ssh/hetzner_fresh root@<VPS_IP> "cat > /home/qt/quantum_trader/microservices/position_state_brain/main.py"

# Deploy proof script
cat ops/p33_proof_e2e_testnet.sh | wsl ssh -i ~/.ssh/hetzner_fresh root@<VPS_IP> "cat > /home/qt/quantum_trader/ops/p33_proof_e2e_testnet.sh && chmod +x /home/qt/quantum_trader/ops/p33_proof_e2e_testnet.sh"
```

### 3. Verify Environment Files

```bash
ssh root@<VPS_IP> << 'EOF'
# Check apply-layer.env has Binance credentials
if [ ! -f /etc/quantum/apply-layer.env ]; then
    echo "ERROR: Missing /etc/quantum/apply-layer.env"
    exit 1
fi

grep -q "BINANCE_TESTNET_API_KEY" /etc/quantum/apply-layer.env || echo "WARN: Missing BINANCE_TESTNET_API_KEY"

# Check position-state-brain.env has P3.3 settings
if [ ! -f /etc/quantum/position-state-brain.env ]; then
    echo "Creating /etc/quantum/position-state-brain.env"
    cat > /etc/quantum/position-state-brain.env << 'ENVEOF'
# P3.3 Position State Brain Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Polling (for snapshot refresh, not permit scanning)
P33_POLL_SEC=5

# Sanity thresholds
P33_STALE_THRESHOLD_SEC=10
P33_COOLDOWN_SEC=15
P33_QTY_TOLERANCE_PCT=1.0

# Permit TTL
P33_PERMIT_TTL=60

# Metrics
P33_METRICS_PORT=8045

# Symbol allowlist
P33_ALLOWLIST=BTCUSDT
ENVEOF
fi

echo "✓ Environment files ready"
EOF
```

### 4. Create Consumer Group

```bash
ssh root@<VPS_IP> << 'EOF'
# Create P3.3 consumer group (idempotent)
redis-cli XGROUP CREATE quantum:stream:apply.plan p33 $ MKSTREAM 2>&1 | grep -v "BUSYGROUP" || true
echo "✓ Consumer group 'p33' ready on quantum:stream:apply.plan"
EOF
```

### 5. Restart Services

```bash
ssh root@<VPS_IP> << 'EOF'
# Restart P3.3 first (so it's ready to consume)
systemctl restart quantum-position-state-brain
sleep 2
systemctl status quantum-position-state-brain --no-pager -l | head -20

# Restart Apply Layer
systemctl restart quantum-apply-layer
sleep 2
systemctl status quantum-apply-layer --no-pager -l | head -20

echo ""
echo "✓ Services restarted"
EOF
```

### 6. Run Proof Script

```bash
ssh root@<VPS_IP> << 'EOF'
cd /home/qt/quantum_trader
./ops/p33_proof_e2e_testnet.sh
EOF
```

Expected output:
```
✓ quantum-apply-layer: RUNNING
✓ quantum-position-state-brain: RUNNING
✓ EXECUTE plan detected: <plan_id>
✓ Governor permit EXISTS (TTL: 58s)
✓ P3.3 permit EXISTS (TTL: 57s)
✓ P3.3 logged permit decision: ALLOW plan <plan_id> (safe_qty=0.0080, exchange_amt=0.0100)
✓ Apply Layer execution log: permits ready after 120ms (Governor + P3.3)
✓✓✓ EXECUTION SUCCESSFUL (executed=True)
```

## Verification Checklist

- [ ] Both services running (`systemctl status quantum-{apply-layer,position-state-brain}`)
- [ ] P3.3 logs show "event-driven mode" startup message
- [ ] Consumer group exists: `redis-cli XINFO GROUPS quantum:stream:apply.plan | grep p33`
- [ ] Apply Layer logs show "Both permits ready after Xms" (not "Governor permit skip")
- [ ] Execution logs show `executed=True` with reduceOnly orders
- [ ] No `permit_timeout` errors in Apply Layer logs
- [ ] P3.3 metrics available: `curl http://localhost:8045/metrics | grep p33_permit_allow_total`

## Rollback

If issues occur:

```bash
ssh root@<VPS_IP> << 'EOF'
# Stop services
systemctl stop quantum-apply-layer quantum-position-state-brain

# Restore backup
BACKUP_DIR=$(ls -td /home/qt/quantum_trader.backup.* | head -1)
if [ -n "$BACKUP_DIR" ]; then
    rm -rf /home/qt/quantum_trader
    cp -r "$BACKUP_DIR" /home/qt/quantum_trader
    echo "✓ Restored from $BACKUP_DIR"
fi

# Restart services
systemctl start quantum-position-state-brain
systemctl start quantum-apply-layer
EOF
```

## Configuration Reference

### Environment Variables (apply-layer.env)
- `BINANCE_TESTNET_API_KEY` - Binance Futures Testnet API key
- `BINANCE_TESTNET_API_SECRET` - Binance Futures Testnet secret
- `REDIS_HOST` - Redis server (default: localhost)
- `REDIS_PORT` - Redis port (default: 6379)
- `APPLY_MODE` - Mode: `testnet` / `dry_run` / `production`

### Environment Variables (position-state-brain.env)
- `P33_POLL_SEC` - Snapshot refresh interval (default: 5, not used for permit scanning)
- `P33_PERMIT_TTL` - Permit expiry in seconds (default: 60)
- `P33_ALLOWLIST` - Comma-separated symbols (e.g., BTCUSDT,ETHUSDT)
- `P33_METRICS_PORT` - Prometheus metrics port (default: 8045)

## Troubleshooting

### "permit_timeout" errors
**Symptom**: Apply Layer logs show "Permit timeout after 1200ms"

**Causes**:
1. P3.3 service not running → Check `systemctl status quantum-position-state-brain`
2. Governor not issuing permits → Manual injection needed in testnet (see below)
3. P3.3 denying permits → Check P3.3 logs for DENY reasons (stale, mismatch, cooldown)

**Fix**:
```bash
# Check P3.3 is consuming stream
redis-cli XINFO CONSUMERS quantum:stream:apply.plan p33

# Manual Governor permit injection (testnet only)
PLAN_ID="<from logs>"
redis-cli SETEX quantum:permit:$PLAN_ID 60 '{"granted":true}'
```

### P3.3 "reconcile_required_qty_mismatch"
**Symptom**: P3.3 logs show DENY with qty mismatch

**Fix**: Reconcile ledger to match exchange snapshot:
```bash
redis-cli HSET quantum:position:ledger:BTCUSDT last_known_amt <actual_exchange_amt>
```

### Consumer group lag
**Symptom**: P3.3 processes old plans

**Fix**:
```bash
# Check pending messages
redis-cli XPENDING quantum:stream:apply.plan p33

# Claim stale messages (if needed)
redis-cli XAUTOCLAIM quantum:stream:apply.plan p33 <consumer_id> 60000 0-0
```

## Performance Expectations

- **Permit latency**: 10-100ms (event-driven vs 1000-5000ms polling)
- **Apply wait time**: 100-300ms typical (vs 1200ms max)
- **P3.3 CPU**: <5% (BLOCK 1000ms reduces busy-wait)
- **Memory**: Apply ~200MB, P3.3 ~150MB

## Next Steps

1. Monitor testnet execution for 24h
2. Verify no permit_timeout errors
3. Check P3.3 metrics for permit_allow vs permit_deny ratio
4. Once stable, consider production deployment with same architecture
