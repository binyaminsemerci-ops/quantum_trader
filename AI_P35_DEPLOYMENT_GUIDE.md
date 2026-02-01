# P3.5 Decision Intelligence Service - Deployment Guide

**Status**: Ready for deployment ✅  
**Date**: 2026-02-01  
**Version**: 1.0.0

---

## Quick Start

```bash
# On VPS as root
cd /home/qt/quantum_trader

# 1. Copy configuration
sudo cp etc/quantum/p35-decision-intelligence.env /etc/quantum/
sudo chown qt:qt /etc/quantum/p35-decision-intelligence.env

# 2. Copy systemd unit
sudo cp etc/systemd/system/quantum-p35-decision-intelligence.service /etc/systemd/system/

# 3. Reload and start
sudo systemctl daemon-reload
sudo systemctl enable quantum-p35-decision-intelligence
sudo systemctl start quantum-p35-decision-intelligence

# 4. Verify deployment
bash scripts/proof_p35_decision_intelligence.sh
```

---

## What P3.5 Does

P3.5 **Decision Intelligence Service** is a lightweight analytics processor that:

1. **Consumes** messages from `quantum:stream:apply.result`
2. **Aggregates** decisions (EXECUTE, SKIP, BLOCKED, ERROR) and reasons (why skipped)
3. **Produces** rolling-window analytics (1m, 5m, 15m, 1h)
4. **Tracks** service health (processed count, pending messages, ACK status)

**Use Case**: Answer questions like:
- "Why aren't plans executing?" → Top reasons in last 5 minutes
- "Is the allowlist working?" → See `not_in_allowlist` count
- "Are positions getting blocked?" → See `kill_score_critical` count
- "Is the service healthy?" → Check XPENDING and processed_total

---

## Architecture

### Input
- **Stream**: `quantum:stream:apply.result`
- **Consumer Group**: `p35_decision_intel`
- **Consumer Name**: `{hostname}-{pid}` (per-instance tracking)
- **Batch Size**: 100 messages (configurable)
- **ACK Interval**: 10 seconds (reliable delivery)

### Processing
1. Read batch of messages from stream
2. For each message:
   - Extract: decision, error, symbol, timestamp
   - Determine bucket: `YYYYMMDDHHMM` format
   - HINCRBY decision count in bucket
   - HINCRBY reason count in bucket
   - Optional: HINCRBY symbol_reason count
3. ACK messages in batch
4. Periodically (60s) recompute snapshots from buckets

### Output (Redis Keys)

**Per-minute buckets** (live data):
```
quantum:p35:bucket:202602011430
  decision:EXECUTE    → 42 (count)
  decision:SKIP       → 150
  reason:no_position  → 75
  reason:none         → 42
  symbol_reason:ETHUSDT:no_position → 20
```

**Snapshot windows** (recomputed every minute):
```
quantum:p35:decision:counts:5m  (HASH)
  EXECUTE  → 210
  SKIP     → 750
  BLOCKED  → 15
  ERROR    → 0

quantum:p35:reason:top:5m  (ZSET, sorted by count)
  "no_position"        → 375 (score)
  "not_in_allowlist"   → 200
  "duplicate_plan"     → 100
  ...
```

**Service status**:
```
quantum:p35:status  (HASH)
  processed_total    → 5042
  pending_estimate   → 0
  last_ts            → 1738351234
  consumer_name      → vps-1951265
  service_start_ts   → 1738350000
```

---

## Performance

**CPU**: ~20% (cgroup limited, configurable)  
**Memory**: 256 MB (cgroup limited)  
**Throughput**: 1,000+ decisions/second  
**Latency**: Buckets updated in real-time, snapshots every ~60 seconds  
**Storage**: ~50 MB per 24h (buckets expire after 48h)

---

## Configuration

### Environment Variables

```bash
# /etc/quantum/p35-decision-intelligence.env

# Redis Connection
REDIS_HOST=localhost           # Redis hostname
REDIS_PORT=6379                # Redis port
REDIS_DB=0                     # Redis database

# Logging
LOG_LEVEL=INFO                 # DEBUG|INFO|WARNING|ERROR

# Features
ENABLE_SYMBOL_BREAKDOWN=true   # Track symbol-level reasons (adds storage)
```

### Tuning

```bash
# For high-throughput systems (>1000 decisions/sec)
LOG_LEVEL=WARNING              # Reduce log spam
BATCH_SIZE=500                 # Increase batch size
ACK_INTERVAL=30                # ACK less frequently

# For debugging
LOG_LEVEL=DEBUG                # Verbose logging
ENABLE_SYMBOL_BREAKDOWN=false  # Reduce storage
```

---

## Verification

### Quick Checks

```bash
# Service is running
systemctl is-active quantum-p35-decision-intelligence

# No pending messages (all ACKed)
redis-cli XPENDING quantum:stream:apply.result p35_decision_intel

# Consumer group exists
redis-cli XINFO GROUPS quantum:stream:apply.result

# Status key exists
redis-cli HGETALL quantum:p35:status
```

### Analytics Queries

```bash
# Top 10 skip reasons (5-minute window)
redis-cli ZREVRANGE quantum:p35:reason:top:5m 0 10 WITHSCORES

# Decision distribution
redis-cli HGETALL quantum:p35:decision:counts:5m

# Service health
redis-cli HGETALL quantum:p35:status

# Consumer lag (should be near 0)
redis-cli XPENDING quantum:stream:apply.result p35_decision_intel 0 1 1

# All available windows
for w in 1m 5m 15m 1h; do
  echo "=== $w ===" 
  redis-cli HGETALL quantum:p35:decision:counts:$w
done
```

### Proof Script

```bash
# Comprehensive deployment verification
bash scripts/proof_p35_decision_intelligence.sh
```

The proof script will:
1. Verify consumer group exists
2. Start/restart service
3. Check service status
4. Display P3.5 status hash
5. Show top skip reasons (if data available)
6. Show decision distribution (if data available)
7. Verify XPENDING is healthy
8. List available analytics windows
9. Provide CLI command examples

---

## Monitoring

### Logs

```bash
# Follow service logs
journalctl -u quantum-p35-decision-intelligence -f

# Recent logs (last 50 lines)
journalctl -u quantum-p35-decision-intelligence -n 50

# Logs since service start
journalctl -u quantum-p35-decision-intelligence --since today

# Search for errors
journalctl -u quantum-p35-decision-intelligence -p err
```

### Metrics

**Processed Count**:
```bash
redis-cli HGET quantum:p35:status processed_total
```

**Pending Messages**:
```bash
redis-cli XPENDING quantum:stream:apply.result p35_decision_intel
```

**Latest Timestamp**:
```bash
redis-cli HGET quantum:p35:status last_ts
```

**Service Uptime**:
```bash
redis-cli HGET quantum:p35:status service_start_ts
```

### Alerts (Recommended)

```bash
# Alert if pending > 100
pending=$(redis-cli XPENDING quantum:stream:apply.result p35_decision_intel | head -1)
if [ "$pending" -gt 100 ]; then
  echo "⚠️  ALERT: P3.5 has $pending pending messages"
fi

# Alert if service down
if ! systemctl is-active --quiet quantum-p35-decision-intelligence; then
  echo "⚠️  ALERT: P3.5 service is down"
fi

# Alert if no messages processed in 5 minutes
last_ts=$(redis-cli HGET quantum:p35:status last_ts)
now=$(date +%s)
if [ $((now - last_ts)) -gt 300 ]; then
  echo "⚠️  ALERT: P3.5 has not processed messages in 5+ minutes"
fi
```

---

## Troubleshooting

### Service Won't Start

```bash
# Check service status
systemctl status quantum-p35-decision-intelligence

# View recent logs
journalctl -u quantum-p35-decision-intelligence -n 50

# Check configuration file exists
test -f /etc/quantum/p35-decision-intelligence.env && echo "✅ Config exists" || echo "❌ Config missing"

# Try manual start (as qt user)
sudo -u qt python3 /home/qt/quantum_trader/microservices/decision_intelligence/main.py
```

### High Pending Messages

```bash
# Check service is running
systemctl is-active quantum-p35-decision-intelligence

# Check service CPU/memory
systemctl status quantum-p35-decision-intelligence

# Check Redis is responsive
redis-cli ping

# Restart service
systemctl restart quantum-p35-decision-intelligence
systemctl status quantum-p35-decision-intelligence
```

### No Analytics Data

P3.5 needs time to collect data:
- **1 minute**: Buckets start appearing
- **5 minutes**: 5m snapshot available
- **15 minutes**: All windows available

```bash
# Check if buckets exist
redis-cli KEYS "quantum:p35:bucket:*" | wc -l

# Check latest bucket
redis-cli KEYS "quantum:p35:bucket:*" | tail -1 | xargs redis-cli HGETALL

# Check if service processing
redis-cli HGET quantum:p35:status processed_total
```

### ACKing Issues

```bash
# Check pending messages
redis-cli XPENDING quantum:stream:apply.result p35_decision_intel

# Check consumer info
redis-cli XINFO CONSUMERS quantum:stream:apply.result p35_decision_intel

# If too many pending, restart service
systemctl restart quantum-p35-decision-intelligence

# Verify ACKs working
sleep 10
redis-cli XPENDING quantum:stream:apply.result p35_decision_intel  # should be near 0
```

---

## Integration Examples

### Python: Get Top Skip Reasons

```python
import redis

r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# Get top 10 skip reasons (5-minute window)
reasons = r.zrevrange('quantum:p35:reason:top:5m', 0, 10, withscores=True)
for reason, count in reasons:
    print(f"{reason}: {count}")
```

### Python: Get Decision Distribution

```python
import redis

r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# Get decision counts
decisions = r.hgetall('quantum:p35:decision:counts:5m')
total = sum(int(c) for c in decisions.values())
for decision, count in sorted(decisions.items(), key=lambda x: -int(x[1])):
    pct = 100 * int(count) / total if total > 0 else 0
    print(f"{decision}: {count} ({pct:.1f}%)")
```

### Bash: Alert on High Skip Rate

```bash
#!/bin/bash

# Get 5-minute stats
decisions=$(redis-cli HGETALL quantum:p35:decision:counts:5m)
execute=$(echo "$decisions" | grep -A1 "EXECUTE" | tail -1 || echo 0)
skip=$(echo "$decisions" | grep -A1 "SKIP" | tail -1 || echo 0)
total=$((execute + skip))

if [ "$total" -eq 0 ]; then
  echo "No decisions in 5m window"
  exit
fi

skip_rate=$((100 * skip / total))

if [ "$skip_rate" -gt 80 ]; then
  echo "⚠️  ALERT: Skip rate is ${skip_rate}% (threshold: 80%)"
fi
```

---

## Deployment Rollback

```bash
# Stop service
systemctl stop quantum-p35-decision-intelligence

# Disable
systemctl disable quantum-p35-decision-intelligence

# Remove unit file
sudo rm /etc/systemd/system/quantum-p35-decision-intelligence.service
sudo systemctl daemon-reload

# Redis data persists (can restart later), no action needed
```

---

## Future Enhancements

- [ ] Symbol-level analytics dashboard
- [ ] Historical trend analysis (day-over-day)
- [ ] Predictive skip detection
- [ ] Integration with alerting system
- [ ] Custom reason codes per strategy
- [ ] Per-market analytics breakdown

---

## References

- **Source**: `microservices/decision_intelligence/main.py`
- **Config**: `/etc/quantum/p35-decision-intelligence.env`
- **Unit**: `/etc/systemd/system/quantum-p35-decision-intelligence.service`
- **Proof**: `scripts/proof_p35_decision_intelligence.sh`
- **Docs**: `ops/README.md` (P3.5 Decision Intelligence Service section)

---

**Deployment ready!** ✅
