# BACKLOG DRAIN DEPLOYMENT & EXECUTION GUIDE

## 📋 Overview

Safely drains 10,014 historical trade.intent events with:
- **Throttling**: 2 events/sec (configurable 1-10)
- **Age filter**: Drop events older than 24 hours
- **Confidence filter**: Drop events with confidence < 0.6
- **Symbol filter**: Optional allowlist
- **Dry-run mode**: Test without executing trades

## 🚀 Deployment Steps

### 1. Upload backlog drain service to VPS

```bash
# From local machine:
scp -i ~/.ssh/hetzner_fresh \
  C:\quantum_trader\backend\services\execution\backlog_drain_service.py \
  root@46.224.116.254:/home/qt/quantum_trader/backend/services/execution/

scp -i ~/.ssh/hetzner_fresh \
  C:\quantum_trader\scripts\run_backlog_drain.sh \
  root@46.224.116.254:/home/qt/quantum_trader/scripts/
```

### 2. Make script executable

```bash
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 \
  "chmod +x /home/qt/quantum_trader/scripts/run_backlog_drain.sh"
```

## 🎯 Execution Phases

### Phase 1: DRY-RUN (Recommended First)

Test with small sample to verify filtering logic:

```bash
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 << 'EOF'
cd /home/qt/quantum_trader
docker exec quantum_backend python -m backend.services.execution.backlog_drain_service \
  --mode=dry-run \
  --throttle=5 \
  --max-age-hours=24 \
  --min-confidence=0.6 \
  --max-events=100
EOF
```

**Expected output:**
```
[BACKLOG_DRAIN] Stream length: 10014 events
[BACKLOG_DRAIN] Starting drain process...
[BACKLOG_DRAIN] [DRY-RUN] Would process: BTCUSDT LONG (conf=0.75, source=ensemble)
[BACKLOG_DRAIN] Filtered 1765950222254-0: too_old (120.5h)
...
[BACKLOG_DRAIN] === PROGRESS STATS ===
  Total processed: 100
  Filtered (age): 95
  Filtered (confidence): 3
  Executed: 2
```

### Phase 2: LIVE with Conservative Settings

Start with very conservative throttle and max events:

```bash
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 << 'EOF'
cd /home/qt/quantum_trader
docker exec quantum_backend python -m backend.services.execution.backlog_drain_service \
  --mode=live \
  --throttle=1 \
  --max-age-hours=6 \
  --min-confidence=0.7 \
  --max-events=50
EOF
```

### Phase 3: Full Drain with Monitoring

Once validated, drain full backlog:

```bash
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 << 'EOF'
cd /home/qt/quantum_trader

# Start drain in background
nohup docker exec quantum_backend python -m backend.services.execution.backlog_drain_service \
  --mode=live \
  --throttle=2 \
  --max-age-hours=24 \
  --min-confidence=0.6 \
  > /tmp/backlog_drain.log 2>&1 &

DRAIN_PID=$!
echo "Drain process PID: $DRAIN_PID"
echo "Monitor with: tail -f /tmp/backlog_drain.log"
EOF
```

## 📊 Monitoring During Drain

### Monitor drain progress:

```bash
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "tail -f /tmp/backlog_drain.log"
```

### Monitor Redis lag:

```bash
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 << 'EOF'
watch -n 2 "docker exec quantum_redis redis-cli XINFO STREAM quantum:stream:trade.intent | grep -E 'length|entries-added'"
EOF
```

### Check stream length:

```bash
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 \
  "docker exec quantum_redis redis-cli XLEN quantum:stream:trade.intent"
```

## 🎛️ Configuration Options

### Throttle Levels

| Throttle | Events/sec | Time for 10k | Risk Level |
|----------|------------|--------------|------------|
| 1        | 1          | ~3 hours     | Very Safe  |
| 2        | 2          | ~1.5 hours   | Safe ✅    |
| 5        | 5          | ~35 min      | Moderate   |
| 10       | 10         | ~17 min      | Aggressive |

**Recommended: 2 events/sec**

### Filter Strategies

#### Strategy A: Drop All Old Events (Safest)
```bash
--max-age-hours=1  # Only process events from last hour
--min-confidence=0.7
```

#### Strategy B: Process High-Confidence Only
```bash
--max-age-hours=24
--min-confidence=0.8  # Only very confident signals
```

#### Strategy C: Symbol Allowlist
```bash
--max-age-hours=12
--min-confidence=0.6
--symbols=BTCUSDT,ETHUSDT,BNBUSDT  # Only major pairs
```

## 🔍 Verification Commands

### Before drain:

```bash
# Get total stream length
docker exec quantum_redis redis-cli XLEN quantum:stream:trade.intent

# Sample events to see age
docker exec quantum_redis redis-cli XRANGE quantum:stream:trade.intent - + COUNT 3

# Check stream info
docker exec quantum_redis redis-cli XINFO STREAM quantum:stream:trade.intent
```

### During drain:

```bash
# Watch stream length decrease
watch -n 5 "docker exec quantum_redis redis-cli XLEN quantum:stream:trade.intent"

# Monitor backend container resources
docker stats quantum_backend --no-stream

# Check for errors
docker logs --tail 50 quantum_backend | grep -i error
```

### After drain:

```bash
# Verify lag is 0
docker exec quantum_redis redis-cli XINFO GROUPS quantum:stream:trade.intent

# Check final stream length
docker exec quantum_redis redis-cli XLEN quantum:stream:trade.intent

# Review drain report
cat /tmp/backlog_drain.log | grep "FINAL DRAIN REPORT" -A 20
```

## ⚠️ Safety Mechanisms

1. **Dry-run mode**: Test without executing (default)
2. **Throttling**: Prevent overwhelming system
3. **Age filter**: Drop stale events automatically
4. **Confidence filter**: Skip low-quality signals
5. **Max events**: Limit processing scope
6. **Detailed logging**: Track every decision

## 🛡️ Recommended Approach

```bash
# Step 1: Analyze first 1000 events (dry-run)
docker exec quantum_backend python -m backend.services.execution.backlog_drain_service \
  --mode=dry-run \
  --max-events=1000 \
  --max-age-hours=24 \
  --min-confidence=0.6

# Step 2: If filter rate > 90%, proceed with live (small batch)
docker exec quantum_backend python -m backend.services.execution.backlog_drain_service \
  --mode=live \
  --throttle=2 \
  --max-age-hours=24 \
  --min-confidence=0.6 \
  --max-events=100

# Step 3: Monitor metrics, then full drain
docker exec quantum_backend python -m backend.services.execution.backlog_drain_service \
  --mode=live \
  --throttle=2 \
  --max-age-hours=24 \
  --min-confidence=0.6
```

## 📈 Expected Results

Given 10,014 events that are 5+ days old:

**With max-age-hours=24:**
- Filtered (age): ~9,500+ events (95%+)
- Filtered (confidence): ~200-300 events
- Executed: ~100-200 events maximum

**Processing time:** 
- At 2 events/sec: 50-100 seconds for remaining events
- Most events filtered in-memory (fast)

## ❌ Emergency Stop

If drain causes issues:

```bash
# Find drain process
docker exec quantum_backend ps aux | grep backlog_drain

# Kill process
docker exec quantum_backend kill <PID>

# Or restart container
docker restart quantum_backend
```

## 📝 Filter Logic Justification

### Why these filters?

1. **Age filter (24 hours):**
   - Events >5 days old are obsolete
   - Market conditions changed dramatically
   - Positions likely closed or expired
   - **Impact**: Filters ~95% of backlog

2. **Confidence filter (0.6):**
   - Ensemble model confidence threshold
   - Below 0.6 = weak signal
   - Prevents low-quality executions
   - **Impact**: Filters ~2-5% of remaining

3. **Throttle (2/sec):**
   - Prevents exchange rate limits
   - Allows monitoring
   - Manageable resource usage
   - **Impact**: Safe for live system

### Result:
- **~95-98% filtered** = Safe backlog elimination
- **~2-5% executed** = Only valid recent signals
- **Controlled pace** = No system overload
