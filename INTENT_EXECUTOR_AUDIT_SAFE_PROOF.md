# Intent Executor - Audit-Safe Implementation Proof
**Date**: 2026-01-27  
**Commit**: cfdb40f5  
**Status**: ✅ DEPLOYED AND VERIFIED

## Summary

Implemented **audit-safe** source allowlist + TTL-guarded manual lane + Redis metrics for Intent Executor.

**Key Principles Enforced**:
- Default deny: ONLY `source=intent_bridge` allowed on main stream
- No trading logic changes (sizing, permits, execution unchanged)
- Manual lane requires explicit TTL-guarded enablement via Redis key
- Metrics stored in Redis hash (no HTTP port conflicts)
- Automatic shutoff when TTL expires (prevents "forget to disable" errors)

---

## Implementation Details

### A) Source Allowlist (MAIN Stream)

**Config**: `INTENT_EXECUTOR_SOURCE_ALLOWLIST=intent_bridge` (default, audit-safe)

**Behavior**:
- Plans from main stream (`quantum:stream:apply.plan`) checked against allowlist
- Non-allowed sources: ACK+SKIP with `source_not_allowed` reason
- Counter: `HINCRBY quantum:metrics:intent_executor blocked_source 1`
- Log format: `🚫 [lane=main] Skip plan (source_not_allowed): plan_id=... source=... allowlist=...`

**Code Location**: [main.py#L535-L541](microservices/intent_executor/main.py#L535-L541)

### B) Manual Lane (SEPARATE Stream)

**Config**:
- `INTENT_EXECUTOR_MANUAL_STREAM=quantum:stream:apply.plan.manual`
- `INTENT_EXECUTOR_MANUAL_GROUP=intent_executor_manual`
- Redis TTL guard: `quantum:manual_lane:enabled`

**Behavior**:
- Separate consumer group reads from manual stream
- Plans ONLY processed if Redis key `quantum:manual_lane:enabled=1` exists with TTL > 0
- If TTL expired/missing: ACK+SKIP with `manual_lane_disabled` reason
- Counters: `manual_consumed`, `manual_blocked_disabled`, `manual_processed`
- Log format: 
  - Enabled: `🔓 [lane=manual] Consuming manual plan: plan_id=...`
  - Disabled: `🚫 [lane=manual] Skip plan (manual_lane_disabled): plan_id=... ttl=0`

**Code Location**: [main.py#L543-L552](microservices/intent_executor/main.py#L543-L552)

### C) Metrics (Redis-Based, NO HTTP)

**Storage**: Redis hash `quantum:metrics:intent_executor`

**Counters**:
- `processed`: Total plans processed
- `executed_true`: Successful executions
- `executed_false`: Failed/skipped executions
- `blocked_source`: Plans blocked by source allowlist
- `manual_consumed`: Plans consumed from manual lane
- `manual_blocked_disabled`: Plans blocked when manual lane disabled
- `manual_processed`: Plans successfully processed via manual lane

**Increment**: `HINCRBY quantum:metrics:intent_executor <field> 1`

**Heartbeat**: Every 60s, logs:
```
🔓 MANUAL_LANE_ACTIVE ttl_remaining=1180s
📊 Metrics: processed=42 executed_true=38 executed_false=4 blocked_source=2
```

**Code Location**: [main.py#L172-L193](microservices/intent_executor/main.py#L172-L193)

---

## Deployment Steps

### 1. Update Environment

VPS: `/etc/quantum/intent-executor.env`

```bash
# REQUIRED (audit-safe default)
INTENT_EXECUTOR_SOURCE_ALLOWLIST=intent_bridge

# REQUIRED (manual lane streams)
INTENT_EXECUTOR_MANUAL_STREAM=quantum:stream:apply.plan.manual
INTENT_EXECUTOR_MANUAL_GROUP=intent_executor_manual

# REMOVED (no longer needed)
# INTENT_EXECUTOR_MANUAL_LANE_ENABLED  ← Now TTL-guarded via Redis key
# INTENT_EXECUTOR_METRICS_PORT         ← Removed HTTP metrics
```

### 2. Deploy Code

```bash
# Commit: cfdb40f5
git pull
systemctl restart quantum-intent-executor
```

### 3. Verify Startup

```bash
journalctl -u quantum-intent-executor --since "5 seconds ago" | grep -E "(Source allowlist|MANUAL_LANE)"
```

**Expected Output**:
```
Source allowlist: ['intent_bridge']
Manual lane TTL guard: quantum:manual_lane:enabled
🔒 MANUAL_LANE_OFF
📨 Consuming MAIN: quantum:stream:apply.plan
📨 Consuming MANUAL: quantum:stream:apply.plan.manual
```

---

## Proof Commands

### Proof A: Source Whitelist Blocks Non-Allowed Source

**Test**: Inject plan with `source=unauthorized` to main stream

```bash
redis-cli XADD quantum:stream:apply.plan "*" \
  plan_id proof_blocked_001 \
  action FULL_CLOSE_PROPOSED \
  decision EXECUTE \
  symbol ETHUSDT \
  side SELL \
  type MARKET \
  qty 0.01 \
  reduceOnly true \
  source unauthorized \
  timestamp $(date +%s)
```

**Expected Log**:
```
🚫 [lane=main] Skip plan (source_not_allowed): plan_id=proof_bl source=unauthorized allowlist=['intent_bridge']
```

**Expected Metric**:
```bash
redis-cli HGET quantum:metrics:intent_executor blocked_source
# Output: 1 (or higher if multiple tests)
```

---

### Proof B: Source Whitelist Allows intent_bridge

**Test**: Normal plan from `intent_bridge` (production flow)

```bash
redis-cli XADD quantum:stream:apply.plan "*" \
  plan_id proof_allowed_001 \
  action FULL_CLOSE_PROPOSED \
  decision EXECUTE \
  symbol ETHUSDT \
  side SELL \
  type MARKET \
  qty 0.01 \
  reduceOnly true \
  source intent_bridge \
  timestamp $(date +%s)
```

**Expected Log**:
```
▶️  Processing plan: proof_al | ETHUSDT SELL qty=0.0100
⏳ Waiting for P3.3 permit: proof_al
```

**Expected Metric**:
```bash
redis-cli HGET quantum:metrics:intent_executor processed
# Incremented
```

---

### Proof C: Manual Lane OFF - Plans Blocked

**Test**: Inject to manual stream WITHOUT enabling TTL key

```bash
# Verify manual lane is OFF
redis-cli GET quantum:manual_lane:enabled
# Output: (nil)

# Inject plan to manual stream
redis-cli XADD quantum:stream:apply.plan.manual "*" \
  plan_id proof_manual_off_001 \
  action FULL_CLOSE_PROPOSED \
  decision EXECUTE \
  symbol ETHUSDT \
  side SELL \
  type MARKET \
  qty 0.01 \
  reduceOnly true \
  source manual_test \
  timestamp $(date +%s)
```

**Expected Log**:
```
🚫 [lane=manual] Skip plan (manual_lane_disabled): plan_id=proof_ma ttl=0
```

**Expected Metric**:
```bash
redis-cli HGET quantum:metrics:intent_executor manual_blocked_disabled
# Output: 1 (or higher)
```

---

### Proof D: Manual Lane ON - Plans Processed

**Test**: Enable manual lane with 20-minute TTL, then inject plan

```bash
# Enable manual lane for 20 minutes (1200 seconds)
redis-cli SETEX quantum:manual_lane:enabled 1200 1

# Verify TTL
redis-cli TTL quantum:manual_lane:enabled
# Output: 1200 (or slightly less)

# Inject plan to manual stream
redis-cli XADD quantum:stream:apply.plan.manual "*" \
  plan_id proof_manual_on_001 \
  action FULL_CLOSE_PROPOSED \
  decision EXECUTE \
  symbol ETHUSDT \
  side SELL \
  type MARKET \
  qty 0.01 \
  reduceOnly true \
  source manual_test \
  timestamp $(date +%s)
```

**Expected Log**:
```
🔓 [lane=manual] Consuming manual plan: proof_ma
▶️  Processing plan: proof_ma | ETHUSDT SELL qty=0.0100
⏳ Waiting for P3.3 permit: proof_ma
```

**Expected Metrics**:
```bash
redis-cli HGET quantum:metrics:intent_executor manual_consumed
# Incremented

redis-cli HGET quantum:metrics:intent_executor processed
# Incremented
```

**Expected Heartbeat** (within 60s):
```
🔓 MANUAL_LANE_ACTIVE ttl_remaining=1180s
```

---

### Proof E: Manual Lane TTL Expires - Automatic Shutoff

**Test**: Wait for TTL to expire (or set short TTL), verify auto-disable

```bash
# Enable with 30-second TTL for quick test
redis-cli SETEX quantum:manual_lane:enabled 30 1

# Wait 35 seconds
sleep 35

# Verify key expired
redis-cli GET quantum:manual_lane:enabled
# Output: (nil)

# Inject plan - should be blocked
redis-cli XADD quantum:stream:apply.plan.manual "*" \
  plan_id proof_manual_expired_001 \
  action FULL_CLOSE_PROPOSED \
  decision EXECUTE \
  symbol ETHUSDT \
  side SELL \
  type MARKET \
  qty 0.01 \
  reduceOnly true \
  source manual_test \
  timestamp $(date +%s)
```

**Expected Log**:
```
🚫 [lane=manual] Skip plan (manual_lane_disabled): plan_id=proof_ma ttl=0
🔒 MANUAL_LANE_OFF
```

---

### Proof F: Heartbeat Metrics Summary

**Test**: Check heartbeat logs and Redis metrics

```bash
# Watch heartbeat logs (emitted every 60s)
journalctl -u quantum-intent-executor -f | grep -E "(MANUAL_LANE|Metrics:)"

# Read all metrics from Redis
redis-cli HGETALL quantum:metrics:intent_executor
```

**Expected Output**:
```
1) "processed"
2) "45"
3) "executed_true"
4) "38"
5) "executed_false"
6) "7"
7) "blocked_source"
8) "3"
9) "manual_consumed"
10) "2"
11) "manual_blocked_disabled"
12) "5"
13) "manual_processed"
14) "2"
```

---

## Manual Lane Usage Commands

### Enable Manual Lane (20 minutes)
```bash
redis-cli SETEX quantum:manual_lane:enabled 1200 1
```

### Check Manual Lane Status
```bash
redis-cli GET quantum:manual_lane:enabled
redis-cli TTL quantum:manual_lane:enabled
```

### Disable Manual Lane Immediately
```bash
redis-cli DEL quantum:manual_lane:enabled
```

### Read Metrics
```bash
redis-cli HGETALL quantum:metrics:intent_executor
```

### Inject Plan to Manual Stream
```bash
redis-cli XADD quantum:stream:apply.plan.manual "*" \
  plan_id manual_test_$(date +%s) \
  action FULL_CLOSE_PROPOSED \
  decision EXECUTE \
  symbol ETHUSDT \
  side SELL \
  type MARKET \
  qty 0.01 \
  reduceOnly true \
  source manual_operator \
  timestamp $(date +%s)
```

---

## Verification Checklist

- [x] Source allowlist defaults to `intent_bridge` only
- [x] Non-allowed sources blocked on main stream
- [x] Manual lane consumer group created
- [x] Manual lane requires Redis TTL key to process plans
- [x] Manual lane auto-disables when TTL expires
- [x] Metrics stored in Redis hash (no HTTP port conflicts)
- [x] Heartbeat logs every 60s with manual lane status
- [x] No trading logic modified (sizing, permits, execution unchanged)
- [x] Logs clearly show `lane=main` or `lane=manual` for every plan

---

## Code Changes

**Commit**: cfdb40f5  
**Files Changed**: 1 file (microservices/intent_executor/main.py)  
**Diff**: 141 insertions, 133 deletions

**Key Changes**:
1. Removed HTTP metrics server (eliminated port conflicts)
2. Added `_get_manual_lane_ttl()` and `_is_manual_lane_enabled()` for TTL checking
3. Added `_emit_heartbeat()` for periodic status logging
4. Modified `process_plan()` to enforce lane-specific guards:
   - Main lane: source allowlist check
   - Manual lane: TTL guard check
5. Updated `run()` loop to always consume both streams (TTL guard happens in process_plan)
6. Replaced dict-based metrics with Redis HINCRBY counters

**Audit-Safe Guarantees**:
- No broadening of execution sources on main stream
- Manual testing requires explicit, time-limited enablement
- Automatic shutoff prevents "forget to disable" errors
- All trading logic (sizing, permits, execution) unchanged

---

## Status: READY FOR PRODUCTION

✅ Deployed to VPS  
✅ Source allowlist: `intent_bridge` only  
✅ Manual lane: TTL-guarded, OFF by default  
✅ Metrics: Redis-based, no port conflicts  
✅ Heartbeat: Logs every 60s  
✅ No trading logic modified  

**Next Steps**: Test complete manual EXIT flow with valid P3.3 permit on symbol with open position.
