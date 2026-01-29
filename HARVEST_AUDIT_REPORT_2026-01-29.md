# QUANTUM TRADER HARVEST SYSTEM AUDIT REPORT
**Date**: 2026-01-29 15:31 UTC  
**Auditor**: Sonnet (VPS Operator + Auditor)  
**Server**: quantumtrader-prod-1 (46.224.116.254)  
**Commit**: 9141fbf7

---

## VERDICT: **FAIL**

---

## EXECUTIVE SUMMARY

The Quantum Trader HARVEST system has **critical integration failure** between P2.6 Portfolio Heat Gate and downstream consumers (Apply Layer). While all services are running and generating data, the Heat Gate is operating in **ENFORCE mode with ZERO processed proposals** (p26_proposals_processed_total=0.0, p26_hash_writes_total=0.0), indicating a **complete processing stall** despite harvest proposals being published every 10 seconds. The Heat Gate metrics show p26_enforce_mode=1.0 but no throughput, suggesting the consumer loop is not reading from the input stream quantum:stream:harvest.proposal (XLEN=25). The calibrated stream exists (XLEN=12) with recent entries but appears disconnected from live processing. Apply Layer continues reading directly from hash keys (quantum:harvest:proposal:BTCUSDT) which contain stale data (last_update_epoch=1769700655 vs current ~1769700693), bypassing the Heat Gate's calibration entirely. This is a **PRODUCTION BLOCKER** - harvest proposals are being generated but Heat Gate is idle.

---

## EVIDENCE BUNDLE

### STEP 0: Environment + Time
```
Thu Jan 29 03:29:52 PM UTC 2026
quantumtrader-prod-1
9141fbf7
```

### STEP 1: Services Status

**Active Services** (all running):
```
quantum-harvest-proposal.service: active
quantum-portfolio-heat-gate.service: active  
quantum-apply-layer.service: active
quantum-governor.service: active
quantum-position-state-brain.service: active
```

**Harvest Proposal Publisher**:
- Status: Active (running) since Thu 2026-01-29 06:53:36 UTC (8h ago)
- PID: 907828
- Memory: 19.5M / 512.0M
- CPU: 16.612s
- **Recent Activity** (15:29:15 UTC):
  ```
  Published harvest for BTCUSDT: action=FULL_CLOSE_PROPOSED R=7.01 K=0.758 SL=100.2 reasons=harvest_partial_75,profit_lock
  Published harvest for ETHUSDT: action=FULL_CLOSE_PROPOSED R=9.80 K=0.761 SL=50.1 reasons=harvest_partial_75,profit_lock
  ```
- Runs every 10 seconds with "=== Harvest Proposal Publish Cycle ===" log
- **VERDICT**: ✅ OPERATIONAL - producing proposals regularly

**Portfolio Heat Gate**:
- Status: Active (running) since Thu 2026-01-29 06:53:36 UTC (8h ago)
- PID: 907854
- Memory: 20.4M / 512.0M
- CPU: 1min 9.479s
- **Logs**: `-- No entries --` (journal rotated, no recent activity visible)
- **VERDICT**: ⚠️ RUNNING BUT SILENT (metrics reveal idle state)

**Apply Layer**:
- Status: Active (running) since Thu 2026-01-29 06:53:36 UTC (8h ago)
- PID: 907768
- **Recent Logs** (15:30:19 UTC):
  ```
  BTCUSDT: Kill score 0.758 >= 0.6 but action FULL_CLOSE_PROPOSED is close (OK)
  BTCUSDT: Plan 5ccb55e0c09e77cb published (decision=EXECUTE, steps=1)
  ETHUSDT: Plan 3f58da1bdb590b3b published (decision=EXECUTE, steps=1)
  [PERMIT_WAIT] BLOCK plan=3f58da1bdb590b3b symbol=ETHUSDT wait_ms=1207 info={'reason': 'missing_p26', 'gov_ttl': 58, 'p33_ttl': 59, 'p26_ttl': -2}
  ```
- **VERDICT**: ✅ OPERATIONAL - creating and publishing plans, but WARNING shows p26_ttl=-2 (negative TTL = stale/missing P2.6 data)

### STEP 2: Metrics

**Heat Gate Metrics** (http://127.0.0.1:8056/metrics at 15:31:33 UTC):
```
p26_heat_value 0.008825
p26_bucket{state="COLD"} 1.0
p26_bucket{state="WARM"} 0.0
p26_bucket{state="HOT"} 0.0
p26_stream_lag_ms_count 0.0
p26_proposals_processed_total 0.0          # ❌ ZERO PROCESSING
p26_hash_writes_total 0.0                  # ❌ ZERO OUTPUT
p26_hash_write_fail_total 0.0
p26_enforce_mode 1.0                       # ✅ ENFORCE ACTIVE
```

**CRITICAL FINDINGS**:
- **p26_proposals_processed_total = 0.0**: Heat Gate has processed ZERO proposals despite running for 8+ hours
- **p26_hash_writes_total = 0.0**: Heat Gate has written ZERO calibrated outputs
- **p26_enforce_mode = 1.0**: Service is in ENFORCE mode (not shadow)
- **p26_stream_lag_ms_count = 0.0**: No stream reads detected
- **VERDICT**: ❌ HEAT GATE IS COMPLETELY IDLE - not consuming input stream

### STEP 3: Redis Streams/Keys

**Harvest Proposal Stream** (quantum:stream:harvest.proposal):
```
EXISTS: 1 (exists)
XLEN: 25
Latest 3 entries (timestamps 1769581431, 1769581428, 1769581424):
  plan_id: 8dfbb9ca6040b95a, 7a3c98346b3e1a3f, 6f2a0b06461bd59d
  symbol: BTCUSDT
  action: FULL_CLOSE_PROPOSED
  decision: EXECUTE
  kill_score: 0.8
```
- **VERDICT**: ✅ INPUT STREAM ACTIVE - receiving proposals from publisher

**Calibrated Stream** (quantum:stream:harvest.calibrated):
```
EXISTS: 1 (exists)
XLEN: 12
Latest 3 entries (timestamps 1769581431, 1769581428, 1769581424):
  trace_id: 1468d162-191a-49ec-b210-fe5c0547b773
  plan_id: 8dfbb9ca6040b95a
  symbol: BTCUSDT
  original_action: FULL_CLOSE_PROPOSED
  calibrated_action: FULL_CLOSE_PROPOSED
  heat_value: 0.008825
  heat_bucket: COLD
  mode: enforce
  calibrated: true
  reason: non_full_close_unchanged
```
- **VERDICT**: ⚠️ CALIBRATED STREAM EXISTS BUT STALE (last entry 1769581431 = ~33 hours ago, current time ~1769700693)

**Hash Keys** (quantum:harvest:proposal:*):
```
Found: BTCUSDT, ETHUSDT, SOLUSDT

quantum:harvest:proposal:BTCUSDT:
  ts: 0.038395
  harvest_action: FULL_CLOSE_PROPOSED
  position_side: LONG
  kill_score: 0.758285465877316
  R_net: 7.013614233516237
  new_sl_proposed: 100.2
  reason_codes: harvest_partial_75,profit_lock,kill_score_triggered,regime_flip,ts_drop
  computed_at_utc: 2026-01-29T15:30:15.765694
  last_update_epoch: 1769700655  # ⚠️ 15:30:15 UTC (now ~15:31:33, 78 seconds stale)
```
- **VERDICT**: ✅ HASH KEYS EXIST - Apply Layer reads from here, but data is NOT from Heat Gate (no calibrated fields visible in hash)

### STEP 4: Apply Layer Integration (CRITICAL)

**Code Inspection**:
```python
# /home/qt/quantum_trader/microservices/apply_layer/main.py:501
def get_harvest_proposal(self, symbol: str) -> Optional[Dict[str, Any]]:
    """Read harvest proposal from Redis"""
    try:
        key = f"quantum:harvest:proposal:{symbol}"  # ❌ READS HASH DIRECTLY
        data = self.redis.hgetall(key)
        ...
        "harvest_action": data.get("harvest_action"),  # ❌ NOT READING CALIBRATED_ACTION
```

**Findings**:
- Apply Layer reads `quantum:harvest:proposal:{symbol}` hash keys (line 504)
- Apply Layer does **NOT** read from `quantum:stream:harvest.calibrated`
- Apply Layer extracts `harvest_action` field (line 520), **NOT** `calibrated_action`
- No grep results for "calibrated" in apply_layer code
- **VERDICT**: ❌ INTEGRATION BROKEN - Apply Layer bypasses Heat Gate entirely

**Live Behavior**:
```
Apply Plan Stream (quantum:stream:apply.plan) latest entries:
  plan_id: c499ee105e5431fe (ETHUSDT), 04a1841a05e66f74 (BTCUSDT)
  action: FULL_CLOSE_PROPOSED
  kill_score: 0.761, 0.758
  decision: EXECUTE
  timestamp: 1769700679, 1769700678 (recent, ~15s ago)
```

Apply Layer logs show:
```
15:30:20 ETHUSDT: Plan 3f58da1bdb590b3b published (decision=EXECUTE, steps=1)
15:30:22 [PERMIT_WAIT] BLOCK plan=3f58da1bdb590b3b symbol=ETHUSDT wait_ms=1207 info={'reason': 'missing_p26', 'p26_ttl': -2}
```

**CRITICAL**: Apply Layer log shows `'p26_ttl': -2` (negative TTL) indicating it's looking for P2.6 data but finding stale/expired entries. This confirms the Heat Gate is NOT updating the hash keys Apply Layer reads.

---

## BLOCKERS (FAIL CONDITIONS)

### BLOCKER 1: Heat Gate Consumer Loop Completely Idle
**Why it fails definition**: Violates condition C - Heat Gate is deployed and running but **NOT processing proposals** (0.0 processed, 0.0 hash writes) despite being in ENFORCE mode.

**Evidence**:
```bash
# Metrics at 2026-01-29 15:31:33 UTC
curl -s http://127.0.0.1:8056/metrics | grep p26_proposals_processed_total
# p26_proposals_processed_total 0.0

curl -s http://127.0.0.1:8056/metrics | grep p26_hash_writes_total  
# p26_hash_writes_total 0.0

# Input stream has 25 entries waiting
redis-cli XLEN quantum:stream:harvest.proposal
# 25
```

**Root Cause Hypothesis**: Heat Gate consumer loop is not reading from `quantum:stream:harvest.proposal` stream. Possible causes:
- Consumer group not created
- Stream key name mismatch
- Consumer loop crashed/blocked without logging
- Missing XREADGROUP call in code

**Smallest Patch Suggestion**:
```bash
# Check consumer group existence
redis-cli XINFO GROUPS quantum:stream:harvest.proposal

# If no groups, create manually to test
redis-cli XGROUP CREATE quantum:stream:harvest.proposal p26_gate $ MKSTREAM

# Restart heat gate and monitor logs
systemctl restart quantum-portfolio-heat-gate
journalctl -u quantum-portfolio-heat-gate -f
```

---

### BLOCKER 2: Apply Layer Reads Hash Instead of Calibrated Stream
**Why it fails definition**: Violates condition D - Apply Layer integration exists but **bypasses calibrated output**, reading stale hash keys instead of Heat Gate results.

**Evidence**:
```python
# apply_layer/main.py line 504
key = f"quantum:harvest:proposal:{symbol}"  # Reads hash
data = self.redis.hgetall(key)              # Not reading stream

# Hash data shows last_update_epoch=1769700655 (78s stale)
# Calibrated stream last entry: 1769581431 (~33 hours old)
```

**Integration Mismatch**: Heat Gate writes to `quantum:stream:harvest.calibrated` (12 entries) but Apply Layer reads from `quantum:harvest:proposal:*` hash keys. Even if Heat Gate starts processing, Apply Layer won't see calibrated results.

**Smallest Patch Suggestion**:
```python
# Option A: Heat Gate writes to hash in ENFORCE mode (quick fix)
# In portfolio_heat_gate main loop after calibration:
if enforce_mode:
    hash_key = f"quantum:harvest:proposal:{symbol}"
    self.redis.hset(hash_key, "calibrated_action", calibrated_action)
    self.redis.hset(hash_key, "heat_bucket", bucket)
    self.redis.expire(hash_key, 60)  # TTL to prevent stale reads

# Option B: Apply Layer reads from calibrated stream (architectural fix)
# In apply_layer/main.py replace get_harvest_proposal():
# Read latest entry from quantum:stream:harvest.calibrated for symbol
```

---

### BLOCKER 3: No Live Proof of Heat Gate Calibration Impact
**Why it fails definition**: Violates condition E - Cannot prove calibration is working because Heat Gate has 0 throughput.

**Evidence**:
```
p26_proposals_processed_total 0.0
p26_actions_downgraded_total: <metric not found>  # No downgrades recorded
quantum:stream:harvest.calibrated: 12 entries (oldest 33 hours)
```

**Required Proof**: After fixing Blocker 1, need evidence showing:
- Heat Gate processes proposal with action=PARTIAL_75 in HOT state
- Calibrated output shows calibrated_action=HOLD (downgraded)
- Metric p26_actions_downgraded_total increments

**Smallest Patch Suggestion**:
```bash
# After fixing Blocker 1, inject test proposal
redis-cli HMSET quantum:state:portfolio equity_usd 10000  # Force HOT state
redis-cli XADD quantum:stream:harvest.proposal "*" \
  plan_id TEST_$(date +%s) symbol TESTUSDT \
  action PARTIAL_75 kill_score 0.5 timestamp $(date +%s)

# Wait 2s, then verify
sleep 2
curl -s http://127.0.0.1:8056/metrics | grep p26_proposals_processed_total
redis-cli XREVRANGE quantum:stream:harvest.calibrated + - COUNT 1
```

---

## SUMMARY OF FAILURES

1. **Heat Gate Consumer Idle**: ENFORCE mode but 0 proposals processed (BLOCKER 1)
2. **Integration Bypass**: Apply Layer reads hash, not calibrated stream (BLOCKER 2)
3. **No Calibration Proof**: Cannot verify calibration logic without live processing (BLOCKER 3)
4. **Stale Data**: Calibrated stream last entry 33 hours old, hash keys updated by publisher (not Heat Gate)
5. **Negative P26 TTL**: Apply Layer logs show `p26_ttl=-2`, confirming it expects P2.6 data but finds expired entries

---

## NEXT ACTIONS (PRIORITY ORDER)

1. **URGENT**: Debug why Heat Gate consumer loop is idle (check consumer groups, stream reads, code logs)
2. **CRITICAL**: Fix integration path - Heat Gate must update hash keys OR Apply Layer must read calibrated stream
3. **VERIFY**: After fixes, run live test with injected proposal to prove calibration works
4. **DOCUMENT**: Create proof script showing end-to-end flow (publish → calibrate → apply → execute)

---

**END OF AUDIT REPORT**
