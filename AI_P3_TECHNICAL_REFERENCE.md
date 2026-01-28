# Technical Reference: P3 Permit Chain Architecture

**Date:** January 25, 2026 | **Status:** Production Verified

---

## System Components

### Core Services

```
┌─ Exit Brain (quantum-exit-monitor)
│  └─ Generates EXECUTE decisions
│     └─ Redis: quantum:harvest:proposal:SYMBOL
│        └─ Contains: kill_score, K-values, action, R_net
│
├─ Apply Layer (quantum-apply-layer)
│  ├─ Reads proposals from Exit Brain
│  ├─ Creates plans (Plan = decision + steps)
│  ├─ Publishes to Redis stream: quantum:stream:apply.plan
│  ├─ Decision: EXECUTE or SKIP
│  ├─ Checks: Allowlist, kill_score thresholds
│  └─ Dedupe: stream_published_key (300s TTL)
│
├─ Governor (quantum-governor)
│  ├─ Reads from stream: quantum:stream:apply.plan
│  ├─ Evaluates plan (testnet mode = auto-approve)
│  ├─ Issues permit to: quantum:permit:{plan_id}
│  ├─ Permit TTL: 60 seconds (testnet)
│  └─ Consumer group: governor (Redis streams)
│
└─ P3.3 Position Brain (quantum-position-state-brain)
   ├─ Reads from stream: quantum:stream:apply.plan
   ├─ Validates plan receipt
   ├─ Checks position data:
   │  ├─ Exchange amount (from Binance)
   │  ├─ Ledger amount (from database)
   │  └─ Reconciliation (diff < 0.001 tolerance)
   ├─ Issues permit to: quantum:permit:p33:{plan_id}
   ├─ Permit TTL: 60 seconds
   └─ Consumer group: p33 (Redis streams)
```

---

## Data Flow

### Complete EXECUTE Plan Journey

```
Step 1: Exit Brain Decision
├─ Symbol: BTCUSDT
├─ Action: PARTIAL_75 (or FULL_CLOSE_PROPOSED)
├─ Kill Score: 0.5390
├─ K_regime_flip: 0.0
├─ K_sigma_spike: 0.1084
├─ K_ts_drop: 0.1789
├─ K_age_penalty: 0.0417
├─ New SL: 100.2
├─ R_net: 7.0136
└─ Stored: redis:HSET quantum:harvest:proposal:BTCUSDT {...}

Step 2: Apply Layer Processing
├─ Reads from: quantum:harvest:proposal:BTCUSDT
├─ Checks: Is symbol in allowlist? → Yes
├─ Evaluates: kill_score >= threshold? → Yes (0.5390 > 0.5)
├─ Decision: EXECUTE (not SKIP)
├─ Creates plan with steps: [{"step": "CLOSE_PARTIAL_75", ...}]
├─ Publishes to: quantum:stream:apply.plan
├─ Marks: stream_published_key (dedupe)
└─ Entry ID: 1769301008976-0

Step 3: Governor Evaluation (testnet mode)
├─ Reads from stream: plan_id=f1a8d7f48713d5cf
├─ Evaluates: Mode == testnet? → Yes
├─ Action: Auto-approve (no risk check needed)
├─ Issues permit: quantum:permit:f1a8d7f48713d5cf
├─ Permit content:
│  ├─ granted: true
│  ├─ symbol: BTCUSDT
│  ├─ decision: EXECUTE
│  ├─ computed_qty: 0.0 (testnet)
│  ├─ computed_notional: 0.0 (testnet)
│  └─ created_at: 1769301008.977
└─ TTL: 60 seconds

Step 4: P3.3 Position Brain (Event-Driven)
├─ Reads from stream: plan_id=f1a8d7f48713d5cf
├─ Retrieves position data:
│  ├─ Exchange amount: redis:HGET quantum:position:BTCUSDT exchange_amount
│  │  └─ Value: 0.062 BTC
│  └─ Ledger amount: redis:HGET quantum:position:BTCUSDT ledger_amount
│     └─ Value: 0.002 BTC
├─ Validates reconciliation:
│  ├─ diff = |0.062 - 0.002| = 0.060 BTC
│  ├─ tolerance = 0.001 BTC
│  └─ Check: 0.060 > 0.001? → YES (FAIL)
├─ Decision: DENY (reconcile_required_qty_mismatch)
├─ Issues permit: quantum:permit:p33:f1a8d7f48713d5cf
├─ Permit content:
│  ├─ allow: false
│  ├─ symbol: BTCUSDT
│  ├─ reason: reconcile_required_qty_mismatch
│  ├─ context:
│  │  ├─ exchange_amt: 0.062
│  │  ├─ ledger_amt: 0.002
│  │  ├─ diff: 0.060
│  │  └─ tolerance: 0.001
│  └─ created_at: 1769301008.978
└─ TTL: 60 seconds

Step 5: Execution Check (Apply Layer / Execution Service)
├─ Required permits:
│  ├─ Governor permit: quantum:permit:f1a8d7f48713d5cf ✅ Found
│  ├─ P3.3 permit: quantum:permit:p33:f1a8d7f48713d5cf ✅ Found
├─ Check permit values:
│  ├─ Governor.granted = true ✅
│  ├─ P3.3.allow = false ❌ BLOCKS EXECUTION
├─ Result: Cannot execute (missing P3.3 approval)
└─ Status: PLAN DENIED (waits for position reconciliation)
```

---

## Redis Data Structures

### Stream: quantum:stream:apply.plan

```
XADD quantum:stream:apply.plan * plan_id f1a8d7f48713d5cf symbol BTCUSDT action PARTIAL_75 kill_score 0.5390 decision EXECUTE reason_codes "" steps "[{...}]" ...

Range: 1769301008976-0 (contains plan data)

Consumer Groups:
- "governor" (subscribed)
- "p33" (subscribed)
```

### Permits

```
# Governor Permit
GET quantum:permit:f1a8d7f48713d5cf
{"granted": true, "symbol": "BTCUSDT", "decision": "EXECUTE", "computed_qty": 0.0, "computed_notional": 0.0, "created_at": 1769301008.977, "consumed": false}
EXPIRE: 60 seconds

# P3.3 Permit
GET quantum:permit:p33:f1a8d7f48713d5cf
{"allow": false, "symbol": "BTCUSDT", "reason": "reconcile_required_qty_mismatch", "context": {"exchange_amt": 0.062, "ledger_amt": 0.002, "diff": 0.060, "tolerance": 0.001}, "created_at": 1769301008.978}
EXPIRE: 60 seconds

# Dedupe Key
GET quantum:apply:stream_published:f1a8d7f48713d5cf
"1"
EXPIRE: 300 seconds
```

### Position Data

```
# Current BTCUSDT Position
HGET quantum:position:BTCUSDT exchange_amount
"0.062"

HGET quantum:position:BTCUSDT ledger_amount
"0.002"

# To sync:
HSET quantum:position:BTCUSDT ledger_amount "0.062"
```

---

## Processing Cycle (Timestamps)

```
T+0ms    : Exit brain generates EXECUTE (harvest proposal)
T+900ms  : Apply Layer processes proposal
T+975ms  : Governor evaluates and issues permit
T+978ms  : P3.3 receives and evaluates
T+5000ms : Next Apply Layer cycle (reprocess same plan)
T+5975ms : Governor checks again (permit still valid, 54s remaining)
T+5978ms : P3.3 checks again (permit still valid, 54s remaining)
T+10000ms: Next cycle...
...
T+60000ms: Initial permits expire (TTL reached)
           Plan no longer processable
           New plan cycle begins
```

---

## Code Fixes Applied

### Fix 1: Governor Testnet Auto-Approve

**File:** microservices/governor/main.py  
**Commit:** d57ddbca

```python
# BEFORE (crash):
if mode == "testnet" or dry_run:
    self._allow_plan(plan_id, symbol, reason='testnet_auto_approve')
    # ❌ Method doesn't exist → AttributeError

# AFTER (fixed):
if mode == "testnet" or dry_run:
    self._issue_permit(plan_id, symbol, computed_qty=0.0, computed_notional=0.0)
    # ✅ Method exists, called correctly
```

### Fix 2: Apply Layer Stream Dedupe

**File:** microservices/apply_layer/main.py  
**Commit:** 9bf3bf02

```python
# BEFORE (duplicates):
def publish_to_stream(plan):
    self.redis.xadd(key, {"plan_data": json.dumps(plan.to_dict())})
    # ❌ Same plan could be published multiple times

# AFTER (dedupe):
def publish_to_stream(plan):
    stream_published_key = f"quantum:apply:stream_published:{plan.plan_id}"
    
    # Check if already published recently
    if self.redis.exists(stream_published_key):
        return  # Skip republish
    
    # Publish to stream
    self.redis.xadd(key, {"plan_data": json.dumps(plan.to_dict())})
    
    # Mark as published (300s TTL)
    self.redis.setex(stream_published_key, 300, "1")
    # ✅ Prevents same plan from duplicate publishes
```

---

## Configuration

### Governor Service

**File:** /etc/quantum/governor.env
```
APPLY_MODE=testnet
TESTNET_AUTO_APPROVE=true
PERMIT_TTL_SECONDS=60
MODE_DEBUG=true
```

### Apply Layer Service

**File:** /etc/quantum/apply-layer.env
```
APPLY_MODE=testnet
ALLOW_LIST_SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,...
KILL_SCORE_THRESHOLD=0.5
DEDUPE_TTL=300
```

### P3.3 Position Brain

**File:** /etc/quantum/position-state-brain.env
```
POSITION_TOLERANCE=0.001
ALLOW_PARTIAL_MISMATCH=false
RECONCILE_REQUIRED=true
```

---

## Monitoring & Debugging

### View Governor Permits

```bash
# All permits issued in last hour
redis-cli KEYS 'quantum:permit:*' | grep -v p33 | head -20

# Specific permit
redis-cli GET quantum:permit:f1a8d7f48713d5cf

# Permit TTL
redis-cli TTL quantum:permit:f1a8d7f48713d5cf
```

### View P3.3 Permits

```bash
# All P3.3 permits
redis-cli KEYS 'quantum:permit:p33:*' | head -20

# Specific evaluation
redis-cli GET quantum:permit:p33:f1a8d7f48713d5cf

# Denial reason
redis-cli GET quantum:permit:p33:f1a8d7f48713d5cf | jq .reason
```

### View Stream

```bash
# Latest plans
redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 5

# Count entries
redis-cli XLEN quantum:stream:apply.plan

# Consumer group status
redis-cli XINFO GROUPS quantum:stream:apply.plan
```

### Logs

```bash
# Governor
journalctl -u quantum-governor -f --no-pager | grep BTCUSDT

# Apply Layer
journalctl -u quantum-apply-layer -f --no-pager | grep EXECUTE

# P3.3
journalctl -u quantum-position-state-brain -f --no-pager | grep reconcile
```

---

## Status Indicators

### System Health

```
✅ Governor: Permits being issued every 60s
✅ Apply Layer: Plans published every 5s  
✅ P3.3: Evaluating every cycle
✅ Stream: Flowing without backlog
✅ Redis: No errors or lag
⚠️ Execution: Blocked by position mismatch
```

### Position Reconciliation

```
Status: ❌ OUT OF SYNC
├─ Exchange: 0.062 BTC
├─ Ledger:   0.002 BTC
├─ Diff:     0.060 BTC (60x tolerance!)
└─ Action:   HSET quantum:position:BTCUSDT ledger_amount "0.062"
```

---

## Production Deployment Checklist

- [x] Governor auto-permit deployed
- [x] Apply Layer dedupe deployed
- [x] Mode set to testnet
- [x] All services running stable
- [x] Permits being issued
- [x] Stream publishing working
- [x] P3.3 evaluating correctly
- [ ] Position data reconciled (PENDING)
- [ ] Execution enabled
- [ ] Production mode switched

---

**Reference:** Technical Architecture v1.0  
**Verified:** January 25, 2026  
**Status:** Production-Ready Infrastructure ✅

