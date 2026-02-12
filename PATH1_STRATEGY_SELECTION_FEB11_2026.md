# PATH 1 — HARVEST BRAIN REPAIR STRATEGY

**Context**: AUTHORITY_FREEZE active, BSC deployed, execution pipeline broken  
**Goal**: Restore ONE working execution path: CLOSE decision → Binance order → Confirmation  
**Date**: 2026-02-11  
**Canonical Status**: A1 runtime map completed, PATH 1 repair ready to begin

---

## 🔑 LOCKED TRUTH (From A1 Runtime Map)

**Proven Facts** (immutable, do not re-investigate):

| Component | Status | Evidence |
|-----------|--------|----------|
| `harvest.intent` | ✅ PRODUCING | 3,054 events, lag=0 |
| `intent_executor_harvest` | ✅ CONSUMING | 13 consumers, lag=0 |
| `apply.plan` | ✅ PRODUCING | 10,012 events |
| `apply_layer` | ✅ APPROVING | Logs show "CLOSE allowed" |
| `apply.result` | ❌ **STOPPING** | `executed=False`, `error=kill_score_close_ok` |
| `harvest_brain:execution` | ❌ **DEAD** | LAG **164,515** events (2+ days frozen) |
| `execution_service` | ❌ **STARVED** | Listens to `trade.intent` (WRONG stream) |

**First Hard Stop**: `harvest_brain:execution` consumer (dead since Feb 8)  
**Second Issue**: `execution_service` on wrong input stream

**Latest apply.result Event** (2026-02-11):
```json
{
  "plan_id": "dce754dfd1412d0b",
  "symbol": "ETHUSDT",
  "decision": "SKIP",
  "executed": "False",
  "would_execute": "False",
  "error": "kill_score_close_ok"
}
```

---

## 🎯 STRATEGIC CHOICE (Choose ONE)

### PATH 1A — Resurrect Harvest Brain ⚰️➡️🧠

**Approach**: Revive the dead consumer, preserve architecture

**What We Do**:
1. Restart/fix `harvest_brain:execution` consumer
2. Let it read `apply.result` stream
3. Let it call Binance (or `execution_service`)
4. Set `executed=True` after confirmation

**When This is Right**:
- ✅ You want to preserve existing architecture
- ✅ You believe Harvest Brain was meant to be executor
- ✅ You minimize redesign
- ✅ You trust Harvest Brain codebase

**Risks**:
- ⚠️ Inherits complexity (Harvest Brain is 2000+ lines)
- ⚠️ May hide legacy logic/bugs
- ⚠️ Takes longer to audit (code archaeology required)
- ⚠️ Consumer has been dead 2+ days (unknown root cause)

**Implementation Estimate**: 2-4 hours (find consumer registration, restart, verify)

---

### PATH 1B — Bypass Harvest Brain (RECOMMENDED) 🔀

**Approach**: Remove dead component, restore flow via existing healthy service

**What We Do**:
1. Change `execution_service` subscription: `trade.intent` → `apply.result`
2. Parse `apply.result` events: if `decision=CLOSE` → execute
3. Ignore Harvest Brain entirely (let it stay DEAD/OBSERVER)
4. Execution service becomes sole executor

**When This is Right**:
- ✅ You want **fastest restoration** of execution
- ✅ You tolerate letting Harvest Brain die permanently
- ✅ You want to **reduce system complexity**
- ✅ You prefer simple, auditable execution path
- ✅ You follow BSC pattern (direct, bypass broken pipeline)

**Benefits**:
- ✅ `execution_service` already has ALL order logic (futures_create_order)
- ✅ Service is running and healthy (0 crashes)
- ✅ Minimal code change (1 stream name + event parsing)
- ✅ Easy to audit (single responsibility: execute approved orders)
- ✅ Matches BSC philosophy (direct API, no complex pipeline)

**Risks**:
- ⚠️ Requires small code modification to `execution_service.py`
- ⚠️ Harvest Brain becomes permanently orphaned (acceptable if it stays OBSERVER)
- ⚠️ Requires new audit (but simpler than 1A)

**Implementation Estimate**: 30-60 minutes (change stream, test, verify)

---

## 🧪 BEVISKRAV FOR PATH 1 (Either A or B)

**Minimum Viable Execution (MVE)** - must prove ALL 6:

1. **CLOSE intent generated** (harvest.intent)  
   Evidence: Redis XLEN shows growth

2. **Service consumes intent** (intent_executor OR execution_service)  
   Evidence: Consumer lag = 0

3. **Binance order placed** (futures_create_order called)  
   Evidence: Order ID in logs

4. **Order confirmation received** (Binance API response)  
   Evidence: Order status = FILLED

5. **Position goes to 0** (close successful)  
   Evidence: futures_position_information shows qty=0

6. **System survives restart** (service auto-recovers)  
   Evidence: systemctl restart → consumer resumes within 10s

**NO PnL optimization required**. NO scoring. NO intelligence. Just **existence proof**.

---

## 📊 CURRENT STATE ANALYSIS

### execution_service Investigation

**File**: `/root/quantum_trader/services/execution_service.py` (1,078 lines)

**Key Functions**:
```python
Line 384: async def execute_order(signal: RiskApprovedSignal)
Line 472: async def execute_order_from_intent(intent: TradeIntent)
Line 881: async def order_consumer()  # <-- CURRENTLY LISTENS TO WRONG STREAM
Line 903:     "quantum:stream:trade.intent"  # <-- CHANGE TO apply.result
```

**Current Subscription**:
```python
# Line 882-903 (approximate)
async def order_consumer():
    """Background task: consume approved orders from quantum:stream:trade.intent"""
    # ...
    streams = {"quantum:stream:trade.intent": ">"}
```

**Proposed Change (PATH 1B)**:
```python
async def order_consumer():
    """Background task: consume approved CLOSE orders from quantum:stream:apply.result"""
    # ...
    streams = {"quantum:stream:apply.result": ">"}
    
    # Parse event:
    # if event["decision"] == "CLOSE" and event.get("approved", False):
    #     execute_order(symbol, side, quantity)
```

### apply.result Consumer Groups

**Current Consumers**:
1. `harvest_brain:execution` - **LAG 164,515** (DEAD since Feb 8)
2. `trade_history_logger` - LAG 0 (healthy, but only logs)

**After PATH 1B**:
1. `harvest_brain:execution` - IGNORED (orphaned, eventually remove)
2. `trade_history_logger` - LAG 0 (unchanged)
3. `execution_service_exits` - LAG 0 ✅ **NEW** (executes approved closes)

---

## 🪜 RECOMMENDED: PATH 1B IMPLEMENTATION PLAN

### Phase 1: Code Modification (15 min)

**File**: `/root/quantum_trader/services/execution_service.py`

**Change 1** - Update stream subscription:
```python
# OLD (line ~903):
streams = {"quantum:stream:trade.intent": ">"}

# NEW:
streams = {"quantum:stream:apply.result": ">"}
```

**Change 2** - Update event parsing logic:
```python
# OLD: Expects TradeIntent structure
# NEW: Parse apply.result structure

for stream_name, messages in response.items():
    for message_id, data in messages:
        decision = data.get("decision")
        symbol = data.get("symbol")
        executed = data.get("executed") == "True"  # String from Redis
        
        # Only execute CLOSE decisions not already executed
        if decision == "CLOSE" and not executed:
            # Extract order details from apply.result
            # Execute close order
            # Log execution
```

**Change 3** - Update consumer group name:
```python
# OLD:
group_name = "execution_service"

# NEW:
group_name = "execution_service_exits"  # Clearly indicates exit-only
```

### Phase 2: Dry-Run Test (15 min)

**Method**: Paper trading / Testnet

1. Restart `execution_service`
2. Trigger 1 CLOSE intent (harvest generates it)
3. Watch logs:
   ```bash
   journalctl -u quantum-ex
   
ecution.service -f
   ```
4. Verify:
   - Event consumed from `apply.result`
   - Order placed to Binance
   - Order ID logged
   - Position closed

### Phase 3: Verification (15 min)

**6 BEVISKRAV Checklist**:
- [ ] CLOSE intent in harvest.intent (XLEN grows)
- [ ] execution_service_exits consumer lag = 0
- [ ] Binance order ID in logs
- [ ] Order status = FILLED (Binance API confirmation)
- [ ] Position quantity = 0 (futures_position_information)
- [ ] Service restart → consumer auto-resumes

### Phase 4: Authority Audit (15 min)

**Before granting CONTROLLER authority**, verify:
1. ✅ Only CLOSE orders executed (no entries)
2. ✅ No unintended orders
3. ✅ reduceOnly=True honored (if applicable)
4. ✅ Execution path traceable (Redis events → Binance order)
5. ✅ Fail-open behavior (service crash ≠ panic close)

**If ALL pass**: Update `PNL_AUTHORITY_MAP_CANONICAL_FEB10_2026.md`

---

## 🎯 DECISION POINT

**Choose NOW**:

### Option A: PATH 1A (Resurrect Harvest Brain)
**Pros**: Preserve architecture, minimal redesign  
**Cons**: Slower, complex, audit-heavy  
**Command**: `I choose PATH 1A`

### Option B: PATH 1B (Bypass Harvest Brain) ⭐ RECOMMENDED
**Pros**: Fast, simple, auditable, matches BSC pattern  
**Cons**: Orphans Harvest Brain (acceptable tradeoff)  
**Command**: `I choose PATH 1B`

---

**Next Step**: After you choose, I will provide:
1. Exact code changes (line-by-line diff)
2. Deployment commands
3. Verification checklist
4. Post-deployment audit prompt

---

**Status**: AWAITING STRATEGY SELECTION  
**Recommendation**: PATH 1B (bypass Harvest Brain via execution_service re-wiring)  
**Estimated Time to MVE**: 60 minutes (PATH 1B) vs 2-4 hours (PATH 1A)

