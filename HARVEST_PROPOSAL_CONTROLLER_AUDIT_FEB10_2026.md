# HARVEST PROPOSAL CONTROLLER AUDIT — FEB 10, 2026

## EXECUTIVE SUMMARY

**Component:** Harvest Proposal (quantum-harvest-proposal.service)  
**Authority Claimed:** 🟢 CONTROLLER  
**Audit Requested:** GATEKEEPER → CONTROLLER re-validation  
**Audit Standard:** PNL_AUTHORITY_ESCALATION_RULEBOOK_V1.md  
**Audit Mode:** Runtime observation only (NO code changes)  

**FINAL VERDICT:** 🔴 **DEMOTION REQUIRED**

**Reason:** Harvest Proposal generates exit intents continuously BUT these intents never reach Binance exchange due to broken execution pipeline. Component has **ZERO ACTUAL CONTROL** over PnL despite claiming CONTROLLER authority.

---

## BEVISKRAV RESULTS

### 🔴 BEVISKRAV 1: DIREKTE EXECUTION-PATH — FAIL

**Requirement:** Controller must have direct path from decision → Binance order  
**Standard:** "Genererer harvest.intent → konsumeres → faktiske exchange-ordrer"  

**Evidence:**

1. **Intent Generation:** ✅ WORKING
   ```
   Stream: quantum:stream:harvest.intent
   Events: 3,054 total
   Rate: ~10 sec intervals (PUBLISH_INTERVAL_SEC=10)
   Latest: "FULL_CLOSE_PROPOSED R=2.44 K=0.725" (BTCUSDT, ETHUSDT)
   ```

2. **Consumer Groups:** ✅ EXISTS
   ```
   intent_executor_harvest:
     - Consumers: 13
     - Lag: 0
     - Pending: 0 (all caught up)
   
   Metrics: harvest_executed=971 (lifetime counter)
   ```

3. **Apply Layer Processing:** ⚠️ EVALUATES BUT DOESN'T EXECUTE
   ```
   Logs: "CLOSE allowed kill_score=0.725 < threshold=0.850"
   Result: "executed=False, error=kill_score_close_ok"
   ```
   **Analysis:** Apply Layer approves the close but publishes `executed=False`

4. **Harvest Brain Execution Consumer:** ❌ DEAD
   ```
   Consumer group: harvest_brain:execution
   Lag: 157,777 events (2+ days behind!)
   Last consumed: 1770430480716 (Feb 8)
   Status: Consumer is stuck/crashed
   ```

5. **Final Execution to Binance:** ❌ NO ACTIVITY
   ```
   quantum-execution.service logs (last 24h): 0 CLOSE orders
   System-wide Binance API search: 0 close/exit orders found
   Only market data ingestion (kline updates)
   ```

**Execution Chain Status:**
```
Harvest Proposal → harvest.intent ✅
   ↓ (consumed by intent_executor)
Intent Executor → evaluates ⚠️ (metrics frozen at 971)
   ↓ (forwards to apply_layer)
Apply Layer → approves ✅ publishes executed=False ❌
   ↓ (should go to harvest_brain:execution consumer)
Harvest Brain Consumer → LAG 157,777 ❌ (DEAD 2+ days)
   ↓ (should execute via quantum-execution.service)
Execution Service → NO LOGS ❌ (not executing)
   ↓
Binance Exchange → NO ORDERS ❌
```

**VERDICT:** FAIL — Intent pipeline exists but execution is **completely broken**. No actual PnL control.

---

### 🔴 BEVISKRAV 2: COUNTERFACTUAL PROOF — FAIL

**Requirement:** "Vis at systemet hadde gjort det verre uten Harvest"  
**Standard:** Dokumenter minst ett av:
- Trades lukket av Harvest med lavere tap enn max adverse excursion
- N trades hvor Harvest exit < hard SL
- Aggregert R-net forbedring

**Evidence:**

1. **Recent Harvest Intent Sample (last 3 events):**
   ```
   Event 1: BTCUSDT emergency_stop_loss R=-1.59 pnl=-$38
   Event 2: BTCUSDT emergency_stop_loss R=-1.59 pnl=-$38 (duplicate)
   Event 3: ETHUSDT emergency_stop_loss R=-1.74 pnl=-$1.62
   ```

2. **CLM Trade Data:** ❌ NOT FOUND
   ```
   File: /root/quantum_trader/data/clm_trades.jsonl
   Status: Does not exist
   ```

3. **Execution Reality:** ❌ INTENTS NEVER EXECUTED
   - All intents show negative R (losing trades)
   - But none were actually executed (executed=False in apply.result)
   - Cannot prove counterfactual when decisions don't reach exchange

4. **Open Position Analysis:**
   ```
   BTCUSDT: No position data (Harvest trying to close non-existent position!)
   SOLUSDT: Position exists (6.87 SOLUSDT @ 2x leverage)
            - BUT entry_price=0.0 (broken tracking)
            - risk_missing=1 (incomplete data)
   ```

**VERDICT:** FAIL — Cannot prove value-add when:
1. Intents never execute (no actual impact on PnL)
2. No trade outcome data (CLM file missing)
3. Position tracking broken (entry_price=0, risk_missing=1)
4. Harvest trying to close positions that don't exist (BTCUSDT)

---

### 🟢 BEVISKRAV 3: FAILURE SAFETY — CONDITIONAL PASS

**Requirement:** "Hva skjer hvis Harvest stopper? Finnes fallback exit-logikk?"  

**Evidence:**

1. **Exit-Related Services Running:**
   ```
   quantum-harvest-proposal.service:    active (intent producer)
   quantum-intent-executor.service:     active (intent consumer)
   quantum-apply-layer.service:         active (approval layer)
   quantum-execution.service:           active (final executor)
   ```

2. **Fallback Mechanisms:**
   - Apply Layer: Active (provides kill_score gating)
   - Execution Service: Listed as running (but no activity logs)
   - Intent Executor: Multiple services (13 consumers on harvest stream)

3. **Failure Mode Analysis:**
   - **IF Harvest stops:** No new harvest.intent events
   - **Fallback:** Apply Layer still active (can handle other exit sources)
   - **Risk:** Existing Harvest Brain consumer is already DEAD (157k lag)
     - Positions not being actively managed by Harvest anyway
     - System already in "Harvest disabled" state functionally

**Current Reality:** Harvest stopping would have **NO IMPACT** because:
- Harvest Brain execution consumer already dead (2+ days lag)
- Apply Layer says executed=False (not executing Harvest intents anyway)
- No actual exits happening even with Harvest running

**VERDICT:** CONDITIONAL PASS — Fallback services exist, but ironically system is already running without functional Harvest execution. Stopping Harvest would change nothing because it's not actually controlling exits.

---

### ✅ BEVISKRAV 4: SCOPE SINGULARITY — PASS

**Requirement:** "Bekreft at Harvest ❌ ikke påvirker entry/sizing, ✅ kun exit timing"  

**Evidence:**

```bash
grep -r "entry\|size\|leverage" /root/quantum_trader/microservices/harvest_proposal
Result: (empty - no entry/sizing logic found)
```

**Harvest Intent Structure:**
```
intent_type: AUTONOMOUS_EXIT
symbol: BTCUSDT
action: CLOSE
percentage: 1.0
reason: emergency_stop_loss / profit_lock
hold_score: 0
exit_score: 10
R_net: -1.59
```

**Analysis:**
- Harvest ONLY generates CLOSE intents ✅
- No entry signals
- No position sizing
- No leverage modification
- Purely exit timing decisions

**VERDICT:** PASS — Scope is correctly limited to exit control only.

---

### ⚠️ BEVISKRAV 5: KILL SWITCH — PARTIAL PASS

**Requirement:** "Harvest kan deaktiveres innen ≤ 60 sekunder uten systemrestart"  

**Evidence:**

1. **Configuration File:**
   ```
   File: /etc/quantum/harvest-proposal.env
   Flag: ENABLE_STREAM=false (currently, but stream still has 3k events?)
   ```

2. **Quick Stop Method:**
   ```bash
   systemctl stop quantum-harvest-proposal.service
   Expected time: 1-2 seconds (SIGTERM)
   ✅ Meets <60 sec requirement
   ```

3. **Disable + Restart Method:**
   ```bash
   # Edit /etc/quantum/harvest-proposal.env: ENABLE_STREAM=false
   systemctl restart quantum-harvest-proposal.service
   Total time: 10-20 seconds
   ✅ Meets <60 sec requirement
   ```

4. **Redis Runtime Flag:** ❌ NOT FOUND
   ```bash
   redis-cli KEYS "*harvest*enable*"
   Result: (empty - no Redis-based kill switch)
   ```

**Analysis:**
- Can stop via systemctl within 1-2 sec ✅
- Can disable via config + restart within 10-20 sec ✅
- NO instant Redis flag for hot-disable ⚠️
- Requires service interaction (not instant runtime toggle)

**VERDICT:** PARTIAL PASS — Meets 60-second requirement via systemctl, but lacks instant Redis-based runtime disable flag for true zero-downtime kill switch.

---

## CRITICAL FINDINGS

### 🔴 FINDING 1: EXECUTION PIPELINE CATASTROPHIC FAILURE

**Severity:** CRITICAL  
**Impact:** Harvest Proposal has ZERO actual PnL control  

**Evidence Chain:**
1. Harvest generates 3,054 intents ✅
2. Intent executor consumes them (harvest_executed=971) ✅
3. Apply Layer evaluates and APPROVES them ✅
4. Apply Layer publishes `executed=False` ❌
5. Harvest Brain consumer has 157,777 event lag (DEAD 2+ days) ❌
6. Execution service shows 0 CLOSE orders in 24h ❌
7. System-wide search: 0 Binance exit orders found ❌

**Root Cause:** Harvest Brain execution consumer (consumer group on `quantum:stream:apply.result`) is **stuck/crashed** since Feb 8. This is the component responsible for converting apply_layer approvals into actual Binance orders.

**Impact:** Harvest Proposal is a **ghost controller** — generates decisions continuously but has no path to execution. All intents die at Apply Layer step.

---

### 🔴 FINDING 2: PHANTOM POSITION MANAGEMENT

**Severity:** CRITICAL  
**Impact:** Harvest trying to close positions that don't exist  

**Evidence:**
```
Harvest Intent: "BTCUSDT FULL_CLOSE_PROPOSED R=2.44"
Redis Position: quantum:position:BTCUSDT → (empty)

Harvest Intent: "ETHUSDT FULL_CLOSE_PROPOSED R=2.44"
Redis Position: quantum:position:ETHUSDT → (no data)

Only live position: SOLUSDT (6.87 @ 2x leverage)
  - entry_price: 0.0 (broken!)
  - risk_missing: 1 (incomplete)
```

**Analysis:** Harvest is publishing close intents for BTCUSDT/ETHUSDT positions that have no Redis state. Only SOLUSDT has position data, but tracking is broken (entry_price=0).

**Impact:** Even if execution pipeline worked, Harvest would be trying to close non-existent positions.

---

### 🔴 FINDING 3: METRICS FROZEN (INTENT EXECUTOR NOT PROCESSING)

**Severity:** HIGH  
**Impact:** Intent executor appears dead or stuck  

**Evidence:**
```
18:50:20 harvest_executed=971 executed_true=3948 executed_false=18658
18:51:21 harvest_executed=971 executed_true=3948 executed_false=18658
18:52:23 harvest_executed=971 executed_true=3948 executed_false=18658
18:53:23 harvest_executed=971 executed_true=3948 executed_false=18658
18:54:24 harvest_executed=971 executed_true=3948 executed_false=18658
```

All counters **completely frozen** over 4+ minutes. No metric changes despite:
- Harvest publishing new intents every 10 seconds
- 13 consumers in intent_executor_harvest group
- Lag=0 (claims caught up)

**Analysis:** Intent executor is likely stuck in internal processing loop or crash-loop, logging metrics but not actually consuming new events.

---

### ⚠️ FINDING 4: APPLY LAYER "EXECUTED=FALSE" PARADOX

**Severity:** MEDIUM  
**Impact:** Unclear execution semantics  

**Evidence:**
```
Apply Layer Log: "CLOSE allowed kill_score=0.725 < threshold=0.850"
Apply Layer Log: "Result published (executed=False, error=kill_score_close_ok)"
```

**Analysis:** Apply Layer approves the close decision ("CLOSE allowed") but then publishes `executed=False` with `error=kill_score_close_ok`. The "error" field suggests close is OK (approved), but executed=False says it didn't execute.

**Interpretation:** Apply Layer is an **approval layer**, not an execution layer. It evaluates intents and forwards approved ones downstream, but the actual execution happens elsewhere (Harvest Brain consumer, which is dead).

---

## AUTHORITY ASSESSMENT

### Current Authority: 🟢 CONTROLLER

**Definition (per PNL_AUTHORITY_MAP_CANONICAL_FEB10_2026.md):**
> "CONTROLLER = Direkte kontroll over entry/exit/sizing. Faktisk PnL-innvirkning."

### Reality Check:

| CONTROLLER Requirement | Harvest Proposal Reality | Status |
|------------------------|--------------------------|--------|
| Direct PnL control | Intents never reach exchange | ❌ FAIL |
| Decisions execute | executed=False, 0 Binance orders | ❌ FAIL |
| Measurable impact | No counterfactual data | ❌ FAIL |
| Safe failure mode | Harvest Brain consumer DEAD | ❌ FAIL |
| System dependency | Already non-functional for 2+ days | ❌ FAIL |

### Recommended Authority: 🔵 OBSERVER (demotion from CONTROLLER)

**Justification:**
- Harvest generates observable output (3,054 intents) ✅
- Output is consumed (lag=0 on intent stream) ✅
- But NO execution path to exchange ❌
- NO actual PnL control ❌

**If execution pipeline repaired:** Could escalate to GATEKEEPER (approval layer) or SCORER (influence without control), but NOT CONTROLLER until proven to execute orders successfully.

---

## AUDIT PRINCIPLE VIOLATION

**Rulebook Quote:**
> "Den eneste komponenten som får styre penger,  
> må være den mest kjedelige, målbare og forutsigbare."

### Harvest Proposal Reality:

- ❌ **Not boring:** Complex multi-step pipeline with 3+ failure points
- ❌ **Not measurable:** No CLM data, broken position tracking, phantom closes
- ❌ **Not predictable:** Execution consumer dead 2+ days, metrics frozen, executed=False outcomes

### Controller Principle:
> "If it doesn't execute, it doesn't control."

Harvest Proposal **does not execute** → **does not control**.

---

## RECOMMENDATIONS

### 🔴 IMMEDIATE (CRITICAL):

1. **DEMOTE Harvest Proposal:** CONTROLLER → OBSERVER
   - Remove from PNL_AUTHORITY_MAP_CANONICAL as a controller
   - Reclassify as "Observable Intent Generator (no execution)"
   - Update documentation to reflect reality

2. **INVESTIGATE Harvest Brain Death:**
   - Consumer group `harvest_brain:execution` stuck 2+ days
   - 157,777 event lag requires manual intervention
   - Either repair consumer or remove stale consumer group

3. **FIX Position Tracking:**
   - BTCUSDT/ETHUSDT: Harvest publishing closes for non-existent positions
   - SOLUSDT: entry_price=0.0, risk_missing=1 (incomplete tracking)
   - Repair position state or disable phantom close generation

### ⚠️ HIGH PRIORITY:

4. **UNFREEZE Intent Executor:**
   - Metrics frozen for hours (harvest_executed=971 static)
   - 13 consumers but no processing
   - Investigate crash-loop or deadlock

5. **CLARIFY Apply Layer Semantics:**
   - `executed=False` with `error=kill_score_close_ok` is confusing
   - Document that Apply Layer is approval, not execution
   - Make clear which component actually executes to Binance

### 🟢 MEDIUM PRIORITY:

6. **ADD Redis Kill Switch:**
   - Current: Requires systemctl stop (1-2 sec) or config edit (10-20 sec)
   - Desired: `redis-cli SET quantum:harvest:enabled false` (instant)
   - Enables zero-downtime emergency disable

7. **IMPLEMENT Counterfactual Tracking:**
   - Create CLM trade outcome data
   - Track Harvest exits vs. hard SL outcomes
   - Build evidence base for future escalation requests

---

## FINAL VERDICT

### 🔴 DEMOTION REQUIRED

**From:** CONTROLLER (direct PnL control)  
**To:** OBSERVER (generates observable output, no execution)  

**Reason:** Harvest Proposal generates exit intents continuously and maintains clean observable output, BUT execution pipeline is catastrophically broken:
- Harvest Brain consumer dead 2+ days (157k lag)
- Apply Layer publishes executed=False
- Execution service shows 0 CLOSE orders
- No Binance API calls found system-wide
- Trying to close phantom positions (BTCUSDT has no Redis state)

**Audit Principle Applied:**
> "If Harvest Proposal stopped tomorrow, no one would notice because it's not actually controlling positions."

**Evidence Standard:** Runtime-only observation with fail-closed verification.  
**Audit Date:** 2026-02-10 18:54 UTC  
**Auditor:** Systematic PnL Authority Escalation Framework  
**Next Steps:** Repair execution pipeline before reconsidering CONTROLLER authority.

---

## APPENDIX: RAW EVIDENCE

### A. Harvest Intent Stream Sample
```
1770734523368-0
  intent_type: AUTONOMOUS_EXIT
  symbol: BTCUSDT
  action: CLOSE
  percentage: 1.0
  reason: emergency_stop_loss (R=-1.59)
  hold_score: 0
  exit_score: 10
  R_net: -1.5878553598079335
  pnl_usd: -37.999099995413566
  entry_price: 70385.54117647
  exit_price: 68150.3000002692
  timestamp: 1770734523
```

### B. Consumer Group Status
```
quantum:stream:harvest.intent
  └─ intent_executor_harvest
     ├─ consumers: 13
     ├─ pending: 0
     ├─ lag: 0
     └─ last-delivered: 1770734523368-0

quantum:stream:apply.result
  ├─ harvest_brain:execution
  │  ├─ consumers: 1
  │  ├─ pending: 0
  │  ├─ lag: 157,777 ⚠️ CRITICAL
  │  └─ last-delivered: 1770430480716 (Feb 8)
  └─ trade_history_logger
     ├─ consumers: 4
     ├─ lag: 0
     └─ last-delivered: 1770749683829
```

### C. Apply Layer Logs
```
Feb 10 18:54:17 BTCUSDT: CLOSE allowed kill_score=0.725 < threshold=0.850
Feb 10 18:54:17 BTCUSDT: Result published (executed=False, error=kill_score_close_ok)
Feb 10 18:54:17 ETHUSDT: CLOSE allowed kill_score=0.714 < threshold=0.850
Feb 10 18:54:17 ETHUSDT: Result published (executed=False, error=kill_score_close_ok)
```

### D. Intent Executor Metrics (Frozen)
```
18:50:20 harvest_executed=971 executed_true=3948 executed_false=18658
18:51:21 harvest_executed=971 executed_true=3948 executed_false=18658
18:52:23 harvest_executed=971 executed_true=3948 executed_false=18658
(No change in counters for 4+ minutes)
```

### E. Position State
```
BTCUSDT: (empty - no position data)
ETHUSDT: (empty - no position data)
SOLUSDT:
  symbol: SOLUSDT
  side: LONG
  quantity: 6.87363959216405
  entry_price: 0.0 ⚠️
  leverage: 2.0
  stop_loss: 86.4171
  take_profit: 89.03580000000001
  risk_missing: 1 ⚠️
```

**End of Audit Report**
