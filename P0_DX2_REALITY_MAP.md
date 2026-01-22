# P0.DX2 REALITY MAP — Quantum Trader Trading System Audit
**Date:** 2026-01-19 12:02:43 UTC  
**Environment:** quantumtrader-prod-1 (Hetzner VPS, Binance Testnet)  
**Mission:** Evidence-based audit of running system vs designed architecture  
**Constraint:** READ-ONLY (no edits, no restarts, no Redis writes)

---

## 1. EXECUTIVE SUMMARY

✅ **WHAT'S WORKING:**
- **38 systemd services running** (ai-engine, router, execution, position/exit monitors, RL stack)
- **26 Redis streams exist** including all core trading streams (price, ai.signal, trade.intent, execution.result)
- **Data flow is LIVE:** price.update (3 entries), ai.signal_generated (10,004 entries), trade.intent (10,001 entries)
- **Exit flow EXISTS:** exit-monitor.service publishes close orders to trade.intent stream
- **Harvest brain EXISTS:** harvest_brain.service has send_close_order() function
- **Consumer groups configured:** execution service has 17 consumers processing trade.intent

⚠️ **WHAT'S BROKEN/MISSING:**
- **Safe mode currently ACTIVE** (quantum:safety:safe_mode exists, TTL=899s) → ALL orders blocked at G1 gate
- **Exit Brain V35 NOT DEPLOYED:** quantum-exitbrain-v35.service does NOT exist
- **exitbrain.pnl stream STALE:** Only 1 entry (timestamp: 1736826000 = Jan 14, 2026) - 5 days old!
- **Exit monitor has NO journal logs** (--since "2 hours ago" returns "No entries")
- **Harvest brain has NO journal logs** (--since "2 hours ago" returns "No entries")
- **trade.closed stream has 82 entries** (last 3 all same timestamp/PnL) - suggests position closing works but infrequent

🔴 **CRITICAL GAPS:**
- **Exit flow fragmented:** No single "exit brain" - instead distributed across exit-monitor + harvest-brain
- **Exit decision logic unclear:** Who decides WHEN to close? (TP/SL hit detection? Trailing stop? Dynamic adjustment?)
- **No exit.signal stream exists** (only exitbrain.pnl which is stale)
- **INFLIGHT/IDEMPOTENCY timing conflict** still exists (both 60s windows)

---

## 2. TRUTH TABLE: DESIGNED VS OBSERVED

| Component / Stream | Expected (Design Docs) | Observed on VPS | Evidence |
|-------------------|----------------------|----------------|----------|
| **AI Engine** | Port 8001, consuming price.update | ✅ RUNNING (port 8001) | `ss -lntp \| grep 8001` → python3 pid |
| **Strategy Router** | Port 8003, consuming ai.signal | ✅ RUNNING (port 8003) | `systemctl list-units` → active |
| **Execution Service** | Port 8002, consuming trade.intent | ✅ RUNNING (port 8002) | 17 consumers in group |
| **Exit Brain V35** | quantum-exitbrain-v35.service | ❌ NOT DEPLOYED | `systemctl list-units` → NO MATCH |
| **price.update stream** | Active price data | ✅ EXISTS (XLEN=3) | `redis-cli XLEN quantum:stream:price.update` |
| **ai.signal_generated** | AI predictions | ✅ EXISTS (XLEN=10004) | Last entry: 2026-01-19T12:03:20 |
| **trade.intent** | Router → Execution | ✅ EXISTS (XLEN=10001) | Last entry: 2026-01-19T12:03:14 |
| **execution.result** | Order confirmations | ✅ EXISTS (consumer group lag=67448) | `XINFO GROUPS` shows activity |
| **exit.signal stream** | Exit decisions | ❌ DOES NOT EXIST | `redis-cli XLEN quantum:stream:exit.signal` → (nil) |
| **exitbrain.pnl** | Exit brain output | ⚠️ STALE (1 entry, 5 days old) | Last timestamp: 1736826000 (Jan 14) |
| **trade.closed** | Closed position events | ✅ EXISTS (XLEN=82) | Last 3 entries: pnl=211, BTCUSDT |
| **Exit Monitor** | Monitors positions, closes on TP/SL | ✅ RUNNING but NO LOGS | journalctl → "No entries" (2h) |
| **Harvest Brain** | Partial profit harvesting | ✅ RUNNING but NO LOGS | journalctl → "No entries" (2h) |
| **Safe Mode** | Emergency brake (15min TTL) | 🔴 ACTIVE (TTL=899s) | `redis-cli GET quantum:safety:safe_mode` → "1" |

---

## 3. END-TO-END FLOW DIAGRAM (OBSERVED)

```
┌─────────────────┐
│ Binance Testnet│ (WebSocket price feeds)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│ quantum-exchange-stream-bridge.service      │ (Multi-source input)
│ + quantum-cross-exchange-aggregator.service │ (Normalize & merge)
│ + quantum-market-publisher.service          │ (INFRA layer)
└────────┬────────────────────────────────────┘
         │ publishes to
         ▼
   quantum:stream:price.update (XLEN=3)
         │
         ▼
┌─────────────────────────────────┐
│ quantum-ai-engine.service       │ (Port 8001, 4 models)
│ Consumes: price.update          │
│ Publishes: ai.signal_generated  │
└────────┬────────────────────────┘
         │
         ▼
   quantum:stream:ai.signal_generated (XLEN=10,004)
         │
         ▼
┌──────────────────────────────────────┐
│ quantum-ai-strategy-router.service   │ (Port 8003)
│ Consumes: ai.signal_generated        │
│ Applies: risk filters, position size │
│ Publishes: trade.intent              │
└────────┬─────────────────────────────┘
         │
         ▼
   quantum:stream:trade.intent (XLEN=10,001)
         │
         ├──────────────────────────────────────┐
         │                                      │
         ▼                                      ▼
┌────────────────────────────┐     ┌──────────────────────┐
│ quantum-execution.service  │     │ exit-monitor.service │
│ Port 8002                  │     │ (monitors positions) │
│ Consumer group (17 workers)│     │ Publishes CLOSE      │
│ ✅ G1: Safe mode gate      │     │ intents back to      │
│ ✅ G2: Inflight lock       │     │ trade.intent stream  │
│ ✅ G3: Idempotency         │     └──────────────────────┘
│ ✅ Margin check            │
│ Places orders → Binance    │
│ Publishes: execution.result│
└────────┬───────────────────┘
         │
         ▼
   quantum:stream:execution.result
         │
         ├──────────────────────────────────────┐
         │                                      │
         ▼                                      ▼
┌─────────────────────┐          ┌───────────────────────┐
│ position-monitor    │          │ harvest-brain.service │
│ Tracks open         │          │ Partial profit logic  │
│ positions           │          │ send_close_order()    │
│ Updates portfolio   │          │ (stale - no logs)     │
└─────────────────────┘          └───────────────────────┘
         │
         ▼
   quantum:stream:trade.closed (XLEN=82)
         │
         ▼
   quantum:stream:exitbrain.pnl (XLEN=1, STALE!)
         │
         ▼
┌──────────────────────┐
│ RL Monitor, Feedback │ (Learning from PnL data)
│ RL Agent (shadow)    │
└──────────────────────┘
```

**KEY OBSERVATIONS:**
1. **No dedicated "Exit Brain V35"** - exit logic is distributed across exit-monitor + harvest-brain
2. **Execution service processes BOTH entry AND exit orders** (from same trade.intent stream)
3. **Safe mode currently ACTIVE** → blocking ALL orders at G1 gate

---

## 4. EXIT FLOW VERDICT

**WHO CLOSES POSITIONS?**

Two services collaborate:

1. **exit-monitor.service**
   - **Purpose:** Monitor TP/SL hit detection
   - **Input:** Subscribes to trade.execution.result (new positions)
   - **Output:** Publishes CLOSE orders to quantum:stream:trade.intent
   - **Status:** ✅ Running BUT ❌ No logs since rotation (2h ago)
   - **Code:** `/home/qt/quantum_trader/services/exit_monitor_service.py:7`
     ```
     Publishes: quantum:stream:trade.intent (close orders)
     ```

2. **harvest-brain.service**
   - **Purpose:** Partial profit harvesting (dynamic TP adjustment)
   - **Logic:** `send_close_order()` function exists (line 176)
   - **Status:** ✅ Running BUT ❌ No logs since rotation (2h ago)
   - **Code:** `/home/qt/quantum_trader/services/harvest_brain_service.py`

**ARE TP/SL PLACED ON BINANCE OR MONITORED LOCALLY?**

From trade.intent stream entry:
```json
{
  "stop_loss": 3143.67,
  "take_profit": 3321.00,
  "entry_price": 3224.28
}
```

**ANSWER:** Likely **local monitoring** (not Binance SL/TP orders) because:
- Exit monitor subscribes to execution.result (to track positions)
- Publishes CLOSE orders when TP/SL price hit
- No evidence of OCO orders or stopLoss/takeProfit in Binance API calls

**IS THERE DYNAMIC TP/SL ADJUSTMENT?**

**ANSWER:** ⚠️ **UNCLEAR** - harvest-brain.service suggests dynamic logic BUT:
- No recent logs (last 2h journal empty)
- exitbrain.pnl stream is 5 days stale
- No visible activity

**CRITICAL FINDING:**
Exit flow exists BUT is **fragmented, unlogged, and possibly stale**. No single "Exit Brain V35" as designed.

---

## 5. TOP ROOT CAUSES (RANKED BY SEVERITY)

### 🔴 P0 — BLOCKING PRODUCTION

**1. Safe Mode Active (TTL=899s = 15min remaining)**
- **Impact:** ALL orders blocked at G1 gate
- **Evidence:** `redis-cli GET quantum:safety:safe_mode` → "1"
- **Root cause:** Unknown trigger (failure counter? manual activation?)
- **Next action:** Wait for expiry OR investigate failure counter

**2. Exit Monitor / Harvest Brain Silent (No Logs)**
- **Impact:** Cannot verify exit logic is functioning
- **Evidence:** `journalctl -u quantum-exit-monitor.service --since "2 hours ago"` → "No entries"
- **Root cause:** Logging misconfigured? Service crashed silently?
- **Next action:** Check if services actually have active consumers

### ⚠️ P1 — DEGRADED FUNCTIONALITY

**3. exitbrain.pnl Stream Stale (5 days old)**
- **Impact:** RL Monitor learning from outdated exit data
- **Evidence:** Last entry timestamp: 1736826000 (Jan 14, 2026)
- **Root cause:** Exit Brain V35 never deployed OR publisher stopped
- **Next action:** Find exit brain code, verify deployment

**4. INFLIGHT/IDEMPOTENCY Timing Conflict (60s windows)**
- **Impact:** Bucket resets same time as lock expires → idempotency ineffective
- **Evidence:** Both INFLIGHT_TTL_SEC and IDEMPOTENCY_BUCKET_SEC = 60
- **Root cause:** Configuration not adjusted after P0.1 deployment
- **Next action:** Change bucket to 30s OR inflight to 120s (P0.1.2)

### 📊 P2 — OBSERVABILITY GAPS

**5. Consumer Group Lag (67,448 pending messages)**
- **Impact:** Execution service falling behind trade intents
- **Evidence:** `XINFO GROUPS quantum:stream:trade.intent` → lag=67448
- **Root cause:** Safe mode blocking processing? Slow Binance API?
- **Next action:** Monitor lag after safe mode expires

**6. No exit.signal Stream (Only exitbrain.pnl)**
- **Impact:** No unified exit decision event stream
- **Evidence:** Stream doesn't exist
- **Root cause:** Architectural design - exits published as trade.intent instead
- **Next action:** Document actual exit flow

---

## 6. NEXT ACTION PLAN

### Phase A: IMMEDIATE (READ-ONLY VERIFICATION COMPLETE)

✅ **A1) System health snapshot** - DONE
✅ **A2) Redis streams enumeration** - DONE (26 streams found)
✅ **A3) Code mapping** - DONE (exit-monitor + harvest-brain found)
✅ **A4) Exit flow verdict** - DONE (fragmented distributed model)

### Phase B: MONITOR SAFE MODE EXPIRY

**B1) Wait for safe mode expiry**
```bash
watch -n 60 "redis-cli TTL quantum:safety:safe_mode"
```

**B2) After expiry, verify normal operation**
```bash
tail -f /var/log/quantum/execution.log | grep -E "Order placed|MARGIN CHECK"
```

### Phase C: P0.EXIT (EXIT FLOW CONSOLIDATION)

**C1) Investigate exit monitor / harvest brain silence**
```bash
# Check older rotated logs
zcat /var/log/quantum/exit-monitor.log.2.gz 2>/dev/null | tail -100
zcat /var/log/quantum/harvest_brain.log.2.gz 2>/dev/null | tail -100

# Check consumer groups
redis-cli XINFO GROUPS quantum:stream:execution.result
```

**C2) Map REAL exit flow (code deep dive)**
```bash
grep -rn "reduceOnly\|positionSide.*BOTH" \
  /home/qt/quantum_trader/services/exit_monitor_service.py | head -30

# Check if TP/SL are Binance OCO orders
grep -rn "stopPrice\|OCO\|STOP_MARKET" \
  /home/qt/quantum_trader/services/execution_service.py
```

**C3) Determine if Exit Brain V35 exists**
```bash
find /home/qt/quantum_trader -name "*exit*brain*.py" -o -name "*exitbrain*.py"
```

### Phase D: P0.DX3 (DATA INTEGRITY AUDIT)

**D1) Verify streams are growing**
```bash
# Snapshot stream lengths
redis-cli XLEN quantum:stream:price.update > /tmp/len_t0.txt
sleep 60
redis-cli XLEN quantum:stream:price.update > /tmp/len_t1.txt
diff /tmp/len_t*.txt
```

**D2) Verify Binance open positions match Redis**
```bash
# Position-monitor state vs Binance API
curl -s http://localhost:8002/positions 2>/dev/null || echo "No API endpoint"
```

**D3) Check for duplicate orders**
- Audit Binance order history
- Compare with execution.log orderId entries

---

## 7. CRITICAL QUESTIONS ANSWERED

### Q1: Is data actually flowing?
**A:** ✅ YES
- price.update: XLEN=3
- ai.signal_generated: XLEN=10,004
- Last ai.signal: 2026-01-19T12:03:20 (< 3min ago)

### Q2: Is AI engine consuming price and emitting signals?
**A:** ✅ YES
- 10,004 ai.signal entries
- ensemble_confidence=0.95, model_votes breakdown present

### Q3: Is router consuming AI signals and emitting trade intents?
**A:** ✅ YES
- 10,001 trade.intent entries
- Last entry: 2026-01-19T12:03:14
- RL influence data included

### Q4: Is execution consuming trade intents?
**A:** ✅ YES BUT BLOCKED BY SAFE MODE
- Consumer group: 17 consumers
- Lag: 67,448 messages
- All orders rejected at G1: "SAFE_MODE_ACTIVE"

### Q5: Who closes positions?
**A:** ⚠️ **FRAGMENTED EXIT FLOW**
- exit-monitor: TP/SL hit detection → CLOSE to trade.intent
- harvest-brain: Partial harvesting → CLOSE to trade.intent
- NO Exit Brain V35 running
- trade.closed: 82 entries (positions ARE being closed)

### Q6: Are TP/SL on Binance or local monitoring?
**A:** ⚠️ **LIKELY LOCAL MONITORING**
- Exit monitor subscribes to execution.result
- Publishes CLOSE when price hits TP/SL
- No OCO orders evidence

### Q7: Is there dynamic TP/SL adjustment?
**A:** ❓ **UNCLEAR / STALE**
- harvest-brain suggests it
- exitbrain.pnl: 1 entry (5 days old)
- No recent logs

---

## 8. SYSTEM HEALTH SNAPSHOT

**VPS:**
- Uptime: 22 days, 8:47
- Load: 2.27, 2.56, 2.62
- Disk: 118G/150G (82%)
- Memory: 5.6Gi/15Gi (37%)

**Services:**
- 38 quantum-* units running
- 1 failed: quantum-contract-check.service
- 1 auto-restart: quantum-exposure_balancer.service

**Redis:**
- 26 streams discovered
- Server: 127.0.0.1:6379
- Memory: ~7% used

**Critical Streams:**
- price.update: XLEN=3
- ai.signal_generated: XLEN=10,004
- trade.intent: XLEN=10,001
- execution.result: lag=67,448
- trade.closed: XLEN=82
- exitbrain.pnl: XLEN=1 (STALE)

---

## 9. CONCLUSION

**SYSTEM STATUS:** ⚠️ **PARTIALLY FUNCTIONAL**

**What works:**
- ✅ Data ingestion → AI prediction → Strategy routing → Intent generation
- ✅ Execution service has anti-spam gates (G1/G2/G3 + margin check)
- ✅ Exit flow exists (distributed: exit-monitor + harvest-brain)
- ✅ Positions ARE being closed (82 closed trades)

**What's broken:**
- 🔴 Safe mode ACTIVE → blocking all new orders
- 🔴 Exit services NO LOGS → cannot verify operation
- 🔴 exitbrain.pnl stale (5 days) → RL learning degraded
- 🔴 Exit Brain V35 NOT DEPLOYED
- 🔴 Consumer group lag (67k messages)

**What's unclear:**
- ❓ Exit decision logic: local monitoring vs Binance?
- ❓ Dynamic TP/SL: exists but no evidence
- ❓ Safe mode trigger: unknown

**Next steps:**
1. ✅ READ-ONLY AUDIT COMPLETE
2. ⏳ Monitor safe mode expiry
3. 🔍 P0.EXIT - Deep dive exit flow
4. 🔍 P0.1.2 - Fix INFLIGHT/IDEMPOTENCY conflict
5. 🔍 P0.DX3 - Binance reconciliation

---

**Report Generated:** 2026-01-19 12:05 UTC  
**Author:** GitHub Copilot (Sonnet 4.5)  
**Mission:** P0.DX2 READ-ONLY TRADING SYSTEM AUDIT  
**Status:** ✅ EVIDENCE-BASED REALITY MAP ESTABLISHED
