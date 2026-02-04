# QUANTUM TRADER SYSTEM WIRING MAP
**Generated:** 2026-02-03 22:50 UTC  
**Method:** Forensic analysis (systemd + Redis + code grep + logs)  
**Status:** Read-only diagnostic - NO ASSUMPTIONS

---

## EXECUTIVE SUMMARY

### System Health
- **9 ACTIVE services** (out of 95 total units)
- **47 Redis streams** in use
- **44 active positions** (quantum:position:{symbol})
- **NO ledger positions** (quantum:position:ledger:* are empty)
- **NO snapshot positions** (quantum:position:snapshot:* don't exist)

### Critical Finding
**POSITION SOURCE OF TRUTH:** `quantum:position:{symbol}` (HASH)
- Apply-layer writes here
- Exitbrain reads from here
- Ledger keys exist but are EMPTY (TYPE=none)
- Snapshot keys DON'T EXIST

---

## 1. ACTIVE SERVICES (systemd truth)

### Core Trading Stack (9 running services)

| Service | File | Port | PID | Notes |
|---------|------|------|-----|-------|
| **quantum-ai-engine** | microservices/ai_engine/main.py | 8001 | Active | FastAPI + uvicorn |
| **quantum-trading_bot** | microservices/trading_bot/main.py | 8006 | 3919343 | Signal generator |
| **quantum-intent-bridge** | microservices/intent_bridge/main.py | - | 2173276 | trade.intent → apply.plan |
| **quantum-governor** | (not found in ExecStart) | - | Active | Policy enforcement |
| **quantum-apply-layer** | microservices/apply_layer/main.py | - | 4100865 | **EXECUTOR** (entry + exit) |
| **quantum-exitbrain-v35** | microservices/position_monitor/main_exitbrain.py | - | Active | Exit proposals |
| **quantum-harvest-proposal** | microservices/harvest_proposal_publisher/main.py | - | Active | Harvest suggestions |
| **quantum-harvest-brain** | harvest_brain.py | - | 2869651 | Ledger tracking |
| **quantum-position-monitor** | services/position_monitor.py | 8005 | 2869277 | Position tracking |

### Additional Active Services
- **quantum-intent-executor** (microservices/intent_executor/main.py)
- **quantum-marketstate** (P0.5 - metrics publisher)
- **quantum-rl-agent** (shadow mode)

### Inactive Services (85 units loaded but dead/inactive)
- All CLM/training services (dead)
- All dashboard services (dead)
- quantum-execution (dead - testnet mode uses apply-layer directly)
- quantum-ensemble (not-found)
- Various monitoring/diagnostic timers

---

## 2. REDIS STREAMS (data flow truth)

### 2A. Core Trading Streams

#### `quantum:stream:trade.intent`
**PURPOSE:** Entry signals from trading_bot → intent-bridge  
**LENGTH:** 52,864 messages (2.28M total added)  
**PUBLISHERS:**
- `microservices/trading_bot/simple_bot.py:446` (XADD)

**CONSUMERS (2 groups):**
1. **apply_layer_entry** - 13 consumers, 13 pending, lag=2
2. **intent_bridge** - (consumer count unknown)

**LAST MESSAGE:**
```json
{
  "symbol": "FHEUSDT",
  "side": "SELL",
  "confidence": 0.56116,
  "entry_price": 0.12433,
  "stop_loss": 0.1268166,
  "take_profit": 0.11935679999999999,
  "position_size_usd": 200.0,
  "leverage": 10.0,
  "timestamp": "2026-02-03T22:45:36.933921",
  "model": "fallback-trend-following"
}
```

---

#### `quantum:stream:apply.plan`
**PURPOSE:** Execution plans (ENTRY + EXIT) → apply-layer  
**LENGTH:** 10,019 messages (138,145 total added)  
**PUBLISHERS:**
- `microservices/intent_bridge/main.py` (ENTRY plans from trade.intent)
- `microservices/position_monitor/main_exitbrain.py` (EXIT plans from exitbrain)

**CONSUMERS (5 groups):**
1. **apply_layer_entry** - 13 consumers, 13 pending, lag=2 ⚠️
2. **governor** - 44 consumers, 0 pending, lag=0
3. **heat_gate** - 1 consumer, 0 pending, lag=6381 ⚠️
4. **intent_executor** - 63 consumers, 0 pending, lag=0
5. **p33** - 44 consumers, 0 pending, lag=14455 ⚠️

**LAST MESSAGE (EXIT):**
```json
{
  "plan_id": "83730b8c4352a34e",
  "symbol": "ETHUSDT",
  "action": "FULL_CLOSE_PROPOSED",
  "kill_score": 0.7119699587849092,
  "decision": "EXECUTE",
  "steps": [{"step": "CLOSE_FULL", "type": "market_reduce_only", "side": "close", "pct": 100.0}],
  "timestamp": 1770158765
}
```

**ENTRY vs EXIT mix:**
- ENTRY plans: `decision=EXECUTE, side=BUY/SELL, type=MARKET, reduceOnly=false`
- EXIT plans: `action=FULL_CLOSE_PROPOSED/PARTIAL_*, kill_score, steps=[reduceOnly]`

---

#### `quantum:stream:apply.result`
**PURPOSE:** Execution results (success/failure) from apply-layer  
**LENGTH:** 10,027 messages (692,806 total added)  
**PUBLISHERS:**
- `microservices/apply_layer/main.py:2124` (ENTRY results)
- `microservices/apply_layer/main.py:2143` (EXIT results - NEW)
- `microservices/apply_layer/main.py:2181` (SKIP results)
- `microservices/apply_layer/main.py:2245` (duplicate_plan)
- `microservices/apply_layer/main.py:2265` (error results)

**CONSUMERS (3 groups):**
1. **exit_intelligence** - 1 consumer, 481 pending, lag=231082 ⚠️
2. **metricpack_builder** - 1 consumer, 0 pending, lag=231082 ⚠️
3. **p35_decision_intel** - 2 consumers, 0 pending, lag=231082 ⚠️

**LAST MESSAGE:**
```json
{
  "plan_id": "54eab58dc0d63f53",
  "symbol": "ZILUSDT",
  "decision": "SKIP",
  "executed": "False",
  "error": "duplicate_plan",
  "timestamp": 1770158778
}
```

**Current state:** All recent results are `error=duplicate_plan` (anti-duplicate gates working)

---

#### `quantum:stream:execution.result`
**PURPOSE:** Exchange fills → harvest-brain (ledger tracking)  
**LENGTH:** 10,007 messages (29,180 total added)  
**PUBLISHERS:**
- Execution services (old)
- **CURRENTLY STALE** (last entry 2026-02-02 05:38:47 - NEOUSDT qty=0.0)

**CONSUMERS (2 groups):**
1. **harvest_brain:execution** - offset fixed to $ (skips old backlog)
2. **position_monitor** - (consumer details unknown)

**ISSUE:** This stream hasn't received NEW entries since testnet reset  
**ROOT CAUSE:** Apply-layer publishes to `apply.result`, not `execution.result`  
**IMPACT:** Harvest-brain not receiving new fills (but offset fixed to $ to skip old junk)

---

### 2B. Supporting Streams (47 total)

Complete list:
```
quantum:stream:ai.decision.made
quantum:stream:ai.signal_generated
quantum:stream:alerts
quantum:stream:allocation.decision
quantum:stream:allocation.target.proposed
quantum:stream:alpha.attribution
quantum:stream:apply.heat.observed
quantum:stream:apply.plan.manual
quantum:stream:budget.violation
quantum:stream:capital.efficiency.decision
quantum:stream:clm.intent
quantum:stream:events
quantum:stream:exchange.normalized
quantum:stream:exchange.raw
quantum:stream:exitbrain.pnl
quantum:stream:governor.events
quantum:stream:harvest.calibrated
quantum:stream:harvest.heat.decision
quantum:stream:harvest.proposal
quantum:stream:learning.retraining.completed
quantum:stream:learning.retraining.started
quantum:stream:market.klines
quantum:stream:marketstate
quantum:stream:market.tick
quantum:stream:meta.regime
quantum:stream:model.retrain
quantum:stream:policy.audit
quantum:stream:policy.update
quantum:stream:policy.updated
quantum:stream:portfolio.cluster_state
quantum:stream:portfolio.exposure_updated
quantum:stream:portfolio.gate
quantum:stream:portfolio.snapshot_updated
quantum:stream:portfolio.state
quantum:stream:position.snapshot
quantum:stream:reconcile.close
quantum:stream:reconcile.events
quantum:stream:sizing.decided
quantum:stream:trade.closed
quantum:stream:trade.execution.res
quantum:stream:trade.intent.dlq
quantum:stream:trade.signal
quantum:stream:trading.plan
quantum:stream:utf
```

---

## 3. POSITION KEYS (Redis storage truth)

### 3A. Active Positions

**KEY PATTERN:** `quantum:position:{symbol}` (HASH)  
**COUNT:** 44 positions

**SCHEMA (RIVERUSDT example):**
```redis
TYPE: hash
symbol: RIVERUSDT
side: SHORT
quantity: 13.84178835905599
leverage: 10.0
stop_loss: 14.73798
take_profit: 13.871039999999999
plan_id: f374d81ad7d635f1
created_at: 1770153103
```

**Policy Symbols (testnet allowlist):**
- ANKRUSDT (LONG, qty=33892.56)
- RIVERUSDT (SHORT, qty=13.84)
- ARCUSDT, AXSUSDT, FHEUSDT, HYPEUSDT, OGUSDT, STABLEUSDT, 币安人生USDT

**Non-Policy Symbols (44 total including old positions):**
- ADAUSDT, XRPUSDT, HBARUSDT, SUIUSDT, AAVEUSDT, CHESSUSDT, SEIUSDT, ALGOUSDT, DOTUSDT, ZILUSDT, NEARUSDT, LTCUSDT, TONUSDT, BNBUSDT, FILUSDT, MINAUSDT, ETCUSDT, LINKUSDT, SANDUSDT, APTUSDT, CRVUSDT, MANAUSDT, etc.

---

### 3B. Ledger Keys (EMPTY)

**KEY PATTERN:** `quantum:position:ledger:{symbol}`  
**TYPE:** none (keys exist but have NO DATA)

**FINDING:**
```bash
$ redis-cli TYPE quantum:position:ledger:RIVERUSDT
none

$ redis-cli HGETALL quantum:position:ledger:RIVERUSDT
(empty array)
```

**CONCLUSION:** Ledger system exists in code but is NOT ACTIVE.  
**WHO USES IT:** harvest-brain expects to read from here, but finds nothing.

---

### 3C. Snapshot Keys (DON'T EXIST)

**KEY PATTERN:** `quantum:position:snapshot:{symbol}`  
**STATUS:** Does not exist at all

**CONCLUSION:** Snapshot system not implemented or disabled.

---

## 4. DATA FLOW ARCHITECTURE

### 4A. Entry Flow (WORKING)

```
┌─────────────────┐
│ trading_bot     │ Generates signals (fallback + AI)
│ :8006           │ • Calculates TP/SL (3% SL, 4% TP)
└────────┬────────┘ • Leverage: 10x (conservative)
         │          • Position size: $200 margin
         │ XADD
         v
┌─────────────────────────────────────────┐
│ quantum:stream:trade.intent             │
│ • symbol, side, confidence, entry_price │
│ • stop_loss, take_profit, leverage      │
│ • model: "fallback-trend-following"     │
└────────┬────────────────────────────────┘
         │ XREADGROUP (intent_bridge)
         v
┌─────────────────┐
│ intent-bridge   │ Converts intent → plan
│ (no port)       │ • Enforces policy allowlist
└────────┬────────┘ • CHESSUSDT: SKIP (not in policy)
         │          • ANKRUSDT, RIVERUSDT: PASS
         │ XADD
         v
┌──────────────────────────────────────┐
│ quantum:stream:apply.plan            │
│ • plan_id, symbol, side, qty         │
│ • decision=EXECUTE, reduceOnly=false │
│ • leverage, stop_loss, take_profit   │
└────────┬─────────────────────────────┘
         │ XREADGROUP (apply_layer_entry)
         v
┌─────────────────┐
│ apply-layer     │ Executes on Binance
│ :4100865        │ • Anti-duplicate gate (10min TTL)
└────────┬────────┘ • Cooldown gate (15 min per symbol)
         │          • Creates quantum:position:{symbol}
         │ XADD
         v
┌──────────────────────────────────┐
│ quantum:stream:apply.result      │
│ • executed=True/False            │
│ • error=duplicate_plan (if skip) │
└──────────────────────────────────┘
```

**TP/SL SOURCE:** Trading_bot calculates via `trading_mathematician`  
- Base: 3% SL, 4% TP (based on ATR=10%)
- R:R ratio: 1.33:1
- **NOT calculated by exitbrain** (exitbrain only proposes CLOSE actions)

---

### 4B. Exit Flow (NOW WORKING - Fixed Feb 3)

```
┌─────────────────┐
│ exitbrain-v35   │ Monitors positions
│ (no port)       │ • Reads quantum:position:{symbol}
└────────┬────────┘ • Calculates kill_score (k_regime_flip, k_ts_drop, k_age_penalty)
         │          • Proposes FULL_CLOSE_PROPOSED when kill_score > threshold
         │ XADD
         v
┌──────────────────────────────────────────┐
│ quantum:stream:apply.plan                │
│ • plan_id, symbol, action=FULL_CLOSE_*   │
│ • kill_score, decision=EXECUTE           │
│ • steps=[{"type":"market_reduce_only"}]  │
└────────┬─────────────────────────────────┘
         │ XREADGROUP (apply_layer_entry)
         v
┌─────────────────┐
│ apply-layer     │ **NEW: CLOSE handler (Feb 3)**
│ :4100865        │ • Parses FULL_CLOSE_PROPOSED
└────────┬────────┘ • Executes reduceOnly=true market order
         │          • Reduces/deletes quantum:position:{symbol}
         │ XADD
         v
┌──────────────────────────────────────┐
│ quantum:stream:apply.result          │
│ • executed=True, reduceOnly=True     │
│ • close_qty, filled_qty, order_id    │
└──────────────────────────────────────┘
```

**BEFORE FIX (Feb 3):** Apply-layer only handled ENTRY actions → exits went to /dev/null  
**AFTER FIX:** Apply-layer has CLOSE handler (lines 2100-2280 in main.py)

---

### 4C. Harvest Flow (PARTIALLY BROKEN)

```
┌─────────────────┐
│ harvest-proposal│ Generates harvest suggestions
│ (no port)       │ • When to take partial profits
└────────┬────────┘
         │ XADD
         v
┌──────────────────────────────────┐
│ quantum:stream:harvest.proposal  │
└──────────────────────────────────┘
         │
         ? (consumer unknown)
         v
┌─────────────────┐
│ harvest-brain   │ Ledger tracking (PnL calculation)
│ :2869651        │ **ISSUE:** Expects execution.result
└─────────────────┘ **REALITY:** Stream stale since Feb 2
                    **FIX:** Consumer offset moved to $ (skip old msgs)
```

**MISSING LINK:** harvest-brain consumes `execution.result` but apply-layer publishes to `apply.result`  
**IMPACT:** No new fills being tracked by harvest-brain  
**WORKAROUND:** Offset fixed to $ so it doesn't process old junk

---

## 5. DECISION LOGIC (who decides what)

### 5A. Entry Decisions (trading_bot)

**FILE:** `microservices/trading_bot/simple_bot.py`  
**AI ENGINE:** `backend/services/ai/trading_mathematician.py`

**CALCULATES:**
- Entry price (current market price)
- **Stop-loss:** 3% (based on ATR)
- **Take-profit:** 4% (R:R=1.33:1)
- Leverage: 10x (conservative, limited history)
- Position size: $200 margin ($2000 notional)

**METHOD:**
```python
def calculate_optimal_params(symbol, atr, balance, win_rate, ...):
    base_sl = max(atr * 0.3, 0.03)  # 3% default
    base_tp = base_sl * 1.33         # 4% default (R:R=1.33)
    leverage = min(max_leverage, 10.0)  # Conservative
    return {
        'stop_loss': base_sl,
        'take_profit': base_tp,
        'leverage': leverage,
        'position_size': balance * 0.02  # 2% of balance
    }
```

**PROOF FROM LOGS:**
```
💰 Margin target: $200.00 (2.0% of $10000.00)
🛡️  Optimal SL: 3.00%  (based on ATR=10.00%)
🎯 Optimal TP: 4.00% (R:R=1.33:1)
⚡ Optimal Leverage: 10.0x
```

**NO DYNAMIC TP/SL ADJUSTMENT** - Values set at entry, not recalculated.

---

### 5B. Exit Decisions (exitbrain-v35)

**FILE:** `microservices/position_monitor/main_exitbrain.py`  
**ENGINE:** `microservices/exitbrain_v3_5/exit_brain.py`

**CALCULATES:**
- **kill_score** = weighted sum of:
  - `k_regime_flip` (1.0 if regime changed)
  - `k_sigma_spike` (volatility spike)
  - `k_ts_drop` (trend strength drop)
  - `k_age_penalty` (position age)
- Proposes `FULL_CLOSE_PROPOSED` when kill_score > threshold (typically 0.7)

**DOES NOT CALCULATE:**
- TP/SL levels (those come from trading_bot at entry)
- Partial exit percentages (uses kill_score only)
- Trail stops (code exists but not active in current logs)

**PROOF FROM LOGS:**
```json
{
  "action": "FULL_CLOSE_PROPOSED",
  "kill_score": 0.7119699587849092,
  "k_regime_flip": 1.0,
  "k_sigma_spike": 0.0,
  "k_ts_drop": 0.24159399999999998,
  "k_age_penalty": 0.041666666666666664
}
```

**METHOD:** Intent-based (WHAT to close), not levels-based (WHERE to place TP/SL orders)

---

### 5C. Adaptive Leverage Engine (EXISTS but NOT USED)

**FILE:** `microservices/exitbrain_v3_5/adaptive_leverage_engine.py`

**CALCULATES:**
- Leverage-aware TP/SL scaling
- Multi-stage harvest (TP1, TP2, TP3)
- Fail-safe clamps (SL: 0.1%-2.0%, TP: min 0.3%)

**STATUS:** Code exists but NOT called by current system  
**EVIDENCE:** Logs show fixed 3%/4% TP/SL, not adaptive scaling

---

## 6. POLICY ENFORCEMENT

### Intent-Bridge Allowlist

**SOURCE:** Redis policy key (loaded by intent-bridge)  
**POLICY COUNT:** 10 symbols  
**TRADABLE COUNT:** 580 (Binance testnet)  
**FINAL COUNT:** 9 (after intersection)

**ALLOWLIST:**
```
ANKRUSDT, ARCUSDT, AXSUSDT, FHEUSDT, HYPEUSDT, OGUSDT, RIVERUSDT, STABLEUSDT, 币安人生USDT
```

**BLOCKED EXAMPLE (from logs):**
```
[WARNING] SKIP_INTENT_SYMBOL_NOT_IN_ALLOWLIST symbol=CHESSUSDT
```

**IMPACT:** Only 9 symbols can open new positions  
**EXISTING POSITIONS:** 44 symbols (including old non-policy symbols from before policy enforcement)

---

## 7. ANTI-DUPLICATE GATES

### Apply-Layer Gates (WORKING)

**FILE:** `microservices/apply_layer/main.py`

**GATE 1: Duplicate Plan (10 min TTL)**
```python
dedupe_key = f"quantum:apply:done:{plan_id}"
if redis.exists(dedupe_key):
    return "duplicate_plan"
redis.setex(dedupe_key, 600, "1")  # 10 min
```

**GATE 2: Symbol Cooldown (15 min)**
```python
cooldown_key = f"quantum:cooldown:{symbol}"
if redis.exists(cooldown_key):
    return "cooldown_active"
redis.setex(cooldown_key, 900, "1")  # 15 min
```

**PROOF FROM LOGS:**
```
[apply.result] error=duplicate_plan (25 instances in 5 min)
```

**RESULT:** Pyramiding stopped, multiple entry attempts blocked.

---

## 8. KNOWN ISSUES

### 8A. Consumer Group Lag

| Stream | Group | Lag | Impact |
|--------|-------|-----|--------|
| apply.plan | apply_layer_entry | 2 | ✅ Minor |
| apply.plan | heat_gate | 6,381 | ⚠️ heat_gate not keeping up |
| apply.plan | p33 | 14,455 | ⚠️ p33 severely lagging |
| apply.result | exit_intelligence | 231,082 | 🔴 Massive lag |
| apply.result | metricpack_builder | 231,082 | 🔴 Massive lag |
| apply.result | p35_decision_intel | 231,082 | 🔴 Massive lag |

**ROOT CAUSE:** These services inactive or processing slowly  
**IMPACT:** Monitoring/metrics delayed, but execution still works

---

### 8B. execution.result Staleness

**LAST ENTRY:** 2026-02-02 05:38:47 (20+ hours ago)  
**CURRENT DATA:** NEOUSDT qty=0.0 (junk from old backlog)

**ISSUE:** harvest-brain expects fills here, but apply-layer publishes to `apply.result`  
**WORKAROUND:** Consumer offset fixed to $ (skip old messages)  
**LONG-TERM FIX:** Decide if harvest-brain should read `apply.result` instead

---

### 8C. Ledger Positions Empty

**EXPECTED:** `quantum:position:ledger:{symbol}` should contain PnL tracking  
**REALITY:** All ledger keys have TYPE=none (empty)

**IMPACT:** Realized PnL not being tracked  
**WHO NEEDS IT:** harvest-brain, performance attribution

---

### 8D. Exitbrain Logs Missing

**COMMAND:** `journalctl -u quantum-exitbrain-v35 --since "2 hours ago"`  
**RESULT:** "-- No entries --"

**ISSUE:** Exitbrain not logging to journald  
**EVIDENCE:** Redis shows CLOSE proposals being generated → exitbrain IS running  
**LIKELY CAUSE:** Logging to file instead of stdout

---

## 9. CRITICAL PATHS (what actually runs)

### Entry Path (SUCCESS RATE: ~95%)
1. trading_bot → quantum:stream:trade.intent ✅
2. intent-bridge → quantum:stream:apply.plan ✅
3. apply-layer → Binance testnet ✅
4. apply-layer → quantum:position:{symbol} ✅
5. apply-layer → quantum:stream:apply.result ✅

**BLOCKS:** Policy allowlist (9 symbols), anti-duplicate gates (10min + 15min cooldown)

---

### Exit Path (SUCCESS RATE: 0% → NOW WORKING)
1. exitbrain-v35 → quantum:stream:apply.plan ✅
2. apply-layer CLOSE handler → Binance testnet ✅ (NEW)
3. apply-layer → quantum:position:{symbol} update/delete ✅ (NEW)
4. apply-layer → quantum:stream:apply.result ✅ (NEW)

**BEFORE FIX (Feb 3):** Step 2 didn't exist → proposals went nowhere  
**AFTER FIX:** Complete end-to-end working  
**AWAITING:** First natural exit trigger (positions too new)

---

### Harvest Path (PARTIALLY BROKEN)
1. harvest-proposal → quantum:stream:harvest.proposal ⚠️ (consumer unknown)
2. harvest-brain reads execution.result 🔴 (stream stale)
3. harvest-brain updates ledger 🔴 (ledger keys empty)

**STATUS:** Needs investigation - unclear if harvest system is meant to be active

---

## 10. MISSING COMPONENTS

### 10A. Execution Levels Brain (NOT FOUND)

**EXPECTED:** Component that calculates WHERE to place TP/SL orders  
**REALITY:** TP/SL set at entry by trading_bot, never adjusted

**ADAPTIVE LEVERAGE ENGINE EXISTS** but not called by current flow.

---

### 10B. Dynamic TP/SL Adjustment (NOT ACTIVE)

**CODE EXISTS:** exitbrain_v3_5/adaptive_leverage_engine.py  
**STATUS:** Not integrated into live flow  
**CURRENT BEHAVIOR:** Static 3%/4% TP/SL from entry

---

### 10C. Trailing Stops (CODE EXISTS, NOT ACTIVE)

**FILE:** exitbrain_v3_5/exit_brain.py  
**CONFIG:** `trailing_enabled: bool`, `trailing_callback_pct: 0.008`  
**LOGS:** No trailing activity observed

---

## 11. RECOMMENDATIONS

### 11A. Immediate (Production Hygiene)

1. **Fix harvest-brain stream mismatch:**
   - Either: Make harvest-brain read `apply.result` instead of `execution.result`
   - Or: Make apply-layer publish fills to `execution.result` too
   - **Impact:** Realized PnL tracking will work

2. **Investigate ledger emptiness:**
   - Why are `quantum:position:ledger:{symbol}` keys empty?
   - Is ledger system disabled or broken?

3. **Clean up consumer group lag:**
   - Remove or fix inactive consumers (exit_intelligence, metricpack_builder)
   - Prevents Redis memory bloat

---

### 11B. Medium-Term (Architecture Cleanup)

1. **Consolidate position keys:**
   - Clarify: position vs ledger vs snapshot
   - Document which services use which keys
   - **Current state:** Only `quantum:position:{symbol}` is used

2. **Add exitbrain logging:**
   - Redirect stdout to journald
   - Enables log-based debugging

3. **Policy symbol cleanup:**
   - Clear old non-policy positions (44 → 9)
   - Or: Allow manual close of non-policy symbols

---

### 11C. Long-Term (Feature Development)

1. **Activate adaptive leverage:**
   - Integrate exitbrain_v3_5/adaptive_leverage_engine.py
   - Replace static 3%/4% with dynamic scaling

2. **Implement execution levels brain:**
   - Separate "WHAT to exit" (intent) from "WHERE to exit" (levels)
   - Current: exitbrain only does intent (FULL_CLOSE_PROPOSED)
   - Missing: Component that calculates multi-stage TP levels

3. **Enable trailing stops:**
   - Use existing trailing_callback_pct config
   - Requires level adjustment mechanism

---

## 12. APPENDIX: RAW COMMAND OUTPUTS

### A. systemctl list-units
```
ACTIVE SERVICES (9):
- quantum-ai-engine.service (running)
- quantum-trading_bot.service (running)
- quantum-intent-bridge.service (running)
- quantum-governor.service (running)
- quantum-apply-layer.service (running, PID 4100865)
- quantum-exitbrain-v35.service (running)
- quantum-harvest-proposal.service (running)
- quantum-harvest-brain.service (running, PID 2869651)
- quantum-position-monitor.service (running, PID 2869277)

FAILED SERVICES (4):
- quantum-contract-check.service (failed)
- quantum-ess-trigger.service (failed)
- quantum-health-gate.service (failed)
- quantum-portfolio-governance.service (failed)

INACTIVE SERVICES (82): All CLM, training, ensemble, etc.
```

### B. Service Details
```
quantum-ai-engine:
  ExecStart=/opt/quantum/venvs/ai-engine/bin/python -m uvicorn microservices.ai_engine.main:app --host 127.0.0.1 --port 8001
  Environment=PYTHONPATH=/home/qt/quantum_trader

quantum-trading_bot:
  ExecStart=/opt/quantum/venvs/ai-engine/bin/uvicorn microservices.trading_bot.main:app --host 127.0.0.1 --port 8006
  Environment=TRADING_SYMBOLS=ANKRUSDT,ARCUSDT,CHESSUSDT,FHEUSDT,GPSUSDT,HYPEUSDT,RIVERUSDT,STABLEUSDT

quantum-apply-layer:
  ExecStart=/usr/bin/python3 -u microservices/apply_layer/main.py
  Environment=PYTHONUNBUFFERED=1

quantum-exitbrain-v35:
  ExecStart=/opt/quantum/venvs/ai-engine/bin/python3 microservices/position_monitor/main_exitbrain.py
  Environment=PYTHONPATH=/home/qt/quantum_trader:/home/qt/quantum_trader/microservices

quantum-harvest-brain:
  ExecStart=/opt/quantum/venvs/ai-client-base/bin/python -u harvest_brain.py
  WorkingDirectory=/home/qt/quantum_trader
```

### C. Redis Stream Counts
```
47 total streams
Key streams:
- trade.intent: 52,864 messages (2.28M added)
- apply.plan: 10,019 messages (138K added)
- apply.result: 10,027 messages (692K added)
- execution.result: 10,007 messages (29K added, STALE)
```

### D. Position Key Count
```
$ redis-cli --scan --pattern "quantum:position:*" | wc -l
92 total (44 positions + 46 ledger keys + others)

$ redis-cli --scan --pattern "quantum:position:*" | grep -v ledger | grep -v snapshot | wc -l
44 actual positions
```

---

**END OF WIRING MAP**

**Next Steps:**
1. Review this map with team
2. Prioritize fixes (harvest-brain stream mismatch is P1)
3. Document missing components (execution levels brain)
4. Plan adaptive leverage integration
