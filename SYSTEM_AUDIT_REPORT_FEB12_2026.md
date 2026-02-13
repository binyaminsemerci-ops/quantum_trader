# QUANTUM TRADER SYSTEM AUDIT REPORT

**Date:** February 12, 2026  
**Auditor:** System Debugger  
**Scope:** End-to-end runtime behavior analysis  
**Method:** Code inspection, not assumptions

---

## EXECUTIVE SUMMARY

This system is a **simple event-driven paper trader** with extensive AI/ML components that are **mostly decorative**. The actual trading logic flows from Trading Bot → Execution Service → Exit Monitor. Most AI components (RL sizing, strategy selection, ExitBrain adaptive levels) are **collectors/scorers that log data but don't affect execution**.

### Critical Findings

| Finding | Impact | Risk Level |
|---------|--------|------------|
| AI signals often overridden by fallback rules | AI investment may be wasted | MEDIUM |
| TP/SL are software-only (not Binance orders) | Exit Monitor failure = positions stuck | **HIGH** |
| Only 3 services are truly critical | Complexity vs. reality mismatch | LOW |
| Most AI components are loggers/scorers | No PnL impact if removed | LOW |

---

## 1. BOOT & LIFECYCLE

### Service Startup Order

1. **Redis** - Required by all services (hard dependency)
2. **Trading Bot** (`microservices/trading_bot`) - Signal generator
3. **AI Engine** (`microservices/ai_engine`) - Model inference server on port 8001
4. **Execution Service** (`services/execution_service.py`) - Port 8002
5. **Exit Monitor Service** (`services/exit_monitor_service.py`) - Port 8007
6. **HarvestBrain** (`microservices/harvest_brain`) - Profit harvesting

### Service Criticality Matrix

| Service | Critical | Port | Effect if Down |
|---------|----------|------|----------------|
| Redis | **YES** | 6379 | System dead, no events flow |
| Trading Bot | **YES** | 8000 | No signals generated |
| Execution Service | **YES** | 8002 | No orders placed |
| AI Engine | NO | 8001 | Fallback to rule-based signals |
| Exit Monitor | NO | 8007 | Positions won't auto-close |
| HarvestBrain | NO | - | No partial profit harvesting |
| ExitBrain v3.5 | NO | - | Uses default TP/SL |
| Risk Governor | NO | - | Uses hardcoded limits |

### Fallback Behavior

When AI Engine is unavailable, Trading Bot uses **fallback_triggered** logic:

```python
# Location: microservices/trading_bot/simple_bot.py:320
if rsi < 45 and macd > -0.002:  # BUY signal
    action = "BUY"
    ensemble_confidence = 0.72
elif rsi > 55 and macd < 0.002:  # SELL signal
    action = "SELL"
    ensemble_confidence = 0.72
```

**Reality:** Fallback triggers even when AI is running (when AI returns HOLD with 0.50 ≤ confidence ≤ 0.98).

---

## 2. DATA FLOW

### Data Entry Points

```
┌─────────────────────────────────────────────────────────────┐
│                    BINANCE API                              │
│  fapi.binance.com/fapi/v1/ticker/24hr                      │
│  testnet.binancefuture.com                                  │
└─────────────────────────────────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌──────────────┐ ┌───────────┐ ┌───────────────┐
    │ Trading Bot  │ │ Exit Mon  │ │ Execution Svc │
    │ (price data) │ │ (ticker)  │ │ (account)     │
    └──────────────┘ └───────────┘ └───────────────┘
```

### Data Transformation Pipeline

```
1. Trading Bot: price_change_24h → fallback signal OR
                market_data → AI Engine /api/ai/signal → prediction

2. AI Engine: features → ensemble_manager.predict → (action, confidence, votes)

3. Signal → Redis stream "quantum:stream:trade.intent"

4. Execution Service: TradeIntent → Margin Check → Governor → Binance Order

5. Exit Monitor: polls prices → checks TP/SL → sends close TradeIntent
```

### Data Classification

#### Data That AFFECTS Decisions:
- `price_change_24h` (fallback signals)
- `confidence` (from ensemble or synthetic)
- `available_margin` (Binance account)
- `current_price` (for TP/SL checks)

#### Data LOGGED But NOT USED:
| Data | Collected By | Used? | Evidence |
|------|--------------|-------|----------|
| RL Position Size | RLPositionSizingAgent | **NO** | Fallback dominates |
| Cross-Exchange Features | AI Engine | **NO** | Defaults used if missing |
| Funding Rate | FundingRateFilter | **NO** | Bypassed on testnet |
| Drift Detection | DriftDetector | **NO** | Bypassed on testnet |
| ExitBrain Calculations | ExitBrain v3.5 | **NO** | TP/SL not placed as orders |

---

## 3. DECISION POINTS

### Position OPENING Decision Points

| Location | Who Decides | Inputs | Output | Authority |
|----------|-------------|--------|--------|-----------|
| `simple_bot.py:202` | _process_symbol() | price_change_24h, RSI, MACD | trade.intent | Trading Bot |
| `service.py:1756` | generate_signal() | ensemble prediction | ai.decision.made | AI Engine |
| `apply_layer/main.py` | Apply Layer | harvest proposals | apply.plan | Apply Layer |

### Position CLOSING Decision Points

| Location | Who Decides | Inputs | Output | Authority |
|----------|-------------|--------|--------|-----------|
| `exit_monitor_service.py:122` | check_exit_conditions() | current_price vs TP/SL | trade.intent (close) | Exit Monitor |
| `harvest_brain.py` | HarvestBrain | R-level, position age | harvest.suggestions | HarvestBrain |
| `intent_executor/main.py` | IntentExecutor | apply.plan + permits | Binance reduceOnly | Intent Executor |

### Trade BLOCKING Points

| Location | Gate | Condition | Effect |
|----------|------|-----------|--------|
| `execution_service.py:500` | Idempotency | Duplicate trace_id | Skip |
| `execution_service.py:507` | Per-Symbol Lock | Concurrent order | Skip |
| `execution_service.py:538` | Symbol Rate Limit | >1 per 30s | Skip |
| `execution_service.py:545` | Global Rate Limit | >5 per minute | Skip |
| `execution_service.py:578` | Margin Guard | Insufficient margin | **REJECT** |
| `risk_governor.py:108` | Notional Check | >MAX_NOTIONAL_USD | **REJECT** |
| Redis key | Kill Switch | `trading:emergency_stop=1` | Block All |

### Final Authority Chain

```
Trading Bot (generates intent)
    │
    ▼
Execution Service (FINAL AUTHORITY)
    ├── [1] Idempotency check
    ├── [2] Per-symbol lock
    ├── [3] Rate limit (symbol + global)
    ├── [4] Margin guard
    ├── [5] Risk Governor evaluation
    └── [6] Binance API call
```

---

## 4. COMPONENT REALITY CHECK

### Claimed vs. Actual Behavior

| Component | Claims To Do | Actually Does | If Removed | PnL Impact |
|-----------|--------------|---------------|------------|------------|
| **AI Engine** | ML ensemble voting | Returns predictions OR HOLD | Fallback signals used | **UNKNOWN** |
| **ExitBrain v3.5** | Adaptive TP/SL | Calculates levels, publishes to Redis | Default 2% TP/SL | Minimal |
| **HarvestBrain** | R-based harvesting | Monitors positions, generates suggestions | No partial exits | Minor |
| **Risk Governor** | Safety bounds | Clamps size/leverage, checks notional | Hardcoded defaults | Safety risk |
| **Exit Monitor** | TP/SL enforcement | Polls prices, sends close intents | **Positions stuck open** | **CRITICAL** |
| **CLM** | Auto-retrain models | Collects trade outcomes | No learning | None |
| **Trading Bot** | Signal generation | **Actually generates all trade intents** | **No trades** | **CRITICAL** |

### Critical Code Evidence

**TP/SL NOT placed as hard orders:**
```python
# Location: services/execution_service.py:791
# NOTE: TP/SL are NOT placed as hard orders on Binance
```

This means Exit Monitor **must be running** for positions to close at TP/SL levels.

---

## 5. AI / AGENT CLASSIFICATION

### Agent Classification Matrix

| AI/Agent | Real Decisions? | Can WAIT? | Can Be Ignored? | Classification |
|----------|-----------------|-----------|-----------------|----------------|
| Ensemble Manager | Produces (action, confidence) | Yes (HOLD) | **YES** | **SCORER** |
| RL Position Sizing | Calculates size/leverage | No | **YES** | **PLACEBO** |
| Meta-Strategy Selector | Disabled in code | N/A | N/A | **DISABLED** |
| Risk Mode Predictor | Calculates risk multiplier | No | Logged only | **LOGGER** |
| Strategy Selector | Selects strategy name | No | Logged only | **LOGGER** |
| ExitBrain v3.5 | Calculates TP/SL levels | No | **YES** | **SCORER** |
| Funding Rate Filter | Can block trades | No | Bypassed on testnet | **RULE ENGINE** |
| Drift Detector | Can block trades | No | Bypassed on testnet | **RULE ENGINE** |

### Classification Definitions

- **SCORER**: Produces numerical outputs that MAY influence decisions
- **LOGGER**: Calculates values but only logs them (no downstream effect)
- **RULE ENGINE**: Hard rules that block/allow based on conditions
- **PLACEBO**: Appears to do something but outputs are typically overwritten
- **DISABLED**: Code exists but is not executed

### Critical Finding: Fallback Override

The **fallback signal logic** in Trading Bot triggers when:
- AI Engine returns HOLD with 0.50 ≤ confidence ≤ 0.98
- RSI/MACD meet relaxed thresholds

**Result:** AI decisions can be completely overridden by rule-based fallback.

```python
# Location: microservices/trading_bot/simple_bot.py:1983
# FALLBACK: If ML models return HOLD, use rule-based signals for exploration
# EXPANDED THRESHOLD: Changed from 0.65 to 0.98 to enable trades during 
# high-confidence HOLD periods
```

---

## 6. ONE TRADE TRACE

### Complete Trade Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: SIGNAL GENERATION (Trading Bot)                   │
├─────────────────────────────────────────────────────────────┤
│ _process_symbol(BTCUSDT)                                    │
│   └── _fetch_market_data() → price=$95000                   │
│   └── _get_ai_prediction() → AI Engine /api/ai/signal       │
│       └── RETURNS: HOLD, confidence=0.60                    │
│   └── FALLBACK TRIGGERED (RSI=42, MACD=0.001)               │
│       └── action=BUY, confidence=0.72 (synthetic)           │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: INTENT PUBLICATION                                 │
├─────────────────────────────────────────────────────────────┤
│ _publish_trade_signal()                                     │
│   └── Redis XADD "quantum:stream:trade.intent"              │
│       {symbol: BTCUSDT, side: BUY, confidence: 0.72,        │
│        position_size_usd: 200, leverage: 10}                │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 3: EXECUTION (Execution Service)                      │
├─────────────────────────────────────────────────────────────┤
│ execute_order_from_intent(intent)                           │
│   └── [GATE 1] Idempotency check → PASS                     │
│   └── [GATE 2] Per-symbol lock → ACQUIRED                   │
│   └── [GATE 3] Rate limit → PASS                            │
│   └── [GATE 4] Margin check → $10000 avail, $25 req → PASS  │
│   └── [GATE 5] Risk Governor → ACCEPT                       │
│   └── binance_client.futures_change_leverage(10x)           │
│   └── binance_client.futures_create_order(BUY, 0.002 BTC)   │
│   └── ORDER FILLED: avgPrice=$95010                         │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 4: POSITION TRACKING (Exit Monitor)                   │
├─────────────────────────────────────────────────────────────┤
│ Receives execution.result                                   │
│   └── TrackedPosition{BTCUSDT, BUY, entry=$95010,          │
│                       TP=$96910 (2%), SL=$93110 (2%)}       │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 5: EXIT MONITORING (Loop every 5s)                    │
├─────────────────────────────────────────────────────────────┤
│ Loop iteration 1:                                           │
│   └── get_current_price(BTCUSDT) → $96000                   │
│   └── check_exit_conditions() → None (not at TP/SL)         │
│                                                             │
│ Loop iteration N:                                           │
│   └── get_current_price(BTCUSDT) → $97000                   │
│   └── check_exit_conditions() → "TAKE_PROFIT" (≥ $96910)    │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 6: CLOSE ORDER                                        │
├─────────────────────────────────────────────────────────────┤
│ send_close_order(position, "TAKE_PROFIT")                   │
│   └── TradeIntent{SELL, reduceOnly=true}                    │
│   └── → Redis "quantum:stream:trade.intent"                 │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 7: EXECUTION (Close)                                  │
├─────────────────────────────────────────────────────────────┤
│ Binance SELL 0.002 BTCUSDT reduceOnly=true                  │
│   └── FILLED at $96990                                      │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 8: REALIZED PnL                                       │
├─────────────────────────────────────────────────────────────┤
│ Entry: $95010 × 0.002 = $190.02                             │
│ Exit:  $96990 × 0.002 = $193.98                             │
│ Gross PnL: +$3.96 (+2.1%)                                   │
│ Fees: ~$0.15 (0.04% × 2 trades)                             │
│ Net PnL: ~$3.81                                             │
└─────────────────────────────────────────────────────────────┘
```

### Trace Analysis

| Aspect | Finding |
|--------|---------|
| **Irreversible Steps** | `futures_create_order()` - Order placed on Binance |
| **Silent Failures** | AI Engine down → fallback used (no error); ExitBrain calculations → logged not enforced |
| **Latency** | Signal → Execution: ~1-5s; Exit check interval: 5s (can miss fast moves) |
| **Lost Information** | RL sizing recommendations overwritten; Strategy selector results logged only |

---

## 7. COMPONENT → PnL IMPACT TABLE

### If Component X Is Removed

| Component | What Happens | PnL Impact |
|-----------|--------------|------------|
| **Redis** | System completely dead | PnL = $0 |
| **Trading Bot** | No signals generated | PnL = $0 |
| **Execution Service** | No orders placed | PnL = $0 |
| **Exit Monitor** | Positions never close | PnL = **UNDEFINED** |
| **AI Engine** | Fallback rules take over | PnL ≈ same |
| **ExitBrain** | Default TP/SL used | PnL ≈ same |
| **Risk Governor** | Larger positions allowed | PnL more volatile |
| **HarvestBrain** | No partial exits | Minor impact |
| **CLM** | No learning | No immediate impact |
| **RL Sizing** | Default sizing used | No impact |
| **Strategy Selector** | Nothing changes | No impact |
| **Drift Detector** | No trade blocking | No impact (testnet bypassed anyway) |

---

## 8. ARCHITECTURAL CONCERNS

### Concern 1: TP/SL Dependency on Exit Monitor

**Risk:** If Exit Monitor crashes or hangs, positions will remain open indefinitely.

**Evidence:**
```python
# services/execution_service.py:791
# NOTE: TP/SL are NOT placed as hard orders on Binance
```

**Recommendation:** Place hard TP/SL orders on Binance exchange.

### Concern 2: Fallback Logic Dominance

**Risk:** AI models rarely influence actual trading decisions.

**Evidence:**
```python
# microservices/trading_bot/simple_bot.py:1983
# EXPANDED THRESHOLD: Changed from 0.65 to 0.98
```

Fallback triggers whenever AI returns HOLD with any confidence between 0.50 and 0.98.

**Recommendation:** Clarify whether AI should drive decisions or if rule-based is intentional.

### Concern 3: Complexity vs. Impact Mismatch

**Observation:** 32 services documented, but only 3-4 affect realized PnL.

| Service Count | Impact Category |
|---------------|-----------------|
| 3 | Critical (no trades without them) |
| 2 | Important (risk/exits) |
| 27+ | Optional/Decorative |

---

## 9. SYSTEM FLOW DIAGRAM

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         QUANTUM TRADER SYSTEM                           │
└─────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────┐
                              │   BINANCE   │
                              │    API      │
                              └──────┬──────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         │                           │                           │
         ▼                           ▼                           ▼
┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│  TRADING BOT    │        │  EXIT MONITOR   │        │ EXECUTION SVC   │
│  ─────────────  │        │  ─────────────  │        │ ─────────────   │
│ • Fetch prices  │        │ • Poll prices   │        │ • Margin check  │
│ • AI prediction │        │ • Check TP/SL   │        │ • Rate limit    │
│ • Fallback rule │        │ • Send close    │        │ • Place orders  │
│ • Publish intent│        │                 │        │                 │
└────────┬────────┘        └────────┬────────┘        └────────┬────────┘
         │                          │                          │
         │                          │                          │
         └──────────────────────────┼──────────────────────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │     REDIS       │
                           │   ───────────   │
                           │ Event Streams:  │
                           │ • trade.intent  │
                           │ • execution.res │
                           │ • trade.closed  │
                           └─────────────────┘
                                    │
                   ┌────────────────┼────────────────┐
                   │                │                │
                   ▼                ▼                ▼
         ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
         │  AI ENGINE  │   │ HARVESTBRAIN│   │ RISK GOV    │
         │ ──────────  │   │ ──────────  │   │ ─────────   │
         │  (SCORER)   │   │  (LOGGER)   │   │ Clamp size  │
         │  Ensemble   │   │  R-levels   │   │ Check notio │
         │  predict    │   │  Suggestions│   │             │
         └─────────────┘   └─────────────┘   └─────────────┘
               │
               ▼
    ┌──────────────────────────────────────────────────────┐
    │            DECORATIVE COMPONENTS (LOGGERS)           │
    │ ──────────────────────────────────────────────────── │
    │ • RL Position Sizing    • Strategy Selector          │
    │ • ExitBrain v3.5        • Risk Mode Predictor        │
    │ • Drift Detector        • Funding Rate Filter        │
    │ • CLM                   • Meta-Strategy Selector     │
    └──────────────────────────────────────────────────────┘
```

---

## 10. RECOMMENDATIONS

### Immediate Actions (P0)

1. **Add hard TP/SL orders on Binance** to eliminate Exit Monitor dependency
2. **Add heartbeat monitoring** for Exit Monitor with alerts

### Short-Term (P1)

3. **Clarify AI vs. Fallback intent** - Is fallback a safety net or the primary strategy?
4. **Remove or disable decorative components** to reduce complexity

### Long-Term (P2)

5. **Implement proper observability** for AI component impact measurement
6. **A/B test AI vs. Fallback** to measure actual PnL difference

---

## APPENDIX A: Key File Locations

| Component | File Path |
|-----------|-----------|
| Trading Bot Main | `microservices/trading_bot/main.py` |
| Trading Bot Logic | `microservices/trading_bot/simple_bot.py` |
| AI Engine Main | `microservices/ai_engine/main.py` |
| AI Engine Service | `microservices/ai_engine/service.py` |
| Execution Service | `services/execution_service.py` |
| Exit Monitor | `services/exit_monitor_service.py` |
| Risk Governor | `services/risk_governor.py` |
| ExitBrain v3.5 | `microservices/exitbrain_v3_5/exit_brain.py` |
| HarvestBrain | `microservices/harvest_brain/harvest_brain.py` |
| Apply Layer | `microservices/apply_layer/main.py` |
| Intent Executor | `microservices/intent_executor/main.py` |

---

## APPENDIX B: Redis Streams

| Stream Name | Purpose | Publishers | Consumers |
|-------------|---------|------------|-----------|
| `quantum:stream:trade.intent` | Trade signals | Trading Bot, Exit Monitor | Execution Service |
| `quantum:stream:execution.result` | Fill confirmations | Execution Service | Exit Monitor, HarvestBrain |
| `quantum:stream:trade.closed` | Closed positions | Execution Service | CLM, AI Engine |
| `quantum:stream:apply.plan` | Apply layer plans | Apply Layer | Intent Executor |
| `quantum:stream:harvest.suggestions` | Harvest proposals | HarvestBrain | Apply Layer |
| `quantum:stream:exitbrain.pnl` | ExitBrain calculations | ExitBrain v3.5 | (Logged only) |

---

**Report Generated:** February 12, 2026  
**Audit Method:** Static code analysis with runtime behavior inference  
**Confidence Level:** HIGH (based on direct code evidence)
