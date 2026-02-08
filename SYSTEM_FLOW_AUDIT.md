# QUANTUM TRADER - FULL SYSTEM FLOW AUDIT
# ===========================================

## CURRENT ARCHITECTURE - WHAT ACTUALLY RUNS

### PHASE 1: SIGNAL GENERATION
**Active Components:**
1. **AI Engine** (microservices/ai_engine/service.py)
   - Generates trading signals (BUY/SELL)
   - Sets: position_size_usd, leverage, stop_loss, take_profit
   - Publishes to: `quantum:stream:trade.intent`
   - **WHO SETS TP/SL:** AI Engine (line 2301-2302)

2. **Trading Bot** (microservices/trading_bot/simple_bot.py)  
   - Fallback signal generator
   - Also sets TP/SL
   - Publishes to same stream

### PHASE 2: INTENT PROCESSING
3. **Intent Bridge** (microservices/intent_bridge/main.py)
   - Reads: `quantum:stream:trade.intent`
   - Passes through TP/SL unchanged
   - Publishes to: `quantum:stream:apply.plan`

### PHASE 3: EXIT PLANNING (OPTIONAL?)
4. **ExitBrain v3.5** (microservices/exitbrain_v3_5/exit_brain.py)
   - **QUESTION:** Does this override TP/SL from AI Engine?
   - Uses: Adaptive Leverage Engine
   - Publishes: Exit plans

### PHASE 4: EXECUTION GATE
5. **Apply Layer** (microservices/apply_layer/main.py)
   - Reads: `quantum:stream:apply.plan`
   - **Stores position with:** stop_loss, take_profit (line ~2619)
   - Calls Binance to open position
   - Stores in: `quantum:position:{symbol}`

### PHASE 5: POSITION MONITORING
6. **Harvest Brain** (microservices/harvest_brain/harvest_brain.py)
   - Scans positions every 5 seconds
   - **NEW:** Has HARD SL TRIGGER (line 487-533)
   - Checks for:
     - SL breach → EMERGENCY_SL_CLOSE
     - Profit targets (R=2/4/6)
     - Regime flip (kill_score >= 0.6)
   - Publishes close intents to: `quantum:stream:apply.plan`

7. **Position Monitor** (microservices/position_monitor/main.py)
   - Alternative monitoring?

### PHASE 6: CLOSE EXECUTION  
8. **Apply Layer** (again)
   - Receives close intents
   - Executes reduce-only orders

---

## KEY QUESTIONS TO ANSWER:

1. **WHO SETS FINAL TP/SL?**
   - AI Engine sets initial values
   - Does ExitBrain override?
   - Apply Layer just stores what it receives?

2. **WHO DECIDES WHEN TO EXIT?**
   - Harvest Brain (profit harvesting + SL trigger)
   - ExitBrain v3.5 (what does this do?)
   - Are they competing?

3. **WHAT IS REDUNDANT?**
   - Multiple exit systems?
   - Multiple monitors?

4. **POSITION SIZE - WHO DECIDES?**
   - AI Engine: RL Sizing Agent (line ~2220)
   - Testnet cap: $1000 (line ~2239)
   - But ZECUSDT was $8,800! → BUG!

---

## CURRENT PROBLEMS IDENTIFIED:

1. **NO POSITION SIZE ENFORCEMENT**
   - ZECUSDT: 36 units × 244 = $8,800 notional
   - Should be max $1000 on testnet
   - **BUG:** Testnet cap not enforced?

2. **SL/TP DIRECTION BUGS** (FIXED for existing positions)
   - SHORT positions had SL below entry (wrong)
   - LONG positions with TP below entry (ZECUSDT)

3. **MULTIPLE EXIT SYSTEMS - UNCLEAR HIERARCHY**
   - Harvest Brain
   - ExitBrain v3.5
   - Position Monitor
   - Which one is "in charge"?

4. **SL TRIGGER WAS MISSING** (NOW FIXED)
   - Harvest Brain didn't check SL until yesterday
   - Now has HARD SL TRIGGER

---

## DESIRED STATE (USER REQUEST):

**"Full autonomi AI skal bestemme alt"**

### CLEAN ARCHITECTURE:
```
[AI Engine] → [Apply Layer] → [Harvest Brain] → [Close]
     ↓              ↓              ↓              
  Size/SL/TP    Execute      Monitor & Close
```

### REMOVE/DISABLE:
- ExitBrain v3.5? (redundant with Harvest Brain?)
- Position Monitor? (redundant with Harvest Brain?)
- Trading Bot fallback? (if AI Engine is autonomous)

### ENSURE:
- AI Engine: Full control over entry (size, leverage, SL, TP)
- Harvest Brain: Full control over exit (dynamic profit taking, emergency SL)
- Apply Layer: Just executes, no logic
- Position size limits: ENFORCED (max $1000 testnet)

---

## NEXT STEPS:

1. **INVESTIGATE:**
   - Why was ZECUSDT $8,800 when cap is $1000?
   - Does ExitBrain override AI Engine's TP/SL?
   - Is Position Monitor active?

2. **FIX:**
   - Enforce position size cap
   - Remove redundant exit systems
   - Ensure clear responsibility chain

3. **VERIFY:**
   - AI Engine has full control
   - No hidden overrides
   - No competing exit logic

