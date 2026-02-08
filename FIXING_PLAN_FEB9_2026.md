# QUANTUM TRADER - FIXING PLAN
**Dato:** 9. februar 2026  
**Status:** AWAITING APPROVAL  
**Estimert tid:** 2-3 timer (alle 6 steg)

---

## 🎯 MÅL
Få hele trading pipeline til å fungere:
```
AI Engine → Autonomous Trader → Intent Bridge → Apply Layer → Intent Executor → Binance
   ✅            ✅                 ❌               ⚠️              ❌            ❌
CURRENT        CURRENT           BUG #11        STREAM?        ISOLATED      NO ORDERS

                                    ↓ FIX ALL ↓

   ✅            ✅                 ✅               ✅              ✅            ✅
  AFTER          AFTER            AFTER           AFTER          AFTER        ORDERS!
```

**Success criteria:** Minimum 1 order plassert på Binance innen 30 minutter etter deploy.

---

## 📋 STEG-FOR-STEG PLAN

### ✅ STEG 0: PRE-DEPLOYMENT CHECK (5 min)

**Hva:**
- Check current VPS service status
- Backup current working code
- Note current stream lengths for before/after comparison

**Commands:**
```bash
# Create backup tag
git tag backup-before-bug11-12-fix-$(date +%Y%m%d_%H%M%S)
git push origin --tags

# Check service status
ssh root@46.224.116.254 'systemctl status quantum-* --no-pager | grep Active'

# Record current metrics
ssh root@46.224.116.254 '
redis-cli xlen quantum:stream:ai.signal_generated
redis-cli xlen quantum:stream:trade.intent  
redis-cli xlen quantum:stream:apply.plan
redis-cli xlen quantum:stream:apply.plan.manual
'
```

**Why:** Safety first - kan rollback hvis noe går galt.

---

### 🔥 STEG 1: FIX BUG #11 - INTENT BRIDGE PARSER (30 min)

**Problem:**
Intent Bridge forventer `qty` field, men Autonomous Trader sender `position_usd` + `leverage`.

**Impact:** 99% av intents blir rejected (75 av 76).

**Løsning:** Utvid Intent Bridge parser til å støtte BEGGE formater.

**File:** `microservices/intent_bridge/main.py`

**Changes:**

**1.1 - Find current parse_intent function:**
```bash
grep -n "def parse_intent\|def _parse_intent\|qty = " microservices/intent_bridge/main.py
```

**1.2 - Modify parser to support both formats:**
```python
def _parse_intent_quantity(self, intent_data: dict, symbol: str) -> Optional[dict]:
    """
    Parse quantity from intent - supports BOTH formats:
    - Format 1: Direct 'qty' field (legacy)
    - Format 2: 'position_usd' + 'leverage' (from Autonomous Trader)
    """
    try:
        # Format 1: Direct qty (legacy/manual intents)
        if "qty" in intent_data and intent_data["qty"]:
            qty = float(intent_data["qty"])
            price = float(intent_data.get("price", 0))
            if not price:
                price = self._get_current_price(symbol)
            
            logger.info(f"[PARSE] Format 1 (direct qty): qty={qty}, price={price}")
            return {"qty": qty, "price": price, "leverage": int(intent_data.get("leverage", 1))}
        
        # Format 2: position_usd + leverage (Autonomous Trader)
        elif "position_usd" in intent_data and "leverage" in intent_data:
            position_usd = float(intent_data["position_usd"])
            leverage = float(intent_data["leverage"])
            
            # Get current market price
            price = self._get_current_price(symbol)
            if not price:
                logger.error(f"[PARSE] Cannot fetch price for {symbol}")
                return None
            
            # Calculate quantity: (position_usd * leverage) / price
            # Example: ($300 * 2x) / $71,000 = 0.00845 BTC
            qty = (position_usd * leverage) / price
            
            logger.info(f"[PARSE] Format 2 (position_usd): position_usd={position_usd}, "
                       f"leverage={leverage}, price={price}, calculated_qty={qty}")
            return {"qty": qty, "price": price, "leverage": int(leverage)}
        
        else:
            logger.warning(f"[PARSE] Invalid intent format - missing qty or position_usd: {intent_data}")
            return None
            
    except (ValueError, TypeError, KeyError) as e:
        logger.error(f"[PARSE] Failed to parse quantity: {e}, intent={intent_data}")
        return None
```

**1.3 - Add price fetcher if not exists:**
```python
def _get_current_price(self, symbol: str) -> Optional[float]:
    """Fetch current market price from Redis market data."""
    try:
        # Try Redis cache first
        price_key = f"quantum:market:{symbol}:price"
        price = self.redis_client.get(price_key)
        
        if price:
            return float(price)
        
        # Fallback: Try last known price from Apply Layer
        last_plan = self.redis_client.xrevrange(f"quantum:stream:apply.plan", count=1)
        if last_plan:
            plan_data = last_plan[0][1]
            if plan_data.get(b"symbol", b"").decode() == symbol:
                return float(plan_data.get(b"price", 0))
        
        logger.warning(f"[PRICE] No price found for {symbol}")
        return None
        
    except Exception as e:
        logger.error(f"[PRICE] Error fetching price for {symbol}: {e}")
        return None
```

**1.4 - Update main intent processing:**
```python
async def process_intent(self, intent_id: str, intent_data: dict):
    """Process a single intent from the stream."""
    symbol = intent_data.get("symbol")
    action = intent_data.get("action")
    
    # Step 1: Validate symbol against policy
    if not self._validate_symbol(symbol):
        logger.warning(f"[INTENT] Symbol {symbol} not in allowlist")
        return
    
    logger.info(f"✅ Symbol {symbol} in allowlist, proceeding")
    
    # Step 2: Parse quantity (UPDATED - supports both formats)
    parsed = self._parse_intent_quantity(intent_data, symbol)
    if not parsed:
        logger.warning(f"⚠️  Invalid quantity: {intent_data}")
        return
    
    qty = parsed["qty"]
    price = parsed["price"]
    leverage = parsed["leverage"]
    
    logger.info(f"✅ Parsed intent: {symbol} {action} qty={qty} price={price} leverage={leverage}")
    
    # Step 3: Create execution plan
    plan = {
        "plan_id": str(uuid.uuid4())[:8],
        "symbol": symbol,
        "side": action,
        "qty": qty,
        "price": price,
        "leverage": leverage,
        "stop_loss": self._calculate_sl(price, action, intent_data),
        "take_profit": self._calculate_tp(price, action, intent_data),
        "reduceOnly": intent_data.get("reduceOnly", "false"),
        "source": "intent_bridge",
        "timestamp": int(time.time())
    }
    
    # Step 4: Publish to apply.plan stream
    await self.redis_client.xadd("quantum:stream:apply.plan", plan)
    logger.info(f"📤 Published plan: {plan['plan_id']} | {symbol} {action} qty={qty}")
```

**Expected outcome:**
- Parser nå støtter begge formater
- Success rate: 99% → 100%
- Log vil vise "Format 2 (position_usd)" for intents fra Autonomous Trader

**Risk:** LOW - backwards compatible (støtter fortsatt gamle formater)

---

### 🔍 STEG 2: INVESTIGATE APPLY LAYER OUTPUT (15 min)

**Problem:**
Apply Layer consumer fra `apply.plan` men Intent Executor leser `apply.plan.manual`.
Ikke klart hvor Apply Layer publiserer ENTRY plans.

**Hva vi skal finne:**

**2.1 - Check Apply Layer source code:**
```bash
# Find output stream configuration
grep -n "xadd\|publish\|OUTPUT_STREAM" microservices/apply_layer/*.py

# Find hvor ENTRY plans håndteres
grep -n "ENTRY\|entry\|ACTION.*OPEN" microservices/apply_layer/*.py

# Check environment config
cat /home/qt/quantum_trader/.env | grep APPLY_LAYER
ssh root@46.224.116.254 'cat /etc/quantum/apply-layer.env'
```

**2.2 - Check actual behavior in logs:**
```bash
ssh root@46.224.116.254 '
# Find the one successful ENTRY plan (21:55:02)
journalctl -u quantum-apply-layer --since "21:55:00" --until "21:55:05" \
  | grep -E "ENTRY|2e99efa9|xadd|publish"
'
```

**2.3 - Check stream consumer groups:**
```bash
ssh root@46.224.116.254 '
redis-cli xinfo groups quantum:stream:apply.plan
redis-cli xinfo groups quantum:stream:apply.plan.manual
'
```

**Expected findings:**
- Apply Layer kan ha conditional logic: CLOSE → auto, ENTRY → manual
- Eller Apply Layer kan ha bug som ikke publiserer ENTRY
- Eller `.manual` stream er legacy/unused

**Possible outcomes:**

**A) Apply Layer publishes ENTRY to .manual (intended design):**
→ Går til STEG 3A: Fix Intent Executor config

**B) Apply Layer has bug - doesn't publish ENTRY:**
→ Fix Apply Layer output logic

**C) .manual stream is legacy/unused:**
→ Går til STEG 3A: Fix Intent Executor config

---

### 🔧 STEG 3A: FIX INTENT EXECUTOR STREAM CONFIG (5 min)

**Scenario:** Hvis Apply Layer publishes til `apply.plan` (ikke `.manual`)

**Problem:** Intent Executor konfigurert til å lese feil stream.

**File:** `/etc/quantum/intent-executor.env` på VPS

**Change:**
```bash
# Before:
INTENT_EXECUTOR_MANUAL_STREAM=quantum:stream:apply.plan.manual

# After:
INTENT_EXECUTOR_MANUAL_STREAM=quantum:stream:apply.plan
```

**Deploy:**
```bash
ssh root@46.224.116.254 '
# Backup current config
cp /etc/quantum/intent-executor.env /etc/quantum/intent-executor.env.backup

# Update config
sed -i "s/apply.plan.manual/apply.plan/" /etc/quantum/intent-executor.env

# Restart service
systemctl restart quantum-intent-executor

# Verify
sleep 2
systemctl status quantum-intent-executor
journalctl -u quantum-intent-executor -n 20
'
```

**Expected outcome:**
- Intent Executor nå leser fra riktig stream
- Vil få 10,002 messages immediately (backlog)
- Log vil vise "Processing plan" messages

**Risk:** LOW - config change only

---

### 🔧 STEG 3B: FIX APPLY LAYER OUTPUT (20 min)

**Scenario:** Hvis Apply Layer har bug og ikke publiserer ENTRY plans

**Problem:** Apply Layer logic error - skipper ENTRY plans.

**File:** `microservices/apply_layer/apply_layer.py`

**Investigation first:**
```bash
# Find where plans are published
grep -n "xadd\|publish" microservices/apply_layer/apply_layer.py

# Find ENTRY handling logic
grep -n "if.*ENTRY\|if.*entry\|if action ==" microservices/apply_layer/apply_layer.py
```

**Likely fix:**
```python
# Ensure ALL plans are published (ENTRY and CLOSE)
async def publish_plan(self, plan: dict):
    """Publish plan to output stream for Intent Executor."""
    output_stream = "quantum:stream:apply.plan"
    
    # OLD (bug): Only published CLOSE plans?
    # if plan.get("action") == "CLOSE":
    #     await self.redis_client.xadd(output_stream, plan)
    
    # NEW: Publish ALL plans
    await self.redis_client.xadd(output_stream, plan)
    logger.info(f"📤 Published {plan.get('action')} plan: {plan.get('plan_id')}")
```

**Deploy:**
```bash
git add microservices/apply_layer/apply_layer.py
git commit -m "fix: Apply Layer now publishes ENTRY plans"
git push origin main

ssh root@46.224.116.254 '
cd /home/qt/quantum_trader
git pull origin main
systemctl restart quantum-apply-layer
sleep 2
systemctl status quantum-apply-layer
'
```

**Expected outcome:**
- Apply Layer nå publiserer ENTRY plans
- Output stream vil få nye ENTRY messages

**Risk:** MEDIUM - depends på current logic

---

### 💰 STEG 4: FIX BUG #12 - NOTIONAL CALCULATION (15 min)

**Problem:**
Orders for små - $70.61 notional < $100.00 minimum.
To issues:
1. Intent Bridge calculation feil (leverage i feil plass)
2. ALLOW_UPSIZE=false (kan ikke auto-adjust)

**Løsninger (pick one eller begge):**

**OPTION A: Fix Calculation (Intent Bridge) - ALREADY DONE IN STEP 1**

Step 1.2 already fixes this:
```python
# OLD (wrong):
qty = position_usd / (price * leverage)  # 300 / (71000 * 2) = 0.00211 → $150 ✓ but wrong logic

# NEW (correct):
qty = (position_usd * leverage) / price  # (300 * 2) / 71000 = 0.00845 → $600 ✓✓✓
```

Med Bug #11 fix vil qty være korrekt og notional vil være > $100.

**OPTION B: Enable ALLOW_UPSIZE (Intent Executor) - RECOMMENDED AS SAFETY**

Selv med fix, vi bør enable auto-upsize som safety net.

**File:** `/etc/quantum/intent-executor.env`

**Change:**
```bash
ssh root@46.224.116.254 '
# Add ALLOW_UPSIZE to config
echo "ALLOW_UPSIZE=true" >> /etc/quantum/intent-executor.env

# Restart service
systemctl restart quantum-intent-executor

# Verify
sleep 2
systemctl status quantum-intent-executor
'
```

**Expected outcome:**
- Hvis qty for noen reason er under min notional, executor vil auto-adjust
- Safety net for edge cases

**Risk:** LOW - safety improvement

---

### 🚀 STEG 5: DEPLOY ALL FIXES (20 min)

**Deployment sequence:**

**5.1 - Commit all local changes:**
```bash
git add microservices/intent_bridge/main.py
git add microservices/apply_layer/apply_layer.py  # hvis endret
git commit -m "fix(bug-11-12): Intent Bridge multi-format parser + notional fix

- Intent Bridge now supports both qty and position_usd+leverage formats
- Fixes 99% rejection rate (Bug #11)
- Correct leverage calculation fixes notional issue (Bug #12)
- Backwards compatible with legacy intent formats
"

git push origin main
```

**5.2 - Deploy to VPS:**
```bash
ssh root@46.224.116.254 '
cd /home/qt/quantum_trader

# Pull latest code
git pull origin main

# Stop affected services
systemctl stop quantum-intent-bridge
systemctl stop quantum-intent-executor
systemctl stop quantum-apply-layer  # hvis endret

# Apply config changes
# (already done in steps 3 and 4)

# Start services in correct order
systemctl start quantum-intent-bridge
sleep 2
systemctl start quantum-apply-layer
sleep 2
systemctl start quantum-intent-executor

# Check all services
systemctl status quantum-intent-bridge --no-pager
systemctl status quantum-apply-layer --no-pager
systemctl status quantum-intent-executor --no-pager
'
```

**5.3 - Verify no errors:**
```bash
ssh root@46.224.116.254 '
# Check recent logs for errors
journalctl -u quantum-intent-bridge -n 50 --no-pager | grep -i error
journalctl -u quantum-apply-layer -n 50 --no-pager | grep -i error
journalctl -u quantum-intent-executor -n 50 --no-pager | grep -i error
'
```

**Expected outcome:**
- All services running
- No error logs
- Ready for verification

**Risk:** MEDIUM - multiple service restarts

---

### ✅ STEG 6: VERIFY FULL PIPELINE (30 min)

**Verification checklist:**

**6.1 - Verify Intent Bridge parsing (Bug #11 fix):**
```bash
ssh root@46.224.116.254 '
# Wait for new intents
sleep 60

# Check success rate
journalctl -u quantum-intent-bridge --since "5 minutes ago" | grep -E "Published plan|Invalid quantity"

# Should see:
# - "Format 2 (position_usd)" logs
# - "Published plan" messages (not "Invalid quantity")
# - Success rate should be ~100%
'
```

**6.2 - Verify Apply Layer forwarding:**
```bash
ssh root@46.224.116.254 '
# Check stream growth
redis-cli xlen quantum:stream:apply.plan

# Should be increasing (new ENTRY plans arriving)
'
```

**6.3 - Verify Intent Executor receiving plans:**
```bash
ssh root@46.224.116.254 '
# Check Intent Executor logs
journalctl -u quantum-intent-executor --since "3 minutes ago" | grep "Processing plan"

# Should see:
# - "▶️  Processing plan: XXXX | BTCUSDT ..."
# - "✅ P3.3 permit granted"
# - "ORDER_SUBMITTED" (not "ORDER_BLOCKED")
'
```

**6.4 - Verify Binance order placement:**
```bash
ssh root@46.224.116.254 '
# Check for successful orders
journalctl -u quantum-intent-executor --since "5 minutes ago" | grep -E "ORDER_SUBMITTED|POSITION_OPENED"

# Check Redis positions
redis-cli keys "quantum:position:*"
redis-cli hgetall "quantum:position:BTCUSDT"  # or whatever opened

# Should see:
# - At least 1 "ORDER_SUBMITTED" log
# - Position hash with entry details
'
```

**6.5 - Verify notional values (Bug #12 fix):**
```bash
ssh root@46.224.116.254 '
# Check order sizes
journalctl -u quantum-intent-executor --since "5 minutes ago" | grep "notional"

# Should see:
# - notional values > $100 (not $70)
# - NO "ORDER_BLOCKED" due to notional
# - Or "ALLOW_UPSIZE adjusted qty" if needed
'
```

**6.6 - Full pipeline metrics:**
```bash
ssh root@46.224.116.254 '
echo "=== PIPELINE METRICS ==="
echo "AI Signals: $(redis-cli xlen quantum:stream:ai.signal_generated)"
echo "Trade Intents: $(redis-cli xlen quantum:stream:trade.intent)"
echo "Apply Plans: $(redis-cli xlen quantum:stream:apply.plan)"
echo "Positions: $(redis-cli keys "quantum:position:*" | wc -l)"
echo ""
echo "=== RECENT ACTIVITY ==="
journalctl -u quantum-intent-executor --since "2 minutes ago" | tail -20
'
```

**SUCCESS CRITERIA:**
✅ Intent Bridge: 0 "Invalid quantity" errors  
✅ Intent Executor: At least 1 "ORDER_SUBMITTED" log  
✅ Binance: At least 1 position opened  
✅ Notional: All orders > $100  

**FAILURE CRITERIA:**
❌ Still seeing "Invalid quantity" (Bug #11 not fixed)  
❌ Intent Executor silent (still isolated)  
❌ "ORDER_BLOCKED" due to notional (Bug #12 not fixed)  

---

## ⏱️ TIMELINE

| Steg | Task | Time | Cumulative |
|------|------|------|------------|
| 0 | Pre-deployment check | 5 min | 5 min |
| 1 | Fix Bug #11 (Intent Bridge) | 30 min | 35 min |
| 2 | Investigate Apply Layer | 15 min | 50 min |
| 3 | Fix stream config/logic | 5-20 min | 70 min |
| 4 | Fix Bug #12 (ALLOW_UPSIZE) | 15 min | 85 min |
| 5 | Deploy all fixes | 20 min | 105 min |
| 6 | Verify pipeline | 30 min | **135 min** |

**Total:** ~2 timer 15 minutter (2.5 timer med buffer)

---

## 🎲 RISK ASSESSMENT

**LOW RISK:**
- Steg 0: Backup only ✅
- Steg 1: Backwards compatible parser ✅
- Steg 3A: Config change only ✅
- Steg 4: Safety feature enable ✅

**MEDIUM RISK:**
- Steg 3B: Logic change (if needed) ⚠️
- Steg 5: Multiple service restarts ⚠️

**MITIGATION:**
- Git tag before start (can rollback)
- Config backups before changes
- Services restart one-by-one (not all at once)
- Verification at each step

**ROLLBACK PLAN:**
```bash
# If something goes wrong:
ssh root@46.224.116.254 '
cd /home/qt/quantum_trader
git checkout backup-before-bug11-12-fix-XXXXXX
systemctl restart quantum-*
'
```

---

## 📊 EXPECTED OUTCOME

**Before fixes:**
```
AI Engine (8,401 signals) → Autonomous Trader (10,010 intents)
  → Intent Bridge (1 plan / 75 rejected)
  → Apply Layer (processing but isolated output)
  → Intent Executor (0 plans received)
  → Binance (0 orders)

Pipeline success rate: 0.00%
```

**After fixes:**
```
AI Engine (signals) → Autonomous Trader (intents)
  → Intent Bridge (~100% success)
  → Apply Layer (forwarding all plans)
  → Intent Executor (processing all plans)
  → Binance (PLACING ORDERS! 🎉)

Pipeline success rate: >90% (some legitimt rejected by risk limits)
```

**Within 30 minutes:**
- At least 1-3 positions opened on Binance
- Logs showing normal trading activity
- No "Invalid quantity" errors
- No "ORDER_BLOCKED" due to notional

---

## ❓ QUESTIONS BEFORE WE START

1. **Godkjenner du denne planen?** Noen endringer du vil ha?

2. **Timing:** Kan vi starte nå? Eller skal vi vente til en bestemt tid?

3. **Risk tolerance:** OK med 2-3 minutters downtime under service restarts?

4. **Monitoring:** Vil du følge med live under verification (Step 6)?

5. **Backup strategy:** Er git tag + config backups nok, eller vil du ha full VM snapshot?

---

**Status:** Venter på din godkjenning for å starte...

