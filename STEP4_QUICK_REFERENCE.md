# STEP 4 QUICK REFERENCE: TRADE SHUTDOWN GATES

## 🚨 Shutdown Conditions

| Condition | Trigger | Action | SAFE Limit | AGGRESSIVE Limit |
|-----------|---------|--------|------------|------------------|
| **EXTREME_VOL** | Volatility = EXTREME | Immediate shutdown | N/A | N/A |
| **Daily DD** | Drawdown <= limit | Session shutdown | -2.5% | -6.0% |
| **Max Positions** | Open trades >= limit | Block new entries | 5 | 10 |
| **Exposure Limit** | Total exposure >= limit | Block new entries | 10% | 20% |
| **Losing Streak** | Consecutive losses >= limit | Risk reduction (30%) | 3 | 5 |

## ✅ Verification Commands

```powershell
# Check if Step 4 is active
Get-Content c:\quantum_trader\backend\services\orchestrator_config.py | Select-String "use_for_trading_gate"
# Expected: use_for_trading_gate=True

# Monitor for shutdown events
Get-Content -Path "backend_terminal.log" -Wait | Select-String "TRADE SHUTDOWN ACTIVE|TRADING PAUSED"

# Check current positions (should still be monitored during shutdown)
python check_current_positions.py
```

## 📊 Log Signatures

**Shutdown Active:**
```
🚨 TRADE SHUTDOWN ACTIVE 🚨
   Reason: <condition>
   Risk Profile: NO_NEW_TRADES
   🛑 NO NEW TRADES - Exits only
```

**Gate Closed:**
```
⏭️ Skipping signal processing - trading gate CLOSED
   ✅ Existing positions continue to be monitored
   🚫 New entries BLOCKED
```

**Trading Resumed:**
```
✅ Trading resumed - conditions cleared
```

## 🔧 Config Location

**File:** `backend/services/orchestrator_config.py`  
**Method:** `create_live_mode_gradual()`  
**Line:** ~315

```python
use_for_trading_gate=True,  # ✅ Step 4: NOW ACTIVE
```

## 🎯 What Continues During Shutdown

✅ Position monitoring  
✅ TP/SL enforcement  
✅ Exit signal processing  
✅ PnL tracking  
✅ Risk calculations  

🚫 New trades  
🚫 Signal processing for entries  
🚫 Order placement  

## 🚀 Status

**✅ DEPLOYED:** 2025-11-22  
**Mode:** LIVE  
**Active:** Yes  
**Enforcement:** Event-driven executor  

---

**For full details:** See `LIVE_MODE_STEP4_TRADE_GATES.md`
