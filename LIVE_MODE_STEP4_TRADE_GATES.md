# LIVE MODE STEP 4: TRADE SHUTDOWN GATES

**Status:** ✅ **DEPLOYED** (2025-11-22)

## 📋 Overview

Step 4 implements **trading shutdown gates** to automatically **BLOCK NEW TRADES** when dangerous conditions are detected. This is a critical safety mechanism that prevents the system from opening positions during high-risk scenarios while still allowing existing positions to be monitored and exited.

---

## 🚨 Shutdown Conditions

The OrchestratorPolicy monitors these conditions and sets `allow_new_trades = False` when triggered:

### 1. **EXTREME VOLATILITY**
- **Trigger:** Volatility level = "EXTREME"
- **Action:** Immediate shutdown
- **Risk Profile:** `NO_NEW_TRADES`
- **Reason:** Market too unpredictable for safe entry
- **Note:** "EXTREME volatility - no new trades"

### 2. **DAILY DRAWDOWN LIMIT**
- **Trigger:** `current_drawdown_pct <= -daily_dd_limit`
- **Action:** Session shutdown (no new trades for rest of day)
- **Risk Profile:** `NO_NEW_TRADES`
- **Reason:** Daily loss limit reached
- **Example:** SAFE profile: -2.5% DD triggers shutdown
- **Note:** "Daily DD limit hit (-2.50%)"

### 3. **MAX POSITIONS REACHED**
- **Trigger:** `open_trades_count >= max_open_positions`
- **Action:** No new entries until a position closes
- **Reason:** Position limit management
- **Example:** SAFE: 5 positions, AGGRESSIVE: 10 positions
- **Note:** "Max positions reached (5)"

### 4. **EXPOSURE LIMIT EXCEEDED**
- **Trigger:** `total_exposure_pct >= total_exposure_limit`
- **Action:** No new entries until exposure drops
- **Reason:** Total capital at risk too high
- **Example:** SAFE: 10%, AGGRESSIVE: 20%
- **Note:** "Exposure limit hit (10.5%)"

### 5. **LOSING STREAK LIMIT** *(Risk Reduction, Not Full Shutdown)*
- **Trigger:** `losing_streak >= losing_streak_limit`
- **Action:** Risk reduction (30% of normal sizing)
- **Reason:** Consecutive losses indicate unfavorable conditions
- **Note:** "Losing streak, reducing risk to 30%"

---

## ⚙️ Implementation

### **Config Flag**
```python
# backend/services/orchestrator_config.py
@classmethod
def create_live_mode_gradual(cls):
    return cls(
        enable_orchestrator=True,
        mode=OrchestratorMode.LIVE,
        use_for_signal_filter=True,           # ✅ Step 1
        use_for_confidence_threshold=True,    # ✅ Step 1
        use_for_risk_sizing=True,             # ✅ Step 2
        use_for_exit_mode=True,               # ✅ Step 3
        use_for_trading_gate=True,            # ✅ Step 4: NOW ACTIVE
        use_for_position_limits=False,        # ⏳ Step 5
        log_all_signals=True
    )
```

### **Gate Enforcement**
```python
# backend/services/event_driven_executor.py (lines 346-370)

# Step 4: Trading gate enforcement
if self.orch_config.use_for_trading_gate:
    actual_trading_allowed = policy.allow_new_trades
    if not actual_trading_allowed:
        logger.warning(
            f"🚨 TRADE SHUTDOWN ACTIVE 🚨\n"
            f"   Reason: {policy.note}\n"
            f"   Risk Profile: {policy.risk_profile}\n"
            f"   Regime: {regime_tag} | Vol: {vol_level}\n"
            f"   🛑 NO NEW TRADES - Exits only\n"
            f"   ⏳ Will check for recovery in next cycle"
        )

# Early exit if trading gate closed
if not actual_trading_allowed:
    logger.info(
        "⏭️ Skipping signal processing - trading gate CLOSED\n"
        "   ✅ Existing positions continue to be monitored\n"
        "   ✅ Exits will be processed normally\n"
        "   🚫 New entries BLOCKED"
    )
    return  # Skip signal processing, continue loop
```

---

## 📊 Log Examples

### **Normal Operation (Trading Allowed)**
```
🔴 LIVE MODE - Policy ENFORCED: Regime=TRENDING_UP | Vol=NORMAL
📋 Policy Controls: allow_trades=True, min_conf=0.45, blocked_symbols=0, exit_mode=TREND_FOLLOW
🎯 Strong signals: BTCUSDT=BUY(0.78,xgb), ETHUSDT=BUY(0.65,ensemble)
```

### **Shutdown Active (EXTREME_VOL)**
```
🔴 LIVE MODE - Policy ENFORCED: EXTREME volatility - no new trades
📋 Policy Controls: allow_trades=False, min_conf=0.70, blocked_symbols=0, exit_mode=DEFENSIVE_TRAIL
⚠️ TRADING PAUSED: EXTREME volatility - no new trades

🚨 TRADE SHUTDOWN ACTIVE 🚨
   Reason: EXTREME volatility - no new trades
   Risk Profile: NO_NEW_TRADES
   Regime: SIDEWAYS | Vol: EXTREME
   🛑 NO NEW TRADES - Exits only
   ⏳ Will check for recovery in next cycle

⏭️ Skipping signal processing - trading gate CLOSED
   ✅ Existing positions continue to be monitored
   ✅ Exits will be processed normally
   🚫 New entries BLOCKED
```

### **Shutdown Active (Daily DD Limit)**
```
🔴 LIVE MODE - Policy ENFORCED: Daily DD limit hit (-2.50%)
📋 Policy Controls: allow_trades=False, min_conf=0.60, blocked_symbols=0, exit_mode=FAST_TP
⚠️ TRADING PAUSED: Daily DD limit hit (-2.50%)

🚨 TRADE SHUTDOWN ACTIVE 🚨
   Reason: Daily DD limit hit (-2.50%)
   Risk Profile: NO_NEW_TRADES
   Regime: TRENDING_DOWN | Vol: HIGH
   🛑 NO NEW TRADES - Exits only
   ⏳ Will check for recovery in next cycle
```

### **Shutdown Active (Max Positions)**
```
🔴 LIVE MODE - Policy ENFORCED: Max positions reached (5)
📋 Policy Controls: allow_trades=False, min_conf=0.45, blocked_symbols=0, exit_mode=TREND_FOLLOW
⚠️ TRADING PAUSED: Max positions reached (5)

🚨 TRADE SHUTDOWN ACTIVE 🚨
   Reason: Max positions reached (5)
   Risk Profile: SAFE
   Regime: TRENDING_UP | Vol: NORMAL
   🛑 NO NEW TRADES - Exits only
   ⏳ Will check for recovery in next cycle
```

---

## 🔄 Recovery Process

### **Automatic Recovery**
The system checks every cycle (default: 10 seconds) whether shutdown conditions have cleared:

1. **EXTREME_VOL → NORMAL/HIGH:** Trading resumes
2. **Daily DD recovers:** Trading resumes
3. **Position closes:** If under max, trading resumes
4. **Exposure drops:** If under limit, trading resumes

### **Log Example: Recovery**
```
🔴 LIVE MODE - Policy ENFORCED: Volatility normalized
📋 Policy Controls: allow_trades=True, min_conf=0.45, blocked_symbols=0, exit_mode=TREND_FOLLOW
✅ Trading resumed - conditions cleared
```

---

## ✅ Verification Commands

### **Check Current Policy Status**
```python
# From Python console or script
from backend.services.orchestrator_policy import OrchestratorPolicy
from backend.services.orchestrator_config import OrchestratorIntegrationConfig

config = OrchestratorIntegrationConfig.create_live_mode_gradual()
orchestrator = OrchestratorPolicy(config.get_orchestrator_config())
policy = orchestrator.update_policy(...)

print(f"Trading Allowed: {policy.allow_new_trades}")
print(f"Reason: {policy.note}")
print(f"Risk Profile: {policy.risk_profile}")
```

### **Monitor Logs for Shutdown Events**
```powershell
# Watch for shutdown warnings in real-time
Get-Content -Path "backend_terminal.log" -Wait | Select-String "TRADE SHUTDOWN ACTIVE"

# Check if trading is paused
Get-Content -Path "backend_terminal.log" -Wait | Select-String "TRADING PAUSED"
```

### **Check Active Positions During Shutdown**
```python
python check_current_positions.py
```
Positions should still show:
- Active monitoring
- TP/SL orders intact
- Exit signals being processed
- No new entries

---

## 🎯 Safety Guarantees

### **What CONTINUES During Shutdown:**
✅ Existing positions monitored  
✅ TP/SL orders enforced  
✅ Exit signals processed  
✅ TradeLifecycleManager active  
✅ Position PnL tracking  
✅ Risk calculations  

### **What is BLOCKED During Shutdown:**
🚫 New signal processing  
🚫 New order placement  
🚫 Position size increases  
🚫 Rebalancing actions  

---

## 📈 Operational Impact

### **Step 4 vs Previous Steps**

| Feature | Step 1 | Step 2 | Step 3 | Step 4 |
|---------|--------|--------|--------|--------|
| Signal Filtering | ✅ | ✅ | ✅ | ✅ |
| Confidence Threshold | ✅ | ✅ | ✅ | ✅ |
| Risk Scaling | ❌ | ✅ | ✅ | ✅ |
| Exit Mode Override | ❌ | ❌ | ✅ | ✅ |
| **Trading Gate** | ❌ | ❌ | ❌ | **✅ NEW** |

### **Risk Reduction:**
- **EXTREME_VOL:** Prevents entries in chaotic markets
- **Daily DD:** Stops trading before catastrophic losses
- **Max Positions:** Prevents over-concentration
- **Exposure Limit:** Caps total capital at risk

---

## 🔧 Configuration (backend/services/orchestrator_config.py)

### **Risk Profiles:**

#### **SAFE Profile:**
```python
daily_dd_limit = 2.5          # Stop at -2.5% daily drawdown
losing_streak_limit = 3        # Reduce risk after 3 losses
max_open_positions = 5         # Max 5 concurrent positions
total_exposure_limit = 10.0    # Max 10% total exposure
```

#### **MODERATE Profile:**
```python
daily_dd_limit = 4.0          # Stop at -4% daily drawdown
losing_streak_limit = 4        # Reduce risk after 4 losses
max_open_positions = 8         # Max 8 concurrent positions
total_exposure_limit = 15.0    # Max 15% total exposure
```

#### **AGGRESSIVE Profile:**
```python
daily_dd_limit = 6.0          # Stop at -6% daily drawdown
losing_streak_limit = 5        # Reduce risk after 5 losses
max_open_positions = 10        # Max 10 concurrent positions
total_exposure_limit = 20.0    # Max 20% total exposure
```

---

## 🧪 Testing Scenarios

### **Scenario 1: EXTREME_VOL Shutdown**
1. Simulate extreme volatility spike
2. **Expected:** Immediate shutdown, no new trades
3. **Expected Log:** "EXTREME volatility - no new trades"
4. **Verify:** Existing positions still monitored

### **Scenario 2: Daily DD Shutdown**
1. Simulate -2.5% daily drawdown (SAFE profile)
2. **Expected:** Session shutdown
3. **Expected Log:** "Daily DD limit hit (-2.50%)"
4. **Verify:** No new trades for rest of day

### **Scenario 3: Max Positions Shutdown**
1. Open 5 positions (SAFE limit)
2. **Expected:** 6th signal blocked
3. **Expected Log:** "Max positions reached (5)"
4. **Verify:** New trade only after one position closes

### **Scenario 4: Exposure Limit Shutdown**
1. Reach 10% total exposure (SAFE limit)
2. **Expected:** No new trades
3. **Expected Log:** "Exposure limit hit (10.0%)"
4. **Verify:** Trading resumes when exposure drops

---

## 🚀 Next Steps

### **Step 5: Position Limits (Future)**
```python
use_for_position_limits=True  # Per-symbol position sizing
```

### **Step 6: Full Control (Future)**
```python
use_for_all=True  # Complete orchestrator override
```

---

## 📝 Technical Details

### **Code Locations:**

**Configuration:**
- `backend/services/orchestrator_config.py` (line 308-324)

**Policy Computation:**
- `backend/services/orchestrator_policy.py` (line 274-400+)
  - Shutdown logic at lines 330-380

**Gate Enforcement:**
- `backend/services/event_driven_executor.py` (line 346-370)
  - Trading gate check at line 355
  - Early exit at line 365

**Lifecycle Management:**
- `backend/services/trade_lifecycle_manager.py` (receives policy)
- Continues to process exits during shutdown

---

## 📞 Support

**If trading gate not working:**
1. Check `use_for_trading_gate=True` in config
2. Verify backend restarted after changes
3. Check logs for "TRADE SHUTDOWN ACTIVE" warnings
4. Confirm policy.allow_new_trades=False in conditions
5. Review orchestrator_policy.update_policy() logic

**Emergency Override:**
```python
# Temporarily disable gates (NOT RECOMMENDED)
config = OrchestratorIntegrationConfig.create_live_mode_gradual()
config.use_for_trading_gate = False  # Bypass shutdown enforcement
```

---

**✅ STEP 4 COMPLETE**

Trade shutdown gates are now **ACTIVE** and will automatically block new trades when dangerous conditions are detected, providing critical safety for live trading operations.
