# 🚀 FULL LIVE MODE - Quick Reference

**Last Updated:** 2025-11-22  
**Status:** ✅ ACTIVE

---

## ⚡ Active Subsystems

| Subsystem | Flag | Status | Quick Check |
|-----------|------|--------|-------------|
| **Signal Filter** | `use_for_signal_filter` | ✅ | Look for "BLOCKED_BY_POLICY_FILTER" |
| **Confidence Gate** | `use_for_confidence_threshold` | ✅ | Look for "BLOCKED_BY_CONFIDENCE" |
| **Risk Scaling** | `use_for_risk_sizing` | ✅ | Check position size variance |
| **Exit Mode** | `use_for_exit_mode` | ✅ | Check for "exit_mode=TREND_FOLLOW\|FAST_TP\|DEFENSIVE" |
| **Trade Gate** | `use_for_trading_gate` | ✅ | Look for "TRADE SHUTDOWN ACTIVE" |
| **Position Limits** | `use_for_position_limits` | ✅ | Check "position_limits=ACTIVE" |

---

## 🔍 Quick Verification

### **1-Minute Health Check**
```powershell
# Check config
Get-Content backend/services/orchestrator_config.py | Select-String "use_for_position_limits"
# Should show: use_for_position_limits=True

# Check logs for FULL LIVE MODE
Get-Content backend_logs.txt | Select-String "FULL LIVE MODE" | Select-Object -Last 5

# Check latest policy
Get-Content backend_logs.txt | Select-String "Policy Controls" | Select-Object -Last 1
```

### **Expected Output:**
```
🔴 FULL LIVE MODE - Policy ENFORCED: Normal market conditions
📋 Policy Controls: allow_trades=True, min_conf=0.45, blocked_symbols=0, risk_pct=2.00%, exit_mode=TREND_FOLLOW, position_limits=ACTIVE
```

---

## 📊 Policy State Matrix

| Condition | allow_trades | min_conf | risk_pct | exit_mode | risk_profile |
|-----------|--------------|----------|----------|-----------|--------------|
| **Normal** | ✅ True | 0.45 | 1.5-2% | TREND_FOLLOW | MODERATE |
| **High Vol** | ✅ True | 0.60 | 0.8-1% | FAST_TP | REDUCED |
| **Extreme Vol** | 🚫 False | 0.70 | 0.5% | DEFENSIVE_TRAIL | NO_NEW_TRADES |
| **DD Limit** | 🚫 False | 0.60 | 0.5% | FAST_TP | NO_NEW_TRADES |
| **Max Positions** | 🚫 False | 0.45 | 1% | Current | MODERATE |
| **Fallback** | ✅ True | 0.65 | 1% | DEFENSIVE_TRAIL | FALLBACK |

---

## 🚨 Log Signatures

### **Normal Operation:**
```
🔴 FULL LIVE MODE - Policy ENFORCED
📋 Policy Controls: allow_trades=True, min_conf=0.45, risk_pct=2.00%, exit_mode=TREND_FOLLOW
✅ Policy confidence active: 0.45
✅ Trading gate: OPEN
✅ Position limits: ACTIVE
```

### **Volatility Spike:**
```
🔴 FULL LIVE MODE - Policy ENFORCED: High volatility detected
📋 Policy Controls: allow_trades=True, min_conf=0.60, risk_pct=1.50%, exit_mode=FAST_TP
✅ Policy confidence active: 0.60 (default: 0.45)
```

### **Shutdown Event:**
```
🚨 TRADE SHUTDOWN ACTIVE 🚨
   Reason: Daily DD limit hit (-2.60%)
   Risk Profile: NO_NEW_TRADES
   Regime: TRENDING_DOWN | Vol: HIGH
⏭️ Skipping signal processing - trading gate CLOSED
```

### **Orchestrator Failure (Fallback):**
```
⚠️ Orchestrator policy update failed: <error>
🛡️ Using SAFE FALLBACK policy
📋 Policy Controls: allow_trades=True, min_conf=0.65, risk_pct=1.00%, exit_mode=DEFENSIVE_TRAIL
```

### **Signal Blocking:**
```
🚫 Signal BLOCKED: NEARUSDT (confidence: 0.48 < threshold: 0.60)
🚫 Signal BLOCKED: APTUSDT (symbol in disallowed_symbols)
```

---

## 🎯 Risk Profile Scaling

| Profile | Base Risk | Actual Risk | Confidence | Position Limit | Use Case |
|---------|-----------|-------------|------------|----------------|----------|
| **SAFE** | 1% | 1% | 0.45 | 5 | DD recovery, high vol |
| **MODERATE** | 1.5% | 1.5% | 0.45 | 8 | Normal conditions |
| **AGGRESSIVE** | 2% | 2% | 0.45 | 10 | Low vol, trending |
| **REDUCED** | 1.5% | 1.05% (×0.7) | 0.55 | 5 | Elevated risk |
| **NO_NEW_TRADES** | 0.5% | 0.05% (×0.1) | 0.70 | 0 | SHUTDOWN |
| **FALLBACK** | 1% | 1% | 0.65 | 5 | Orchestrator failed |

---

## 🔄 Exit Mode Quick Reference

| Mode | TP | SL | Trail | Partial | When Used |
|------|----|----|-------|---------|-----------|
| **TREND_FOLLOW** | 8% | 4% | 3% | 50% @ 1.5R | TRENDING + NORMAL/LOW vol |
| **FAST_TP** | 5% | 3% | 2% | 60% @ 1R | HIGH vol, any regime |
| **DEFENSIVE_TRAIL** | 4% | 2% | 1.5% | 70% @ 0.8R | EXTREME vol, CHOPPY |

---

## ⚙️ Command Cheat Sheet

### **Configuration**
```powershell
# View config
Get-Content backend/services/orchestrator_config.py | Select-String "use_for_" -Context 0,1

# Check mode
Get-Content backend/services/orchestrator_config.py | Select-String "mode="
```

### **Live Monitoring**
```powershell
# Follow logs
Get-Content -Path backend_logs.txt -Wait | Select-String "Policy|SHUTDOWN|BLOCKED"

# Policy updates only
Get-Content -Path backend_logs.txt -Wait | Select-String "FULL LIVE MODE|Policy Controls"

# Shutdown events only
Get-Content -Path backend_logs.txt -Wait | Select-String "TRADE SHUTDOWN"

# Signal decisions
Get-Content -Path backend_logs.txt -Wait | Select-String "Signal BLOCKED|Signal ACCEPTED"
```

### **Status Checks**
```powershell
# Current policy state
Get-Content backend_logs.txt | Select-String "Policy Controls" | Select-Object -Last 1

# Shutdown status
Get-Content backend_logs.txt | Select-String "TRADE SHUTDOWN|Trading gate" | Select-Object -Last 5

# Exit mode switches
Get-Content backend_logs.txt | Select-String "exit_mode=" | Select-Object -Last 10
```

### **Python Quick Checks**
```python
# Check orchestrator status
from backend.services.orchestrator_config import OrchestratorIntegrationConfig
config = OrchestratorIntegrationConfig.create_live_mode_gradual()
print(f"Live Mode: {config.is_live_mode()}")
print(f"All flags: {config.use_for_signal_filter and config.use_for_risk_sizing and config.use_for_exit_mode and config.use_for_trading_gate and config.use_for_position_limits}")

# Get current policy
from backend.services.orchestrator_policy import OrchestratorPolicy
orchestrator = OrchestratorPolicy(config)
policy = orchestrator.get_current_policy()
print(f"Allow trades: {policy.allow_new_trades}")
print(f"Min confidence: {policy.min_confidence}")
print(f"Risk %: {policy.max_risk_pct}")
print(f"Exit mode: {policy.exit_mode}")
```

---

## 🛡️ Safety Guarantees

### **What CANNOT Happen:**
- ❌ Trading during EXTREME volatility
- ❌ Positions exceeding 2% risk in high vol
- ❌ New trades after daily DD limit (-2.5% SAFE, -6% AGG)
- ❌ More than max positions (5 SAFE, 10 AGG)
- ❌ Exposure exceeding limits (10% SAFE, 20% AGG)
- ❌ Signals below confidence threshold

### **What ALWAYS Happens:**
- ✅ Existing positions monitored
- ✅ TP/SL orders enforced
- ✅ Exit signals processed
- ✅ PnL tracked
- ✅ Fallback policy if orchestrator fails
- ✅ All decisions logged

---

## 🚀 Deployment Steps

### **1. Pre-Deployment Checks**
```powershell
# Verify config changes
git diff backend/services/orchestrator_config.py
git diff backend/services/event_driven_executor.py

# Expected changes:
# - use_for_position_limits=True
# - Enhanced policy logging
# - Safety fallback added
```

### **2. Deployment**
```powershell
# Stop backend
Get-Process -Name uvicorn -ErrorAction SilentlyContinue | Stop-Process -Force

# Start with FULL LIVE MODE
cd C:\quantum_trader\backend
$env:PYTHONPATH='C:\quantum_trader'
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### **3. Post-Deployment Verification**
```powershell
# Wait 30 seconds for first policy cycle
Start-Sleep -Seconds 30

# Check logs
Get-Content backend_logs.txt | Select-String "FULL LIVE MODE" | Select-Object -Last 3
Get-Content backend_logs.txt | Select-String "position_limits=ACTIVE" | Select-Object -Last 1

# Expected: Both should show recent entries
```

---

## 🐛 Troubleshooting

### **Issue: "FULL LIVE MODE" not in logs**
**Cause:** Orchestrator not enabled or not in LIVE mode  
**Fix:**
```python
# Check: backend/services/orchestrator_config.py
enable_orchestrator=True  # Must be True
mode=OrchestratorMode.LIVE  # Must be LIVE
```

### **Issue: "position_limits=ACTIVE" not appearing**
**Cause:** Flag not enabled  
**Fix:**
```python
# Check: backend/services/orchestrator_config.py line 308
use_for_position_limits=True  # Must be True
```

### **Issue: All signals blocked**
**Cause:** Too high confidence threshold or shutdown active  
**Check:**
```powershell
Get-Content backend_logs.txt | Select-String "Policy Controls" | Select-Object -Last 1
# Look for: min_conf=0.70, allow_trades=False
```
**Explanation:**
- High min_conf (0.70): EXTREME volatility detected
- allow_trades=False: Shutdown gate active (DD limit, max positions, etc.)

### **Issue: Orchestrator errors**
**Cause:** Orchestrator failure, fallback active  
**Check:**
```powershell
Get-Content backend_logs.txt | Select-String "SAFE FALLBACK"
```
**Action:**
- System continues safely with fallback policy
- Review error logs for root cause
- Orchestrator will retry next cycle

---

## 📋 Daily Checklist

### **Morning (Pre-Market):**
- [ ] Check backend is running: `Get-Process -Name uvicorn`
- [ ] Verify FULL LIVE MODE active: `Get-Content backend_logs.txt | Select-String "FULL LIVE MODE" | Select-Object -Last 1`
- [ ] Check current policy state: `... | Select-String "Policy Controls" | Select-Object -Last 1`
- [ ] Confirm no shutdown gates: `... | Select-String "TRADE SHUTDOWN" | Select-Object -Last 5`

### **During Trading:**
- [ ] Monitor policy switches: Watch for exit_mode changes
- [ ] Watch for shutdown events: "TRADE SHUTDOWN ACTIVE"
- [ ] Check signal filtering: "Signal BLOCKED" should be rare unless high vol

### **Evening (Post-Market):**
- [ ] Review policy transitions: Count exit_mode switches
- [ ] Analyze shutdown events: When/why/duration
- [ ] Check DD impact: Compare to no-orchestrator baseline
- [ ] Verify position limits enforced: Max 5/10 positions respected

---

## 🎯 Performance Metrics to Track

| Metric | Pre-Orchestrator | Target | How to Check |
|--------|------------------|--------|--------------|
| **Max Daily DD** | -3% to -5% | -2% to -3% | Daily PnL logs |
| **Avg Position Risk** | 1.5-2% | 0.8-1.5% (dynamic) | Position size variance |
| **Shutdown Events** | 0 | 1-3/day (high vol days) | Count "TRADE SHUTDOWN" |
| **Losing Streak Avg** | 3-5 trades | 2-3 trades | Streak logs |
| **Exit Timing Quality** | Fixed | Adaptive | Compare TP hit rates by exit_mode |

---

## 💡 Pro Tips

1. **Watch for policy transitions:** If exit_mode switches frequently (>5/hour), volatility is high
2. **Shutdown = OK:** Shutdown events are PROTECTIVE, not failures
3. **Fallback = Investigate:** SAFE FALLBACK means orchestrator needs attention
4. **Blocked signals = Working:** In high vol, most signals SHOULD be blocked
5. **Position limits = Capital efficiency:** Hitting limit means good opportunities, not a problem

---

**✅ FULL LIVE MODE OPERATIONAL**

All systems active, all safety mechanisms engaged, fully autonomous trading enabled.

**For Details:** See `FULL_LIVE_MODE_FINAL.md`
