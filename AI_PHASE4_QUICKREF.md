# 🎯 PHASE 4 QUICKREF: Adaptive Policy Reinforcement Layer (APRL)

**Status**: ✅ **LIVE** (Limited Mode - awaits Phase 2/3)  
**Deployed**: 2025-12-20 03:56 UTC  
**Mode**: NORMAL | **Window**: 1000 samples | **Updates**: 0

---

## 🚀 QUICK STATUS CHECK

```bash
# Health check (shows Phase 4 active status)
curl http://46.224.116.254:8000/health | jq '.phases.phase4_aprl'

# Expected output:
{
  "active": true,
  "mode": "NORMAL",
  "metrics_tracked": 0,
  "policy_updates": 0
}
```

```bash
# Detailed Phase 4 status
curl http://46.224.116.254:8000/health/phase4 | jq

# Expected output:
{
  "active": true,
  "mode": "NORMAL",
  "policy_updates": 0,
  "performance_samples": 0,
  "current_metrics": {
    "mean": 0.0,
    "std": 0.0,
    "drawdown": 0.0,
    "sharpe": 0.0,
    "sample_count": 0
  },
  "thresholds": {
    "drawdown_defensive": -0.05,
    "volatility_defensive": 0.02,
    "performance_aggressive": 0.01
  }
}
```

---

## 🧠 HOW IT WORKS

### 1️⃣ Input: P&L Signals
```python
# Record new P&L observation
await aprl.record_pnl(pnl_value)
```

### 2️⃣ Calculate Metrics
- **Mean P&L**: Average performance
- **Std Dev**: Volatility measurement
- **Max Drawdown**: Worst decline from peak
- **Sharpe Ratio**: Risk-adjusted return (annualized)

### 3️⃣ Determine Mode
| **Condition**                                       | **Mode**       |
|-----------------------------------------------------|----------------|
| Drawdown ≤ -5% OR Volatility ≥ 2%                   | **DEFENSIVE**  |
| Performance > 1% AND DD > -2% AND Vol < 1%          | **AGGRESSIVE** |
| Otherwise                                           | **NORMAL**     |

### 4️⃣ Adjust Policy
| **Mode**       | **Leverage** | **Position Size** |
|----------------|--------------|-------------------|
| **DEFENSIVE**  | 0.5x         | 50% max           |
| **NORMAL**     | 1.0x         | 70% max           |
| **AGGRESSIVE** | 1.5x         | 80% max           |

### 5️⃣ Update Components
- `governor.update_policy(max_leverage, max_position_size)`
- `risk_brain.update_limits(volatility_threshold, drawdown_threshold)`
- `event_bus.publish("policy_adjusted", data)`

---

## 📊 CURRENT STATUS

### ✅ What's Working
- APRL module initialized and running
- Metrics calculation ready (mean, std, drawdown, sharpe)
- Mode determination logic active (DEFENSIVE/NORMAL/AGGRESSIVE)
- Health endpoints responding
- Error handling graceful

### ⚠️ Limited Functionality (awaits Phase 2/3)
- **No Governor**: Cannot adjust leverage/position size
- **No Risk Brain**: Cannot update risk limits
- **No EventBus**: Cannot publish policy change events
- **No Background Loop**: Continuous adjustment not started

**Why?** Current main.py is from commit 16aa5d2f (pre-Phase 2 baseline).

---

## 🔧 INTEGRATION POINTS

### Current (Limited)
```python
# APRL initialized with None parameters
aprl = AdaptivePolicyReinforcement(
    governor=None,          # ⚠️ Not available
    risk_brain=None,        # ⚠️ Not available
    event_bus=None,         # ⚠️ Not available
    max_window=1000
)
```

### Full Integration (Requires Phase 2/3)
```python
# APRL with live components
aprl = AdaptivePolicyReinforcement(
    governor=app.state.safety_governor,     # ✅ Phase 3
    risk_brain=app.state.risk_brain,        # ✅ Phase 2
    event_bus=app.state.event_bus,          # ✅ Phase 2
    max_window=1000
)

# Start continuous background loop
asyncio.create_task(aprl.run_continuous())
```

---

## 📡 API ENDPOINTS

### `/health` - Main Health Check
**Method**: GET  
**Port**: 8000  
**Response**:
```json
{
  "status": "ok",
  "phases": {
    "phase4_aprl": {
      "active": true,
      "mode": "NORMAL",
      "metrics_tracked": 0,
      "policy_updates": 0
    }
  }
}
```

### `/health/phase4` - Detailed APRL Status
**Method**: GET  
**Port**: 8000  
**Response**: Full metrics, thresholds, mode, performance samples

---

## 🚀 NEXT STEPS FOR FULL ACTIVATION

### Phase 2: Restore Autonomous Intelligence
```bash
# Need to restore:
- CEO Brain (strategic oversight)
- Strategy Brain (tactical optimization)
- Risk Brain (risk assessment)
- EventBus (inter-brain communication)
```

### Phase 3: Restore Safety & Self-Healing
```bash
# Need to restore:
- Safety Governor (policy enforcement)
- ESS (Early Stop System)
- Self-Healing Monitor (autonomous recovery)
```

### Phase 4: Connect Live Components
```python
# Update APRL initialization in main.py
aprl = AdaptivePolicyReinforcement(
    governor=app.state.safety_governor,    # Connect Phase 3
    risk_brain=app.state.risk_brain,       # Connect Phase 2
    event_bus=app.state.event_bus          # Connect Phase 2
)

# Start continuous background loop
asyncio.create_task(aprl.run_continuous())
```

---

## 🧪 TESTING COMMANDS

### Check Container Status
```bash
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 'systemctl list-units --filter name=quantum_backend'
```

### View Phase 4 Logs
```bash
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 'journalctl -u quantum_backend.service 2>&1 | grep -E "\[PHASE 4\]|\[APRL\]"'
```

### Test Health Endpoints
```bash
# Main health
curl http://46.224.116.254:8000/health | jq

# Phase 4 detailed
curl http://46.224.116.254:8000/health/phase4 | jq
```

### Verify APRL File Deployed
```bash
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 'ls -lh ~/quantum_trader/backend/services/adaptive_policy_reinforcement.py'
# Expected: -rw-rw-r-- 1 qt qt 12K
```

---

## 📝 KEY FILES

| **File**                                           | **Size** | **Status**      |
|----------------------------------------------------|----------|-----------------|
| `backend/services/adaptive_policy_reinforcement.py`| 12KB     | ✅ Deployed     |
| `backend/main.py`                                  | 120 lines| ✅ Phase 4 added|
| `backend/main.py.backup_phase3`                    | 120 lines| 💾 Backup       |
| `backend/main.py.backup_before_health_update`      | varies   | 💾 Backup       |

---

## ⚙️ CONFIGURATION

### Thresholds
```python
drawdown_defensive = -0.05       # -5% drawdown triggers DEFENSIVE
volatility_defensive = 0.02      # 2% volatility triggers DEFENSIVE
performance_aggressive = 0.01    # +1% performance allows AGGRESSIVE
```

### Performance Window
```python
max_window = 1000                # Last 1000 P&L samples tracked
```

### Adjustment Frequency
```python
run_continuous_interval = 3600   # Policy adjustment every 1 hour
```

---

## 🎓 UNDERSTANDING THE MODES

### DEFENSIVE Mode (Risk Reduction)
- **Trigger**: Drawdown ≤ -5% OR Volatility ≥ 2%
- **Leverage**: 0.5x (half risk)
- **Position Size**: 50% max (conservative)
- **Goal**: Protect capital during volatility

### NORMAL Mode (Balanced)
- **Trigger**: Default state, balanced conditions
- **Leverage**: 1.0x (standard risk)
- **Position Size**: 70% max (moderate)
- **Goal**: Steady growth with managed risk

### AGGRESSIVE Mode (Growth)
- **Trigger**: Performance > 1% AND Drawdown > -2% AND Volatility < 1%
- **Leverage**: 1.5x (increased risk)
- **Position Size**: 80% max (maximum deployment)
- **Goal**: Capitalize on favorable conditions

---

## 📊 METRICS EXPLAINED

### Mean P&L
- Average performance over rolling window
- Positive = profitable, Negative = losing
- Used to assess overall strategy performance

### Standard Deviation (Volatility)
- Measures consistency of returns
- Higher = more volatile (risky)
- Lower = more stable (predictable)

### Maximum Drawdown
- Worst peak-to-trough decline
- Always negative (e.g., -5% = 5% loss from peak)
- Critical for risk management

### Sharpe Ratio
- Risk-adjusted return (annualized)
- Formula: `(mean * √252) / std` (if std > 0)
- Higher = better risk-adjusted performance
- Industry benchmark: > 1.0 is good

---

## 🔍 TROUBLESHOOTING

### APRL Not Showing in /health
```bash
# Check if APRL initialized
journalctl -u quantum_backend.service 2>&1 | grep "PHASE 4"

# Expected: "[PHASE 4] ✅ Adaptive Policy Reinforcement initialized"
```

### Policy Updates Not Incrementing
**Reason**: No Governor/Risk Brain connected (current limitation)  
**Fix**: Restore Phase 2/3 infrastructure

### Mode Stuck on NORMAL
**Reason**: No P&L data recorded yet (performance_samples = 0)  
**Fix**: Call `aprl.record_pnl(pnl_value)` with live trading data

### Health Endpoint Returns 500 Error
```bash
# Check logs for stack trace
journalctl -u quantum_backend.service 2>&1 | tail -50

# Common issue: AttributeError accessing private APRL attributes
# Fix: Use aprl.get_status() instead of direct attribute access
```

---

## 🎉 SUCCESS INDICATORS

✅ **Container Running**: `systemctl list-units` shows "Up X seconds"  
✅ **Health Returns 200**: `curl /health` returns `"status": "ok"`  
✅ **Phase 4 Active**: `/health` shows `"phase4_aprl": {"active": true}`  
✅ **Logs Show Init**: `grep PHASE 4` finds initialization messages  
✅ **Mode Determined**: `/health/phase4` shows `"mode": "NORMAL"`  
✅ **APRL File Exists**: `ls` shows 12K file on VPS  

---

**Last Updated**: 2025-12-20 03:56 UTC  
**Quick Reference**: Phase 4 Adaptive Policy Reinforcement Layer  
**Full Documentation**: See AI_PHASE4_APRL_DEPLOYMENT_COMPLETE.md

