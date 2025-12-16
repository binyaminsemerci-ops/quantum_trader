# Risk Gate v3 - Quick Reference

**Location:** `backend/risk/risk_gate_v3.py`  
**Tests:** `tests/risk/test_risk_gate_v3.py` (13/13 passing ✅)  
**Status:** Ready for execution integration

---

## 🚀 Quick Start

### 1. Initialize Risk Gate (at startup)
```python
from backend.risk.risk_gate_v3 import init_risk_gate, RiskStateFacade
from backend.services.risk.emergency_stop_system import EmergencyStopSystem

# Initialize
risk_facade = RiskStateFacade(risk_api_url="http://localhost:8001")
ess = EmergencyStopSystem(...)
init_risk_gate(risk_facade=risk_facade, ess=ess)
```

### 2. Use in Execution Flow
```python
from backend.risk.risk_gate_v3 import evaluate_order_risk

# Before placing order
risk_result = await evaluate_order_risk(
    account_name="PRIVATE_MAIN",
    exchange_name="binance",
    strategy_id="neo_scalper_1",
    order_request={
        "symbol": "BTCUSDT",
        "side": "BUY",
        "size": 1000.0,
        "leverage": 2.0,
    }
)

# Handle decision
if risk_result.decision == "block":
    logger.warning(f"Order BLOCKED: {risk_result.reason}")
    return  # Skip order

elif risk_result.decision == "scale_down":
    logger.info(f"Order SCALED: {risk_result.scale_factor}x")
    size = size * risk_result.scale_factor

# Proceed with order
await client.place_order(...)
```

---

## 📋 Decision Model

### RiskGateResult
```python
@dataclass
class RiskGateResult:
    decision: "allow" | "block" | "scale_down"
    reason: str  # Human-readable explanation
    scale_factor: float = 1.0  # For scale_down decisions
    risk_level: Optional[str] = None  # "INFO" | "WARNING" | "CRITICAL"
    ess_active: bool = False
    timestamp: datetime
```

### Evaluation Hierarchy (First Failure Wins)
1. **ESS halt** → BLOCK (highest priority, no exceptions)
2. **Global Risk CRITICAL** → BLOCK
3. **Risk v3 ESS action required** → BLOCK
4. **Strategy not in whitelist** → BLOCK
5. **Leverage exceeds limit** → BLOCK
6. **Single-trade risk too large** → BLOCK
7. **All checks pass** → ALLOW

---

## 🔧 Configuration

### Risk v3 API URL
```python
risk_facade = RiskStateFacade(risk_api_url="http://localhost:8001")
```

### Capital Profiles (from EPIC-P10)
- **micro**: allowed_leverage=1, max_single_trade=0.2%
- **low**: allowed_leverage=2, max_single_trade=0.5%
- **normal**: allowed_leverage=3, max_single_trade=1.0%
- **aggressive**: allowed_leverage=5, max_single_trade=2.0%

---

## 🎯 Integration Points

### Execution Service
**File:** `backend/services/execution/execution.py`

**Where to Add:**
```python
def execute_signal(...):
    # Existing: check_profile_limits_for_signal()
    
    # NEW: Add here, before place_order()
    risk_result = await evaluate_order_risk(...)
    if risk_result.decision == "block":
        return  # Skip order
    
    # Existing: await client.place_order(...)
```

---

## 📊 Block Reasons

### Common Block Reasons:
- `"ess_trading_halt_active"` → ESS active
- `"global_risk_critical: ..."` → Risk v3 CRITICAL
- `"risk_v3_ess_action_required"` → Risk v3 recommends ESS
- `"strategy_not_allowed: ..."` → Strategy not in profile whitelist
- `"leverage_exceeds_limit: Xx > Yx"` → Leverage too high
- `"single_trade_risk_exceeds_limit: X% > Y%"` → Trade risk too large
- `"risk_gate_not_initialized"` → Risk gate not set up

---

## 🧪 Testing

### Run Tests
```bash
pytest tests/risk/test_risk_gate_v3.py -v
```

### Test Coverage (13/13 ✅)
1. ESS halt blocks order
2. Global Risk CRITICAL blocks order
3. Risk v3 ESS action blocks order
4. Strategy not in whitelist blocks order
5. Leverage exceeds limit blocks order
6. Single-trade risk too large blocks order
7. Happy path allows order
8. Risk v3 unavailable fallback
9. Global instance initialization
10. Convenience function
11. Convenience function not initialized
12. Different capital profiles
13. Risk level propagation

---

## ⚠️ Known Limitations (TODOs)

### 1. Single-Trade Risk (Stub)
**Current:** Hardcoded to 0.1% risk per trade
**TODO:** Integrate with portfolio service for real equity
```python
# Current stub:
single_trade_risk_pct = 0.1  # Stub

# Future implementation:
single_trade_risk_pct = (order_size / total_equity) * 100
```

### 2. Daily/Weekly Loss Limits (Stub)
**Current:** Not implemented
**TODO:** Integrate with analytics service for real PnL
```python
# TODO: Check against:
# - profile.max_daily_loss_pct
# - profile.max_weekly_loss_pct
```

### 3. Execution Integration (Not Done)
**Current:** Risk gate exists but not wired into execution
**TODO:** Add risk gate call before `client.place_order()` in execution.py

---

## 🚨 Emergency Procedures

### Bypass Risk Gate (Emergency Only)
```python
# Option 1: Disable ESS (if safe to do so)
ess.reset()  # Manual reset

# Option 2: Temporarily skip risk gate
# (Add flag to order_request? Requires admin approval?)

# ⚠️ ALL BYPASSES MUST BE LOGGED AND AUDITED
```

### Debug Risk Gate Decision
```python
# Check current risk state
signal = await risk_facade.get_global_risk_signal()
print(f"Risk Level: {signal['risk_level']}")
print(f"Critical Issues: {signal['critical_issues']}")
print(f"ESS Action: {signal['ess_action_required']}")

# Check capital profile
from backend.policies.capital_profiles import get_profile
from backend.policies.account_config import get_capital_profile_for_account
profile_name = get_capital_profile_for_account("PRIVATE_MAIN")
profile = get_profile(profile_name)
print(f"Profile: {profile_name}, Max Leverage: {profile.allowed_leverage}")

# Check ESS
print(f"ESS Active: {ess.is_active()}")
```

---

## 📞 Support

**Created Files:**
- `backend/risk/risk_gate_v3.py` (415 lines)
- `tests/risk/test_risk_gate_v3.py` (540 lines)
- `EPIC_RISK3_EXEC_001_SUMMARY.md`
- `RISK_GATE_V3_QUICKREF.md` (this file)

**Related Systems:**
- Global Risk v3: `backend/services/risk_v3/`
- Capital Profiles (EPIC-P10): `backend/policies/capital_profiles.py`
- ESS: `backend/services/risk/emergency_stop_system.py`
- Execution: `backend/services/execution/execution.py` (TODO: integrate)

**Next Steps:**
1. ⏳ Wire risk gate into execution.py
2. ⏳ Initialize at startup
3. 📊 Integrate real equity for single-trade risk
4. 📊 Integrate daily/weekly PnL for loss limits

---

**EPIC-RISK3-EXEC-001:** ✅ Implementation Complete | ⏳ Execution Integration Pending
