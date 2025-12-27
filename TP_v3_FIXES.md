# Take-Profit Pipeline v3 - Analysis & Fixes

## DISCOVERED ARCHITECTURE

### Current TP Flow (3-Layer)
```
[1] RL v3 Manager (rl_manager_v3.py)
    ├─ Input: obs_dict (market data)
    └─ Output: {'action', 'confidence', 'value'}  ❌ NO TP OUTPUT
    
[2] Dynamic TP/SL Calculator (dynamic_tpsl.py)
    ├─ Input: confidence, action, risk_mode
    ├─ Calculates: tp_percent, sl_percent, trail_percent
    └─ Output: DynamicTPSLOutput
    
[3] Hybrid TP/SL (hybrid_tpsl.py)
    ├─ Blends: baseline risk + AI overlays
    ├─ Places: SL + BASE_TP + TRAILING_STOP orders
    └─ Uses: SafeOrderExecutor (retry logic)
    
[4] Position Monitor (position_monitor.py)
    └─ Manages: Live TP/SL order updates
```

### Files Modified
- `backend/services/execution/dynamic_tpsl.py` ✅ Working
- `backend/services/execution/hybrid_tpsl.py` ✅ Working  
- `backend/services/monitoring/position_monitor.py` ✅ Working
- `backend/domains/learning/rl_v3/rl_manager_v3.py` ⚠️ Missing TP output

---

## CRITICAL GAPS IDENTIFIED

### Gap 1: RL v3 Does Not Output TP Ranges
**Issue:** RL Manager returns only `action` and `confidence`. Dynamic TP/SL must derive TP entirely from confidence scaling.

**Current Behavior:**
```python
# rl_manager_v3.py predict()
return {
    'action': action,
    'confidence': confidence,
    'value': value
}
# Missing: 'tp_zone', 'exit_price_range'
```

**Impact:** TP decisions are not statistically optimized by RL - only confidence-based scaling applied.

### Gap 2: No TP-Specific Reward Signal in RL Training
**Issue:** RL environment (`env_v3.py`) rewards PnL but does not specifically reward optimal TP placement.

**Missing:**
- Reward bonus for TP hits within predicted zone
- Penalty for TP set too tight/wide
- TP zone prediction accuracy tracking

### Gap 3: Risk v3 Integration Missing
**Issue:** Dynamic TP/SL calculator does not query Risk v3 for:
- Current portfolio exposure (ESS)
- Systemic risk levels
- Volatility regime adjustments

**Current:**
```python
# dynamic_tpsl.py - only uses confidence + optional volatility
volatility_scale = 1.0
if market_conditions and 'volatility' in market_conditions:
    volatility = market_conditions['volatility']
    volatility_scale = 0.8 + (volatility / 0.03) * 0.4
```

**Missing:**
- ESS-based TP tightening
- Correlation risk adjustments
- Systemic event detection → defensive TP

---

## STEP 3 — EXECUTION VALIDATION

### ✅ WORKING CORRECTLY
1. **Hybrid TP/SL Order Placement:**
   - Baseline SL always placed
   - BASE_TP order placed
   - TRAILING_STOP conditionally placed
   - Uses `SafeOrderExecutor` with retry logic

2. **Partial TP Support:**
   - Enabled for confidence >= 0.80
   - Multi-level TP zones calculated
   - Position Monitor handles partial fills

3. **Slippage Protection:**
   - Prices rounded to tick size
   - Quantities rounded to step size
   - Orders validated before placement

### ⚠️ MISSING FEATURES
1. **Dynamic Trailing TP Rearm:**
   - Trailing stop placed initially but NOT dynamically adjusted as position profits
   - Should tighten callback % as unrealized PnL increases

2. **TP Order Replacement Logic:**
   - Stale orders cancelled but no explicit "TP replace" workflow
   - If market moves significantly, TP should be recalculated and replaced

3. **Race Condition Guard:**
   - No mutex around simultaneous TP updates from:
     - Position Monitor (scheduled)
     - Event-driven executor (signal-based)
   - Risk of duplicate/conflicting TP orders

---

## STEP 4 — DYNAMIC TP v3 IMPROVEMENTS

### Patch 1: Add TP Zone Output to RL v3

**File:** `backend/domains/learning/rl_v3/rl_manager_v3.py`

```python
def predict(self, obs_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Predict action with TP zone suggestion."""
    state = build_feature_vector(obs_dict)
    action, log_prob, value = self.agent.act(state, deterministic=True)
    
    confidence = min(1.0, abs(value) / 10.0)
    
    # NEW: Suggest TP zone based on value estimate
    # Positive value → wider TP, negative value → tighter TP
    tp_multiplier = 1.0 + (value / 20.0)  # Range: 0.5x to 1.5x
    tp_multiplier = max(0.5, min(1.5, tp_multiplier))
    
    return {
        'action': action,
        'confidence': confidence,
        'value': value,
        'tp_zone_multiplier': tp_multiplier,  # NEW
        'suggested_tp_pct': 0.06 * tp_multiplier  # NEW: Base 6% TP
    }
```

### Patch 2: Integrate Risk v3 Adjustments

**File:** `backend/services/execution/dynamic_tpsl.py`

```python
def calculate(
    self,
    signal_confidence: float,
    action: str,
    market_conditions: Optional[Dict[str, Any]] = None,
    risk_mode: str = "NORMAL",
    risk_v3_context: Optional[Dict[str, Any]] = None  # NEW
) -> DynamicTPSLOutput:
    """Calculate dynamic TP/SL with Risk v3 integration."""
    
    # ... existing confidence + volatility scaling ...
    
    # NEW: Apply Risk v3 adjustments
    if risk_v3_context:
        ess_factor = risk_v3_context.get('ess_factor', 1.0)
        systemic_risk = risk_v3_context.get('systemic_risk_level', 0.0)
        
        # High ESS → tighten TP to reduce exposure
        if ess_factor > 1.5:
            tp_percent *= 0.85
            self.logger.info(f"[Risk v3] TP tightened due to ESS={ess_factor:.2f}")
        
        # Systemic risk detected → defensive TP
        if systemic_risk > 0.7:
            tp_percent *= 0.75
            sl_percent *= 1.2
            self.logger.warning(f"[Risk v3] Defensive mode: systemic_risk={systemic_risk:.2f}")
    
    # ... rest of calculation ...
```

### Patch 3: Dynamic Trailing TP Rearm

**File:** `backend/services/monitoring/position_monitor.py`

```python
async def _adjust_trailing_stop_dynamically(
    self,
    position: Dict[str, Any],
    unrealized_pnl_pct: float
) -> None:
    """Tighten trailing stop as position profits increase."""
    if unrealized_pnl_pct < 0.02:  # Only for profitable positions
        return
    
    symbol = position['symbol']
    current_callback = position.get('trailing_callback_pct', 0.02)
    
    # Profit-based callback tightening
    if unrealized_pnl_pct > 0.10:  # 10%+ profit
        new_callback = current_callback * 0.5  # Halve callback
    elif unrealized_pnl_pct > 0.05:  # 5%+ profit
        new_callback = current_callback * 0.75
    else:
        return  # No adjustment needed
    
    new_callback = max(0.005, new_callback)  # Min 0.5% callback
    
    if new_callback < current_callback:
        # Cancel old trailing stop and place new tighter one
        await self._replace_trailing_stop(symbol, new_callback)
        self.logger.info(
            f"[Dynamic TP] Tightened trailing stop for {symbol}: "
            f"{current_callback:.2%} → {new_callback:.2%}"
        )
```

### Patch 4: TP Replacement Mutex

**File:** `backend/services/execution/event_driven_executor.py`

```python
import asyncio

class EventDrivenExecutor:
    def __init__(self):
        # ... existing ...
        self._tp_update_locks: Dict[str, asyncio.Lock] = {}
    
    async def _update_tp_with_lock(self, symbol: str, new_tp_params: Dict):
        """Thread-safe TP update."""
        if symbol not in self._tp_update_locks:
            self._tp_update_locks[symbol] = asyncio.Lock()
        
        async with self._tp_update_locks[symbol]:
            # Cancel old TP orders
            await self._cancel_tp_orders(symbol)
            
            # Place new TP orders
            await self._place_tp_orders(symbol, new_tp_params)
            
            self.logger.info(f"[TP-MUTEX] Updated TP for {symbol}")
```

---

## STEP 5 — TESTS REQUIRED

### Test Coverage Matrix

| Test | Component | Status |
|------|-----------|--------|
| `test_rl_tp_zone_generation` | RL v3 Manager | ❌ Missing |
| `test_dynamic_tpsl_confidence_scaling` | Dynamic TP/SL | ✅ Partial |
| `test_risk_v3_tp_adjustment` | Risk Integration | ❌ Missing |
| `test_hybrid_order_placement` | Hybrid TP/SL | ✅ Exists |
| `test_partial_tp_execution` | Position Monitor | ⚠️ Integration only |
| `test_trailing_stop_rearm` | Position Monitor | ❌ Missing |
| `test_tp_replacement_mutex` | Executor | ❌ Missing |

### Priority Tests to Create

**1. Test: RL v3 TP Zone Output**
```python
# tests/test_rl_v3_tp_output.py
def test_rl_predicts_tp_zone():
    manager = RLv3Manager()
    obs = {'price': 50000, 'rsi': 65, 'macd': 0.02}
    
    result = manager.predict(obs)
    
    assert 'tp_zone_multiplier' in result
    assert 'suggested_tp_pct' in result
    assert 0.5 <= result['tp_zone_multiplier'] <= 1.5
```

**2. Test: Risk v3 TP Adjustment**
```python
# tests/test_dynamic_tpsl_risk_v3.py
def test_high_ess_tightens_tp():
    calculator = DynamicTPSLCalculator()
    
    base_result = calculator.calculate(
        signal_confidence=0.75,
        action='BUY',
        risk_v3_context={'ess_factor': 1.0}
    )
    
    high_ess_result = calculator.calculate(
        signal_confidence=0.75,
        action='BUY',
        risk_v3_context={'ess_factor': 2.5}
    )
    
    assert high_ess_result.tp_percent < base_result.tp_percent
```

**3. Test: Trailing Stop Rearm**
```python
# tests/test_trailing_rearm.py
@pytest.mark.asyncio
async def test_trailing_stop_tightens_with_profit():
    monitor = PositionMonitor()
    position = {
        'symbol': 'BTCUSDT',
        'unrealized_pnl_pct': 0.12,
        'trailing_callback_pct': 0.02
    }
    
    await monitor._adjust_trailing_stop_dynamically(position, 0.12)
    
    # Verify new trailing stop placed with tighter callback
    assert position['trailing_callback_pct'] < 0.02
```

---

## STEP 6 — TP SMOKE TEST

**File:** `scripts/tp_smoke_test.py`

```python
"""
TP Smoke Test - End-to-end TP pipeline validation
"""
import asyncio
from backend.domains.learning.rl_v3.rl_manager_v3 import RLv3Manager
from backend.services.execution.dynamic_tpsl import get_dynamic_tpsl_calculator
from backend.services.execution.hybrid_tpsl import calculate_hybrid_levels

async def run_tp_smoke_test():
    """Validate TP pipeline end-to-end."""
    print("🧪 TP v3 SMOKE TEST")
    print("=" * 50)
    
    # STEP 1: RL v3 generates signal
    print("\n[1] RL v3 Signal Generation...")
    rl_manager = RLv3Manager()
    obs = {
        'price': 100000.0,
        'rsi': 70,
        'macd': 0.015,
        'volume': 1000000
    }
    rl_output = rl_manager.predict(obs)
    print(f"   Action: {rl_output['action']}")
    print(f"   Confidence: {rl_output['confidence']:.2%}")
    print(f"   Suggested TP: {rl_output.get('suggested_tp_pct', 'N/A')}")
    
    # STEP 2: Dynamic TP/SL calculation
    print("\n[2] Dynamic TP/SL Calculation...")
    calculator = get_dynamic_tpsl_calculator()
    tpsl_result = calculator.calculate(
        signal_confidence=rl_output['confidence'],
        action='BUY',
        market_conditions={'volatility': 0.02},
        risk_mode='NORMAL'
    )
    print(f"   TP: {tpsl_result.tp_percent:.2%}")
    print(f"   SL: {tpsl_result.sl_percent:.2%}")
    print(f"   Trail: {tpsl_result.trail_percent:.2%}")
    print(f"   R:R = {tpsl_result.tp_percent / tpsl_result.sl_percent:.2f}x")
    
    # STEP 3: Hybrid TP/SL blending
    print("\n[3] Hybrid TP/SL Blending...")
    hybrid = calculate_hybrid_levels(
        entry_price=100000.0,
        side='BUY',
        risk_sl_percent=0.025,
        base_tp_percent=0.05,
        ai_tp_percent=tpsl_result.tp_percent,
        ai_trail_percent=tpsl_result.trail_percent,
        confidence=rl_output['confidence']
    )
    print(f"   Final TP: {hybrid['final_tp_percent']:.2%}")
    print(f"   Final SL: {hybrid['final_sl_percent']:.2%}")
    print(f"   Trailing: {hybrid['trail_callback_percent']:.2%}")
    print(f"   Mode: {hybrid['mode']}")
    
    # STEP 4: Validation checks
    print("\n[4] Validation Checks...")
    checks = {
        'RL outputs confidence': rl_output['confidence'] > 0,
        'TP > SL': hybrid['final_tp_percent'] > hybrid['final_sl_percent'],
        'TP >= 5%': hybrid['final_tp_percent'] >= 0.05,
        'R:R >= 1.5x': (hybrid['final_tp_percent'] / hybrid['final_sl_percent']) >= 1.5,
        'Trailing enabled for high conf': hybrid['trail_callback_percent'] > 0 if rl_output['confidence'] > 0.8 else True
    }
    
    all_passed = all(checks.values())
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {check}")
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ TP SMOKE TEST PASSED")
    else:
        print("❌ TP SMOKE TEST FAILED")
    
    return all_passed

if __name__ == "__main__":
    asyncio.run(run_tp_smoke_test())
```

---

## KNOWN LIMITATIONS

1. **RL v3 TP Prediction Not Trained:**
   - Current RL only trained on action selection
   - TP zone prediction requires separate reward component
   - Requires retraining with TP-specific rewards

2. **Risk v3 Integration Incomplete:**
   - Dynamic TP/SL does not query real-time ESS
   - No systemic risk event detection
   - Portfolio correlation not factored into TP width

3. **Trailing Stop Not Fully Dynamic:**
   - Initial callback set at trade entry
   - Not automatically tightened as profit increases
   - Requires Position Monitor enhancement

4. **No TP Performance Tracking:**
   - TP hit rate not tracked
   - Average TP slippage not measured
   - No feedback loop to RL training

---

## HOW TO ENABLE/DISABLE FEATURES

### Enable Dynamic TP/SL (Current Default)
```bash
export QT_AI_DYNAMIC_TPSL_ENABLED=true
export QT_AI_DYNAMIC_TPSL_OVERRIDE=true  # Override legacy Exit Policy Engine
```

### Enable Trailing Stop
```python
# In system_services.py config
dynamic_tpsl_enabled: bool = True
dynamic_tpsl_override_legacy: bool = True
```

### Adjust TP Parameters
```python
# In dynamic_tpsl.py DynamicTPSLCalculator.__init__()
self.base_tp = 0.06  # 6% base TP
self.base_sl = 0.025  # 2.5% base SL
self.min_tp_pct = 0.05  # 5% minimum TP
```

### Disable for Testing
```bash
export QT_AI_DYNAMIC_TPSL_ENABLED=false
# Falls back to Exit Policy Engine
```

---

## NEXT STEPS

1. **Implement RL v3 TP Zone Prediction** (HIGH PRIORITY)
   - Add TP zone output to `rl_manager_v3.py`
   - Add TP reward component to environment
   - Retrain with TP-aware reward function

2. **Integrate Risk v3 Context** (HIGH PRIORITY)
   - Query ESS from Risk v3 during TP calculation
   - Add systemic risk detection → defensive TP
   - Factor in portfolio correlation

3. **Implement Dynamic Trailing Rearm** (MEDIUM PRIORITY)
   - Add profit-based callback tightening to Position Monitor
   - Test with multiple positions simultaneously

4. **Add TP Performance Metrics** (MEDIUM PRIORITY)
   - Track TP hit rate per strategy
   - Measure average TP slippage
   - Feed back to RL training loop

5. **Create Full Test Suite** (MEDIUM PRIORITY)
   - RL TP zone generation tests
   - Risk v3 integration tests
   - Trailing stop rearm tests
   - Mutex/race condition tests

---

**Report Generated:** December 7, 2025  
**Status:** 🟡 PARTIAL - Core pipeline working, enhancements identified  
**Compatibility:** ✅ Backward compatible with v2.0
