# TP v3 QUICK REFERENCE

**Status**: ✅ COMPLETE | **Tests**: 10/10 PASSED ✅

---

## FILES CREATED

```
backend/services/monitoring/dynamic_trailing_rearm.py      (210 lines)
backend/services/risk_management/risk_v3_integration.py    (292 lines)
backend/services/monitoring/tp_performance_tracker.py      (360 lines)
tests/test_tp_v3_enhancements.py                           (360 lines)
TP_v3_ENHANCEMENTS.md                                      (This report)
```

## FILES MODIFIED

```
backend/domains/learning/rl_v3/env_v3.py                   (TP tracking)
backend/domains/learning/rl_v3/reward_v3.py                (TP bonus)
backend/services/execution/dynamic_tpsl.py                 (Risk v3)
backend/services/monitoring/position_monitor.py            (Trailing)
```

---

## ENHANCEMENTS SUMMARY

### 1. RL v3 TP-Specific Reward Component
**What**: RL agent learns optimal TP placement through explicit reward signal
**How**: +5.0 bonus for 100% TP zone accuracy, scales with hit rate
**Impact**: Agent optimizes exit timing over training episodes

### 2. Dynamic Trailing Rearm
**What**: Trailing stops tighten as profits increase
**How**: 5 profit thresholds (2%, 5%, 10%, 15%, 20%) with callback scaling
**Impact**: Locks in gains while allowing upside potential

### 3. Risk v3 Integration
**What**: Real-time risk metrics adjust TP placement
**How**: ESS, systemic risk, correlation → TP adjustment factor (0.5-1.5x)
**Impact**: TP adapts to market stress, protects capital during crises

### 4. TP Performance Tracking
**What**: Continuous monitoring of TP hit rate, slippage, timing
**How**: Persistent metrics per strategy/symbol with RL feedback
**Impact**: Data-driven optimization and performance visibility

---

## USAGE EXAMPLES

### Dynamic Trailing Rearm
```python
from backend.services.monitoring.dynamic_trailing_rearm import DynamicTrailingManager

manager = DynamicTrailingManager()

# Calculate optimal callback for 12% profit
callback = manager.calculate_optimal_callback(
    unrealized_pnl_pct=0.12,
    current_callback_pct=2.0,
    position_age_minutes=15
)
# Returns: 1.0% (tightened from 2.0%)

# Get partial TP levels
levels = manager.get_partial_tp_levels(
    entry_price=100.0,
    tp_price=115.0,
    side="LONG",
    unrealized_pnl_pct=0.12
)
# Returns: [(107.5, 0.30, "TP1"), (111.25, 0.40, "TP2"), (115.0, 0.30, "TP3")]
```

### Risk v3 Integration
```python
from backend.services.risk_management.risk_v3_integration import RiskV3Integrator

integrator = RiskV3Integrator()

# Get real-time risk context
context = integrator.get_risk_context()
# Returns: RiskV3Context(ess_factor=1.2, systemic_risk_level=0.3, ...)

# Check if TP should be tightened
if integrator.should_tighten_tp(context):
    # Apply adjustment factor
    adjustment = integrator.get_tp_adjustment_factor(context)
    tp_percent *= adjustment
```

### TP Performance Tracking
```python
from backend.services.monitoring.tp_performance_tracker import get_tp_performance_tracker

tracker = get_tp_performance_tracker()

# Record TP attempt
tracker.record_tp_attempt(
    strategy_id="momentum_5m",
    symbol="BTCUSDT",
    entry_time=datetime.now(timezone.utc),
    entry_price=50000.0,
    tp_target_price=51000.0,
    side="LONG"
)

# Record TP hit
tracker.record_tp_hit(
    strategy_id="momentum_5m",
    symbol="BTCUSDT",
    exit_time=datetime.now(timezone.utc),
    exit_price=50950.0,
    tp_target_price=51000.0,
    entry_time=entry_time,
    entry_price=50000.0,
    profit_usd=100.0
)

# Get metrics
metrics = tracker.get_metrics(strategy_id="momentum_5m")
print(f"Hit rate: {metrics[0].tp_hit_rate:.1%}")

# Get RL feedback
feedback = tracker.get_feedback_for_rl_training()
# Returns: {'tp_hit_rate': 0.75, 'avg_slippage': 0.002, 'premature_exit_rate': 0.12}
```

### RL v3 TP Reward
```python
from backend.domains.learning.rl_v3.reward_v3 import compute_reward

# Reward with perfect TP accuracy
reward = compute_reward(
    pnl_delta=100.0,
    drawdown=-0.05,
    position_size=0.5,
    regime_alignment=1.0,
    volatility=0.02,
    tp_zone_accuracy=1.0  # ← +5.0 bonus
)
# Returns: ~10007 (base reward + TP bonus)
```

---

## INTEGRATION CHECKLIST

- [x] RL v3 environment tracks TP targets
- [x] RL v3 reward includes TP accuracy bonus
- [x] Dynamic TP/SL queries Risk v3 context
- [x] Position Monitor uses Dynamic Trailing Manager
- [x] Position Monitor initializes TP Performance Tracker
- [x] All tests passing (10/10)
- [x] Documentation complete

---

## MONITORING COMMANDS

```bash
# Run all TP v3 tests
python tests/test_tp_v3.py
python tests/test_tp_v3_enhancements.py

# View TP performance metrics
cat /app/tmp/tp_metrics.json

# Check RL training logs (TP accuracy bonus)
tail -f /app/tmp/rl_v3_training.log | grep "tp_zone_accuracy"

# Monitor trailing stop adjustments
tail -f /app/logs/position_monitor.log | grep "TP v3"

# Check Risk v3 context queries
tail -f /app/logs/dynamic_tpsl.log | grep "Risk v3"
```

---

## KEY THRESHOLDS

| Metric | Value | Action |
|--------|-------|--------|
| **RL v3 TP Accuracy** | 100% | +5.0 reward bonus |
| **RL v3 TP Accuracy** | 20% | +1.0 reward bonus |
| **Trailing Rearm** | 2% profit | Start tightening |
| **Trailing Rearm** | 20% profit | 80% tighter callback |
| **Risk v3 ESS** | 1.5 | Warning (tighten TP 15%) |
| **Risk v3 ESS** | 2.5 | Critical (tighten TP 30%) |
| **Risk v3 Systemic** | 0.6 | Defensive mode |
| **Risk v3 Systemic** | 0.8 | Crisis mode |
| **TP Hit Rate Target** | 75% | Optimal performance |
| **TP Slippage Target** | <0.3% | Acceptable slippage |

---

## TROUBLESHOOTING

### TP Accuracy Low (<50%)
1. Check RL training logs: `grep "tp_zone_accuracy" /app/tmp/rl_v3_training.log`
2. Verify TP zone width: Should be ~6% from entry
3. Review market volatility: May need wider zones
4. Trigger RL retraining: `python activate_retraining_system.py`

### Trailing Stops Too Tight
1. Check profit thresholds in `dynamic_trailing_rearm.py`
2. Adjust `min_callback_pct` (default: 0.05%)
3. Increase `position_age_minutes` threshold (default: 5 min)

### Risk v3 Not Adjusting TP
1. Verify Risk v3 available: Check `dynamic_tpsl.py` logs
2. Query risk context: `integrator.get_risk_context()`
3. Check cache TTL: Default 10s, may need refresh
4. Review ESS calculation: Should be >1.5 for warnings

### TP Performance Metrics Not Saving
1. Check storage path: `/app/tmp/tp_metrics.json`
2. Verify write permissions
3. Review tracker logs: `grep "TP Tracker" /app/logs/position_monitor.log`

---

## ARCHITECTURE DIAGRAM

```
RL v3 Agent → tp_zone_multiplier, suggested_tp_pct
    ↓
Risk v3 Integrator → ess_factor, systemic_risk_level
    ↓
Dynamic TP/SL Calculator → tp_percent, sl_percent
    ↓
Hybrid TP/SL Blender → partial TP strategy
    ↓
Position Monitor → dynamic trailing rearm
    ↓
TP Performance Tracker → hit rate, slippage, timing
    ↓
RL v3 Feedback → tp_hit_rate, premature_exit_rate
```

---

**For full details, see `TP_v3_ENHANCEMENTS.md`**
