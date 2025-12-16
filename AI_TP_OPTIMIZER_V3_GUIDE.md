# TP Optimizer v3 - Usage Guide

## Overview

The TP Optimizer v3 module analyzes Take Profit performance metrics and generates profile adjustment recommendations. It compares actual hit rates and R multiples against configurable targets to identify when TPs are too close or too far.

## Architecture

**Module**: `backend/services/monitoring/tp_optimizer_v3.py`

**Dependencies**:
- `TPPerformanceTracker` - Provides metrics (hit rates, slippage, timing)
- `tp_profiles_v3` - Provides profile definitions and registration API

**Key Components**:
- `TPOptimizationTarget` - Configuration defining target hit rate bands and min avg R
- `TPAdjustmentRecommendation` - Recommendation with direction, scale factor, confidence
- `TPOptimizerV3` - Core optimizer class with evaluation and application logic

## Decision Logic

The optimizer uses a 4-scenario decision tree:

| Scenario | Hit Rate | Avg R | Recommendation | Scale Factor |
|----------|----------|-------|----------------|--------------|
| **1** | Below min (e.g. 30%) | Above min (e.g. 3.3R) | **CLOSER** | 0.95x (5% closer) |
| **2** | Above max (e.g. 85%) | Below min (e.g. 1.18R) | **FURTHER** | 1.05x (5% further) |
| **3** | Within band (e.g. 55%) | Below min (e.g. 1.1R) | **FURTHER** | 1.05x |
| **4** | Within band (e.g. 55%) | Above min (e.g. 1.8R) | **NO CHANGE** | None |

**R Multiple Estimation**: `avg_r ≈ 1 / hit_rate`
- 50% hit rate → 2.0R
- 70% hit rate → 1.4R
- 30% hit rate → 3.3R

## Usage Examples

### 1. Nightly Batch Optimization

```python
from backend.services.monitoring.tp_optimizer_v3 import get_tp_optimizer

# Get singleton optimizer instance
optimizer = get_tp_optimizer()

# Run batch optimization across all tracked pairs
recommendations = optimizer.optimize_all_profiles_once()

# Log recommendations
for rec in recommendations:
    print(f"{rec.symbol} ({rec.strategy_id}): {rec.direction.value} - {rec.reason}")
    print(f"  Scale factor: {rec.suggested_scale_factor:.3f}")
    print(f"  Confidence: {rec.confidence:.2%}")
    print(f"  Metrics: {rec.metrics_snapshot['tp_hit_rate']:.1%} hit, "
          f"{rec.metrics_snapshot['avg_r_multiple']:.2f}R")
    print()

# Apply high-confidence recommendations (optional)
for rec in recommendations:
    if rec.confidence > 0.5:
        adjusted_profile = optimizer.apply_recommendation(rec, persist=True)
        print(f"Applied adjustment to {rec.symbol}")
```

### 2. Strategy-Specific Optimization

```python
from backend.services.monitoring.tp_optimizer_v3 import optimize_profiles_for_strategy

# Optimize all pairs for a specific strategy
recommendations = optimize_profiles_for_strategy("RL_V3")

# Review and apply selectively
for rec in recommendations:
    if rec.direction == AdjustmentDirection.CLOSER:
        print(f"{rec.symbol}: Bringing TPs closer due to {rec.reason}")
        optimizer.apply_recommendation(rec, persist=True)
```

### 3. Single Pair Evaluation

```python
from backend.services.monitoring.tp_optimizer_v3 import get_tp_optimizer

optimizer = get_tp_optimizer()

# Evaluate specific pair
rec = optimizer.evaluate_profile("RL_V3", "BTCUSDT")

if rec:
    print(f"Recommendation: {rec.direction.value}")
    print(f"Reason: {rec.reason}")
    print(f"Suggested scale: {rec.suggested_scale_factor}")
    
    # Apply in log-only mode (no persistence)
    adjusted_profile = optimizer.apply_recommendation(rec, persist=False)
    print(f"Would adjust profile: {adjusted_profile.name}")
else:
    print("Performance optimal - no adjustment needed")
```

### 4. Custom Targets

```python
from backend.services.monitoring.tp_optimizer_v3 import (
    TPOptimizerV3,
    TPOptimizationTarget,
    get_tp_optimizer
)

# Define custom targets for aggressive strategy
custom_targets = [
    TPOptimizationTarget(
        strategy_id="SCALP_V2",
        symbol="*",
        min_hit_rate=0.60,  # Want 60-85% hit rate
        max_hit_rate=0.85,
        min_avg_r=0.8,      # Accept lower R for scalping
        min_attempts=30,    # Need more data
        adjustment_step=0.03  # Smaller adjustments (3%)
    )
]

# Load custom targets
optimizer = get_tp_optimizer()
optimizer.load_targets(custom_targets)

# Evaluate with custom targets
recommendations = optimizer.optimize_all_profiles_once()
```

## Application Modes

### Log-Only Mode (persist=False)
```python
adjusted_profile = optimizer.apply_recommendation(rec, persist=False)
# Returns adjusted profile without modifying runtime state
# Useful for testing and preview
```

### Runtime Override Mode (persist=True)
```python
adjusted_profile = optimizer.apply_recommendation(rec, persist=True)
# Stores scale factor in _runtime_overrides dict (in-memory)
# Marks recommendation as applied
# Future: Could integrate with PolicyStore/Redis
```

### Persistent Profile Registration (future)
```python
# Current implementation stores in runtime dict
# Future enhancement: register custom profile via register_custom_profile()
# Would persist to database for permanent adjustments
```

## Integration Points

### With Nightly Jobs

```python
# In nightly_optimization_job.py
from backend.services.monitoring.tp_optimizer_v3 import get_tp_optimizer

async def run_tp_optimization():
    """Nightly TP optimization task."""
    optimizer = get_tp_optimizer()
    
    # Get recommendations
    recommendations = optimizer.optimize_all_profiles_once()
    
    # Log all recommendations
    for rec in recommendations:
        logger.info(
            f"TP Optimization Recommendation: {rec.symbol} ({rec.strategy_id})",
            extra={
                "direction": rec.direction.value,
                "scale_factor": rec.suggested_scale_factor,
                "confidence": rec.confidence,
                "reason": rec.reason,
                "hit_rate": rec.metrics_snapshot['tp_hit_rate'],
                "avg_r": rec.metrics_snapshot['avg_r_multiple']
            }
        )
    
    # Apply high-confidence recommendations
    applied_count = 0
    for rec in recommendations:
        if rec.confidence >= 0.6:  # Only apply if 60%+ confidence
            optimizer.apply_recommendation(rec, persist=True)
            applied_count += 1
    
    logger.info(f"Applied {applied_count}/{len(recommendations)} TP optimizations")
```

### With CLM (Continuous Learning Manager)

```python
# In continuous_learning_manager.py
from backend.services.monitoring.tp_optimizer_v3 import get_tp_optimizer

class ContinuousLearningManager:
    def __init__(self):
        self.tp_optimizer = get_tp_optimizer()
    
    async def run_periodic_optimization(self):
        """Run TP optimization as part of CLM cycle."""
        # Evaluate all tracked pairs
        recommendations = self.tp_optimizer.optimize_all_profiles_once()
        
        # Store recommendations for review
        for rec in recommendations:
            await self._store_recommendation(rec)
        
        # Auto-apply low-risk adjustments
        for rec in recommendations:
            if self._is_low_risk(rec):
                self.tp_optimizer.apply_recommendation(rec, persist=True)
    
    def _is_low_risk(self, rec: TPAdjustmentRecommendation) -> bool:
        """Determine if recommendation is safe to auto-apply."""
        return (
            rec.confidence > 0.7 and
            0.90 <= rec.suggested_scale_factor <= 1.10 and  # Max 10% adjustment
            rec.metrics_snapshot['tp_attempts'] >= 30  # Good sample size
        )
```

## Configuration

### Default Targets

The optimizer ships with sensible defaults:

```python
# RL_V3 Strategy
TPOptimizationTarget(
    strategy_id="RL_V3",
    symbol="*",
    min_hit_rate=0.45,
    max_hit_rate=0.70,
    min_avg_r=1.2
)

# SCALP_V2 Strategy
TPOptimizationTarget(
    strategy_id="SCALP_V2",
    symbol="*",
    min_hit_rate=0.60,
    max_hit_rate=0.85,
    min_avg_r=0.8
)

# TREND_FOLLOW Strategy
TPOptimizationTarget(
    strategy_id="TREND_FOLLOW",
    symbol="*",
    min_hit_rate=0.35,
    max_hit_rate=0.60,
    min_avg_r=1.8
)
```

### Target Matching

Targets are matched with specificity:
1. Exact match: `(strategy_id="RL_V3", symbol="BTCUSDT")`
2. Wildcard match: `(strategy_id="RL_V3", symbol="*")`
3. No match: Returns `None` (no optimization)

## Monitoring

### Recommendation Metadata

Each recommendation includes:
- `strategy_id`, `symbol` - Identifies the pair
- `direction` - CLOSER, FURTHER, NO_CHANGE
- `suggested_scale_factor` - Multiplier for r_multiples (e.g. 0.95, 1.05)
- `reason` - Human-readable explanation
- `confidence` - 0.0-1.0 based on deviation from targets
- `metrics_snapshot` - Current hit_rate, avg_r, attempts, etc.
- `targets` - Target thresholds used for evaluation
- `timestamp` - When recommendation was generated
- `applied` - Whether recommendation has been applied

### Runtime Overrides

Check active overrides:
```python
# Get override for specific pair
scale = optimizer.get_runtime_override("RL_V3", "BTCUSDT")
if scale:
    print(f"Active override: {scale:.3f}x")

# Clear all overrides (useful for testing)
optimizer.clear_runtime_overrides()
```

## Testing

Run tests:
```bash
pytest tests/services/test_tp_optimizer_v3.py -v
```

Test coverage:
- Configuration models
- Target matching logic
- Decision tree (4 scenarios)
- Profile adjustment calculations
- Application modes (log-only, runtime, persistent)
- Batch optimization
- Runtime override system
- Edge cases (low sample size, missing data, etc.)

## Future Enhancements

1. **Actual R Tracking**: Replace estimated avg_r with real PnL-based R calculation
2. **PolicyStore Integration**: Store runtime overrides in Redis for persistence
3. **ML-Based Optimization**: Train models to predict optimal TP distances
4. **Symbol-Specific Targets**: Auto-tune targets per symbol based on volatility
5. **Gradual Rollout**: A/B test adjustments before full deployment
6. **Reversion Detection**: Auto-revert adjustments if performance degrades

## FAQ

**Q: When should I run optimization?**
A: Run nightly or weekly after accumulating 20+ TP attempts per pair. Avoid running during market turmoil.

**Q: Are adjustments applied automatically?**
A: No. The optimizer produces recommendations only. You must call `apply_recommendation()` to apply.

**Q: Can I preview adjustments?**
A: Yes. Use `persist=False` mode: `optimizer.apply_recommendation(rec, persist=False)` returns the adjusted profile without modifying state.

**Q: What if I disagree with a recommendation?**
A: Adjust the targets for that strategy/symbol pair. Lower `min_avg_r` or widen the hit_rate band.

**Q: How do I revert an adjustment?**
A: Call `optimizer.clear_runtime_overrides()` to remove all overrides, or manually adjust the target to force opposite recommendation.

**Q: What's the confidence score?**
A: Measures deviation from target thresholds. Higher deviation = higher confidence in recommendation. Range: 0.0-1.0.

## Summary

The TP Optimizer v3 provides a systematic, data-driven approach to TP profile optimization:

1. **Analyzes** metrics from TPPerformanceTracker
2. **Compares** to configurable target bands
3. **Generates** adjustment recommendations with confidence scores
4. **Applies** adjustments via flexible modes (log-only, runtime, persistent)
5. **Integrates** with nightly jobs and CLM for automated optimization

Use it to incrementally tune TP distances based on actual trading performance, improving hit rates without sacrificing R multiples.
