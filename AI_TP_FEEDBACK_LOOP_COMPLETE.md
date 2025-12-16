# TP Feedback Loop - CLM v3 ↔ RL v3 Integration

## Overview

Successfully implemented a **closed-loop feedback system** that connects production TP performance metrics to RL v3 training, enabling the system to automatically adjust reward weights based on real trading outcomes.

## Data Flow

```
TPPerformanceTracker 
    ↓ (metrics per strategy/symbol)
get_strategy_tp_feedback()
    ↓ (hit_rate, avg_r_multiple, attempts)
CLM v3 Orchestrator
    ↓ (compute tp_reward_weight)
TrainingJob.training_params['tp_reward_weight']
    ↓
TradingEnvV3(tp_reward_weight=X)
    ↓
compute_reward(..., tp_reward_weight=X)
    ↓
RL Agent Training (modulated TP bonus)
```

## Components Modified

### 1. TPPerformanceTracker (`backend/services/monitoring/tp_performance_tracker.py`)

**Added**: `get_strategy_tp_feedback(strategy_id, symbol=None, min_attempts=10)`

Returns aggregated TP metrics for a specific strategy/symbol pair:
```python
{
    'tp_hit_rate': 0.30,          # 30% hit rate
    'avg_r_multiple': 3.33,       # Estimated using 1/hit_rate
    'total_attempts': 20,
    'total_hits': 6,
    'total_misses': 14,
    'premature_exit_rate': 0.10
}
```

**Key Features**:
- Aggregates metrics across all symbols if `symbol=None`
- Returns `None` if insufficient data (< `min_attempts`)
- Uses inverse relationship: `avg_r ≈ 1 / hit_rate` (matches TPOptimizerV3 logic)

**Added**: `get_tp_tracker()` alias for convenience

### 2. RL v3 Reward Function (`backend/domains/learning/rl_v3/reward_v3.py`)

**Modified**: `compute_reward()` to accept `tp_reward_weight` parameter

```python
def compute_reward(
    pnl_delta: float,
    drawdown: float,
    position_size: float,
    regime_alignment: float,
    volatility: float = 0.02,
    tp_zone_accuracy: float = 0.0,
    tp_reward_weight: float = 1.0  # NEW: Configurable weight
) -> float:
    # ... existing logic ...
    
    # TP bonus with configurable weight
    if tp_zone_accuracy > 0:
        tp_accuracy_bonus = tp_zone_accuracy * 5.0 * tp_reward_weight
        reward += tp_accuracy_bonus
    
    return reward
```

**Impact**:
- `tp_reward_weight=2.0` → doubles TP accuracy bonus (encourages better TP prediction)
- `tp_reward_weight=0.5` → halves TP accuracy bonus (deprioritizes TP accuracy)
- `tp_reward_weight=1.0` → default behavior (backward compatible)

### 3. Trading Environment (`backend/domains/learning/rl_v3/env_v3.py`)

**Modified**: `TradingEnvV3.__init__()` to accept `tp_reward_weight` parameter

```python
def __init__(
    self,
    config: RLv3Config,
    market_data_provider: Optional[MarketDataProvider] = None,
    tp_reward_weight: float = 1.0  # NEW: Set by CLM v3
):
    self.tp_reward_weight = tp_reward_weight
    # ... rest of initialization ...
```

**Modified**: `step()` to pass `tp_reward_weight` to `compute_reward()`

```python
reward = compute_reward(
    pnl_delta / self.balance,
    drawdown,
    abs(self.position_size) / self.equity if self.equity > 0 else 0.0,
    regime_alignment,
    volatility,
    tp_zone_accuracy=self.tp_zone_accuracy,
    tp_reward_weight=self.tp_reward_weight  # Pass through
)
```

### 4. CLM v3 Orchestrator (`backend/services/clm_v3/orchestrator.py`)

**Added**: `_enrich_rl_training_with_tp_feedback(job: TrainingJob)`

Fetches TP performance metrics and computes `tp_reward_weight` based on production performance:

```python
async def _enrich_rl_training_with_tp_feedback(self, job: TrainingJob) -> None:
    """
    Enrich RL training job with TP performance feedback.
    
    Logic:
    - Low hit rate (< 0.45) + good R → increase weight (2.0)
    - High hit rate (> 0.70) + low R → decrease weight (0.5)
    - Optimal → default weight (1.0)
    """
    tp_tracker = get_tp_tracker()
    tp_feedback = tp_tracker.get_strategy_tp_feedback(strategy_id, symbol)
    
    if not tp_feedback:
        job.training_params['tp_reward_weight'] = 1.0
        return
    
    hit_rate = tp_feedback['tp_hit_rate']
    avg_r = tp_feedback['avg_r_multiple']
    
    # Compute weight based on performance
    if hit_rate < 0.45 and avg_r >= 1.2:
        tp_reward_weight = 2.0  # Encourage better TP prediction
    elif hit_rate > 0.70 and avg_r < 1.2:
        tp_reward_weight = 0.5  # Deprioritize TP accuracy
    elif hit_rate >= 0.45 and hit_rate <= 0.70 and avg_r < 1.2:
        tp_reward_weight = 1.2  # Moderate increase
    else:
        tp_reward_weight = 1.0  # Optimal
    
    job.training_params['tp_reward_weight'] = tp_reward_weight
```

**Added**: `_log_tp_optimizer_recommendations(job, strategy_id)` (optional)

Optionally logs TPOptimizerV3 recommendations for observability (if `enable_tp_optimizer_logging=True` in config).

**Modified**: `handle_training_job()` to call enrichment before training

```python
# Step 0: Enrich RL training with TP feedback
await self._enrich_rl_training_with_tp_feedback(job)

# Step 1: Fetch training data
training_data = await self._fetch_training_data(job)

# Step 2: Train model (job.training_params now contains tp_reward_weight)
model_version = await self._train_model(job, training_data)
```

## Decision Logic

The CLM orchestrator uses a 4-scenario decision tree:

| Scenario | Hit Rate | Avg R | tp_reward_weight | Reasoning |
|----------|----------|-------|------------------|-----------|
| **1** | < 45% | ≥ 1.2 | **2.0** | TPs too far. Increase weight to encourage agent to learn better TP prediction. |
| **2** | > 70% | < 1.2 | **0.5** | TPs too close. Decrease weight to deprioritize TP accuracy, focus on other metrics. |
| **3** | 45-70% | < 1.2 | **1.2** | Hit rate ok but R low. Moderate increase to improve R while maintaining hit rate. |
| **4** | 45-70% | ≥ 1.2 | **1.0** | Optimal performance. Use default weight (no adjustment needed). |

## Usage Example

### Nightly RL Training with TP Feedback

```python
from backend.services.clm_v3.models import TrainingJob, ModelType, TriggerReason
from backend.services.clm_v3.orchestrator import ClmOrchestrator

# Create RL v3 training job
job = TrainingJob(
    model_type=ModelType.RL_V3,
    symbol="BTCUSDT",
    trigger_reason=TriggerReason.PERIODIC,
    training_params={
        'strategy_id': 'RL_V3',
        'learning_rate': 0.0003,
        'batch_size': 64
    }
)

# Orchestrator enriches with TP feedback automatically
model_version = await orchestrator.handle_training_job(job)

# job.training_params now contains:
# {
#     'strategy_id': 'RL_V3',
#     'learning_rate': 0.0003,
#     'batch_size': 64,
#     'tp_hit_rate': 0.30,            # From TP tracker
#     'avg_tp_r_multiple': 3.33,      # Estimated
#     'tp_reward_weight': 2.0,        # Computed by CLM
#     'tp_feedback_reason': 'Low hit rate (30.0%) but good R (3.33) - increasing TP weight'
# }
```

### Using TP Reward Weight in Training

```python
from backend.domains.learning.rl_v3.env_v3 import TradingEnvV3
from backend.domains.learning.rl_v3.config_v3 import RLv3Config

# Extract tp_reward_weight from job config
tp_reward_weight = job.training_params.get('tp_reward_weight', 1.0)

# Create environment with modulated reward
config = RLv3Config(...)
env = TradingEnvV3(config, tp_reward_weight=tp_reward_weight)

# Train PPO agent (TP accuracy bonus is now weighted)
# ... training loop ...
```

## Testing

### CLM v3 Integration Tests

**File**: `tests/services/test_clm_v3_tp_integration.py`

**Coverage**: ✅ All 11 tests passing

- ✅ Low hit rate + high R → increases weight to 2.0
- ✅ High hit rate + low R → decreases weight to 0.5
- ✅ Optimal metrics → default weight 1.0
- ✅ Acceptable hit rate + low R → moderate increase to 1.2
- ✅ Insufficient data → fallback to default 1.0
- ✅ Non-RL models skip enrichment
- ✅ RL training job receives TP feedback
- ✅ Multi-symbol training aggregates metrics
- ✅ TPOptimizer recommendations logged (optional)
- ✅ TPOptimizer disabled by default
- ✅ TP tracker error falls back to default

### Reward Function Tests

**File**: `tests/rl_v3/test_reward_v3_tp_weighted.py`

**Coverage**: Comprehensive test scenarios for weighted TP bonus

- Default weight=1.0 matches baseline behavior
- Double weight=2.0 doubles TP bonus (+4.0 for 80% accuracy)
- Half weight=0.5 halves TP bonus
- Zero weight=0.0 eliminates TP bonus
- High weight=5.0 makes TP accuracy dominant
- Backward compatibility (omitting weight uses default)

## Benefits

### 1. Adaptive Learning
- System automatically adjusts RL training focus based on production performance
- When TPs are hitting poorly → increase focus on TP prediction
- When TPs are hitting too often but R is low → deprioritize TP accuracy, focus on letting trades run

### 2. Closed-Loop Feedback
- Real trading outcomes directly influence learning
- No manual tuning of TP reward weights required
- Continuous improvement cycle

### 3. Observability
- All TP feedback stored in `TrainingJob.training_params`
- Can track:
  - `tp_hit_rate`: Actual production hit rate
  - `avg_tp_r_multiple`: Estimated R from hit rate
  - `tp_reward_weight`: Computed weight
  - `tp_feedback_reason`: Human-readable explanation
- Optional TPOptimizer recommendations logged for manual review

### 4. Backward Compatible
- Default `tp_reward_weight=1.0` preserves existing behavior
- Non-RL models unaffected
- Existing RL training code works without modification

## Configuration

### Enable TPOptimizer Logging

```python
config = {
    'enable_tp_optimizer_logging': True,  # Log TPOptimizer recommendations
    'promotion_criteria': {...}
}

orchestrator = ClmOrchestrator(
    registry=registry,
    training_adapter=training_adapter,
    backtest_adapter=backtest_adapter,
    config=config
)
```

When enabled, CLM will call TPOptimizerV3 and log recommendations in `TrainingJob.training_params['tp_optimizer_recommendations']` for manual review.

### Custom TP Feedback Thresholds

Edit `_enrich_rl_training_with_tp_feedback()` thresholds:

```python
# Current defaults
if hit_rate < 0.45 and avg_r >= 1.2:  # Low hit rate threshold
    tp_reward_weight = 2.0
elif hit_rate > 0.70 and avg_r < 1.2:  # High hit rate threshold
    tp_reward_weight = 0.5
```

## Future Enhancements

1. **Dynamic Thresholds**: Learn optimal hit_rate/avg_r thresholds per symbol based on volatility
2. **Gradual Weight Adjustment**: Use exponential moving average instead of discrete jumps
3. **Multi-Objective Optimization**: Balance TP accuracy vs. R multiple vs. PnL
4. **A/B Testing**: Deploy models with different weights and compare performance
5. **Actual R Tracking**: Replace estimated avg_r with real PnL-based R calculation

## Summary

The TP feedback loop creates a **self-improving system** where:

1. **TPPerformanceTracker** collects real TP performance metrics from production
2. **CLM v3** analyzes metrics and computes optimal `tp_reward_weight`
3. **RL v3** uses weighted reward to adjust learning focus
4. **Agent** learns better TP prediction when needed, or deprioritizes it when optimal
5. **Cycle repeats** with each training run, continuously adapting to performance

This closes the gap between production reality and training objectives, enabling the system to automatically correct course when TP performance diverges from targets. 🚀
