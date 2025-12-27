# RL v2 Quick Reference Guide

**Version:** 2.0 (Domain Architecture)  
**Date:** December 2, 2025  
**Status:** Production-Ready

---

## Quick Start

### 1. Basic Usage

RL v2 is **automatically enabled** when the backend starts. It integrates seamlessly with EventFlow:

```python
# RL v2 agents are initialized in main.py
# They respond to events automatically:
# - SIGNAL_GENERATED → Meta strategy decision
# - TRADE_EXECUTED → Position sizing decision  
# - POSITION_CLOSED → Learning updates
```

### 2. Check RL v2 Status

```python
# Via API endpoint (if exposed)
GET /api/rl/stats

# Programmatically
from backend.domains.learning.rl_v2.meta_strategy_agent_v2 import MetaStrategyAgentV2
from backend.domains.learning.rl_v2.position_sizing_agent_v2 import PositionSizingAgentV2

meta_agent = MetaStrategyAgentV2()
sizing_agent = PositionSizingAgentV2()

meta_stats = meta_agent.get_stats()
sizing_stats = sizing_agent.get_stats()
```

### 3. Monitor Q-Tables

```python
# Q-tables are automatically saved every 100 updates
# Location:
# - data/rl_v2/meta_strategy_q_table.json
# - data/rl_v2/position_sizing_q_table.json

# Check Q-learning stats
stats = meta_agent.get_stats()
print(f"Total states learned: {stats['q_learning_stats']['total_states']}")
print(f"Total updates: {stats['q_learning_stats']['update_count']}")
print(f"Exploration rate: {stats['q_learning_stats']['epsilon']}")
```

---

## Configuration

### Hyperparameters

Edit in agent initialization (`main.py`):

```python
meta_agent = MetaStrategyAgentV2(
    alpha=0.01,        # Learning rate (how fast to update Q-values)
    gamma=0.99,        # Discount factor (future reward importance)
    epsilon=0.1        # Exploration rate (random action probability)
)

sizing_agent = PositionSizingAgentV2(
    alpha=0.01,
    gamma=0.99,
    epsilon=0.1
)
```

### Recommended Settings

| Scenario | Alpha | Gamma | Epsilon |
|----------|-------|-------|---------|
| **Fast Learning** | 0.05 | 0.95 | 0.2 |
| **Stable Learning** | 0.01 | 0.99 | 0.1 |
| **Conservative** | 0.005 | 0.99 | 0.05 |

### Reward Tuning

Edit in `reward_engine_v2.py`:

```python
# Meta Strategy Reward Weights
meta_reward = (
    pnl_pct
    - 0.5 * max_drawdown_pct      # Drawdown penalty (default: 0.5)
    + 0.2 * sharpe_signal          # Sharpe bonus (default: 0.2)
    + 0.15 * regime_alignment      # Regime bonus (default: 0.15)
)

# Position Sizing Reward Weights
size_reward = (
    pnl_pct
    - 0.4 * risk_penalty           # Risk penalty (default: 0.4)
    + 0.1 * volatility_adjustment  # Volatility bonus (default: 0.1)
)
```

---

## Common Operations

### Manual Q-Table Save

```python
meta_agent.save_q_table()
sizing_agent.save_q_table()
```

### Reset Agent State

```python
meta_agent.reset()
sizing_agent.reset()
```

### Force Action Selection

```python
# Build market data
market_data = {
    "symbol": "BTCUSDT",
    "price_history": [50000, 51000, 52000],
    "volume_history": [1000, 1100, 1200],
    "account_balance": 10000.0,
    "confidence": 0.75,
    "portfolio_exposure": 0.3,
    "volatility": 0.02,
    "equity_history": [10000, 10100, 10200],
    "recent_trades": [
        {"pnl": 100, "result": "win"},
        {"pnl": -50, "result": "loss"}
    ]
}

# Get meta strategy action
meta_action = meta_agent.select_action(market_data)
print(f"Strategy: {meta_action.strategy}")
print(f"Model: {meta_action.model}")
print(f"Weight: {meta_action.weight}")

# Get position sizing action
sizing_action = sizing_agent.select_action(market_data)
print(f"Size Multiplier: {sizing_action.size_multiplier}")
print(f"Leverage: {sizing_action.leverage}")
```

### Manual Update

```python
# After trade closes
result_data = {
    "symbol": "BTCUSDT",
    "pnl": 200.0,
    "pnl_percentage": 2.0,
    "drawdown": 0.03,
    "sharpe_ratio": 1.8,
    "regime": "TREND",
    "predicted_regime": "TREND",
    "leverage": 20,
    "position_size_usd": 1500.0,
    "account_balance": 10200.0,
    "volatility": 0.02
}

meta_agent.update(result_data)
sizing_agent.update(result_data)
```

---

## Monitoring & Debugging

### Get Comprehensive Stats

```python
from backend.events.subscribers.rl_subscriber_v2 import RLSubscriberV2

# Assuming you have access to the subscriber instance
stats = rl_subscriber.get_stats()

print("Meta Agent Stats:")
print(f"  Agent Type: {stats['meta_agent_stats']['agent_type']}")
print(f"  Total States: {stats['meta_agent_stats']['q_learning_stats']['total_states']}")
print(f"  Updates: {stats['meta_agent_stats']['q_learning_stats']['update_count']}")
print(f"  Epsilon: {stats['meta_agent_stats']['q_learning_stats']['epsilon']:.4f}")

print("\nSizing Agent Stats:")
print(f"  Agent Type: {stats['sizing_agent_stats']['agent_type']}")
print(f"  Total States: {stats['sizing_agent_stats']['q_learning_stats']['total_states']}")
print(f"  Updates: {stats['sizing_agent_stats']['q_learning_stats']['update_count']}")
print(f"  Epsilon: {stats['sizing_agent_stats']['q_learning_stats']['epsilon']:.4f}")
```

### Check Logs

```bash
# Filter RL v2 logs
grep "RL.*v2" logs/quantum_trader.log

# Check meta agent decisions
grep "Meta Strategy Agent v2" logs/quantum_trader.log

# Check position sizing decisions
grep "Position Sizing Agent v2" logs/quantum_trader.log

# Check Q-learning updates
grep "Q-value updated" logs/quantum_trader.log
```

### Verify Q-Table Growth

```python
import json
from pathlib import Path

# Load Q-table
q_table_path = Path("data/rl_v2/meta_strategy_q_table.json")
with open(q_table_path, 'r') as f:
    data = json.load(f)

q_table = data.get("q_table", {})
print(f"Total unique states: {len(q_table)}")
print(f"Total state-action pairs: {sum(len(actions) for actions in q_table.values())}")
print(f"Current epsilon: {data.get('epsilon', 'N/A')}")
print(f"Total updates: {data.get('update_count', 'N/A')}")
```

---

## Troubleshooting

### Issue: Agents Not Learning

**Symptoms:**
- Q-table stays empty
- No Q-value updates in logs

**Solutions:**
1. Check if events are firing:
   ```bash
   grep "SIGNAL_GENERATED\|TRADE_EXECUTED\|POSITION_CLOSED" logs/quantum_trader.log
   ```

2. Verify RL is enabled:
   ```python
   # Check PolicyStore
   profile = await policy_store.get_active_risk_profile()
   print(f"RL Enabled: {profile.enable_rl}")
   ```

3. Check for errors:
   ```bash
   grep "ERROR.*RL.*v2" logs/quantum_trader.log
   ```

### Issue: Q-Table Growing Too Large

**Symptoms:**
- Q-table file > 100 MB
- Slow action selection

**Solutions:**
1. Increase state discretization:
   - Round float values more aggressively
   - Reduce state feature count

2. Implement state aggregation:
   - Group similar states together
   - Use function approximation (future: DQN)

3. Clear old Q-tables:
   ```bash
   # Backup and reset
   mv data/rl_v2/meta_strategy_q_table.json data/rl_v2/meta_strategy_q_table.backup.json
   # Agent will create new Q-table on next update
   ```

### Issue: Poor Action Selection

**Symptoms:**
- Consistently choosing bad strategies
- Low rewards

**Solutions:**
1. Increase exploration:
   ```python
   meta_agent.q_learning.epsilon = 0.3  # More random exploration
   ```

2. Adjust reward weights:
   - Increase drawdown penalty if too risky
   - Increase regime alignment if regime mismatch

3. Reset and retrain:
   ```python
   meta_agent.reset()
   sizing_agent.reset()
   ```

### Issue: Epsilon Not Decaying

**Symptoms:**
- Epsilon stays at 0.1 (initial value)

**Check:**
- Epsilon decay happens during Q-value updates
- Verify updates are happening:
  ```bash
  grep "Q-value updated" logs/quantum_trader.log | wc -l
  ```

**Solution:**
- Ensure updates are being called after position closes

---

## Performance Optimization

### 1. Reduce State Space

```python
# In state_builder_v2.py, increase rounding
def build_meta_strategy_state(data):
    # Instead of:
    discretized[k] = round(float(v), 2)
    
    # Use:
    discretized[k] = round(float(v), 1)  # Less precision = fewer unique states
```

### 2. Batch Updates

```python
# Instead of saving Q-table every 100 updates
# Save every 500 updates for better I/O performance

if self.q_learning.update_count % 500 == 0:  # Changed from 100
    self.save_q_table()
```

### 3. Async Saves

```python
import asyncio

async def save_q_table_async(self):
    await asyncio.to_thread(self.q_learning.save_q_table, self.q_table_path)
```

---

## Best Practices

### 1. Always Monitor Stats
- Check Q-table growth weekly
- Monitor epsilon decay
- Track average rewards

### 2. Tune Gradually
- Change one hyperparameter at a time
- Test for at least 100 trades before adjusting
- Keep backup of working Q-tables

### 3. Version Q-Tables
```bash
# Before major changes
cp data/rl_v2/meta_strategy_q_table.json \
   data/rl_v2/backups/meta_strategy_q_table_$(date +%Y%m%d).json
```

### 4. Use Realistic Test Data
- Test with real market scenarios
- Include edge cases (crashes, pumps)
- Verify regime detection accuracy

### 5. Document Tuning
```python
# In a tuning log file
"""
2025-12-02: Initial deployment
- Alpha: 0.01
- Gamma: 0.99
- Epsilon: 0.1
- Results: TBD

2025-12-09: Increased exploration
- Epsilon: 0.1 → 0.15
- Reason: Too much exploitation early
- Results: +5% reward improvement
"""
```

---

## API Integration (Future)

### Planned Endpoints

```python
# Get RL stats
GET /api/rl/v2/stats

# Force Q-table save
POST /api/rl/v2/save

# Update hyperparameters
PATCH /api/rl/v2/config
{
    "alpha": 0.02,
    "gamma": 0.99,
    "epsilon": 0.15
}

# Reset agent
POST /api/rl/v2/reset/{agent_type}  # meta | sizing

# Get Q-table summary
GET /api/rl/v2/qtable/summary

# Export Q-table
GET /api/rl/v2/qtable/export/{agent_type}
```

---

## References

- **Implementation Doc:** `docs/RL_V2_IMPLEMENTATION.md`
- **Verification Report:** `docs/RL_V2_VERIFICATION_REPORT.md`
- **Design Doc:** `docs/RL_V2.md`
- **Test Suite:** `tests/integration/test_rl_v2_pipeline.py`

---

**Last Updated:** December 2, 2025  
**Maintainer:** Quantum Trader AI Team
