# RL v3 (PPO) System Documentation

## Overview

RL v3 is a **Proximal Policy Optimization (PPO)** based reinforcement learning system for autonomous trading. It runs **independently** alongside the existing RL v2 (Q-learning) system.

### Key Differences: RL v2 vs RL v3

| Feature | RL v2 (Q-learning) | RL v3 (PPO) |
|---------|-------------------|-------------|
| **Algorithm** | Value-based (TD-learning) | Policy gradient |
| **Framework** | Custom Q-tables | PyTorch neural networks |
| **Actions** | 100 discrete (60 meta + 40 sizing) | 6 discrete (HOLD, LONG, SHORT, REDUCE, CLOSE, FLATTEN) |
| **State** | 11 features | 64-dimensional feature vector |
| **Training** | Online Q-learning with replay | PPO with GAE and clipped objective |
| **Integration** | EventBus v2, RiskGuard, Strategy | Standalone (future EventBus integration) |
| **Status** | Production-ready | Experimental/Research |

## Architecture

```
backend/domains/learning/rl_v3/
├── __init__.py              # Module exports
├── config_v3.py             # PPO hyperparameters
├── features_v3.py           # Feature extraction (64-dim)
├── reward_v3.py             # Reward function
├── policy_network_v3.py     # PyTorch policy network
├── value_network_v3.py      # PyTorch value network
├── ppo_buffer_v3.py         # Experience buffer with GAE
├── ppo_agent_v3.py          # PPO agent (act/evaluate)
├── ppo_trainer_v3.py        # PPO training loop
├── env_v3.py                # Gym trading environment
└── rl_manager_v3.py         # Main interface
```

## Installation

### Required Dependencies

```bash
# Install core dependencies (REQUIRED)
pip install torch numpy gym

# If your project uses gymnasium instead:
pip install torch numpy gymnasium
# Then change: import gym → import gymnasium as gym
```

### Important Notes

⚠️ **Gym vs Gymnasium**
- Currently using deprecated `gym` package
- Warning message is safe to ignore for now
- For gymnasium: Change `import gym` to `import gymnasium as gym` in `env_v3.py`
- Spaces API is compatible between gym and gymnasium

⚠️ **Virtual Environment**
- Ensure you install dependencies in your project's venv
- Check with: `pip list | grep torch`
- Activate venv first: `source .venv/bin/activate` (Linux/Mac) or `.venv\Scripts\activate` (Windows)

## Quick Start

### 1. Basic Usage

```python
from backend.domains.learning.rl_v3.rl_manager_v3 import RLv3Manager

# Create manager
manager = RLv3Manager()

# Train for 100 episodes
metrics = manager.train(num_episodes=100)
print(f"Average reward: {metrics['avg_reward']:.2f}")

# Save trained model
manager.save()

# Predict action
obs_dict = {
    'price_change_1m': 0.001,
    'price_change_5m': 0.005,
    'volatility': 0.02,
    'rsi': 55.0,
    'position_size': 0.0,
    'balance': 10000.0,
    'equity': 10000.0,
    'regime': 'TREND',
    # ... other features
}

result = manager.predict(obs_dict)
print(f"Action: {result['action']}, Confidence: {result['confidence']:.2f}")
```

### 2. Run Sandbox

```bash
python scripts/rl_v3_sandbox.py
```

### 3. Run Tests

```bash
python tests/integration/test_rl_v3_basic.py
```

## Configuration

Edit `backend/domains/learning/rl_v3/config_v3.py`:

```python
@dataclass
class RLv3Config:
    # Network architecture
    state_dim: int = 64
    hidden_dim: int = 128
    action_dim: int = 6
    
    # PPO hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99              # Discount factor
    lambda_gae: float = 0.95         # GAE lambda
    clip_range: float = 0.2          # PPO clip epsilon
    entropy_coef: float = 0.01       # Entropy bonus
    
    # Training
    n_epochs: int = 10               # Epochs per update
    batch_size: int = 64
    buffer_size: int = 2048
    
    # Environment
    max_steps_per_episode: int = 1000
    initial_balance: float = 10000.0
```

## Action Space

RL v3 uses **6 discrete actions**:

- **0: HOLD** - Do nothing
- **1: LONG** - Open long position (10% of equity)
- **2: SHORT** - Open short position (10% of equity)
- **3: REDUCE** - Reduce position by 50%
- **4: CLOSE** - Close entire position
- **5: EMERGENCY_FLATTEN** - Emergency exit all positions

## State Space (64 dimensions)

1. **Price Changes** (3): 1m, 5m, 15m
2. **Technical Indicators** (3): volatility, RSI, MACD
3. **Position Info** (2): size, side
4. **Account Metrics** (2): equity/balance ratio, PnL ratio
5. **Regime Encoding** (4): one-hot (TREND, RANGE, BREAKOUT, MEAN_REVERSION)
6. **Additional Features** (4): trend_strength, volume_ratio, spread, time_of_day
7. **Padding**: Remaining dimensions padded to 64

## Reward Function

```python
reward = (
    pnl_delta * 100                    # Profit/loss (scaled)
    - drawdown² * 50                   # Drawdown penalty
    - position_penalty                 # High volatility penalty
    + regime_alignment * 2             # Regime bonus
    + 0.1                              # Survival bonus
)
```

## Training Process

1. **Collect trajectories** using current policy
2. **Store experience** in PPO buffer (2048 steps)
3. **Compute advantages** using GAE (λ=0.95)
4. **Update policy** using clipped surrogate objective
5. **Update value network** using MSE loss
6. **Repeat** for multiple epochs (10) with mini-batches (64)

## PPO Algorithm Details

### Clipped Surrogate Objective

```
L^CLIP(θ) = E[min(r(θ)·A, clip(r(θ), 1-ε, 1+ε)·A)]

where:
  r(θ) = π_θ(a|s) / π_old(a|s)  (probability ratio)
  A = advantage estimate
  ε = 0.2  (clip range)
```

### Generalized Advantage Estimation (GAE)

```
A_t = Σ(γλ)^k δ_{t+k}

where:
  δ_t = r_t + γV(s_{t+1}) - V(s_t)  (TD error)
  γ = 0.99  (discount)
  λ = 0.95  (GAE lambda)
```

## Future Integration with Quantum Trader

### Option 1: EventBus Integration

```python
# Subscribe to market events
event_bus.subscribe('market.tick', rl_v3_handler)
event_bus.subscribe('signal.generated', rl_v3_handler)

# Publish RL v3 decisions
event_bus.publish('rl_v3.decision', {
    'action': action,
    'confidence': confidence,
    'value': value
})
```

### Option 2: API Route

```python
@router.post("/api/v1/rl/v3/predict")
async def predict_v3(obs: ObservationDict):
    result = manager.predict(obs.dict())
    return {
        'action': result['action'],
        'confidence': result['confidence']
    }
```

### Option 3: Shadow Mode

Run RL v3 in parallel with RL v2 for A/B testing:

```python
# RL v2 (Q-learning) - production
rl_v2_decision = rl_v2_manager.predict(obs)

# RL v3 (PPO) - shadow mode
rl_v3_decision = rl_v3_manager.predict(obs)

# Compare performance
log_comparison(rl_v2_decision, rl_v3_decision, actual_outcome)
```

## Coexistence Strategy

Both systems run independently:

- **RL v2 (Q-learning)**: Integrated with EventBus, RiskGuard, Strategy layers
- **RL v3 (PPO)**: Standalone module for experimentation

No conflicts - different module paths:
- `backend/domains/learning/rl_v2/`
- `backend/domains/learning/rl_v3/`

## Performance Notes

- **Training time**: ~2 episodes/second on CPU
- **Inference**: <1ms per prediction
- **Memory**: ~100MB for networks + buffer
- **GPU support**: Automatic if CUDA available

## Known Limitations

1. **Gym Warning**: Using deprecated `gym` instead of `gymnasium`. Migration recommended for future versions.
2. **Synthetic Prices**: Environment uses random walk. Replace with real/replay data for production.
3. **Simplified Reward**: Current reward function is basic. Tune for specific trading objectives.
4. **No Risk Management**: Unlike RL v2, RL v3 doesn't integrate with RiskGuard yet.

## ⚠️ Critical Warnings

### PPOBuffer.get() Requires Full Buffer

```python
# ❌ WRONG - Will crash
buffer = PPOBuffer(size=2048, state_dim=64)
buffer.store(...)  # Only 100 steps
data = buffer.get()  # AssertionError!

# ✅ CORRECT - Fill buffer completely
buffer = PPOBuffer(size=2048, state_dim=64)
while buffer.ptr < buffer.size:
    buffer.store(...)
buffer.finish_path()
data = buffer.get()  # Works!
```

**Why**: PPO needs complete trajectories for advantage estimation.

### Environment Uses Synthetic Prices

```python
# Current: Random walk prices
env = TradingEnvV3(config)  # prices = random walk

# For production: Use real data
# Option 1: Historical replay
env = TradingEnvV3(config, price_data=historical_df)

# Option 2: Live feed
env = TradingEnvV3(config, price_source='live')
```

**Impact**: Training on random walk != training on real market dynamics.

### Dependencies Must Match Project

```bash
# Check your project's environment
pip list | grep gym

# If using gymnasium:
# 1. Install: pip install gymnasium
# 2. Change in env_v3.py:
#    import gymnasium as gym
# 3. Spaces API is compatible
```

## Troubleshooting

### Import Errors

```bash
# Make sure you're in the project root
cd c:\quantum_trader

# Ensure dependencies are installed
pip install torch numpy gym
```

### CUDA Errors

```python
# Force CPU mode if CUDA issues
import torch
torch.set_default_device('cpu')
```

### Low Rewards

- Increase `initial_balance`
- Reduce `max_steps_per_episode`
- Adjust reward function weights
- Train for more episodes (500+)

## Roadmap

- [ ] Migrate from `gym` to `gymnasium`
- [ ] Add real price data loader
- [ ] Integrate with EventBus v2
- [ ] Add RiskGuard compatibility layer
- [ ] Multi-asset support
- [ ] Hyperparameter tuning script
- [ ] Tensorboard logging
- [ ] Benchmark vs RL v2 on historical data

## Migration Guides

### Migrating to Gymnasium

If your project uses `gymnasium` instead of `gym`:

**Step 1**: Install gymnasium
```bash
pip uninstall gym
pip install gymnasium
```

**Step 2**: Update imports in `backend/domains/learning/rl_v3/env_v3.py`
```python
# OLD
import gym
from gym import spaces

# NEW
import gymnasium as gym
from gymnasium import spaces
```

**Step 3**: Test (no other changes needed)
```bash
python tests/integration/test_rl_v3_basic.py
```

Spaces API is compatible - no code changes required!

### Adding Real Price Data

To replace synthetic prices with real market data:

**Option 1**: Historical replay
```python
# In env_v3.py
class TradingEnvV3:
    def __init__(self, config, price_data=None):
        if price_data is not None:
            self.prices = price_data  # Use provided data
        else:
            self.prices = self._generate_price_series()  # Fallback to random
```

**Option 2**: Live market feed
```python
# In env_v3.py
class TradingEnvV3:
    def __init__(self, config, market_feed=None):
        self.market_feed = market_feed
    
    def step(self, action):
        if self.market_feed:
            self.current_price = self.market_feed.get_price()
        else:
            self.current_price = self.prices[self.current_step]
```

## References

- PPO Paper: https://arxiv.org/abs/1707.06347
- GAE Paper: https://arxiv.org/abs/1506.02438
- OpenAI Spinning Up: https://spinningup.openai.com/en/latest/algorithms/ppo.html

## Status

✅ **Implementation**: Complete  
✅ **Tests**: Passing (2/2)  
✅ **Sandbox**: Working  
⏳ **Integration**: Not started  
⏳ **Production**: Not ready  

---

**Last Updated**: 2025
**Version**: 3.0.0
