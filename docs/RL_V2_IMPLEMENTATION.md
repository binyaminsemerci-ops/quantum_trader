# RL v2 Implementation - Domain Architecture

**Author:** Quantum Trader AI Team  
**Date:** December 2, 2025  
**Version:** 2.0 (Domain Architecture)  
**Status:** ✅ COMPLETE & PRODUCTION-READY

## Overview

Complete implementation of RL v2 system using domain-driven architecture with TD-learning (Q-learning) for meta-strategy and position sizing optimization.

## Architecture

### Domain Structure

```
backend/
├── domains/
│   └── learning/
│       └── rl_v2/
│           ├── __init__.py
│           ├── reward_engine_v2.py
│           ├── state_builder_v2.py
│           ├── action_space_v2.py
│           ├── episode_tracker_v2.py
│           ├── q_learning_core.py
│           ├── meta_strategy_agent_v2.py
│           └── position_sizing_agent_v2.py
├── utils/
│   ├── regime_detector_v2.py
│   ├── volatility_tools_v2.py
│   ├── winrate_tracker_v2.py
│   └── equity_curve_tools_v2.py
└── events/
    └── subscribers/
        └── rl_subscriber_v2.py
```

## Components

### 1. Reward Engine v2 (`reward_engine_v2.py`)

**Purpose:** Advanced reward calculation with regime and risk awareness

**Meta Strategy Reward:**
```
reward = pnl - 0.5×drawdown + 0.2×sharpe + 0.15×regime_alignment
```

**Position Sizing Reward:**
```
reward = pnl - 0.4×risk_penalty + 0.1×volatility_adjustment
```

**Key Features:**
- Regime alignment scoring
- Sharpe ratio normalization
- Risk penalty calculation
- Volatility-based adjustments

### 2. State Builder v2 (`state_builder_v2.py`)

**Purpose:** Build comprehensive state representations

**Meta Strategy State:**
```python
{
    "regime": str,              # TREND, RANGE, BREAKOUT, MEAN_REVERSION
    "volatility": float,        # Market volatility
    "market_pressure": float,   # [-1, 1]
    "confidence": float,        # Signal confidence
    "previous_winrate": float,  # Trailing win rate
    "account_health": float     # [0, 1]
}
```

**Position Sizing State:**
```python
{
    "signal_confidence": float,    # [0, 1]
    "portfolio_exposure": float,   # [0, 1]
    "recent_winrate": float,       # Trailing win rate
    "volatility": float,           # Market volatility
    "equity_curve_slope": float    # Equity trend
}
```

### 3. Action Space v2 (`action_space_v2.py`)

**Purpose:** Define action spaces for agents

**Meta Strategy Actions:**
- Strategy: `dual_momentum`, `mean_reversion`, `momentum_flip`
- Model: `lstm`, `gru`, `transformer`, `ensemble`
- Weight: `0.5` - `1.5`

**Total:** 3 × 4 × 5 = **60 actions**

**Position Sizing Actions:**
- Size Multiplier: `0.5`, `0.75`, `1.0`, `1.5`, `2.0`
- Leverage: `5`, `10`, `15`, `20`, `25`, `30`, `40`, `50`

**Total:** 5 × 8 = **40 actions**

### 4. Episode Tracker v2 (`episode_tracker_v2.py`)

**Purpose:** Manage episode lifecycle with TD-learning

**Features:**
- Episode start/step/end
- Discounted return calculation
- State-action-reward history
- Episode statistics

**Discounted Return:**
```
G = Σ(γ^t × r_t)
```
where γ = 0.99 (discount factor)

### 5. Q-Learning Core (`q_learning_core.py`)

**Purpose:** TD-learning with Q-table management

**TD Update:**
```
Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
```

**Parameters:**
- α (alpha) = 0.01 (learning rate)
- γ (gamma) = 0.99 (discount factor)
- ε (epsilon) = 0.1 (exploration rate)

**Features:**
- Q-table storage (state → action → Q-value)
- Epsilon-greedy action selection
- Q-table persistence (JSON)
- Epsilon decay

### 6. Meta Strategy Agent v2 (`meta_strategy_agent_v2.py`)

**Purpose:** Optimize meta-level strategy decisions

**Optimizes:**
- Strategy selection (dual_momentum, mean_reversion, momentum_flip)
- Model selection (lstm, gru, transformer, ensemble)
- Confidence weighting (0.5 - 1.5)

**Q-Table Path:** `data/rl_v2/meta_strategy_q_table.json`

### 7. Position Sizing Agent v2 (`position_sizing_agent_v2.py`)

**Purpose:** Optimize position sizing decisions

**Optimizes:**
- Size multiplier (0.5 - 2.0)
- Leverage (5 - 50)

**Q-Table Path:** `data/rl_v2/position_sizing_q_table.json`

### 8. RL Subscriber v2 (`rl_subscriber_v2.py`)

**Purpose:** Event-driven RL integration

**Events:**
- `PREDICTION_GENERATED` → Meta strategy decision
- `TRADE_EXECUTED` → Position sizing decision
- `POSITION_CLOSED` → Update both agents

## Utility Modules

### Regime Detector v2 (`regime_detector_v2.py`)

**Regimes:**
- `TREND`: trend_strength > 0.02 && volatility > 0.015
- `RANGE`: trend_strength < 0.01 && volatility < 0.01
- `BREAKOUT`: trend_strength > 0.03 && volatility > 0.02
- `MEAN_REVERSION`: trend_strength < 0.015 && volatility ∈ [0.01, 0.015]

### Volatility Tools v2 (`volatility_tools_v2.py`)

**Methods:**
- `calculate_volatility()`: Standard deviation of returns
- `calculate_market_pressure()`: tanh(price_change × 20) → [-1, 1]

### Winrate Tracker v2 (`winrate_tracker_v2.py`)

**Features:**
- Tracks last 20 trades
- Calculates trailing win rate
- Defaults to 50% if insufficient data

### Equity Curve Tools v2 (`equity_curve_tools_v2.py`)

**Methods:**
- `calculate_equity_curve_slope()`: Linear regression slope
- `calculate_account_health()`: Drawdown-based health score [0, 1]

## Integration

### Main.py Integration

```python
from backend.domains.learning.rl_v2.meta_strategy_agent_v2 import MetaStrategyAgentV2
from backend.domains.learning.rl_v2.position_sizing_agent_v2 import PositionSizingAgentV2
from backend.events.subscribers.rl_subscriber_v2 import RLSubscriberV2

# Create agents
meta_agent = MetaStrategyAgentV2(alpha=0.01, gamma=0.99, epsilon=0.1)
sizing_agent = PositionSizingAgentV2(alpha=0.01, gamma=0.99, epsilon=0.1)

# Create subscriber
rl_subscriber = RLSubscriberV2(
    event_bus=event_bus_v2,
    meta_agent=meta_agent,
    sizing_agent=sizing_agent
)
```

## Testing

### Test Suite (`test_rl_v2_pipeline.py`)

**Tests:**
1. `test_meta_strategy_agent_select_and_update()` - Meta agent workflow
2. `test_position_sizing_agent_select_and_update()` - Sizing agent workflow
3. `test_complete_rl_v2_pipeline()` - Full pipeline integration

**Run Tests:**
```bash
python tests/integration/test_rl_v2_pipeline.py
```

## File Summary

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 38 | Package initialization |
| `reward_engine_v2.py` | 271 | Advanced reward calculation |
| `state_builder_v2.py` | 178 | State representation |
| `action_space_v2.py` | 178 | Action space definition |
| `episode_tracker_v2.py` | 206 | Episode management |
| `q_learning_core.py` | 289 | Q-learning TD updates |
| `meta_strategy_agent_v2.py` | 161 | Meta strategy optimization |
| `position_sizing_agent_v2.py` | 157 | Position sizing optimization |
| `regime_detector_v2.py` | 81 | Market regime detection |
| `volatility_tools_v2.py` | 73 | Volatility calculations |
| `winrate_tracker_v2.py` | 68 | Win rate tracking |
| `equity_curve_tools_v2.py` | 121 | Equity curve analysis |
| `rl_subscriber_v2.py` | 241 | Event integration |
| `test_rl_v2_pipeline.py` | 267 | Integration tests |
| `main.py` (update) | 31 | System integration |

**Total:** 15 files, ~2,360 lines of production code

## Key Improvements Over v1

1. **Domain Architecture**: Clean separation of concerns
2. **TD-Learning**: Q-learning with proper temporal difference updates
3. **Advanced Rewards**: Regime-aware and risk-aware rewards
4. **Enhanced State**: Market pressure, equity slope, account health
5. **Expanded Actions**: 60 meta actions, 40 sizing actions
6. **Episode Tracking**: Proper discounted returns
7. **Q-Table Persistence**: Save/load Q-tables
8. **Utility Modules**: Reusable regime detection, volatility tools, etc.

## Configuration

### Hyperparameters

```python
# Learning parameters
ALPHA = 0.01        # Learning rate
GAMMA = 0.99        # Discount factor
EPSILON = 0.1       # Exploration rate
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.01

# State parameters
WINDOW_SIZE = 30    # Equity curve window
WINRATE_WINDOW = 20 # Win rate window

# Reward parameters
DRAWDOWN_PENALTY = 0.5
SHARPE_WEIGHT = 0.2
REGIME_WEIGHT = 0.15
RISK_PENALTY_WEIGHT = 0.4
VOLATILITY_WEIGHT = 0.1
```

## Production Deployment

### Directory Structure

```bash
data/
└── rl_v2/
    ├── meta_strategy_q_table.json
    └── position_sizing_q_table.json
```

### Startup

1. System initializes agents with saved Q-tables
2. Subscriber registers for events
3. Agents make decisions on prediction/trade events
4. Agents update on position close events
5. Q-tables saved every 100 updates

### Monitoring

**Get Stats:**
```python
stats = rl_subscriber.get_stats()
```

**Returns:**
```python
{
    "meta_agent_stats": {
        "agent_type": "meta_strategy",
        "q_learning_stats": {...},
        "episode_stats": {...},
        "action_space_size": 60
    },
    "sizing_agent_stats": {
        "agent_type": "position_sizing",
        "q_learning_stats": {...},
        "episode_stats": {...},
        "action_space_size": 40
    }
}
```

## Future Enhancements

1. **Deep Q-Networks (DQN)**: Replace Q-tables with neural networks
2. **Multi-Agent Communication**: Agents share information
3. **Hierarchical RL**: Meta-agent controls multiple sub-agents
4. **Experience Replay**: Store and replay past experiences
5. **Prioritized Replay**: Focus on important experiences

## References

- RL_V2.md - Original specification
- Q-Learning: Watkins & Dayan (1992)
- TD-Learning: Sutton & Barto (2018)
- Domain-Driven Design: Eric Evans

---

**Status:** ✅ COMPLETE & PRODUCTION-READY  
**Last Updated:** December 2, 2025  
**Implementation:** Full domain architecture with TD-learning
