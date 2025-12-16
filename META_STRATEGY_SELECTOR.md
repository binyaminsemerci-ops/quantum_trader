# 🧠 META-STRATEGY SELECTOR - AI HEDGE FUND OS

## Overview

The **Meta-Strategy Selector** is an AI-powered reinforcement learning system that dynamically selects the optimal trading strategy for each symbol based on market regime and historical performance.

### Key Features

- **🤖 AI-Driven Strategy Selection**: Uses contextual multi-armed bandit RL
- **📊 Market Regime Detection**: Automatically classifies markets (trending, ranging, volatile, etc.)
- **🎯 Strategy Profiles**: 7 pre-defined strategies from defensive to ultra-aggressive
- **📈 Self-Learning**: Learns which strategies work best over time
- **💾 Persistent Memory**: Saves/loads Q-learning state across restarts
- **⚡ Real-Time Adaptation**: Adjusts to changing market conditions

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    EVENT-DRIVEN EXECUTOR                         │
│                    (Main Trading Loop)                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ 1. AI Signal Generated
                         ▼
          ┌──────────────────────────────────┐
          │   META-STRATEGY INTEGRATION      │
          │   (Orchestration Layer)          │
          └────┬──────────────┬──────────────┘
               │              │
               │              │ 2. Build Market Context
               ▼              ▼
   ┌──────────────────┐   ┌──────────────────┐
   │ REGIME DETECTOR  │   │ MARKET DATA      │
   │ - ATR analysis   │   │ - Volume         │
   │ - Trend strength │   │ - Liquidity      │
   │ - ADX/MAs        │   │ - Spread         │
   └────────┬─────────┘   └──────────────────┘
            │
            │ 3. Detected Regime
            ▼
   ┌──────────────────────────────────┐
   │  META-STRATEGY SELECTOR (RL)     │
   │  - Epsilon-greedy exploration    │
   │  - Q-value exploitation          │
   │  - Context: (symbol, regime)     │
   └─────────────┬────────────────────┘
                 │
                 │ 4. Selected Strategy
                 ▼
   ┌──────────────────────────────────┐
   │  STRATEGY PROFILES              │
   │  - Defensive (1.5R-2.5R)        │
   │  - Moderate (2R-3.5R)           │
   │  - Ultra Aggressive (3R-8R)     │
   │  - Scalp, Trend Rider, etc.     │
   └─────────────┬────────────────────┘
                 │
                 │ 5. TP/SL Config
                 ▼
   ┌──────────────────────────────────┐
   │  TRADING PROFILE / EXECUTION     │
   │  - Apply TP/SL parameters        │
   │  - Execute trade                 │
   └─────────────┬────────────────────┘
                 │
                 │ 6. Trade Closes
                 ▼
   ┌──────────────────────────────────┐
   │  REWARD UPDATE (RL Learning)     │
   │  - Realized R (e.g., +3.5R)      │
   │  - Update Q(symbol, regime, strat)│
   │  - EMA smoothing                 │
   └──────────────────────────────────┘
```

---

## Components

### 1. **Strategy Profiles** (`strategy_profiles.py`)

Defines 7 trading strategies with complete TP/SL parameters:

| Strategy | SL | TP1 | TP2 | TP3 | R:R | Win Rate | Use Case |
|----------|-----|-----|-----|-----|-----|----------|----------|
| **Defensive** | 1.0R | 1.5R | 2.5R | 4.0R | 1.8 | 55% | Conservative, low vol |
| **Moderate** | 1.0R | 2.0R | 3.5R | 5.5R | 2.3 | 52% | Balanced, all markets |
| **Moderate Aggressive** | 1.0R | 2.5R | 4.0R | 6.0R | 3.0 | 50% | Trending, high conf |
| **Balanced Aggressive** | 1.5R | 4.0R | 6.0R | 10R | 4.5 | 48% | Strong trends, BTC/ETH |
| **Ultra Aggressive** | 1.0R | 3.0R | 5.0R | 8.0R | 5.0 | 45% | **DEFAULT** - AI signals |
| **Scalp** | 0.8R | 1.2R | 1.8R | 2.5R | 1.5 | 60% | Range-bound, low vol |
| **Trend Rider** | 2.0R | 5.0R | 8.0R | 12R | 6.5 | 40% | Strong trends, home runs |

**Key Properties:**
- `r_sl`, `r_tp1`, `r_tp2`, `r_tp3`: TP/SL in R (ATR multiples)
- `partial_close_tp1/tp2/tp3`: Position sizing at each TP
- `aggressiveness`: Very Low → Very High
- `suitable_for_trending/ranging/high_vol/low_vol`: Regime filters
- `expected_win_rate`, `expected_risk_reward`: Performance expectations

**API:**
```python
from backend.services.ai.strategy_profiles import get_strategy_profile, StrategyID

profile = get_strategy_profile(StrategyID.ULTRA_AGGRESSIVE)
print(f"TP1: {profile.r_tp1}R, TP2: {profile.r_tp2}R")
# Output: TP1: 3.0R, TP2: 5.0R

tpsl_config = profile.to_tpsl_config()  # Dict for Trading Profile
```

---

### 2. **Regime Detector** (`regime_detector.py`)

Classifies market regime using technical analysis:

**Regimes:**
- `TREND_UP` - Strong uptrend (ADX > 25, trend_strength > 0.3)
- `TREND_DOWN` - Strong downtrend (ADX > 25, trend_strength < -0.3)
- `RANGE_LOW_VOL` - Sideways, low volatility (ADX < 25, ATR < 1.5%)
- `RANGE_HIGH_VOL` - Sideways, high volatility (ADX < 25, ATR > 4%)
- `HIGH_VOLATILITY` - Extreme volatility/whipsaw (ATR > 6%)
- `ILLIQUID` - Low liquidity/dangerous (volume < $2M, depth < $50k, spread > 10bps)
- `UNKNOWN` - Cannot determine

**Inputs (MarketContext):**
```python
from backend.services.ai.regime_detector import RegimeDetector, MarketContext

context = MarketContext(
    symbol="BTCUSDT",
    timeframe="15m",
    current_price=100000.0,
    atr=500.0,
    atr_pct=0.005,  # 0.5% ATR
    adx=35.0,       # ADX above 25 = trending
    trend_strength=0.6,  # +0.6 = strong uptrend
    volume_24h=50_000_000,
    depth_5bps=500_000,
    spread_bps=2.5,
)

detector = RegimeDetector()
result = detector.detect_regime(context)

print(result.regime)  # MarketRegime.TREND_UP
print(result.confidence)  # 0.85
print(result.reasoning)  # "Trending UP: ADX=35.0, trend_strength=0.60, ATR=0.50%"
```

**Detection Logic:**
1. **ILLIQUID CHECK** (highest priority)
   - Volume < $2M OR Depth < $50k OR Spread > 10bps → ILLIQUID
   
2. **EXTREME VOLATILITY**
   - ATR% > 6% → HIGH_VOLATILITY
   
3. **TRENDING**
   - ADX > 25 AND |trend_strength| > 0.3
   - Direction: trend_strength > 0 = UP, < 0 = DOWN
   
4. **RANGING**
   - ADX < 25 AND |trend_strength| < 0.3
   - Volatility: ATR% > 4% = HIGH_VOL, else LOW_VOL

---

### 3. **Meta-Strategy Selector** (`meta_strategy_selector.py`)

**Core RL Engine** - Learns optimal strategies via contextual multi-armed bandit.

**Key Concepts:**

**Q-Table:**
```
Q[(symbol, regime, strategy)] = QStats(
    count: int,          # Number of updates
    ema_reward: float,   # Exponential moving average of rewards
    avg_reward: float,   # Simple average
    win_count: int,      # Wins
    loss_count: int,     # Losses
    total_r: float,      # Cumulative R
)
```

**Epsilon-Greedy Selection:**
```python
if random() < epsilon:
    # EXPLORE: Try random strategy (discover new approaches)
    strategy = random.choice(candidates)
else:
    # EXPLOIT: Use best-performing strategy (use learned knowledge)
    strategy = argmax(Q[(symbol, regime, s)] for s in candidates)
```

**Reward Signal:**
- **Positive:** +3.5R (TP hit at 3.5R)
- **Negative:** -1.0R (SL hit at 1R)
- **Neutral:** +0.2R (breakeven exit)

**EMA Update:**
```python
new_ema = (1 - alpha) * old_ema + alpha * reward
# alpha = 0.2 (default) → recent rewards weighted 20%, history 80%
```

**API:**
```python
from backend.services.ai.meta_strategy_selector import MetaStrategySelector

selector = MetaStrategySelector(epsilon=0.10, alpha=0.20)

# Choose strategy
decision = selector.choose_strategy(
    symbol="BTCUSDT",
    regime=MarketRegime.TREND_UP,
    context=market_context
)

print(decision.strategy_profile.name)  # "Ultra Aggressive"
print(decision.is_exploration)  # False (exploit)
print(decision.confidence)  # 0.85
print(decision.reasoning)  # "Exploitation: highest Q-value=3.245"

# Update after trade closes
selector.update_reward(
    symbol="BTCUSDT",
    regime=MarketRegime.TREND_UP,
    strategy_id=StrategyID.ULTRA_AGGRESSIVE,
    reward=3.5,  # +3.5R trade
)
```

**Persistence:**
- Saves Q-table to `data/meta_strategy_state.json`
- Auto-saves after each update (configurable)
- Loads on startup

---

### 4. **Meta-Strategy Integration** (`meta_strategy_integration.py`)

**Orchestration layer** - Connects all components to EventDrivenExecutor.

**Flow:**
1. **AI Signal Generated** → EventDrivenExecutor
2. **Build Market Context** → MarketContext (ATR, volume, liquidity, etc.)
3. **Detect Regime** → RegimeDetector → MarketRegime
4. **Select Strategy** → MetaStrategySelector → StrategyProfile
5. **Apply TP/SL** → Trading Profile / Execution
6. **Trade Closes** → Update RL Reward

**API:**
```python
from backend.services.meta_strategy_integration import MetaStrategyIntegration

integration = MetaStrategyIntegration(enabled=True, epsilon=0.10, alpha=0.20)

# In EventDrivenExecutor signal processing:
result = await integration.select_strategy_for_signal(
    symbol="BTCUSDT",
    signal=ai_signal,
    market_data=market_data
)

# Use result.tpsl_config for TP/SL parameters
tpsl_config = result.tpsl_config
# {
#   "atr_mult_sl": 1.0,
#   "atr_mult_tp1": 3.0,
#   "atr_mult_tp2": 5.0,
#   "atr_mult_tp3": 8.0,
#   ...
# }

# After trade closes:
await integration.update_strategy_reward(
    symbol="BTCUSDT",
    realized_r=3.5,  # +3.5R profit
    trade_meta={"pnl": 85.0, "duration_hours": 4.2}
)
```

---

## Configuration

### Environment Variables

```bash
# Meta-Strategy Selector
META_STRATEGY_ENABLED=true          # Enable/disable system
META_STRATEGY_EPSILON=0.10          # Exploration rate (10%)
META_STRATEGY_ALPHA=0.20            # EMA smoothing factor
META_STRATEGY_STATE_FILE=data/meta_strategy_state.json

# Regime Detection Thresholds
REGIME_TREND_ADX_THRESHOLD=25.0     # ADX for trending
REGIME_HIGH_VOL_ATR_PCT=0.04        # 4% ATR = high vol
REGIME_ILLIQUID_VOLUME=2000000      # $2M min volume

# Default Strategy (if meta-strategy disabled)
DEFAULT_STRATEGY=ultra_aggressive
```

### Python Configuration

```python
# backend/config/meta_strategy.py
from backend.services.meta_strategy_integration import get_meta_strategy_integration

integration = get_meta_strategy_integration(
    enabled=True,
    epsilon=0.10,  # 10% exploration
    alpha=0.20,    # 20% weight to recent rewards
)
```

---

## Integration with EventDrivenExecutor

### Step 1: Initialize in `__init__`

```python
# backend/services/event_driven_executor.py

from backend.services.meta_strategy_integration import get_meta_strategy_integration

class EventDrivenExecutor:
    def __init__(self, ...):
        # ... existing init code ...
        
        # [NEW] META-STRATEGY SELECTOR
        self.meta_strategy = get_meta_strategy_integration(
            enabled=os.getenv("META_STRATEGY_ENABLED", "true").lower() == "true",
            epsilon=float(os.getenv("META_STRATEGY_EPSILON", "0.10")),
            alpha=float(os.getenv("META_STRATEGY_ALPHA", "0.20")),
        )
        logger.info(f"[OK] Meta-Strategy Selector: {self.meta_strategy.get_metrics()}")
```

### Step 2: Select Strategy Before Execution

```python
# In _check_and_execute() or _execute_signals_direct()

async def _process_signal(self, signal: Dict):
    symbol = signal["symbol"]
    
    # [NEW] Select optimal strategy via Meta-Strategy Selector
    strategy_result = await self.meta_strategy.select_strategy_for_signal(
        symbol=symbol,
        signal=signal,
        market_data=market_data,  # From binance or cache
    )
    
    logger.info(
        f"[META-STRATEGY] {symbol}: {strategy_result.strategy.name} "
        f"(regime={strategy_result.regime.value}, "
        f"explore={strategy_result.decision.is_exploration})"
    )
    
    # [NEW] Override Trading Profile TP/SL with selected strategy
    tpsl_override = strategy_result.tpsl_config
    # Pass to execution.py or hybrid_tpsl.py
    
    # Execute trade with dynamic TP/SL...
```

### Step 3: Update Rewards After Trade Closes

```python
# In trade monitoring / position close handler

async def on_trade_closed(self, symbol: str, entry_price: float, exit_price: float, side: str):
    # Calculate realized R
    # R = (exit_price - entry_price) / ATR (for LONG)
    # R = (entry_price - exit_price) / ATR (for SHORT)
    
    atr = get_atr_for_symbol(symbol)  # From market data
    
    if side == "LONG":
        realized_r = (exit_price - entry_price) / atr
    else:  # SHORT
        realized_r = (entry_price - exit_price) / atr
    
    # [NEW] Update Meta-Strategy RL
    await self.meta_strategy.update_strategy_reward(
        symbol=symbol,
        realized_r=realized_r,
        trade_meta={
            "pnl": calculate_pnl(entry_price, exit_price, position_size),
            "duration_hours": (close_time - open_time).total_seconds() / 3600,
            "side": side,
        }
    )
    
    logger.info(f"[RL UPDATE] {symbol}: R={realized_r:+.2f}")
```

---

## Monitoring & Debugging

### Check Active Strategies

```python
# Get currently active strategies
active = integration._active_strategies
for symbol, result in active.items():
    print(f"{symbol}: {result.strategy.name} (regime={result.regime.value})")
```

### View Q-Table Performance

```python
# Get performance summary
summary = integration.get_performance_summary()
print(json.dumps(summary, indent=2))

# Output:
# {
#   "total_entries": 47,
#   "total_decisions": 152,
#   "total_updates": 38,
#   "exploration_rate": 0.11,
#   "best_strategies": [
#     {
#       "symbol": "BTCUSDT",
#       "regime": "trend_up",
#       "strategy": "ultra_aggressive",
#       "ema_reward": 3.24,
#       "count": 12,
#       "win_rate": 0.67,
#       "total_r": 38.8
#     },
#     ...
#   ]
# }
```

### View Metrics

```python
metrics = integration.get_metrics()
print(metrics)

# Output:
# {
#   "enabled": true,
#   "total_selections": 152,
#   "total_regime_detections": 152,
#   "total_reward_updates": 38,
#   "active_strategies": 3,
#   "epsilon": 0.10,
#   "alpha": 0.20
# }
```

### Logs

```
[META-STRATEGY] Enabled - epsilon=10%, alpha=20%
[REGIME] BTCUSDT: TREND_UP (conf=85%) - Trending UP: ADX=35.0, trend_strength=0.60, ATR=0.50%
[STRATEGY] BTCUSDT: Ultra Aggressive (explore=False, conf=88%) - Exploitation: highest Q-value=3.245
[RL UPDATE] BTCUSDT trend_up ultra_aggressive: R=+3.50, EMA=3.24, count=12, WR=66.7%
```

---

## Performance Optimization

### 1. **Epsilon Tuning** (Exploration Rate)

- **High Epsilon (0.20-0.30)**: More exploration, slower convergence, better for volatile markets
- **Low Epsilon (0.05-0.10)**: More exploitation, faster convergence, better for stable markets
- **Adaptive**: Start high (0.20), decay to low (0.05) over time

```python
# Decay epsilon over time
initial_epsilon = 0.20
min_epsilon = 0.05
decay_rate = 0.995

current_epsilon = max(min_epsilon, initial_epsilon * (decay_rate ** episode))
```

### 2. **Alpha Tuning** (EMA Smoothing)

- **High Alpha (0.30-0.50)**: More weight to recent rewards, adapts faster, more volatile
- **Low Alpha (0.10-0.20)**: More weight to history, stable, slower adaptation
- **Default: 0.20** (balance between stability and adaptation)

### 3. **Regime Detection Thresholds**

Tune based on your symbols:

```python
# More sensitive (more regimes detected)
RegimeDetector(
    trend_adx_threshold=20.0,  # Lower = easier to classify as trending
    high_vol_atr_pct_threshold=0.03,  # Lower = easier to classify as high vol
)

# Less sensitive (fewer regime changes)
RegimeDetector(
    trend_adx_threshold=30.0,  # Higher = stricter trending criteria
    high_vol_atr_pct_threshold=0.05,  # Higher = stricter high vol criteria
)
```

---

## Testing

### Unit Tests

```bash
# Test strategy profiles
pytest tests/test_strategy_profiles.py

# Test regime detector
pytest tests/test_regime_detector.py

# Test meta-strategy selector
pytest tests/test_meta_strategy_selector.py

# Test integration
pytest tests/test_meta_strategy_integration.py
```

### Simulation Test

```python
# Simulate learning over 100 trades
from backend.services.ai.meta_strategy_selector import MetaStrategySelector

selector = MetaStrategySelector(epsilon=0.15, alpha=0.25)

for i in range(100):
    # Choose strategy
    decision = selector.choose_strategy(
        symbol="BTCUSDT",
        regime=MarketRegime.TREND_UP,
        context=market_context
    )
    
    # Simulate reward (biased towards ultra_aggressive in trends)
    if decision.strategy_id == StrategyID.ULTRA_AGGRESSIVE:
        reward = np.random.normal(3.0, 2.0)  # Mean +3R
    else:
        reward = np.random.normal(1.5, 1.5)  # Mean +1.5R
    
    # Update
    selector.update_reward(
        symbol="BTCUSDT",
        regime=MarketRegime.TREND_UP,
        strategy_id=decision.strategy_id,
        reward=reward
    )

# Check learned Q-values
summary = selector.get_performance_summary()
print(summary)
```

---

## Troubleshooting

### Problem: All strategies have Q=0

**Cause:** No rewards updated yet (cold start)

**Solution:** System will use heuristic selection until enough data collected (min 5 samples)

### Problem: System always explores (never exploits)

**Cause:** Epsilon too high OR Q-values below `min_confidence_for_exploit`

**Solution:** Lower epsilon OR wait for more reward updates

### Problem: Regime detector always returns UNKNOWN

**Cause:** Missing market data (ATR, ADX, MAs)

**Solution:** Ensure market_data dict contains required indicators

### Problem: RL not learning (Q-values not changing)

**Cause:** Rewards not being updated

**Solution:** Check `update_strategy_reward()` is called after trade closes

---

## Advanced Topics

### Custom Strategies

Add new strategy to `STRATEGY_PROFILES`:

```python
# backend/services/ai/strategy_profiles.py

StrategyID.CUSTOM = "custom"

STRATEGY_PROFILES[StrategyID.CUSTOM] = StrategyProfile(
    strategy_id=StrategyID.CUSTOM,
    name="My Custom Strategy",
    description="Custom strategy for X market conditions",
    r_sl=1.2,
    r_tp1=2.8,
    r_tp2=4.5,
    r_tp3=7.0,
    # ... rest of parameters
)
```

### Multi-Timeframe Regimes

Extend regime detection to multiple timeframes:

```python
# Detect regime on 15m and 1h
regime_15m = detector.detect_regime(context_15m)
regime_1h = detector.detect_regime(context_1h)

# Combine
if regime_15m.regime == MarketRegime.TREND_UP and regime_1h.regime == MarketRegime.TREND_UP:
    # Strong trend confirmation
    use_ultra_aggressive = True
```

### Contextual Features

Add more context to regime detection:

```python
context = MarketContext(
    # ... existing fields ...
    market_cap_rank=5,  # BTC = 1, ETH = 2, etc.
    correlation_to_btc=0.85,
    recent_news_sentiment=0.6,  # -1 to +1
)

# Use in custom regime detection logic
```

---

## Roadmap

### Phase 1: Core Implementation ✅
- Strategy profiles
- Regime detector
- Meta-strategy selector with RL
- Integration with EventDrivenExecutor

### Phase 2: Enhanced Learning (Q1 2026)
- Per-symbol learning (symbol-specific Q-tables)
- Multi-timeframe regime detection
- Contextual features (market cap, correlation, sentiment)
- Adaptive epsilon decay

### Phase 3: Advanced RL (Q2 2026)
- Policy gradient methods (PPO, A2C)
- Deep Q-Network (DQN) for complex contexts
- Multi-objective optimization (profit, sharpe, drawdown)
- Transfer learning across symbols

### Phase 4: Production Optimization (Q3 2026)
- Real-time backtesting
- A/B testing framework
- Performance dashboards
- Automated hyperparameter tuning

---

## Summary

The Meta-Strategy Selector transforms Quantum Trader from a **static** trading system into an **adaptive AI hedge fund** that:

1. **Learns continuously** - Discovers which strategies work best in different market conditions
2. **Adapts automatically** - Changes strategies based on real-time market regime
3. **Optimizes for profit** - Focuses on maximizing R-multiples, not just win rate
4. **Scales intelligently** - Works across all symbols with symbol-specific learning
5. **Persists knowledge** - Retains learning across restarts

**Result:** Higher profits, better risk-adjusted returns, and true AI-driven trading.

---

**🚀 Ready to maximize profits with AI-driven strategy selection!**
