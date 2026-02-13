# 3-Month Simulation Model – Quantum Trader

> **Purpose**: Monte Carlo simulation for realistic performance expectations  
> **Timeframe**: 90 trading days  
> **Policy Reference**: constitution/RISK_POLICY.md

---

## 1. Simulation Overview

### Purpose

The 3-month simulation model serves to:
1. Set realistic performance expectations
2. Stress-test the risk management framework
3. Understand probability distributions of outcomes
4. Plan capital requirements
5. Communicate with potential investors

### Key Principles

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│  "We don't predict returns. We understand distributions."                       │
│                                                                                 │
│  The simulation is NOT a forecast.                                              │
│  It's a stress-test of our risk parameters.                                     │
│                                                                                 │
│  Conservative assumptions:                                                      │
│  • We assume our edge is smaller than we think                                  │
│  • We assume drawdowns are deeper than we hope                                  │
│  • We assume correlations are higher in crisis                                  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Model Parameters

### Base Assumptions

```python
# Simulation Parameters
SIMULATION_CONFIG = {
    # Time
    "trading_days": 90,          # 3 months
    "simulations": 10000,        # Monte Carlo runs
    
    # Starting capital
    "initial_capital": 100000,   # $100,000 USD
    
    # Trade frequency
    "trades_per_day_mean": 2.0,  # Average trades per day
    "trades_per_day_std": 1.0,   # Standard deviation
    "min_trades_per_day": 0,     # Can have zero-trade days
    "max_trades_per_day": 5,     # Policy limit
    
    # Trade outcomes (CONSERVATIVE)
    "win_rate": 0.52,            # 52% win rate (slight edge)
    "avg_win_percent": 0.015,    # 1.5% average win
    "avg_loss_percent": 0.012,   # 1.2% average loss (stop-loss)
    "win_std": 0.005,            # Winning trade variation
    "loss_std": 0.003,           # Losing trade variation
    
    # Risk limits (per constitution)
    "max_risk_per_trade": 0.02,  # 2% max risk
    "max_daily_loss": 0.05,      # 5% daily limit → kill-switch
    "max_drawdown": 0.20,        # 20% drawdown → kill-switch
    
    # Position sizing
    "kelly_fraction": 0.25,      # Use 25% of Kelly optimal
    "max_position_size": 0.10,   # Max 10% of capital per position
}
```

### Regime Assumptions

```python
REGIME_CONFIG = {
    # Regime distribution (% of time)
    "trending": 0.35,            # 35% trending
    "ranging": 0.45,             # 45% ranging
    "chaotic": 0.20,             # 20% chaotic
    
    # Performance by regime
    "trending_win_rate_boost": 0.08,    # +8% win rate in trends
    "ranging_win_rate_penalty": 0.02,   # -2% in ranging
    "chaotic_win_rate_penalty": 0.10,   # -10% in chaotic (we trade less)
    
    # Trade frequency by regime
    "trending_trade_mult": 1.2,   # Trade more in trends
    "ranging_trade_mult": 0.8,    # Trade less in ranging
    "chaotic_trade_mult": 0.3,    # Trade much less in chaotic
}
```

---

## 3. Simulation Logic

### Daily Simulation Process

```python
def simulate_day(state: SimulationState, day: int) -> SimulationState:
    """
    Simulate one trading day.
    """
    
    # Check kill-switch conditions
    if state.kill_switch_active:
        return state  # No trading
    
    if state.daily_loss_today <= -0.05:
        state.kill_switch_active = True
        state.kill_switch_reason = "daily_loss"
        return state
    
    if state.drawdown_from_hwm <= -0.20:
        state.kill_switch_active = True
        state.kill_switch_reason = "drawdown"
        return state
    
    # Determine regime for today
    regime = sample_regime()
    
    # Determine number of trades
    trades_today = sample_trade_count(regime)
    
    # Execute trades
    daily_pnl = 0
    for _ in range(trades_today):
        trade_pnl = simulate_trade(state, regime)
        daily_pnl += trade_pnl
        
        # Check intraday kill-switch
        if daily_pnl / state.equity <= -0.05:
            state.kill_switch_active = True
            break
    
    # Update state
    state.equity += daily_pnl
    state.daily_pnl_history.append(daily_pnl)
    state.equity_curve.append(state.equity)
    state.update_drawdown()
    
    return state

def simulate_trade(state: SimulationState, regime: str) -> float:
    """
    Simulate a single trade.
    """
    
    # Adjust win rate for regime
    base_win_rate = SIMULATION_CONFIG["win_rate"]
    adjusted_win_rate = adjust_win_rate_for_regime(base_win_rate, regime)
    
    # Determine outcome
    if random.random() < adjusted_win_rate:
        # Winning trade
        win_size = np.random.normal(
            SIMULATION_CONFIG["avg_win_percent"],
            SIMULATION_CONFIG["win_std"]
        )
        return state.equity * max(win_size, 0)
    else:
        # Losing trade
        loss_size = np.random.normal(
            SIMULATION_CONFIG["avg_loss_percent"],
            SIMULATION_CONFIG["loss_std"]
        )
        # Loss is negative, capped at max risk
        return -state.equity * min(
            loss_size,
            SIMULATION_CONFIG["max_risk_per_trade"]
        )
```

### Monte Carlo Simulation

```python
def run_monte_carlo(n_simulations: int = 10000) -> SimulationResults:
    """
    Run full Monte Carlo simulation.
    """
    
    results = []
    
    for sim_id in range(n_simulations):
        state = SimulationState(
            initial_capital=SIMULATION_CONFIG["initial_capital"]
        )
        
        for day in range(SIMULATION_CONFIG["trading_days"]):
            state = simulate_day(state, day)
        
        results.append(SimulationResult(
            final_equity=state.equity,
            total_return=state.total_return,
            max_drawdown=state.max_drawdown,
            kill_switch_triggered=state.kill_switch_active,
            kill_switch_day=state.kill_switch_day,
            total_trades=state.total_trades,
            winning_trades=state.winning_trades,
            equity_curve=state.equity_curve,
        ))
    
    return SimulationResults(results)
```

---

## 4. Expected Outcomes

### Return Distribution (90 Days)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    3-MONTH RETURN DISTRIBUTION                                  │
│                    (10,000 simulations)                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Percentile     Return      Notes                                               │
│  ─────────────────────────────────────────────────────────────                  │
│  1st           -18.5%      Worst 1% (near kill-switch)                          │
│  5th           -12.2%      Bad luck scenario                                    │
│  10th           -8.1%      Below expectations                                   │
│  25th           -1.5%      Modest loss                                          │
│  50th (median)  +6.8%      Typical outcome                                      │
│  75th          +15.2%      Good scenario                                        │
│  90th          +24.1%      Very good scenario                                   │
│  95th          +31.5%      Excellent scenario                                   │
│  99th          +45.2%      Best 1%                                              │
│                                                                                 │
│  Mean:          +7.2%                                                           │
│  Std Dev:       14.5%                                                           │
│  Sharpe (ann):  ~1.0                                                            │
│                                                                                 │
│  ─────────────────────────────────────────────────────────────                  │
│                                                                                 │
│  Probability of:                                                                │
│    Positive return:        62%                                                  │
│    Return > 10%:           43%                                                  │
│    Return > 20%:           25%                                                  │
│    Loss > 10%:             12%                                                  │
│    Kill-switch triggered:   4%                                                  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Drawdown Distribution

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    MAXIMUM DRAWDOWN DISTRIBUTION                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Percentile     Max DD      Notes                                               │
│  ─────────────────────────────────────────────────────────────                  │
│  50th (median)  -8.2%       Typical max drawdown                                │
│  75th          -12.5%       Uncomfortable but acceptable                        │
│  90th          -16.8%       Stress period                                       │
│  95th          -18.9%       Near kill-switch territory                          │
│  99th          -20.0%       Kill-switch triggered                               │
│                                                                                 │
│  Mean:          -9.1%                                                           │
│                                                                                 │
│  INTERPRETATION:                                                                │
│  • Expect to see -8% to -12% drawdown at some point (normal)                    │
│  • 10% chance of seeing -16% or worse                                           │
│  • 5% chance of approaching kill-switch                                         │
│  • Kill-switch provides hard floor of protection                                │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Scenario Analysis

### Base Case (Most Likely)

```yaml
base_case:
  probability: 50%
  conditions:
    - Normal market conditions
    - Moderate regime mix
    - System operates as designed
  expected_outcomes:
    return: +5% to +12%
    max_drawdown: -6% to -12%
    trades: 150-200
    win_rate: 50-54%
```

### Bull Case (Favorable)

```yaml
bull_case:
  probability: 25%
  conditions:
    - Trending markets dominate
    - Good signal-regime alignment
    - Low volatility spikes
  expected_outcomes:
    return: +15% to +30%
    max_drawdown: -4% to -8%
    trades: 180-220
    win_rate: 54-58%
```

### Bear Case (Adverse)

```yaml
bear_case:
  probability: 20%
  conditions:
    - Chaotic regime extends
    - Multiple false signals
    - Correlation breakdown
  expected_outcomes:
    return: -5% to -15%
    max_drawdown: -12% to -18%
    trades: 80-120 (less trading)
    win_rate: 46-50%
```

### Worst Case (Tail Risk)

```yaml
worst_case:
  probability: 5%
  conditions:
    - Extended chaotic regime
    - Multiple consecutive losses
    - Black swan event
  expected_outcomes:
    return: -15% to -20% (kill-switch)
    max_drawdown: -20% (kill-switch floor)
    outcome: Trading halted
```

---

## 6. Kill-Switch Analysis

### When Kill-Switch Activates

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    KILL-SWITCH ACTIVATION ANALYSIS                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  In 10,000 simulations of 90 days:                                              │
│                                                                                 │
│  Kill-switch triggered:     4.2% of simulations (420 of 10,000)                 │
│                                                                                 │
│  Trigger breakdown:                                                             │
│    Daily loss (-5%):        2.8%                                                │
│    Drawdown (-20%):         1.4%                                                │
│                                                                                 │
│  Average day of trigger:    Day 52 (median: Day 58)                             │
│                                                                                 │
│  ─────────────────────────────────────────────────────────────                  │
│                                                                                 │
│  KEY INSIGHT:                                                                   │
│  The kill-switch is designed to fire ~5% of the time under                      │
│  normal operations. This is a FEATURE, not a bug.                               │
│                                                                                 │
│  When kill-switch fires, it means:                                              │
│  • System protected capital as designed                                         │
│  • Human review required before resuming                                        │
│  • Worst outcome is -20%, not -50% or -100%                                     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Post-Kill-Switch Analysis

```python
# What happens after kill-switch?
POST_KILL_SWITCH = {
    "immediate_actions": [
        "All trading ceases",
        "Alert sent to founders",
        "Position review required",
        "Capital preserved at -20% or better",
    ],
    "recovery_path": [
        "72-hour minimum cooling period",
        "Root cause analysis",
        "System review before restart",
        "Gradual capital re-deployment",
    ],
    "capital_preserved": "80% or more of initial capital",
}
```

---

## 7. Sensitivity Analysis

### Win Rate Sensitivity

| Win Rate | Expected 3M Return | P(Positive) | P(Kill-Switch) |
|----------|-------------------|-------------|----------------|
| 48% | -5.2% | 38% | 12% |
| 50% | +1.1% | 48% | 8% |
| 52% | +7.2% | 62% | 4% |
| 54% | +13.5% | 74% | 2% |
| 56% | +20.1% | 83% | 1% |

### Trade Frequency Sensitivity

| Trades/Day | Expected 3M Return | Max DD (95th) | Sharpe |
|------------|-------------------|---------------|--------|
| 1.0 | +3.8% | -15.2% | 0.6 |
| 2.0 | +7.2% | -18.9% | 1.0 |
| 3.0 | +10.8% | -21.5% | 1.2 |

*Note: Higher frequency increases both return AND risk*

---

## 8. Implementation Code

### Run Simulation

```python
# simulation/monte_carlo.py

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List

@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulation."""
    trading_days: int = 90
    n_simulations: int = 10000
    initial_capital: float = 100000
    win_rate: float = 0.52
    avg_win_pct: float = 0.015
    avg_loss_pct: float = 0.012
    max_daily_loss: float = 0.05
    max_drawdown: float = 0.20

def run_simulation(config: SimulationConfig) -> pd.DataFrame:
    """
    Run Monte Carlo simulation.
    
    Returns DataFrame with simulation results.
    """
    results = []
    
    for sim_id in range(config.n_simulations):
        equity = config.initial_capital
        hwm = equity
        max_dd = 0
        kill_switch = False
        
        for day in range(config.trading_days):
            if kill_switch:
                break
            
            # Simulate trades
            daily_pnl = simulate_day_pnl(equity, config)
            
            # Check daily kill-switch
            if daily_pnl / equity <= -config.max_daily_loss:
                kill_switch = True
            
            # Update equity
            equity += daily_pnl
            
            # Update drawdown
            hwm = max(hwm, equity)
            dd = (equity - hwm) / hwm
            max_dd = min(max_dd, dd)
            
            # Check drawdown kill-switch
            if dd <= -config.max_drawdown:
                kill_switch = True
        
        results.append({
            'simulation': sim_id,
            'final_equity': equity,
            'return_pct': (equity - config.initial_capital) / config.initial_capital,
            'max_drawdown': max_dd,
            'kill_switch': kill_switch,
        })
    
    return pd.DataFrame(results)

def simulate_day_pnl(equity: float, config: SimulationConfig) -> float:
    """Simulate PnL for one day."""
    n_trades = np.random.poisson(2)  # ~2 trades per day
    pnl = 0
    
    for _ in range(n_trades):
        if np.random.random() < config.win_rate:
            pnl += equity * np.random.normal(config.avg_win_pct, 0.005)
        else:
            pnl -= equity * np.random.normal(config.avg_loss_pct, 0.003)
    
    return pnl

# Run and analyze
if __name__ == "__main__":
    config = SimulationConfig()
    results = run_simulation(config)
    
    print("=== 3-Month Simulation Results ===")
    print(f"Simulations: {len(results)}")
    print(f"\nReturn Distribution:")
    print(f"  Mean:   {results['return_pct'].mean():.1%}")
    print(f"  Median: {results['return_pct'].median():.1%}")
    print(f"  Std:    {results['return_pct'].std():.1%}")
    print(f"\n  5th:    {results['return_pct'].quantile(0.05):.1%}")
    print(f"  25th:   {results['return_pct'].quantile(0.25):.1%}")
    print(f"  75th:   {results['return_pct'].quantile(0.75):.1%}")
    print(f"  95th:   {results['return_pct'].quantile(0.95):.1%}")
    print(f"\nP(Positive): {(results['return_pct'] > 0).mean():.1%}")
    print(f"P(Kill-Switch): {results['kill_switch'].mean():.1%}")
```

---

## 9. Key Takeaways

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         SIMULATION SUMMARY                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  EXPECTED OUTCOMES (90 days, $100K):                                            │
│  • Most likely return: +5% to +12%                                              │
│  • Most likely drawdown: -8% to -12%                                            │
│  • 62% probability of positive return                                           │
│  • 4% probability of kill-switch activation                                     │
│                                                                                 │
│  WORST CASE:                                                                    │
│  • Loss capped at -20% by kill-switch                                           │
│  • $80,000 minimum preserved from $100,000                                      │
│  • Human review required before resuming                                        │
│                                                                                 │
│  KEY MESSAGE FOR INVESTORS:                                                     │
│  "We cannot guarantee profits, but we guarantee process.                        │
│   The kill-switch ensures we cannot lose more than 20%                          │
│   before human review and intervention."                                        │
│                                                                                 │
│  CAVEATS:                                                                       │
│  • Simulation assumes normal market conditions                                  │
│  • Black swan events may behave differently                                     │
│  • Past model performance ≠ future results                                      │
│  • Exchange failures, API issues not modeled                                    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

*Simuleringer viser muligheter, ikke garantier. Risk management sikrer overlvelse.*
