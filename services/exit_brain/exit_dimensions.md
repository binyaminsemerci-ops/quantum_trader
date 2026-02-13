# Exit Brain — Exit Dimensions (MVP)

**Version**: 1.0  
**Priority**: Mandatory  

---

## Overview

Exit Brain evaluates four exit dimensions. Each runs continuously on every position.
**Profit-targets are secondary. Survival is primary.**

```
Time Exit       ─┐
Regime Exit     ─┼─ ANY triggers → EXIT
Volatility Exit ─┤
Capital Stress  ─┘
```

---

## Dimension A: Time Exit (ALLTID PÅ)

**Question**: Has this position been open too long?

### Rule

```
IF Time_in_Trade > T_max(regime, volatility)
→ EXIT (regardless of PnL)
```

### Time Limits

| Regime | Volatility | Max Hold Time |
|--------|------------|---------------|
| Trending | Normal | 72 hours |
| Trending | High | 48 hours |
| Ranging | Normal | 48 hours |
| Ranging | High | 24 hours |
| Uncertain | Any | 24 hours |
| Chaotic | Any | 12 hours |

### Implementation

```python
def check_time_exit(position: Position, regime: Regime, vol: Volatility) -> ExitSignal:
    time_in_trade = now() - position.opened_at
    max_hold = get_max_hold_time(regime, vol)
    
    if time_in_trade > max_hold:
        return ExitSignal(
            type="TIME",
            action="CLOSE_FULL",
            reason=f"Position exceeded {max_hold}h limit",
            urgency="HIGH"
        )
    
    if time_in_trade > max_hold * 0.8:
        return ExitSignal(
            type="TIME_WARNING",
            action="MONITOR",
            reason=f"Approaching time limit",
            urgency="MEDIUM"
        )
    
    return None
```

---

## Dimension B: Regime Exit (VIKTIGST)

**Question**: Has market regime changed since entry?

### Rule

```
IF Regime_now ≠ Regime_entry
→ PARTIAL EXIT → FULL EXIT
```

### Regime Change Matrix

| Entry Regime | Current Regime | Action |
|--------------|----------------|--------|
| Trending UP | Trending UP | Hold |
| Trending UP | Ranging | Close 50% |
| Trending UP | Trending DOWN | Close 100% |
| Trending UP | Uncertain | Close 75% |
| Ranging | Any different | Close 100% |
| Any | Chaotic | Close 100% |

### Implementation

```python
def check_regime_exit(position: Position, current_regime: Regime) -> ExitSignal:
    if position.entry_regime == current_regime:
        return None  # No change
    
    # Regime changed
    if current_regime == Regime.CHAOTIC:
        return ExitSignal(
            type="REGIME",
            action="CLOSE_FULL",
            reason="Regime changed to CHAOTIC",
            urgency="CRITICAL"
        )
    
    if position.entry_regime.is_opposite(current_regime):
        return ExitSignal(
            type="REGIME",
            action="CLOSE_FULL",
            reason=f"Regime reversed: {position.entry_regime} → {current_regime}",
            urgency="HIGH"
        )
    
    return ExitSignal(
        type="REGIME",
        action="CLOSE_PARTIAL",
        close_percent=0.50,
        reason=f"Regime changed: {position.entry_regime} → {current_regime}",
        urgency="MEDIUM"
    )
```

---

## Dimension C: Volatility Exit

**Question**: Has volatility increased dangerously?

### Rule

```
IF Vol_now > k × Vol_entry (k = 1.5 or 2.0)
→ TIGHTEN STOP → EXIT
```

### Volatility Thresholds

| Vol Multiplier | Action |
|----------------|--------|
| 1.0 - 1.5x | Normal |
| 1.5 - 2.0x | Tighten stop 25% |
| 2.0 - 3.0x | Tighten stop 50% |
| > 3.0x | EXIT immediately |

### Implementation

```python
def check_volatility_exit(position: Position, current_vol: float) -> ExitSignal:
    vol_ratio = current_vol / position.entry_volatility
    
    if vol_ratio > 3.0:
        return ExitSignal(
            type="VOLATILITY",
            action="CLOSE_FULL",
            reason=f"Volatility spiked {vol_ratio:.1f}x",
            urgency="CRITICAL"
        )
    
    if vol_ratio > 2.0:
        return ExitSignal(
            type="VOLATILITY",
            action="TIGHTEN_STOP",
            tighten_percent=0.50,
            reason=f"Volatility elevated {vol_ratio:.1f}x",
            urgency="HIGH"
        )
    
    if vol_ratio > 1.5:
        return ExitSignal(
            type="VOLATILITY",
            action="TIGHTEN_STOP",
            tighten_percent=0.25,
            reason=f"Volatility increased {vol_ratio:.1f}x",
            urgency="MEDIUM"
        )
    
    return None
```

---

## Dimension D: Capital Stress Exit

**Question**: Is the portfolio under stress?

### Rule

```
IF Portfolio_Stress > threshold
→ REDUCE / CLOSE
```

### Stress Indicators

| Indicator | Threshold | Action |
|-----------|-----------|--------|
| Daily loss | > 3% | Reduce 25% |
| Daily loss | > 4% | Reduce 50% |
| Drawdown | > 8% | Close weakest |
| Drawdown | > 12% | Close all |
| Correlation spike | > 0.8 | Reduce correlated |

### Implementation

```python
def check_capital_stress_exit(
    position: Position, 
    portfolio: Portfolio
) -> ExitSignal:
    # Daily loss stress
    if portfolio.daily_pnl_pct < -0.04:
        return ExitSignal(
            type="CAPITAL_STRESS",
            action="CLOSE_FULL",
            reason="Daily loss > 4%, closing position",
            urgency="HIGH"
        )
    
    if portfolio.daily_pnl_pct < -0.03:
        return ExitSignal(
            type="CAPITAL_STRESS",
            action="CLOSE_PARTIAL",
            close_percent=0.25,
            reason="Daily loss > 3%, reducing exposure",
            urgency="MEDIUM"
        )
    
    # Drawdown stress
    if portfolio.drawdown > 0.12:
        return ExitSignal(
            type="CAPITAL_STRESS",
            action="CLOSE_FULL",
            reason="Drawdown > 12%, liquidating",
            urgency="CRITICAL"
        )
    
    if portfolio.drawdown > 0.08:
        if position.is_weakest_in_portfolio:
            return ExitSignal(
                type="CAPITAL_STRESS",
                action="CLOSE_FULL",
                reason="Drawdown > 8%, closing weakest position",
                urgency="HIGH"
            )
    
    return None
```

---

## Exit Signal Priority

When multiple exit signals exist:

```
1. Kill-Switch (from Risk Kernel)  → Immediate
2. Capital Stress (> 12% DD)       → Immediate
3. Regime Change (to CHAOTIC)      → Immediate
4. Volatility (> 3x)               → Immediate
5. Regime Change (other)           → Within 5 min
6. Time Exit                       → Within 5 min
7. Volatility (> 1.5x)             → Tighten stop
8. Capital Stress (minor)          → Partial exit
```

---

## What Exit Brain CANNOT Do

| Action | Allowed? |
|--------|----------|
| Open new position | ❌ NO |
| Increase position size | ❌ NO |
| Widen stop-loss | ❌ NO |
| Ignore Risk Kernel | ❌ NO |
| Delay exit indefinitely | ❌ NO |
