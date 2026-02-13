# Risk Kernel — Risk Dimensions (MVP)

**Version**: 1.0  
**Status**: Production-Ready  

---

## Overview

The Risk Kernel evaluates four independent risk dimensions.
**ALL must pass** for a trade to be approved.

```
Trade Risk      ─┐
Daily Risk      ─┼─ ALL MUST PASS ─→ risk.approved
Drawdown Risk   ─┤
System Integrity─┘

ANY FAIL ─────────────────────────→ risk.rejected
CRITICAL FAIL ────────────────────→ kill_switch.triggered
```

---

## Dimension 1: Trade Risk

**Question**: Does this single trade violate position limits?

### Checks

| Check | Threshold | Failure |
|-------|-----------|---------|
| Risk per trade | ≤ 2% of equity | REJECT |
| Stop-loss exists | Required | REJECT |
| Stop distance | ≤ 3% from entry | REJECT |
| Position size vs capital | Within scaling level | REJECT |
| Worst-case loss | Calculable | REJECT |

### Calculation

```python
def validate_trade_risk(trade: Trade, capital: Capital) -> Result:
    # Stop-loss must exist
    if not trade.stop_loss:
        return REJECT("No stop-loss defined")
    
    # Calculate risk amount
    risk_amount = abs(trade.entry_price - trade.stop_loss) * trade.size
    risk_percent = risk_amount / capital.equity
    
    # Check against policy
    if risk_percent > 0.02:
        return REJECT(f"Risk {risk_percent:.2%} exceeds 2% limit")
    
    return APPROVE()
```

---

## Dimension 2: Daily Risk

**Question**: Has daily loss limit been reached?

### Checks

| Check | Threshold | Failure |
|-------|-----------|---------|
| Realized PnL today | > -5% | KILL_SWITCH |
| Unrealized + Realized | > -5% | REJECT new trades |
| Approaching limit | > -4% | WARNING (reduce size) |

### Calculation

```python
def check_daily_loss(capital: Capital) -> Result:
    daily_pnl_percent = capital.daily_pnl / capital.equity_start_of_day
    
    if daily_pnl_percent <= -0.05:
        return KILL_SWITCH("Daily loss limit -5% reached")
    
    if daily_pnl_percent <= -0.04:
        return REJECT("Approaching daily limit, no new trades")
    
    return APPROVE()
```

---

## Dimension 3: Drawdown Risk

**Question**: Is portfolio drawdown within acceptable limits?

### Checks

| Check | Threshold | Failure |
|-------|-----------|---------|
| Current drawdown | ≤ 15% | REJECT if > 12% |
| Current drawdown | > 20% | KILL_SWITCH |
| Drawdown recovery | Must be gradual | Size limits |

### Calculation

```python
def check_drawdown(capital: Capital) -> Result:
    drawdown = (capital.peak_equity - capital.current_equity) / capital.peak_equity
    
    if drawdown >= 0.20:
        return KILL_SWITCH("Catastrophic drawdown >20%")
    
    if drawdown >= 0.15:
        return REJECT("Drawdown >15%, no new trades")
    
    if drawdown >= 0.12:
        return REJECT("Drawdown >12%, reduce-only mode")
    
    # Return size multiplier for drawdown scaling
    size_mult = calculate_drawdown_multiplier(drawdown)
    return APPROVE(size_multiplier=size_mult)
```

---

## Dimension 4: System Integrity

**Question**: Is the system in a known, valid state?

### Checks

| Check | Requirement | Failure |
|-------|-------------|---------|
| Data freshness | < 5 seconds old | REJECT |
| Data quality score | ≥ 95% | REJECT |
| Position sync | Exchange = Local | KILL_SWITCH if mismatch |
| Kill-switch status | Not active | REJECT all |
| API connectivity | Responsive | REJECT |

### Calculation

```python
def check_system_integrity(state: SystemState) -> Result:
    # Kill-switch already active?
    if state.kill_switch_active:
        return REJECT("Kill-switch is active")
    
    # Data freshness
    if state.data_age_seconds > 5:
        return REJECT("Data is stale")
    
    # Data quality
    if state.data_quality < 0.95:
        return REJECT("Data quality below 95%")
    
    # Position sync
    if state.position_mismatch:
        return KILL_SWITCH("Position mismatch detected")
    
    # API health
    if not state.api_healthy:
        return REJECT("API not responsive")
    
    return APPROVE()
```

---

## Dimension Evaluation Order

**Critical**: Dimensions are evaluated in this exact order:

```
1. System Integrity  (can abort before any calculation)
2. Kill-Switch Check (is system already halted?)
3. Daily Risk        (global limit)
4. Drawdown Risk     (portfolio limit)
5. Trade Risk        (specific trade)
```

This order ensures:
- System health is verified first
- Global limits checked before trade-specific
- No wasted calculation on already-rejected trades

---

## Result Types

```python
class RiskResult:
    APPROVE = "risk.approved"
    REJECT = "risk.rejected"
    KILL_SWITCH = "kill_switch.triggered"
```

**APPROVE**: Trade may proceed to next stage  
**REJECT**: Trade is blocked, system continues  
**KILL_SWITCH**: All trading halts immediately
