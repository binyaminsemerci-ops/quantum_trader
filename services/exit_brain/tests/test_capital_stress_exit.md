# EB-FT-04: Capital Stress Exit Test

**Test ID**: EB-FT-04  
**Category**: Portfolio Protection  
**Priority**: P0 (Must Pass)  

---

## Test Objective

Verify portfolio stress triggers appropriate exits.

---

## Test Cases

### TC-04.1: Daily Loss > 3% (Reduce 25%)

**Given**: Daily PnL = -3.5%  
**When**: Exit Brain evaluates  
**Then**: Result = Reduce 25%

```python
def test_daily_loss_3_percent():
    portfolio = Portfolio(daily_pnl_pct=-0.035)
    position = Position()
    
    result = exit_brain.check_capital_stress_exit(position, portfolio)
    
    assert result.action == "CLOSE_PARTIAL"
    assert result.close_percent == 0.25
```

### TC-04.2: Daily Loss > 4% (Reduce 50%)

**Given**: Daily PnL = -4.2%  
**When**: Exit Brain evaluates  
**Then**: Result = Reduce 50% (or full close)

```python
def test_daily_loss_4_percent():
    portfolio = Portfolio(daily_pnl_pct=-0.042)
    position = Position()
    
    result = exit_brain.check_capital_stress_exit(position, portfolio)
    
    assert result.action in ["CLOSE_PARTIAL", "CLOSE_FULL"]
    assert result.urgency == "HIGH"
```

### TC-04.3: Drawdown > 8% (Close Weakest)

**Given**: Portfolio drawdown = 9%  
**And**: This position is the weakest (worst PnL)  
**When**: Exit Brain evaluates  
**Then**: Result = Full close

```python
def test_drawdown_8_percent_weakest():
    portfolio = Portfolio(drawdown_pct=0.09)
    position = Position(is_weakest=True, pnl=-200)
    
    result = exit_brain.check_capital_stress_exit(position, portfolio)
    
    assert result.action == "CLOSE_FULL"
    assert "weakest" in result.reason.lower()
```

### TC-04.4: Drawdown > 8% (Keep Strongest)

**Given**: Portfolio drawdown = 9%  
**And**: This position is NOT the weakest  
**When**: Exit Brain evaluates  
**Then**: Result = No action (close weakest first)

```python
def test_drawdown_8_percent_strongest():
    portfolio = Portfolio(drawdown_pct=0.09)
    position = Position(is_weakest=False, pnl=100)
    
    result = exit_brain.check_capital_stress_exit(position, portfolio)
    
    # Not weakest, so no exit
    assert result is None or result.action == "MONITOR"
```

### TC-04.5: Drawdown > 12% (Close All)

**Given**: Portfolio drawdown = 13%  
**When**: Exit Brain evaluates ANY position  
**Then**: Result = Full close, urgency = CRITICAL

```python
def test_drawdown_12_percent_critical():
    portfolio = Portfolio(drawdown_pct=0.13)
    position = Position()  # Any position
    
    result = exit_brain.check_capital_stress_exit(position, portfolio)
    
    assert result.action == "CLOSE_FULL"
    assert result.urgency == "CRITICAL"
```

### TC-04.6: No Stress (Normal)

**Given**: Daily PnL = -1%, Drawdown = 3%  
**When**: Exit Brain evaluates  
**Then**: Result = None

```python
def test_no_capital_stress():
    portfolio = Portfolio(daily_pnl_pct=-0.01, drawdown_pct=0.03)
    position = Position()
    
    result = exit_brain.check_capital_stress_exit(position, portfolio)
    
    assert result is None
```

---

## Acceptance Criteria

- [ ] Daily loss triggers partial exit at 3%+
- [ ] Drawdown 8%+ closes weakest only
- [ ] Drawdown 12%+ closes ALL
- [ ] Normal conditions = no action
