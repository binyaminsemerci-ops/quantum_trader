# RK-FT-04: Drawdown Guard Test

**Test ID**: RK-FT-04  
**Category**: Drawdown Protection  
**Priority**: P0 (Must Pass)  

---

## Test Objective

Verify drawdown guards work correctly at all threshold levels.

---

## Drawdown Thresholds

| Drawdown | Action |
|----------|--------|
| 0-5% | Normal trading |
| 5-8% | Size reduced 25% |
| 8-12% | Size reduced 50%, warning |
| 12-15% | REJECT new trades |
| 15-20% | REJECT, close 50% |
| >20% | KILL_SWITCH |

---

## Test Cases

### TC-04.1: Drawdown at 5% (First Reduction)

**Given**: Peak = $10,000, Current = $9,500 (5% DD)  
**When**: Risk Kernel evaluates  
**Then**: Result = `risk.approved` with size_multiplier = 0.75

```python
def test_drawdown_5_percent():
    capital = Capital(peak_equity=10000, current_equity=9500)
    result = risk_kernel.check_drawdown(capital)
    assert result.type == "risk.approved"
    assert result.size_multiplier == 0.75
```

### TC-04.2: Drawdown at 10% (Elevated)

**Given**: Peak = $10,000, Current = $9,000 (10% DD)  
**When**: Risk Kernel evaluates  
**Then**: Result = `risk.approved` with size_multiplier = 0.50

```python
def test_drawdown_10_percent():
    capital = Capital(peak_equity=10000, current_equity=9000)
    result = risk_kernel.check_drawdown(capital)
    assert result.type == "risk.approved"
    assert result.size_multiplier == 0.50
```

### TC-04.3: Drawdown at 12% (Reject Zone)

**Given**: Peak = $10,000, Current = $8,800 (12% DD)  
**When**: Risk Kernel evaluates  
**Then**: Result = `risk.rejected`

```python
def test_drawdown_12_percent():
    capital = Capital(peak_equity=10000, current_equity=8800)
    result = risk_kernel.check_drawdown(capital)
    assert result.type == "risk.rejected"
    assert "drawdown" in result.reason.lower()
```

### TC-04.4: Drawdown at 15% (Close Partial)

**Given**: Peak = $10,000, Current = $8,500 (15% DD)  
**When**: Risk Kernel evaluates  
**Then**: Result = `risk.rejected` with close_positions = True

```python
def test_drawdown_15_percent():
    capital = Capital(peak_equity=10000, current_equity=8500)
    result = risk_kernel.check_drawdown(capital)
    assert result.type == "risk.rejected"
    assert result.close_partial == True
```

### TC-04.5: Drawdown at 20% (Kill-Switch)

**Given**: Peak = $10,000, Current = $8,000 (20% DD)  
**When**: Risk Kernel evaluates  
**Then**: Result = `kill_switch.triggered`

```python
def test_drawdown_20_percent_kill_switch():
    capital = Capital(peak_equity=10000, current_equity=8000)
    result = risk_kernel.check_drawdown(capital)
    assert result.type == "kill_switch.triggered"
    assert "catastrophic drawdown" in result.reason.lower()
```

### TC-04.6: Drawdown Exceeds 20% (25%)

**Given**: Peak = $10,000, Current = $7,500 (25% DD)  
**When**: Risk Kernel evaluates  
**Then**: Result = `kill_switch.triggered`

```python
def test_drawdown_25_percent():
    capital = Capital(peak_equity=10000, current_equity=7500)
    result = risk_kernel.check_drawdown(capital)
    assert result.type == "kill_switch.triggered"
```

### TC-04.7: Drawdown Recovery (Gradual)

**Given**: Was at 10% DD, now at 7% DD  
**When**: Risk Kernel evaluates  
**Then**: Size multiplier reflects CURRENT drawdown, not previous

```python
def test_drawdown_recovery():
    # Recovery: 10% -> 7%
    capital = Capital(peak_equity=10000, current_equity=9300)
    result = risk_kernel.check_drawdown(capital)
    assert result.type == "risk.approved"
    # Still reduced but less than at 10%
    assert result.size_multiplier > 0.50
    assert result.size_multiplier < 1.0
```

### TC-04.8: New Peak Updates Reference

**Given**: Previous peak = $10,000, new equity = $10,500  
**When**: Risk Kernel evaluates  
**Then**: Peak is updated to $10,500

```python
def test_peak_update():
    old_capital = Capital(peak_equity=10000, current_equity=10000)
    new_capital = Capital(peak_equity=10000, current_equity=10500)
    
    risk_kernel.check_drawdown(new_capital)
    
    # Peak should be updated
    assert capital_state.peak_equity == 10500
```

---

## Acceptance Criteria

- [ ] All threshold levels trigger correct response
- [ ] Size multipliers are applied correctly
- [ ] Kill-switch triggers at 20%
- [ ] Peak equity is updated on new highs
- [ ] Recovery is gradual, not instant
