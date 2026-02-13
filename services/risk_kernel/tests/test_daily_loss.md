# RK-FT-03: Daily Loss Kill-Switch Test

**Test ID**: RK-FT-03  
**Category**: Kill-Switch  
**Priority**: P0 (Must Pass)  

---

## Test Objective

Verify that daily loss limit triggers kill-switch at -5%.

---

## Test Cases

### TC-03.1: Daily Loss at -4% (Warning Zone)

**Given**: Daily PnL = -4% of starting equity  
**When**: Risk Kernel evaluates new trade  
**Then**: Result = `risk.rejected` (NO new trades)  
**And**: Kill-switch NOT triggered

```python
def test_daily_loss_warning_zone():
    capital = Capital(
        equity=9600,
        equity_start_of_day=10000,
        daily_pnl=-400  # -4%
    )
    result = risk_kernel.check_daily_loss(capital)
    assert result.type == "risk.rejected"
    assert result.type != "kill_switch.triggered"
```

### TC-03.2: Daily Loss Exactly at -5%

**Given**: Daily PnL = -5.0% of starting equity  
**When**: Risk Kernel evaluates  
**Then**: Result = `kill_switch.triggered`

```python
def test_daily_loss_at_limit():
    capital = Capital(
        equity=9500,
        equity_start_of_day=10000,
        daily_pnl=-500  # -5%
    )
    result = risk_kernel.check_daily_loss(capital)
    assert result.type == "kill_switch.triggered"
    assert "daily loss" in result.reason.lower()
```

### TC-03.3: Daily Loss Exceeds -5% (-5.2%)

**Given**: Daily PnL = -5.2%  
**When**: Risk Kernel evaluates  
**Then**: Result = `kill_switch.triggered`

```python
def test_daily_loss_exceeds_limit():
    capital = Capital(
        equity=9480,
        equity_start_of_day=10000,
        daily_pnl=-520  # -5.2%
    )
    result = risk_kernel.check_daily_loss(capital)
    assert result.type == "kill_switch.triggered"
```

### TC-03.4: Daily Loss at -3% (OK Zone)

**Given**: Daily PnL = -3%  
**When**: Risk Kernel evaluates  
**Then**: Result = `risk.approved`

```python
def test_daily_loss_in_ok_zone():
    capital = Capital(
        equity=9700,
        equity_start_of_day=10000,
        daily_pnl=-300  # -3%
    )
    result = risk_kernel.check_daily_loss(capital)
    assert result.type == "risk.approved"
```

### TC-03.5: Unrealized + Realized Combined

**Given**: Realized -3%, Unrealized -2.5% (Total -5.5%)  
**When**: Risk Kernel evaluates  
**Then**: Result = `kill_switch.triggered`

```python
def test_combined_loss():
    capital = Capital(
        equity=9450,  # After unrealized
        equity_start_of_day=10000,
        daily_pnl=-300,  # Realized
        unrealized_pnl=-250  # Unrealized
    )
    result = risk_kernel.check_daily_loss(capital)
    # Total exposure: -5.5%
    assert result.type == "kill_switch.triggered"
```

### TC-03.6: Kill-Switch Triggers Only Once

**Given**: Daily loss triggers kill-switch  
**When**: Evaluated again  
**Then**: Kill-switch is NOT re-triggered  
**And**: System remains halted

```python
def test_kill_switch_triggers_once():
    # First evaluation - triggers
    result1 = risk_kernel.check_daily_loss(over_limit_capital())
    assert result1.type == "kill_switch.triggered"
    
    # Second evaluation - already halted
    result2 = risk_kernel.evaluate(any_request())
    assert result2.type == "risk.rejected"
    assert "kill-switch active" in result2.reason.lower()
```

---

## Kill-Switch Verification

After kill-switch triggers:

```python
def verify_kill_switch_state():
    assert risk_kernel.is_kill_switch_active() == True
    assert trading_allowed() == False
    assert new_orders_blocked() == True
```

---

## Acceptance Criteria

- [ ] Kill-switch triggers at exactly -5%
- [ ] Kill-switch triggers on combined realized + unrealized
- [ ] Kill-switch does not re-trigger (once is enough)
- [ ] System remains halted until manual reset
