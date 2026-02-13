# RK-FT-02: Trade Risk Test

**Test ID**: RK-FT-02  
**Category**: Risk Limit  
**Priority**: P0 (Must Pass)  

---

## Test Objective

Verify that trades exceeding per-trade risk limits are rejected.

---

## Test Cases

### TC-02.1: Risk Exactly at 2% Limit

**Given**: Trade with risk = 2.0% of equity  
**When**: Risk Kernel evaluates  
**Then**: Result = `risk.approved`

```python
def test_risk_at_2_percent_limit():
    # Equity: $10,000
    # Risk: $200 = 2%
    trade = Trade(
        entry_price=50000,
        stop_loss=49000,  # $1000 per BTC
        size=0.2          # 0.2 * $1000 = $200 risk
    )
    capital = Capital(equity=10000)
    result = risk_kernel.evaluate_trade_risk(trade, capital)
    assert result.type == "risk.approved"
```

### TC-02.2: Risk Exceeds 2% Limit (2.5%)

**Given**: Trade with risk = 2.5% of equity  
**When**: Risk Kernel evaluates  
**Then**: Result = `risk.rejected`

```python
def test_risk_exceeds_2_percent():
    trade = Trade(
        entry_price=50000,
        stop_loss=49000,
        size=0.25  # 0.25 * $1000 = $250 = 2.5%
    )
    capital = Capital(equity=10000)
    result = risk_kernel.evaluate_trade_risk(trade, capital)
    assert result.type == "risk.rejected"
    assert "2%" in result.reason
```

### TC-02.3: No Stop-Loss Defined

**Given**: Trade without stop-loss  
**When**: Risk Kernel evaluates  
**Then**: Result = `risk.rejected`

```python
def test_no_stop_loss():
    trade = Trade(
        entry_price=50000,
        stop_loss=None,  # Missing!
        size=0.1
    )
    result = risk_kernel.evaluate_trade_risk(trade, valid_capital())
    assert result.type == "risk.rejected"
    assert "stop" in result.reason.lower()
```

### TC-02.4: Stop Distance > 3%

**Given**: Stop-loss more than 3% from entry  
**When**: Risk Kernel evaluates  
**Then**: Result = `risk.rejected`

```python
def test_stop_too_far():
    trade = Trade(
        entry_price=50000,
        stop_loss=48000,  # 4% away
        size=0.05
    )
    result = risk_kernel.evaluate_trade_risk(trade, valid_capital())
    assert result.type == "risk.rejected"
    assert "stop distance" in result.reason.lower()
```

### TC-02.5: Worst-Case Loss Incalculable

**Given**: Trade where loss calculation fails (e.g., size = 0)  
**When**: Risk Kernel evaluates  
**Then**: Result = `risk.rejected`

```python
def test_incalculable_loss():
    trade = Trade(
        entry_price=50000,
        stop_loss=49000,
        size=0  # Invalid
    )
    result = risk_kernel.evaluate_trade_risk(trade, valid_capital())
    assert result.type == "risk.rejected"
```

### TC-02.6: Edge Case - Very Small Position

**Given**: Trade with risk = 0.001% (micro position)  
**When**: Risk Kernel evaluates  
**Then**: Result = `risk.approved`

```python
def test_micro_position_approved():
    trade = Trade(
        entry_price=50000,
        stop_loss=49900,
        size=0.001  # Tiny
    )
    result = risk_kernel.evaluate_trade_risk(trade, valid_capital())
    assert result.type == "risk.approved"
```

---

## Acceptance Criteria

- [ ] 2% limit is enforced (inclusive)
- [ ] Stop-loss is mandatory
- [ ] Stop distance is validated
- [ ] Rejection reasons are clear
