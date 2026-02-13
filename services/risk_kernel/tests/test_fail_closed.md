# RK-FT-01: Fail-Closed Default Test

**Test ID**: RK-FT-01  
**Category**: Safety-Critical  
**Priority**: P0 (Must Pass)  

---

## Test Objective

Verify that the Risk Kernel rejects trades when input is missing, unknown, or ambiguous.

---

## Test Cases

### TC-01.1: Missing Trade Data

**Given**: Risk evaluation request with no trade object  
**When**: Risk Kernel receives request  
**Then**: Result = `risk.rejected`  
**And**: Reason contains "missing trade data"

```python
def test_missing_trade_data():
    request = RiskEvaluateRequest(
        trade=None,  # Missing
        capital=valid_capital(),
        system_state=valid_system_state()
    )
    result = risk_kernel.evaluate(request)
    assert result.type == "risk.rejected"
    assert "missing" in result.reason.lower()
```

### TC-01.2: Missing Capital Data

**Given**: Risk evaluation request with no capital object  
**When**: Risk Kernel receives request  
**Then**: Result = `risk.rejected`

```python
def test_missing_capital_data():
    request = RiskEvaluateRequest(
        trade=valid_trade(),
        capital=None,  # Missing
        system_state=valid_system_state()
    )
    result = risk_kernel.evaluate(request)
    assert result.type == "risk.rejected"
```

### TC-01.3: Missing System State

**Given**: Risk evaluation request with no system state  
**When**: Risk Kernel receives request  
**Then**: Result = `risk.rejected`

```python
def test_missing_system_state():
    request = RiskEvaluateRequest(
        trade=valid_trade(),
        capital=valid_capital(),
        system_state=None  # Missing
    )
    result = risk_kernel.evaluate(request)
    assert result.type == "risk.rejected"
```

### TC-01.4: Unknown Symbol

**Given**: Trade with unrecognized symbol  
**When**: Risk Kernel evaluates  
**Then**: Result = `risk.rejected`

```python
def test_unknown_symbol():
    trade = valid_trade()
    trade.symbol = "UNKNOWN123"
    request = RiskEvaluateRequest(trade=trade, ...)
    result = risk_kernel.evaluate(request)
    assert result.type == "risk.rejected"
```

### TC-01.5: Negative Values

**Given**: Trade with negative entry price  
**When**: Risk Kernel evaluates  
**Then**: Result = `risk.rejected`

```python
def test_negative_entry_price():
    trade = valid_trade()
    trade.entry_price = -50000  # Invalid
    request = RiskEvaluateRequest(trade=trade, ...)
    result = risk_kernel.evaluate(request)
    assert result.type == "risk.rejected"
```

### TC-01.6: Timeout During Evaluation

**Given**: System state check times out  
**When**: Risk Kernel evaluates  
**Then**: Result = `risk.rejected`

```python
def test_timeout_handling():
    with mock_timeout("check_system_integrity"):
        result = risk_kernel.evaluate(valid_request())
    assert result.type == "risk.rejected"
    assert "timeout" in result.reason.lower()
```

---

## Acceptance Criteria

- [ ] ALL test cases pass
- [ ] NO test case results in `risk.approved` on invalid input
- [ ] Error messages are descriptive
- [ ] Tests run in < 1 second each

---

## Failure Consequence

If ANY of these tests fail:
- **Block merge to main**
- **System is not production-ready**
