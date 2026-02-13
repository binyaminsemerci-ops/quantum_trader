# EB-FT-03: Volatility Exit Test

**Test ID**: EB-FT-03  
**Category**: Risk-Based Exit  
**Priority**: P0 (Must Pass)  

---

## Test Objective

Verify volatility spikes trigger appropriate exits or stop tightening.

---

## Test Cases

### TC-03.1: Volatility Unchanged (No Action)

**Given**: Entry volatility = 2%  
**And**: Current volatility = 2%  
**When**: Exit Brain evaluates  
**Then**: Result = None

```python
def test_volatility_unchanged():
    position = Position(entry_volatility=0.02)
    
    result = exit_brain.check_volatility_exit(position, current_vol=0.02)
    
    assert result is None
```

### TC-03.2: Volatility 1.5x (Tighten 25%)

**Given**: Entry volatility = 2%  
**And**: Current volatility = 3% (1.5x)  
**When**: Exit Brain evaluates  
**Then**: Result = Tighten stop 25%

```python
def test_volatility_1_5x():
    position = Position(entry_volatility=0.02)
    
    result = exit_brain.check_volatility_exit(position, current_vol=0.03)
    
    assert result.action == "TIGHTEN_STOP"
    assert result.tighten_percent == 0.25
```

### TC-03.3: Volatility 2x (Tighten 50%)

**Given**: Entry volatility = 2%  
**And**: Current volatility = 4% (2x)  
**When**: Exit Brain evaluates  
**Then**: Result = Tighten stop 50%

```python
def test_volatility_2x():
    position = Position(entry_volatility=0.02)
    
    result = exit_brain.check_volatility_exit(position, current_vol=0.04)
    
    assert result.action == "TIGHTEN_STOP"
    assert result.tighten_percent == 0.50
```

### TC-03.4: Volatility 3x+ (Full Exit)

**Given**: Entry volatility = 2%  
**And**: Current volatility = 6.5% (3.25x)  
**When**: Exit Brain evaluates  
**Then**: Result = `close.intent`, urgency = CRITICAL

```python
def test_volatility_3x_exit():
    position = Position(entry_volatility=0.02)
    
    result = exit_brain.check_volatility_exit(position, current_vol=0.065)
    
    assert result.action == "CLOSE_FULL"
    assert result.urgency == "CRITICAL"
    assert "spike" in result.reason.lower()
```

### TC-03.5: Stop Tightening Only Tightens (Never Widens)

**Given**: Previous tightening applied  
**And**: Volatility decreases  
**When**: Exit Brain evaluates  
**Then**: Stop remains at tightened level

```python
def test_stop_never_widens():
    position = Position(
        entry_volatility=0.02,
        stop_loss=49000,
        tightened_stop=49500  # Previously tightened
    )
    
    # Vol decreases
    result = exit_brain.check_volatility_exit(position, current_vol=0.02)
    
    # Stop should not widen back
    assert position.stop_loss >= 49500
```

---

## Acceptance Criteria

- [ ] 1.5x vol tightens 25%
- [ ] 2x vol tightens 50%
- [ ] 3x+ vol exits immediately
- [ ] Stop never widens after tightening
