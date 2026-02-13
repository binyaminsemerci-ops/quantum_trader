# EB-FT-02: Regime Exit Test

**Test ID**: EB-FT-02  
**Category**: Primary Exit Logic  
**Priority**: P0 (Must Pass)  

---

## Test Objective

Verify that regime changes trigger appropriate exit actions.

---

## Test Cases

### TC-02.1: Regime Unchanged (No Exit)

**Given**: Entry regime = TRENDING_UP  
**And**: Current regime = TRENDING_UP  
**When**: Exit Brain evaluates  
**Then**: Result = None (no regime exit)

```python
def test_regime_unchanged():
    position = Position(entry_regime=Regime.TRENDING_UP)
    
    result = exit_brain.check_regime_exit(position, Regime.TRENDING_UP)
    
    assert result is None
```

### TC-02.2: Regime Change to Opposite (Full Exit)

**Given**: Entry regime = TRENDING_UP  
**And**: Current regime = TRENDING_DOWN  
**When**: Exit Brain evaluates  
**Then**: Result = `close.intent` (100%)

```python
def test_regime_reversal():
    position = Position(entry_regime=Regime.TRENDING_UP)
    
    result = exit_brain.check_regime_exit(position, Regime.TRENDING_DOWN)
    
    assert result.action == "CLOSE_FULL"
    assert "reversed" in result.reason.lower()
```

### TC-02.3: Regime Change to CHAOTIC (Full Exit)

**Given**: Entry regime = any  
**And**: Current regime = CHAOTIC  
**When**: Exit Brain evaluates  
**Then**: Result = `close.intent` (100%), urgency = CRITICAL

```python
def test_regime_to_chaotic():
    position = Position(entry_regime=Regime.TRENDING_UP)
    
    result = exit_brain.check_regime_exit(position, Regime.CHAOTIC)
    
    assert result.action == "CLOSE_FULL"
    assert result.urgency == "CRITICAL"
```

### TC-02.4: Regime Change to Ranging (Partial Exit)

**Given**: Entry regime = TRENDING_UP  
**And**: Current regime = RANGING  
**When**: Exit Brain evaluates  
**Then**: Result = `reduce.intent` (50%)

```python
def test_regime_to_ranging():
    position = Position(entry_regime=Regime.TRENDING_UP)
    
    result = exit_brain.check_regime_exit(position, Regime.RANGING)
    
    assert result.action == "CLOSE_PARTIAL"
    assert result.close_percent == 0.50
```

### TC-02.5: Regime Change to UNCERTAIN (75% Exit)

**Given**: Entry regime = TRENDING_UP  
**And**: Current regime = UNCERTAIN  
**When**: Exit Brain evaluates  
**Then**: Result = `reduce.intent` (75%)

```python
def test_regime_to_uncertain():
    position = Position(entry_regime=Regime.TRENDING_UP)
    
    result = exit_brain.check_regime_exit(position, Regime.UNCERTAIN)
    
    assert result.action == "CLOSE_PARTIAL"
    assert result.close_percent == 0.75
```

### TC-02.6: Ranging Entry - Any Change = Full Exit

**Given**: Entry regime = RANGING  
**And**: Current regime = any different  
**When**: Exit Brain evaluates  
**Then**: Result = `close.intent` (100%)

```python
def test_ranging_any_change():
    position = Position(entry_regime=Regime.RANGING)
    
    for new_regime in [Regime.TRENDING_UP, Regime.TRENDING_DOWN, Regime.UNCERTAIN]:
        result = exit_brain.check_regime_exit(position, new_regime)
        assert result.action == "CLOSE_FULL"
```

---

## Acceptance Criteria

- [ ] Opposite regime = full exit
- [ ] CHAOTIC = full exit (critical)
- [ ] Neutral change = partial exit
- [ ] RANGING entry + any change = full exit
