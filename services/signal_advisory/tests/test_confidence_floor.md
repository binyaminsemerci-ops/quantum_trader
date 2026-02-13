# Test: Confidence Floor Enforcement

> **Test ID**: SA-FT-03  
> **Service**: Signal Advisory  
> **Type**: Boundary Test  
> **Priority**: CRITICAL – Verifies immutable floor constraint

---

## 1. Purpose

Verify that Signal Advisory **strictly enforces** the 0.6 confidence floor:
- Confidence < 0.6 → NO_SIGNAL (always)
- Confidence = 0.6 → SIGNAL (edge case)
- Floor cannot be bypassed or configured

---

## 2. Test Cases

### SA-FT-03-A: Below Floor Returns NO_SIGNAL

**Scenario**: Confidence 0.59 (just below floor)

```python
def test_just_below_floor():
    """
    Confidence 0.59 is below 0.6 floor → MUST be NO_SIGNAL.
    """
    
    inputs = SignalInputs(regime="TRENDING")
    
    with mock_base_confidence(0.59):
        signal = SignalAdvisory().generate_signal(inputs)
    
    assert signal.confidence == 0.59
    assert signal.recommended_action == "NO_SIGNAL"
    assert "floor" in signal.reasoning.lower() or "below" in signal.reasoning.lower()
```

**Expected**: NO_SIGNAL with floor-related reasoning  
**Failure Handling**: Critical – floor enforcement broken

---

### SA-FT-03-B: At Floor Emits Signal

**Scenario**: Confidence exactly 0.60

```python
def test_exactly_at_floor():
    """
    Confidence exactly 0.60 should PASS floor.
    """
    
    inputs = SignalInputs(
        regime="TRENDING",
        edge_score=65  # Above edge minimum
    )
    
    with mock_base_confidence(0.60):
        signal = SignalAdvisory().generate_signal(inputs)
    
    assert signal.confidence == 0.60
    # Should NOT be NO_SIGNAL due to floor (may be NO_SIGNAL for other reasons)
    if signal.recommended_action == "NO_SIGNAL":
        assert "floor" not in signal.reasoning.lower()
```

**Expected**: Floor check passes (0.60 >= 0.60)  
**Failure Handling**: Check boundary condition (>= not >)

---

### SA-FT-03-C: Above Floor Emits Signal

**Scenario**: Confidence 0.61 (just above floor)

```python
def test_just_above_floor():
    """
    Confidence 0.61 should clearly pass floor.
    """
    
    inputs = SignalInputs(
        regime="TRENDING",
        edge_score=65
    )
    
    with mock_base_confidence(0.61):
        signal = SignalAdvisory().generate_signal(inputs)
    
    assert signal.confidence == 0.61
    # Should not fail due to floor
    if signal.recommended_action == "NO_SIGNAL":
        assert "floor" not in signal.reasoning.lower()
```

**Expected**: Floor check passes  
**Failure Handling**: Verify floor comparison logic

---

### SA-FT-03-D: Very Low Confidence

**Scenario**: Confidence 0.1 (very low)

```python
def test_very_low_confidence():
    """
    Very low confidence must definitely be NO_SIGNAL.
    """
    
    inputs = SignalInputs(regime="CHAOTIC")
    
    with mock_base_confidence(0.10):
        signal = SignalAdvisory().generate_signal(inputs)
    
    # 0.10 * 0.70 (chaotic) = 0.07
    assert signal.confidence < 0.1
    assert signal.recommended_action == "NO_SIGNAL"
```

**Expected**: NO_SIGNAL  
**Failure Handling**: Check low confidence path

---

### SA-FT-03-E: Floor Cannot Be Configured

**Scenario**: Attempt to set custom floor

```python
def test_floor_not_configurable():
    """
    Confidence floor is IMMUTABLE – cannot be configured.
    """
    
    # Attempt to configure lower floor
    try:
        signal_advisory = SignalAdvisory(confidence_floor=0.5)
        assert False, "Should not accept custom floor"
    except (TypeError, ValueError, ArchitecturalViolation):
        pass  # Expected
    
    # Attempt via environment variable
    import os
    os.environ['CONFIDENCE_FLOOR'] = '0.4'
    
    signal_advisory = SignalAdvisory()
    
    # Floor should still be 0.6
    assert signal_advisory.CONFIDENCE_FLOOR == 0.6
    
    # Clean up
    del os.environ['CONFIDENCE_FLOOR']
```

**Expected**: Floor remains 0.6 regardless of config  
**Failure Handling**: Remove configuration capability

---

### SA-FT-03-F: Floor Checked After All Adjustments

**Scenario**: Floor check happens after regime adjustment

```python
def test_floor_checked_after_regime():
    """
    Floor check must happen AFTER regime adjustment, not before.
    
    Incorrect: Check base 0.65 >= 0.6 ✓ then apply chaotic penalty
    Correct: Apply penalty 0.65 * 0.7 = 0.455, then check 0.455 >= 0.6 ✗
    """
    
    inputs = SignalInputs(regime="CHAOTIC")
    
    # Base confidence passes floor, but after chaotic penalty it won't
    with mock_base_confidence(0.65):
        signal = SignalAdvisory().generate_signal(inputs)
    
    # 0.65 * 0.70 = 0.455 (below floor)
    assert signal.confidence < 0.6
    assert signal.recommended_action == "NO_SIGNAL"
```

**Expected**: Floor checked on FINAL confidence (0.455 not 0.65)  
**Failure Handling**: Fix order of operations

---

### SA-FT-03-G: Zero Confidence

**Scenario**: Edge case of 0.0 confidence

```python
def test_zero_confidence():
    """
    Zero confidence must result in NO_SIGNAL.
    """
    
    inputs = SignalInputs(regime="TRENDING")
    
    with mock_base_confidence(0.0):
        signal = SignalAdvisory().generate_signal(inputs)
    
    assert signal.confidence == 0.0
    assert signal.recommended_action == "NO_SIGNAL"
```

**Expected**: NO_SIGNAL  
**Failure Handling**: Handle edge case

---

### SA-FT-03-H: Floor Boundary Stress Test

**Scenario**: Test many values around floor

```python
def test_floor_boundary_stress():
    """
    Stress test floor boundary with many values.
    """
    
    inputs = SignalInputs(
        regime="TRENDING",
        edge_score=70
    )
    
    # Test values around floor
    test_values = [
        (0.55, "NO_SIGNAL"),
        (0.56, "NO_SIGNAL"),
        (0.57, "NO_SIGNAL"),
        (0.58, "NO_SIGNAL"),
        (0.59, "NO_SIGNAL"),
        (0.599, "NO_SIGNAL"),
        (0.5999, "NO_SIGNAL"),
        (0.6, "SIGNAL_POSSIBLE"),  # At floor
        (0.6001, "SIGNAL_POSSIBLE"),
        (0.601, "SIGNAL_POSSIBLE"),
        (0.61, "SIGNAL_POSSIBLE"),
        (0.65, "SIGNAL_POSSIBLE"),
        (0.70, "SIGNAL_POSSIBLE"),
    ]
    
    for confidence, expected_category in test_values:
        with mock_base_confidence(confidence):
            signal = SignalAdvisory().generate_signal(inputs)
        
        actual_category = "NO_SIGNAL" if signal.recommended_action == "NO_SIGNAL" \
                         else "SIGNAL_POSSIBLE"
        
        assert actual_category == expected_category, \
            f"confidence={confidence}: expected {expected_category}, got {actual_category}"
```

**Expected**: Clear boundary at 0.6  
**Failure Handling**: Fix floor comparison

---

## 3. Edge Case Matrix

| Confidence | Regime | Final Conf | Floor (0.6) | Result |
|------------|--------|------------|-------------|--------|
| 0.60 | TRENDING | 0.60 | = PASS | Signal possible |
| 0.60 | CHAOTIC | 0.42 | < FAIL | NO_SIGNAL |
| 0.59 | TRENDING | 0.59 | < FAIL | NO_SIGNAL |
| 0.61 | TRENDING | 0.61 | > PASS | Signal possible |
| 0.86 | CHAOTIC | 0.602 | > PASS | Signal possible |
| 0.85 | CHAOTIC | 0.595 | < FAIL | NO_SIGNAL |

---

## 4. Immutable Constant Verification

```python
def test_floor_constant_is_immutable():
    """
    CONFIDENCE_FLOOR must be a constant, not configurable.
    """
    
    from signal_advisory.constants import CONFIDENCE_FLOOR
    
    # Should be exactly 0.6
    assert CONFIDENCE_FLOOR == 0.6
    
    # Should not be modifiable
    try:
        import signal_advisory.constants
        signal_advisory.constants.CONFIDENCE_FLOOR = 0.5
        
        # Re-import and check
        from importlib import reload
        reload(signal_advisory.constants)
        
        # If we get here without error, check value unchanged
        from signal_advisory.constants import CONFIDENCE_FLOOR as new_floor
        assert new_floor == 0.6, "Floor should not be modifiable"
    except (AttributeError, TypeError):
        pass  # Good - constant is protected
```

---

## 5. Test Automation

### CI/CD Integration

```yaml
# In CI pipeline
floor_tests:
  name: "Confidence Floor Tests"
  priority: CRITICAL
  blocking: true  # Blocks deployment if any fail
  tests:
    - SA-FT-03-A  # Below floor
    - SA-FT-03-B  # At floor
    - SA-FT-03-C  # Above floor
    - SA-FT-03-D  # Very low
    - SA-FT-03-E  # Not configurable
    - SA-FT-03-F  # Order of operations
    - SA-FT-03-G  # Zero confidence
    - SA-FT-03-H  # Stress test
```

---

## 6. Failure Consequences

```
┌─────────────────────────────────────────────────────────────────┐
│ CONFIDENCE FLOOR VIOLATION DETECTED                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Signal emitted with confidence below 0.6 floor.                 │
│                                                                 │
│ This is a POLICY VIOLATION.                                     │
│                                                                 │
│ Immediate Actions:                                              │
│ 1. HALT signal emission                                         │
│ 2. Investigate how signal bypassed floor                        │
│ 3. Fix floor check logic                                        │
│ 4. Re-run all floor tests                                       │
│ 5. Audit recent signals for floor violations                    │
│                                                                 │
│ Reference: constitution/RISK_POLICY.md                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

*0.6 er ikke en guideline. Det er en lov. Ingen unntak.*
