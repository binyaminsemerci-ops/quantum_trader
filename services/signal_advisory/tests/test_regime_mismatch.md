# Test: Regime Mismatch Handling

> **Test ID**: SA-FT-02  
> **Service**: Signal Advisory  
> **Type**: Logic Verification Test  
> **Priority**: HIGH – Verifies regime-aware behavior

---

## 1. Purpose

Verify that Signal Advisory correctly handles regime mismatches:
- Applies 30% confidence penalty in CHAOTIC regime
- Sets `regime_fit=false` when signal type doesn't match regime
- Adjusts behavior based on regime classification

---

## 2. Test Cases

### SA-FT-02-A: CHAOTIC Regime 30% Penalty

**Scenario**: Base confidence 0.80 in CHAOTIC regime

```python
def test_chaotic_regime_penalty():
    """
    CHAOTIC regime MUST apply 30% confidence penalty.
    """
    
    inputs = SignalInputs(
        regime="CHAOTIC",
        regime_confidence=0.85,
        # Other inputs...
    )
    
    # Mock model to return 0.80 base confidence
    with mock_base_confidence(0.80):
        signal = SignalAdvisory().generate_signal(inputs)
    
    # 0.80 * 0.70 = 0.56 (below 0.6 floor)
    assert signal.confidence == 0.56
    assert signal.regime == "CHAOTIC"
    assert signal.recommended_action == "NO_SIGNAL"
    assert "below floor" in signal.reasoning.lower()
```

**Expected**: Confidence = 0.56, NO_SIGNAL  
**Failure Handling**: Verify regime adjustment logic

---

### SA-FT-02-B: CHAOTIC Requires High Base Confidence

**Scenario**: What base confidence is needed to pass in CHAOTIC?

```python
def test_chaotic_minimum_base_confidence():
    """
    In CHAOTIC regime, base confidence must be >0.857 to reach 0.6 floor.
    0.857 * 0.70 = 0.60
    """
    
    inputs = SignalInputs(regime="CHAOTIC")
    
    # Test 0.85 base → should fail floor
    with mock_base_confidence(0.85):
        signal_low = SignalAdvisory().generate_signal(inputs)
    
    assert signal_low.confidence < 0.6  # 0.85 * 0.70 = 0.595
    assert signal_low.recommended_action == "NO_SIGNAL"
    
    # Test 0.86 base → should pass floor
    with mock_base_confidence(0.86):
        signal_high = SignalAdvisory().generate_signal(inputs)
    
    assert signal_high.confidence >= 0.6  # 0.86 * 0.70 = 0.602
    # May still be NO_SIGNAL for other reasons (edge score, etc.)
```

**Expected**: 0.85 base fails, 0.86 base passes  
**Failure Handling**: Check chaotic multiplier = 0.70

---

### SA-FT-02-C: Trending No Penalty

**Scenario**: TRENDING regime should not reduce confidence

```python
def test_trending_no_penalty():
    """
    TRENDING regime should have 1.0 multiplier (no penalty).
    """
    
    inputs = SignalInputs(regime="TRENDING")
    
    with mock_base_confidence(0.75):
        signal = SignalAdvisory().generate_signal(inputs)
    
    # 0.75 * 1.0 = 0.75
    assert signal.confidence == 0.75
    assert signal.regime == "TRENDING"
    # Should pass floor (0.75 >= 0.6)
    assert signal.recommended_action != "NO_SIGNAL" or "other" in signal.reasoning.lower()
```

**Expected**: Confidence unchanged at 0.75  
**Failure Handling**: Verify TRENDING multiplier = 1.0

---

### SA-FT-02-D: Ranging Moderate Penalty

**Scenario**: RANGING regime applies 10% penalty

```python
def test_ranging_moderate_penalty():
    """
    RANGING regime should have 0.90 multiplier (10% penalty).
    """
    
    inputs = SignalInputs(regime="RANGING")
    
    with mock_base_confidence(0.70):
        signal = SignalAdvisory().generate_signal(inputs)
    
    # 0.70 * 0.90 = 0.63
    assert abs(signal.confidence - 0.63) < 0.01
    assert signal.regime == "RANGING"
```

**Expected**: Confidence = 0.63 (±0.01)  
**Failure Handling**: Verify RANGING multiplier = 0.90

---

### SA-FT-02-E: Signal Type Regime Mismatch

**Scenario**: TREND_FOLLOW signal in CHAOTIC regime

```python
def test_trend_follow_in_chaotic():
    """
    TREND_FOLLOW signal should have regime_fit=false in CHAOTIC.
    """
    
    inputs = SignalInputs(
        regime="CHAOTIC",
        signal_type="TREND_FOLLOW"
    )
    
    signal = SignalAdvisory().generate_signal(inputs)
    
    # TREND_FOLLOW does not fit CHAOTIC regime
    assert signal.regime_fit == False
    assert "regime" in signal.reasoning.lower() or "mismatch" in signal.reasoning.lower()
```

**Expected**: regime_fit=false  
**Failure Handling**: Check regime-signal compatibility matrix

---

### SA-FT-02-F: Mean Reversion in Trending

**Scenario**: MEAN_REVERSION signal in TRENDING regime

```python
def test_mean_reversion_in_trending():
    """
    MEAN_REVERSION signal should have poor regime fit in TRENDING.
    """
    
    inputs = SignalInputs(
        regime="TRENDING",
        signal_type="MEAN_REVERSION"
    )
    
    signal = SignalAdvisory().generate_signal(inputs)
    
    # MEAN_REVERSION does not fit TRENDING well
    assert signal.regime_fit == False
```

**Expected**: regime_fit=false  
**Failure Handling**: Check compatibility matrix

---

### SA-FT-02-G: Perfect Fit Scenarios

**Scenario**: Signal types that match regime perfectly

```python
def test_perfect_regime_fits():
    """
    Test signal types that perfectly fit their regime.
    """
    
    perfect_fits = [
        ("TREND_FOLLOW", "TRENDING"),
        ("MEAN_REVERSION", "RANGING"),
    ]
    
    for signal_type, regime in perfect_fits:
        inputs = SignalInputs(
            regime=regime,
            signal_type=signal_type
        )
        
        signal = SignalAdvisory().generate_signal(inputs)
        
        # Perfect fit scenarios
        assert signal.regime_fit == True, f"{signal_type} should fit {regime}"
```

**Expected**: All perfect fits have regime_fit=true  
**Failure Handling**: Update compatibility matrix

---

### SA-FT-02-H: Unknown Regime Handling

**Scenario**: Unknown or null regime should be treated as CHAOTIC

```python
def test_unknown_regime_defaults_to_chaotic():
    """
    Unknown regime should default to CHAOTIC treatment (40% penalty).
    """
    
    unknown_regimes = [None, "UNKNOWN", "INVALID", ""]
    
    for regime in unknown_regimes:
        inputs = SignalInputs(regime=regime)
        
        with mock_base_confidence(0.80):
            signal = SignalAdvisory().generate_signal(inputs)
        
        # Should apply UNKNOWN penalty (0.60 multiplier) or worse
        # 0.80 * 0.60 = 0.48
        assert signal.confidence <= 0.48 or signal.recommended_action == "NO_SIGNAL"
```

**Expected**: Unknown regimes treated conservatively  
**Failure Handling**: Verify default regime handling

---

## 3. Regime Penalty Matrix Reference

```python
# Reference for test validation
REGIME_PENALTIES = {
    "TRENDING": 1.0,    # No penalty
    "RANGING": 0.90,    # 10% penalty
    "CHAOTIC": 0.70,    # 30% penalty
    "UNKNOWN": 0.60,    # 40% penalty (default)
}

# regime_fit compatibility
REGIME_FIT_MATRIX = {
    ("TREND_FOLLOW", "TRENDING"): True,
    ("TREND_FOLLOW", "RANGING"): False,
    ("TREND_FOLLOW", "CHAOTIC"): False,
    ("MEAN_REVERSION", "TRENDING"): False,
    ("MEAN_REVERSION", "RANGING"): True,
    ("MEAN_REVERSION", "CHAOTIC"): False,
    ("BREAKOUT", "TRENDING"): True,
    ("BREAKOUT", "RANGING"): True,  # Can work at range boundaries
    ("BREAKOUT", "CHAOTIC"): False,
}
```

---

## 4. Test Matrix

| Test ID | Description | Input | Expected |
|---------|-------------|-------|----------|
| SA-FT-02-A | Chaotic 30% penalty | base=0.80, CHAOTIC | conf=0.56, NO_SIGNAL |
| SA-FT-02-B | Chaotic minimum base | varies, CHAOTIC | 0.857 minimum to pass |
| SA-FT-02-C | Trending no penalty | base=0.75, TRENDING | conf=0.75 |
| SA-FT-02-D | Ranging 10% penalty | base=0.70, RANGING | conf=0.63 |
| SA-FT-02-E | Trend in chaotic | TREND_FOLLOW, CHAOTIC | regime_fit=false |
| SA-FT-02-F | MR in trending | MEAN_REVERSION, TRENDING | regime_fit=false |
| SA-FT-02-G | Perfect fits | matched pairs | regime_fit=true |
| SA-FT-02-H | Unknown regime | regime=null | conservative treatment |

---

*Regime-awareness er ikke valgfritt. Signal Advisory må respektere markedsregimet.*
