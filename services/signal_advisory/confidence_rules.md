# Signal Advisory – Confidence Rules

> **Purpose**: Define confidence calculation and regime-aware adjustments  
> **Key Rule**: Confidence < 0.6 → NO SIGNAL  
> **Immutable**: CHAOTIC regime = 30% penalty

---

## 1. Confidence Definition

### What Confidence Represents

```
Confidence (0.0 - 1.0):
  - Probability estimate that signal is correct
  - NOT certainty – even 0.9 confidence can be wrong
  - Regime-adjusted – chaotic markets reduce confidence
  
Confidence is EPISTEMIC:
  - How much we KNOW, not how much we HOPE
  - Based on model certainty, not signal strength
```

### Confidence vs Edge Score

| Metric | Definition | Use |
|--------|------------|-----|
| Edge Score | Signal quality (0-100) | How good is the opportunity? |
| Confidence | Model certainty (0.0-1.0) | How sure are we? |

```
Example:
  - Edge Score: 75 (good opportunity)
  - Confidence: 0.50 (but we're unsure)
  → NO SIGNAL (confidence below floor)

Example:
  - Edge Score: 55 (moderate opportunity)
  - Confidence: 0.75 (we're fairly certain)
  → SIGNAL (subject to Risk Kernel)
```

---

## 2. Confidence Calculation

### Base Confidence Formula

```python
def calculate_base_confidence(
    model_outputs: list,
    data_quality: float,
    temporal_stability: float
) -> float:
    """
    Calculate base confidence before regime adjustment.
    
    Args:
        model_outputs: List of individual model confidences
        data_quality: Quality of input data (0-1)
        temporal_stability: How stable is signal over time (0-1)
    
    Returns:
        Base confidence (0-1)
    """
    
    # Model agreement (50% weight)
    model_confidences = [m.confidence for m in model_outputs]
    model_mean = sum(model_confidences) / len(model_confidences)
    model_std = calculate_std(model_confidences)
    
    # Penalize disagreement
    agreement_penalty = min(model_std * 2, 0.3)  # Max 0.3 penalty
    model_score = model_mean - agreement_penalty
    
    # Data quality (30% weight)
    # Poor data = low confidence
    data_score = data_quality
    
    # Temporal stability (20% weight)
    # Flipping signals = low confidence
    stability_score = temporal_stability
    
    # Weighted combination
    base_confidence = (
        model_score * 0.50 +
        data_score * 0.30 +
        stability_score * 0.20
    )
    
    return max(0.0, min(1.0, base_confidence))
```

### Confidence Components

```python
class ConfidenceComponents:
    """
    Factors that contribute to confidence score.
    """
    
    # Model certainty
    primary_model_confidence: float   # Main model's confidence
    ensemble_agreement: float         # Do models agree?
    
    # Data quality
    data_freshness: float            # How recent is data?
    data_completeness: float         # Any missing data?
    data_reliability: float          # Source quality
    
    # Temporal factors
    signal_stability: float          # Same signal over time?
    prediction_horizon: float        # Shorter = more confident
    
    # Market factors
    regime_clarity: float            # Clear regime = more confident
    volatility_normalcy: float       # Normal vol = more confident
```

---

## 3. Regime Adjustments (MANDATORY)

### Regime Penalties

```python
# IMMUTABLE: Regime confidence adjustments
REGIME_CONFIDENCE_ADJUSTMENTS = {
    "TRENDING": {
        "multiplier": 1.0,      # No penalty
        "description": "Clear trends, high predictability"
    },
    "RANGING": {
        "multiplier": 0.90,     # 10% penalty
        "description": "Bounded moves, moderate predictability"
    },
    "CHAOTIC": {
        "multiplier": 0.70,     # 30% penalty (MANDATORY)
        "description": "Unpredictable, low confidence appropriate"
    },
    "UNKNOWN": {
        "multiplier": 0.60,     # 40% penalty
        "description": "Cannot classify, assume worst"
    },
}

def apply_regime_adjustment(
    base_confidence: float,
    regime: str
) -> float:
    """
    Apply regime-based confidence penalty.
    
    CHAOTIC regime ALWAYS receives 30% penalty.
    This is IMMUTABLE policy.
    """
    
    adjustment = REGIME_CONFIDENCE_ADJUSTMENTS.get(
        regime,
        REGIME_CONFIDENCE_ADJUSTMENTS["UNKNOWN"]  # Default conservative
    )
    
    adjusted = base_confidence * adjustment["multiplier"]
    
    return round(adjusted, 4)

# Example:
# Base confidence: 0.80
# Regime: CHAOTIC
# Adjusted: 0.80 * 0.70 = 0.56 → Below 0.6 floor → NO SIGNAL
```

### Why 30% Chaotic Penalty?

```
Historical Analysis:

TRENDING markets:
  - Signal accuracy: ~58%
  - Predictable patterns
  - Models perform as expected

CHAOTIC markets:
  - Signal accuracy: ~42%
  - Random-like behavior
  - Models perform WORSE than random

30% penalty accounts for:
  - Reduced model accuracy
  - Higher false positive rate
  - Unpredictable regime transitions
  
This is NOT adjustable. It's policy.
```

---

## 4. Confidence Floor (IMMUTABLE)

### The 0.6 Rule

```python
CONFIDENCE_FLOOR = 0.6  # IMMUTABLE

def apply_confidence_floor(confidence: float) -> tuple:
    """
    Apply confidence floor.
    Below floor → NO SIGNAL.
    
    Returns:
        (passes_floor: bool, final_confidence: float)
    """
    
    if confidence < CONFIDENCE_FLOOR:
        return (False, confidence)
    
    return (True, confidence)

# Usage:
passes, final = apply_confidence_floor(0.58)
# passes = False
# Signal should NOT be emitted
```

### Why 0.6?

```
Statistical reasoning:

At 0.6 confidence:
  - Implies ~60% expected accuracy
  - With proper risk management, marginal profitability
  
Below 0.6:
  - Approaches coin-flip territory
  - Transaction costs alone can make unprofitable
  - Better to stand aside
  
The floor is conservative by design.
We prefer missed opportunities over false signals.
```

---

## 5. Confidence Scenarios

### Scenario Matrix

| Base Conf | Regime | Multiplier | Final Conf | Floor (0.6) | Result |
|-----------|--------|------------|------------|-------------|--------|
| 0.90 | TRENDING | 1.0 | 0.90 | ✓ PASS | SIGNAL |
| 0.85 | CHAOTIC | 0.7 | 0.595 | ✗ FAIL | NO SIGNAL |
| 0.70 | RANGING | 0.9 | 0.63 | ✓ PASS | SIGNAL |
| 0.65 | CHAOTIC | 0.7 | 0.455 | ✗ FAIL | NO SIGNAL |
| 0.60 | TRENDING | 1.0 | 0.60 | ✓ PASS | SIGNAL |
| 0.59 | TRENDING | 1.0 | 0.59 | ✗ FAIL | NO SIGNAL |

### Critical Insight

```
To emit a signal in CHAOTIC regime, base confidence must be:
  
  Required final: 0.6
  Chaotic multiplier: 0.7
  Required base: 0.6 / 0.7 = 0.857

Only very high base confidence survives chaotic penalty.
This is INTENTIONAL – trade less in chaotic markets.
```

---

## 6. Confidence Over Time

### Temporal Decay

```python
def apply_temporal_decay(
    confidence: float,
    signal_age_seconds: int,
    max_signal_age: int = 300  # 5 minutes
) -> float:
    """
    Confidence decays as signal ages.
    Old signals are less reliable.
    """
    
    if signal_age_seconds >= max_signal_age:
        return 0.0  # Signal expired
    
    decay_factor = 1.0 - (signal_age_seconds / max_signal_age)
    decay_factor = max(0.5, decay_factor)  # Min 50% of original
    
    return confidence * decay_factor
```

### Signal Freshness Requirements

| Signal Age | Confidence Adjustment | Action |
|------------|----------------------|--------|
| 0-60s | 100% | Full confidence |
| 60-120s | 90% | Slight decay |
| 120-180s | 80% | Moderate decay |
| 180-240s | 70% | Significant decay |
| 240-300s | 60% | Near expiry |
| >300s | EXPIRED | NO SIGNAL |

---

## 7. Implementation

### Full Confidence Pipeline

```python
def calculate_final_confidence(
    model_outputs: list,
    data_quality: float,
    temporal_stability: float,
    regime: str,
    signal_age_seconds: int = 0
) -> dict:
    """
    Complete confidence calculation pipeline.
    
    Returns:
        {
            "base_confidence": float,
            "regime": str,
            "regime_multiplier": float,
            "regime_adjusted": float,
            "temporal_adjusted": float,
            "final_confidence": float,
            "passes_floor": bool,
            "can_emit_signal": bool,
        }
    """
    
    # Step 1: Calculate base confidence
    base_confidence = calculate_base_confidence(
        model_outputs,
        data_quality,
        temporal_stability
    )
    
    # Step 2: Apply regime adjustment
    regime_multiplier = REGIME_CONFIDENCE_ADJUSTMENTS.get(
        regime,
        REGIME_CONFIDENCE_ADJUSTMENTS["UNKNOWN"]
    )["multiplier"]
    
    regime_adjusted = base_confidence * regime_multiplier
    
    # Step 3: Apply temporal decay (if applicable)
    temporal_adjusted = apply_temporal_decay(
        regime_adjusted,
        signal_age_seconds
    )
    
    # Step 4: Check floor
    final_confidence = temporal_adjusted
    passes_floor = final_confidence >= CONFIDENCE_FLOOR
    
    return {
        "base_confidence": round(base_confidence, 4),
        "regime": regime,
        "regime_multiplier": regime_multiplier,
        "regime_adjusted": round(regime_adjusted, 4),
        "temporal_adjusted": round(temporal_adjusted, 4),
        "final_confidence": round(final_confidence, 4),
        "passes_floor": passes_floor,
        "can_emit_signal": passes_floor,
    }
```

### Example Output

```json
{
  "base_confidence": 0.7800,
  "regime": "CHAOTIC",
  "regime_multiplier": 0.70,
  "regime_adjusted": 0.5460,
  "temporal_adjusted": 0.5460,
  "final_confidence": 0.5460,
  "passes_floor": false,
  "can_emit_signal": false
}
```

---

## 8. Monitoring & Alerts

### Confidence Metrics

```python
CONFIDENCE_ALERTS = {
    "high_rejection_rate": {
        "threshold": 0.70,  # >70% of signals rejected
        "window": "1h",
        "action": "Review model calibration"
    },
    "confidence_inflation": {
        "threshold": 0.85,  # Average confidence too high
        "window": "24h",
        "action": "Models may be overconfident"
    },
    "floor_clustering": {
        "threshold": 0.20,  # >20% of signals at exactly floor
        "window": "4h",
        "action": "Check for threshold gaming"
    },
}
```

---

*Confidence under floor er en feature, ikke en bug. Stå stille når usikker.*
