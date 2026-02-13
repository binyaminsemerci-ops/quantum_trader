# Signal Advisory – Scoring Model

> **Purpose**: Define how edge_score (0-100) is calculated  
> **Authority**: Advisory only – scores inform, Risk Kernel decides  
> **Version**: MVP v1.0

---

## 1. Edge Score Definition

### What Edge Score Represents

```
Edge Score (0-100):
  - Probability that signal direction is correct
  - Combined model confidence across multiple inputs
  - NOT a guarantee, NOT execution authority

0-39:   WEAK edge – unlikely to be profitable
40-69:  MODERATE edge – consider with caution
70-89:  STRONG edge – good signal, subject to Risk Kernel
90-100: EXTREME edge – rare, high conviction
```

### Edge Score Components

```python
class EdgeScoreComponents:
    """
    Edge score is computed from multiple model outputs.
    Each component contributes to final score.
    """
    
    # Technical components (40% weight)
    trend_alignment: float     # 0-1, does PA align with trend?
    momentum_score: float      # 0-1, momentum strength
    volume_confirmation: float # 0-1, volume supports move?
    
    # Regime components (30% weight)
    regime_fit: float          # 0-1, signal fits regime?
    regime_stability: float    # 0-1, is regime stable?
    
    # Statistical components (20% weight)
    historical_winrate: float  # 0-1, past performance of similar signals
    expected_rr: float         # Risk-reward expectation
    
    # Model agreement (10% weight)
    model_consensus: float     # 0-1, do models agree?
```

---

## 2. Scoring Formula

### Base Calculation

```python
def calculate_edge_score(components: EdgeScoreComponents) -> int:
    """
    Calculate edge score 0-100.
    
    Higher score = stronger edge (NOT guaranteed profit).
    """
    
    # Weight definitions
    W_TECHNICAL = 0.40
    W_REGIME = 0.30
    W_STATISTICAL = 0.20
    W_CONSENSUS = 0.10
    
    # Technical score (0-40 points)
    technical = (
        components.trend_alignment * 0.4 +
        components.momentum_score * 0.35 +
        components.volume_confirmation * 0.25
    ) * 100 * W_TECHNICAL
    
    # Regime score (0-30 points)
    regime = (
        components.regime_fit * 0.6 +
        components.regime_stability * 0.4
    ) * 100 * W_REGIME
    
    # Statistical score (0-20 points)
    statistical = (
        components.historical_winrate * 0.5 +
        min(components.expected_rr / 3.0, 1.0) * 0.5  # Cap at 3:1 RR
    ) * 100 * W_STATISTICAL
    
    # Consensus score (0-10 points)
    consensus = components.model_consensus * 100 * W_CONSENSUS
    
    # Final score
    edge_score = int(technical + regime + statistical + consensus)
    
    return min(max(edge_score, 0), 100)  # Clamp 0-100
```

### Score Thresholds

| Score Range | Classification | Action |
|-------------|----------------|--------|
| 0-39 | WEAK | NO SIGNAL |
| 40-54 | LOW_MODERATE | Signal only if regime=TRENDING |
| 55-69 | MODERATE | Signal with caveats |
| 70-84 | STRONG | Signal recommended |
| 85-100 | VERY_STRONG | High conviction signal |

---

## 3. Regime-Aware Scoring

### Regime Multipliers

```python
# Edge score is modified based on regime
REGIME_MULTIPLIERS = {
    "TRENDING": 1.0,    # Full edge in trending markets
    "RANGING": 0.85,    # 15% discount in ranging markets
    "CHAOTIC": 0.60,    # 40% discount in chaotic markets
}

def adjust_for_regime(base_score: int, regime: str) -> int:
    """
    Apply regime-based multiplier to edge score.
    """
    multiplier = REGIME_MULTIPLIERS.get(regime, 0.60)  # Default to CHAOTIC
    return int(base_score * multiplier)

# Example:
# Base score: 75
# Regime: CHAOTIC
# Adjusted: 75 * 0.60 = 45 (barely above weak threshold)
```

### Regime-Signal Compatibility

```python
REGIME_SIGNAL_COMPATIBILITY = {
    "TRENDING": {
        "TREND_FOLLOW": 1.0,     # Perfect fit
        "MEAN_REVERSION": 0.5,   # Poor fit
        "BREAKOUT": 0.8,         # Good fit
    },
    "RANGING": {
        "TREND_FOLLOW": 0.4,     # Poor fit
        "MEAN_REVERSION": 1.0,   # Perfect fit
        "BREAKOUT": 0.6,         # Moderate fit
    },
    "CHAOTIC": {
        "TREND_FOLLOW": 0.3,     # Very poor fit
        "MEAN_REVERSION": 0.4,   # Poor fit
        "BREAKOUT": 0.5,         # Poor fit
    },
}
```

---

## 4. Signal Types

### Supported Signal Types

```python
class SignalType(Enum):
    """
    Types of signals Signal Advisory can produce.
    """
    
    TREND_FOLLOW = "TREND_FOLLOW"      # Follow established trend
    MEAN_REVERSION = "MEAN_REVERSION"  # Fade extremes
    BREAKOUT = "BREAKOUT"              # Trade range breakouts
    NO_SIGNAL = "NO_SIGNAL"            # No actionable edge
```

### Signal Type Requirements

| Signal Type | Minimum Edge Score | Preferred Regime | Notes |
|-------------|-------------------|------------------|-------|
| TREND_FOLLOW | 55 | TRENDING | Best in clear trends |
| MEAN_REVERSION | 60 | RANGING | Requires range bounds |
| BREAKOUT | 65 | RANGING→TRENDING | Transition signals |
| NO_SIGNAL | N/A | Any | Default when no edge |

---

## 5. Model Inputs

### Data Requirements

```python
class SignalInputs:
    """
    Required inputs for signal generation.
    Missing inputs → NO SIGNAL (fail-closed).
    """
    
    # Price data (required)
    price_current: float      # Current market price
    price_history: list       # Recent prices (min 100 candles)
    
    # Volume data (required)
    volume_current: float     # Current volume
    volume_history: list      # Recent volumes
    
    # Regime data (required)
    regime: str               # TRENDING | RANGING | CHAOTIC
    regime_confidence: float  # How certain is regime?
    
    # Technical indicators (computed)
    indicators: dict          # MA, RSI, ATR, etc.
    
    # Missing data handling
    def validate(self) -> bool:
        """
        Validate all required inputs present.
        ANY missing → NO SIGNAL.
        """
        if self.price_current is None:
            return False
        if len(self.price_history) < 100:
            return False
        if self.regime is None:
            return False
        return True
```

### Fail-Closed on Missing Data

```python
def generate_signal(inputs: SignalInputs) -> SignalResponse:
    """
    Generate signal from inputs.
    Missing data → NO SIGNAL (fail-closed).
    """
    
    # Validate inputs
    if not inputs.validate():
        return SignalResponse(
            edge_score=0,
            confidence=0.0,
            regime_fit=False,
            recommended_action="NO_SIGNAL",
            reasoning="Missing required data – fail-closed"
        )
    
    # Continue with signal generation...
```

---

## 6. Score Calibration

### Historical Validation

```python
CALIBRATION_REQUIREMENTS = {
    "minimum_backtest_samples": 1000,   # Minimum historical signals
    "minimum_winrate_for_score": {
        "40-54": 0.50,  # 50% winrate for low-moderate
        "55-69": 0.55,  # 55% winrate for moderate
        "70-84": 0.60,  # 60% winrate for strong
        "85-100": 0.65, # 65% winrate for very strong
    },
    "max_score_inflation": 0.10,  # Score should track actual performance ±10%
}
```

### Score Accuracy Monitoring

```python
def monitor_score_accuracy(predictions: list, outcomes: list):
    """
    Monitor if edge scores predict actual outcomes.
    If scores diverge from reality → recalibrate.
    """
    
    buckets = {
        "40-54": {"predicted": 0, "won": 0},
        "55-69": {"predicted": 0, "won": 0},
        "70-84": {"predicted": 0, "won": 0},
        "85-100": {"predicted": 0, "won": 0},
    }
    
    for pred, outcome in zip(predictions, outcomes):
        bucket = get_bucket(pred.edge_score)
        buckets[bucket]["predicted"] += 1
        if outcome.profitable:
            buckets[bucket]["won"] += 1
    
    # Check calibration
    for bucket, data in buckets.items():
        if data["predicted"] > 50:
            actual_winrate = data["won"] / data["predicted"]
            expected_winrate = CALIBRATION_REQUIREMENTS["minimum_winrate_for_score"][bucket]
            
            if abs(actual_winrate - expected_winrate) > 0.10:
                emit_event("signal.calibration.drift", {
                    "bucket": bucket,
                    "expected": expected_winrate,
                    "actual": actual_winrate,
                })
```

---

## 7. Output Examples

### High Confidence Signal

```json
{
  "signal_id": "sig_20260203_001",
  "timestamp": "2026-02-03T10:30:00Z",
  "edge_score": 78,
  "confidence": 0.72,
  "regime_fit": true,
  "regime": "TRENDING",
  "regime_confidence": 0.85,
  "recommended_action": "LONG",
  "reasoning": "Strong trend alignment (0.88), momentum confirmation (0.75), volume supports (0.70). Regime is TRENDING with high stability.",
  "model_version": "v1.0.3",
  "latency_ms": 23,
  "execution_power": "NONE"
}
```

### Weak Signal (No Trade)

```json
{
  "signal_id": "sig_20260203_002",
  "timestamp": "2026-02-03T11:15:00Z",
  "edge_score": 35,
  "confidence": 0.45,
  "regime_fit": false,
  "regime": "CHAOTIC",
  "regime_confidence": 0.72,
  "recommended_action": "NO_SIGNAL",
  "reasoning": "Low edge score (35), confidence below floor (0.45 < 0.60). Regime is CHAOTIC – signal type mismatch. Standing aside.",
  "model_version": "v1.0.3",
  "latency_ms": 18,
  "execution_power": "NONE"
}
```

---

*Edge score er et estimat. Risk Kernel har siste ord. Alltid.*
