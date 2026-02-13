# Signal Advisory – Policy Reference

> **Port**: 8006  
> **Authority Level**: ADVISORY ONLY (Level 0)  
> **Execution Power**: NONE  
> **Source of Truth**: constitution/GOVERNANCE.md, constitution/RISK_POLICY.md

---

## 1. Constitutional Authority

### Grunnlov References

| Grunnlov | Relevant Text | Signal Advisory Role |
|----------|---------------|----------------------|
| §8 | "AI er rådgiver, ikke beslutningstaker" | **DEFINES Signal Advisory** – advisory function only |
| §1 | Kapitalvern er førsteprinsipp | Må akseptere Risk Kernel VETO |
| §2 | Risk Kernel har vetorett | Signal Advisory INGEN override-rett |
| §15 | Systemet KAN stå stille | Signal Advisory kan SI "ingen signal" |

---

## 2. Mandate

### Hva Signal Advisory SKAL Gjøre

```
┌─────────────────────────────────────────────────────────────────┐
│  SIGNAL ADVISORY MANDATE:                                       │
│                                                                 │
│  1. ANALYSE: Evaluer markedsforhold                             │
│  2. SCORE: Produser edge_score (0-100)                          │
│  3. CONFIDENCE: Estimer confidence (0.0-1.0)                    │
│  4. REGIME-FIT: Vurder om signal passer regime                  │
│  5. ADVISE: Gi anbefaling (IKKE ordre)                          │
│                                                                 │
│  Signal Advisory FORESLÅR.                                      │
│  Risk Kernel BESLUTTER.                                         │
│  Execution Engine UTFØRER.                                      │
└─────────────────────────────────────────────────────────────────┘
```

### Hva Signal Advisory IKKE KAN Gjøre

| Handling | Tillatt? | Grunn |
|----------|----------|-------|
| Sende ordre til exchange | ❌ NEI | Ingen execution power |
| Overstyre Risk Kernel VETO | ❌ NEI | Risk Kernel er overordnet |
| Endre posisjonsstørrelse | ❌ NEI | Position Sizer sin oppgave |
| Flytte stop-loss | ❌ NEI | Exit Brain sin oppgave |
| Tvinge trade gjennomføring | ❌ NEI | Advisory only |
| Ignorere regime-analyse | ❌ NEI | Regime-fit er påkrevd |

---

## 3. Output Contract

### Signal Response Format

```python
class SignalResponse:
    """
    Signal Advisory output – ADVISORY ONLY
    
    This response is a RECOMMENDATION, not an ORDER.
    Risk Kernel can VETO regardless of confidence.
    """
    
    signal_id: str           # Unique identifier
    timestamp: datetime      # Signal generation time
    
    # Core advisory output
    edge_score: int          # 0-100, where 100 = maximum edge
    confidence: float        # 0.0-1.0, regime-adjusted
    regime_fit: bool         # Does signal fit current regime?
    
    # Context (informational)
    regime: str              # TRENDING | RANGING | CHAOTIC
    regime_confidence: float # How certain is regime classification?
    
    # Advisory recommendation (NOT binding)
    recommended_action: str  # LONG | SHORT | NO_SIGNAL
    reasoning: str           # Why this recommendation
    
    # Metadata
    model_version: str       # Which model produced this
    latency_ms: int          # Processing time
    
    # EXPLICIT: No execution power
    execution_power: str = "NONE"  # Always "NONE"
```

### Confidence Floors (Immutable)

```python
CONFIDENCE_FLOORS = {
    "minimum_signal": 0.6,      # Below this → NO SIGNAL
    "minimum_regime_fit": 0.5,  # Regime confidence floor
    "chaotic_penalty": 0.30,    # 30% reduction in CHAOTIC regime
}

# Example:
# Base confidence: 0.75
# Regime: CHAOTIC
# Adjusted confidence: 0.75 * (1 - 0.30) = 0.525 → Below 0.6 → NO SIGNAL
```

---

## 4. Integration Rules

### Signal → Risk Kernel Flow

```
┌──────────────┐      ┌──────────────┐      ┌────────────────┐
│   Signal     │      │    Risk      │      │   Execution    │
│   Advisory   │─────▶│    Kernel    │─────▶│    Engine      │
│              │      │              │      │                │
│  ADVISES     │      │  DECIDES     │      │  EXECUTES      │
└──────────────┘      └──────────────┘      └────────────────┘
      │                      │
      │                      │
      ▼                      ▼
  "I suggest             "I APPROVE or
   a LONG with            I VETO.
   confidence 0.72"       Final word."
```

### Risk Kernel Considerations

Risk Kernel CONSIDERS Signal Advisory output, but:

```python
# Risk Kernel logic for signal consideration
def consider_signal(signal: SignalResponse) -> bool:
    """
    Risk Kernel considers signal, but CAN VETO regardless.
    """
    
    # 1. Check confidence floor
    if signal.confidence < 0.6:
        return False  # Below floor
    
    # 2. Check regime fit
    if not signal.regime_fit:
        return False  # Regime mismatch
    
    # 3. Check edge score
    if signal.edge_score < 40:
        return False  # Edge too weak
    
    # 4. Signal accepted for further evaluation
    # BUT: Risk Kernel can STILL VETO based on:
    # - Trade risk (2% limit)
    # - Daily loss (approaching -5%)
    # - Drawdown (approaching -20%)
    # - System integrity issues
    
    return True  # Signal considered (not guaranteed execution)
```

---

## 5. Policy Binding

### Immutable Rules

| Rule | Value | Consequence |
|------|-------|-------------|
| Cannot execute trades | Always | No API access to exchange |
| Cannot override Risk Kernel | Always | VETO is final |
| Must provide confidence | Always | No signal without confidence |
| Must respect regime | Always | Chaotic penalty mandatory |
| Confidence floor | 0.6 | Below = NO SIGNAL |

### Policy References

- §8: "AI er rådgiver, ikke beslutningstaker" → Signal Advisory role definition
- §1: Kapitalvern over profitt → Must accept Risk Kernel VETO
- §2: Risk Kernel vetorett → Advisory subordinate to Risk Kernel
- §15: Systemet KAN stå stille → "NO_SIGNAL" is valid output

---

## 6. Advisory Philosophy

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  "Signal Advisory is the researcher, not the trader."           │
│                                                                 │
│  We ANALYSE.           We do NOT execute.                       │
│  We RECOMMEND.         We do NOT decide.                        │
│  We SCORE.             We do NOT override.                      │
│  We ADVISE.            We do NOT force.                         │
│                                                                 │
│  Our confidence is a probability estimate, not a certainty.     │
│  Risk Kernel has final say. Always.                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

*Signal Advisory vet når den har edge. Risk Kernel bestemmer om det er forsvarlig å bruke den.*
