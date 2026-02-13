# Signal Advisory – Decision Flow

> **Purpose**: Document how Signal Advisory integrates with the system  
> **Key Principle**: ADVISORY ONLY – no execution power  
> **Interaction**: Signal Advisory → Risk Kernel → Execution Engine

---

## 1. Flow Diagram

```
                                    ┌─────────────────┐
                                    │   Market Data   │
                                    │  (Price, Vol)   │
                                    └────────┬────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         SIGNAL ADVISORY                             │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  1. VALIDATE INPUTS                                          │    │
│  │     - Price data complete?                                   │    │
│  │     - Volume data complete?                                  │    │
│  │     - Regime classified?                                     │    │
│  │     ANY MISSING → NO_SIGNAL (fail-closed)                    │    │
│  └───────────────────────────┬─────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  2. CALCULATE EDGE SCORE                                     │    │
│  │     - Technical analysis (40%)                               │    │
│  │     - Regime fit (30%)                                       │    │
│  │     - Statistical edge (20%)                                 │    │
│  │     - Model consensus (10%)                                  │    │
│  │     Score < 40 → NO_SIGNAL                                   │    │
│  └───────────────────────────┬─────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  3. CALCULATE CONFIDENCE                                     │    │
│  │     a) Base confidence (model certainty)                     │    │
│  │     b) Apply regime penalty (CHAOTIC = -30%)                 │    │
│  │     c) Apply temporal decay (if aged)                        │    │
│  │     Confidence < 0.6 → NO_SIGNAL                             │    │
│  └───────────────────────────┬─────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  4. CHECK REGIME FIT                                         │    │
│  │     - Does signal type match regime?                         │    │
│  │     - TREND_FOLLOW in TRENDING? ✓                            │    │
│  │     - TREND_FOLLOW in CHAOTIC? ✗                             │    │
│  │     Regime mismatch → regime_fit = false                     │    │
│  └───────────────────────────┬─────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  5. EMIT SIGNAL (ADVISORY ONLY)                              │    │
│  │     - Publish to: signals.advisory.new                       │    │
│  │     - Include: edge_score, confidence, regime_fit            │    │
│  │     - execution_power: "NONE"                                │    │
│  └───────────────────────────┬─────────────────────────────────┘    │
└──────────────────────────────┼──────────────────────────────────────┘
                               │
                               │ SignalResponse
                               │ (ADVISORY ONLY)
                               ▼
                    ┌──────────────────────┐
                    │     RISK KERNEL      │
                    │                      │
                    │  Receives signal     │
                    │  DECIDES whether     │
                    │  to proceed          │
                    │                      │
                    │  CAN VETO regardless │
                    │  of signal quality   │
                    └──────────┬───────────┘
                               │
                               │ TradeRequest (if approved)
                               ▼
                    ┌──────────────────────┐
                    │   EXECUTION ENGINE   │
                    │                      │
                    │  Receives approved   │
                    │  trade from Risk     │
                    │  Kernel ONLY         │
                    │                      │
                    │  Signal Advisory     │
                    │  has NO ACCESS here  │
                    └──────────────────────┘
```

---

## 2. Signal Generation Pipeline

### Input Validation

```python
def validate_inputs(inputs: SignalInputs) -> ValidationResult:
    """
    First step: Validate all required inputs.
    ANY missing → NO_SIGNAL (fail-closed).
    """
    
    issues = []
    
    # Price data
    if inputs.price_current is None:
        issues.append("Missing current price")
    if len(inputs.price_history) < 100:
        issues.append(f"Insufficient price history: {len(inputs.price_history)}/100")
    
    # Volume data
    if inputs.volume_current is None:
        issues.append("Missing current volume")
    if len(inputs.volume_history) < 100:
        issues.append(f"Insufficient volume history: {len(inputs.volume_history)}/100")
    
    # Regime classification
    if inputs.regime is None:
        issues.append("Missing regime classification")
    if inputs.regime not in ["TRENDING", "RANGING", "CHAOTIC"]:
        issues.append(f"Invalid regime: {inputs.regime}")
    
    # Data freshness
    data_age = now() - inputs.timestamp
    if data_age.seconds > 5:
        issues.append(f"Stale data: {data_age.seconds}s old")
    
    if issues:
        return ValidationResult(
            valid=False,
            issues=issues,
            action="NO_SIGNAL"
        )
    
    return ValidationResult(valid=True, issues=[], action="CONTINUE")
```

### Decision Gates

```python
def generate_signal(inputs: SignalInputs) -> SignalResponse:
    """
    Full signal generation with decision gates.
    """
    
    # Gate 1: Input validation
    validation = validate_inputs(inputs)
    if not validation.valid:
        return SignalResponse(
            recommended_action="NO_SIGNAL",
            reasoning=f"Input validation failed: {validation.issues}"
        )
    
    # Gate 2: Edge score minimum
    edge_score = calculate_edge_score(inputs)
    if edge_score < 40:
        return SignalResponse(
            edge_score=edge_score,
            recommended_action="NO_SIGNAL",
            reasoning=f"Edge score {edge_score} below minimum 40"
        )
    
    # Gate 3: Confidence floor
    confidence_result = calculate_final_confidence(inputs)
    if not confidence_result["passes_floor"]:
        return SignalResponse(
            edge_score=edge_score,
            confidence=confidence_result["final_confidence"],
            recommended_action="NO_SIGNAL",
            reasoning=f"Confidence {confidence_result['final_confidence']} below floor 0.6"
        )
    
    # Gate 4: Regime fit check
    regime_fit = check_regime_fit(inputs.signal_type, inputs.regime)
    
    # All gates passed → Emit advisory signal
    return SignalResponse(
        edge_score=edge_score,
        confidence=confidence_result["final_confidence"],
        regime_fit=regime_fit,
        regime=inputs.regime,
        recommended_action=determine_direction(inputs),
        reasoning=build_reasoning(inputs, edge_score, confidence_result),
        execution_power="NONE"  # Always NONE
    )
```

---

## 3. Event Flow

### Events Published

```yaml
# Signal Advisory publishes to these streams
signals.advisory.new:
  description: New advisory signal generated
  payload:
    signal_id: string
    timestamp: datetime
    edge_score: int (0-100)
    confidence: float (0-1)
    regime_fit: bool
    regime: TRENDING | RANGING | CHAOTIC
    recommended_action: LONG | SHORT | NO_SIGNAL
    reasoning: string
    execution_power: "NONE"
  consumers:
    - risk_kernel  # Primary consumer
    - logger       # For audit
    - dashboard    # For display

signals.advisory.no_signal:
  description: Signal Advisory declined to generate signal
  payload:
    timestamp: datetime
    reason: string
    gate_failed: string
    inputs_summary: object
  consumers:
    - logger
    - dashboard

signals.advisory.calibration:
  description: Calibration drift detected
  payload:
    bucket: string
    expected_winrate: float
    actual_winrate: float
    sample_size: int
  consumers:
    - alerting
    - logger
```

### Events Consumed

```yaml
# Signal Advisory subscribes to these streams
market.data.candle:
  description: New price/volume candle
  use: Input for signal generation

regime.classification.update:
  description: Regime change detected
  use: Update regime for confidence adjustment

system.kill_switch.activated:
  description: Kill switch activated
  use: Stop all signal generation
```

---

## 4. Integration Protocol

### Risk Kernel Handoff

```
Signal Advisory                    Risk Kernel
      │                                  │
      │  SignalResponse                  │
      │  {                               │
      │    edge_score: 72,               │
      │    confidence: 0.68,             │
      │    regime_fit: true,             │
      │    recommended_action: "LONG",   │
      │    execution_power: "NONE"       │
      │  }                               │
      │ ─────────────────────────────▶   │
      │                                  │
      │                                  │ Risk Kernel evaluates:
      │                                  │ - Trade risk (2% max)
      │                                  │ - Daily loss (5% limit)
      │                                  │ - Drawdown (20% limit)
      │                                  │ - System integrity
      │                                  │
      │  ◀───────────────────────────────│
      │                                  │
      │  Risk Kernel can:                │
      │  - APPROVE (proceed)             │
      │  - VETO (reject)                 │
      │  - MODIFY (reduce size)          │
      │                                  │
      │  Signal Advisory CANNOT:         │
      │  - Appeal VETO                   │
      │  - Execute directly              │
      │  - Override decision             │
```

### No Direct Exchange Access

```python
# Signal Advisory architecture
class SignalAdvisory:
    """
    Signal Advisory has NO exchange connectivity.
    """
    
    # What we have
    market_data_feed: MarketDataFeed       # READ-ONLY
    redis_publisher: RedisPublisher         # Event publishing
    model_ensemble: ModelEnsemble           # Signal generation
    
    # What we DO NOT have
    # exchange_client: None                 # NO EXCHANGE ACCESS
    # execution_engine: None                # NO EXECUTION ACCESS
    # order_manager: None                   # NO ORDER ACCESS
    
    def __init__(self):
        # Verify no execution capability
        assert not hasattr(self, 'exchange_client')
        assert not hasattr(self, 'execution_engine')
        assert not hasattr(self, 'order_manager')
```

---

## 5. Timing Requirements

### Response Time Targets

| Operation | Target | Max Allowed |
|-----------|--------|-------------|
| Input validation | <5ms | 10ms |
| Edge score calculation | <20ms | 50ms |
| Confidence calculation | <10ms | 25ms |
| Total signal generation | <50ms | 100ms |
| Event publish | <5ms | 10ms |

### Timeout Handling

```python
SIGNAL_GENERATION_TIMEOUT_MS = 100

def generate_signal_with_timeout(inputs: SignalInputs) -> SignalResponse:
    """
    Signal generation with hard timeout.
    Timeout → NO_SIGNAL (fail-closed).
    """
    
    start = time.time()
    
    try:
        with timeout(SIGNAL_GENERATION_TIMEOUT_MS / 1000):
            return generate_signal(inputs)
    except TimeoutError:
        elapsed = (time.time() - start) * 1000
        return SignalResponse(
            recommended_action="NO_SIGNAL",
            reasoning=f"Signal generation timeout ({elapsed:.0f}ms > {SIGNAL_GENERATION_TIMEOUT_MS}ms)"
        )
```

---

## 6. Error Handling

### Fail-Closed by Default

```python
def handle_error(error: Exception, context: str) -> SignalResponse:
    """
    All errors → NO_SIGNAL (fail-closed).
    Never emit uncertain signal.
    """
    
    log_error(f"Signal Advisory error in {context}: {error}")
    
    emit_event("signals.advisory.error", {
        "context": context,
        "error": str(error),
        "action": "NO_SIGNAL"
    })
    
    return SignalResponse(
        recommended_action="NO_SIGNAL",
        reasoning=f"Error in signal generation: {context}",
        execution_power="NONE"
    )
```

### Error Categories

| Error Type | Handling | Signal Output |
|------------|----------|---------------|
| Missing data | Log, return NO_SIGNAL | NO_SIGNAL |
| Model error | Log, return NO_SIGNAL | NO_SIGNAL |
| Timeout | Log, return NO_SIGNAL | NO_SIGNAL |
| Unknown error | Log, alert, return NO_SIGNAL | NO_SIGNAL |

---

## 7. Monitoring

### Health Metrics

```python
SIGNAL_ADVISORY_METRICS = {
    "signal_rate": "signals per minute",
    "no_signal_rate": "no-signals per minute",
    "rejection_by_gate": {
        "input_validation": "count",
        "edge_score": "count",
        "confidence_floor": "count",
        "regime_fit": "count",
    },
    "average_edge_score": "rolling 1h average",
    "average_confidence": "rolling 1h average",
    "generation_latency_p99": "99th percentile ms",
}
```

### Dashboard Display

```
┌─────────────────────────────────────────────────────────────────┐
│ SIGNAL ADVISORY STATUS                                          │
├─────────────────────────────────────────────────────────────────┤
│ Status: ACTIVE           Signals (1h): 12                       │
│ Last Signal: 2m ago      No-Signals (1h): 48                    │
│                                                                 │
│ Average Edge Score: 58.3      Avg Confidence: 0.64              │
│                                                                 │
│ Rejection Rates:                                                │
│   Input Validation: 5%     Edge Score < 40: 35%                 │
│   Confidence < 0.6: 42%    Regime Mismatch: 18%                 │
│                                                                 │
│ Generation Latency: P50=23ms  P99=67ms                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Summary

```
┌─────────────────────────────────────────────────────────────────┐
│ SIGNAL ADVISORY DECISION FLOW SUMMARY                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 1. VALIDATE → Any issue? → NO_SIGNAL                            │
│ 2. EDGE SCORE → Below 40? → NO_SIGNAL                           │
│ 3. CONFIDENCE → Below 0.6? → NO_SIGNAL                          │
│ 4. EMIT SIGNAL → Always with execution_power="NONE"             │
│ 5. RISK KERNEL → Receives advisory, DECIDES                     │
│                                                                 │
│ KEY POINTS:                                                     │
│ • We ADVISE, we don't EXECUTE                                   │
│ • Risk Kernel can VETO any signal                               │
│ • Fail-closed on ANY error or uncertainty                       │
│ • NO direct exchange access                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

*Signal Advisory gir råd. Risk Kernel bestemmer. Execution Engine utfører.*
