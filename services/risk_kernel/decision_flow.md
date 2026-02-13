# Risk Kernel — Decision Flow

**Version**: 1.0  
**Type**: Fail-Closed  

---

## Decision Flowchart

```
                    ┌─────────────────────┐
                    │ risk.evaluate.request│
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ System Integrity OK? │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │ NO             │ UNKNOWN        │ YES
              ▼                ▼                ▼
        ┌─────────┐      ┌─────────┐     ┌─────────────┐
        │ REJECT  │      │ REJECT  │     │ Kill-Switch │
        └─────────┘      └─────────┘     │   Active?   │
                                         └──────┬──────┘
                                                │
                               ┌────────────────┼────────────┐
                               │ YES                         │ NO
                               ▼                             ▼
                         ┌─────────┐               ┌─────────────┐
                         │ REJECT  │               │ Daily Risk  │
                         │(halted) │               │     OK?     │
                         └─────────┘               └──────┬──────┘
                                                          │
                                          ┌───────────────┼───────────────┐
                                          │ ≤-5%         │ -4% to -5%    │ OK
                                          ▼              ▼               ▼
                                    ┌───────────┐  ┌─────────┐   ┌─────────────┐
                                    │KILL_SWITCH│  │ REJECT  │   │ Drawdown    │
                                    └───────────┘  └─────────┘   │    OK?      │
                                                                 └──────┬──────┘
                                                                        │
                                                    ┌───────────────────┼─────────────┐
                                                    │ >20%             │ >12%        │ OK
                                                    ▼                  ▼             ▼
                                              ┌───────────┐      ┌─────────┐   ┌───────────┐
                                              │KILL_SWITCH│      │ REJECT  │   │ Trade Risk│
                                              └───────────┘      └─────────┘   │    OK?    │
                                                                               └─────┬─────┘
                                                                                     │
                                                                     ┌───────────────┼───────────┐
                                                                     │ NO            │ UNKNOWN   │ YES
                                                                     ▼               ▼           ▼
                                                               ┌─────────┐    ┌─────────┐ ┌───────────┐
                                                               │ REJECT  │    │ REJECT  │ │ APPROVE   │
                                                               └─────────┘    └─────────┘ └───────────┘
```

---

## Fail-Closed Logic (Core)

```python
def evaluate_risk(request: RiskEvaluateRequest) -> RiskResult:
    """
    Fail-Closed Risk Evaluation
    
    If ANY step fails or is uncertain → REJECT
    If CRITICAL fails → KILL_SWITCH
    Only if ALL pass → APPROVE
    """
    
    # Step 1: System Integrity
    integrity = check_system_integrity(request.system_state)
    if integrity.is_kill_switch:
        return trigger_kill_switch(integrity.reason)
    if not integrity.is_approved:
        return reject(integrity.reason)
    
    # Step 2: Kill-Switch Already Active?
    if is_kill_switch_active():
        return reject("Kill-switch is active")
    
    # Step 3: Daily Risk
    daily = check_daily_loss(request.capital)
    if daily.is_kill_switch:
        return trigger_kill_switch(daily.reason)
    if not daily.is_approved:
        return reject(daily.reason)
    
    # Step 4: Drawdown Risk
    drawdown = check_drawdown(request.capital)
    if drawdown.is_kill_switch:
        return trigger_kill_switch(drawdown.reason)
    if not drawdown.is_approved:
        return reject(drawdown.reason)
    
    # Step 5: Trade Risk
    trade = validate_trade_risk(request.trade, request.capital)
    if not trade.is_approved:
        return reject(trade.reason)
    
    # ALL PASSED
    return approve(
        size_multiplier=drawdown.size_multiplier,
        risk_metrics=compile_metrics(integrity, daily, drawdown, trade)
    )
```

---

## Event Flow

### Input Events

```yaml
risk.evaluate.request:
  source: entry_gate
  payload:
    trade:
      symbol: "BTCUSDT"
      direction: "LONG"
      entry_price: 50000.0
      stop_loss: 49000.0
      size: 0.1
    capital:
      equity: 10000.0
      daily_pnl: -200.0
      peak_equity: 10500.0
    system_state:
      data_age_seconds: 2
      data_quality: 0.98
      api_healthy: true
```

### Output Events

```yaml
# Approval
risk.approved:
  trade_id: "uuid"
  size_multiplier: 0.85  # Due to drawdown
  risk_metrics:
    trade_risk_percent: 0.018
    daily_pnl_percent: -0.02
    drawdown_percent: 0.048

# Rejection
risk.rejected:
  trade_id: "uuid"
  reason: "Risk per trade 2.5% exceeds 2% limit"
  dimension: "trade_risk"

# Kill-Switch
kill_switch.triggered:
  trigger: "daily_loss"
  reason: "Daily loss -5.2% exceeds limit"
  timestamp: "ISO8601"
  positions_to_close: ["pos_1", "pos_2"]
```

---

## Response Time Requirements

| Operation | Max Time |
|-----------|----------|
| Full evaluation | < 50ms |
| Single dimension | < 10ms |
| Kill-switch trigger | < 10ms |

---

## Logging Requirements

Every evaluation logs:

```json
{
    "evaluation_id": "uuid",
    "trade_id": "uuid",
    "timestamp": "ISO8601",
    "result": "APPROVE|REJECT|KILL_SWITCH",
    "dimensions": {
        "system_integrity": {"passed": true, "detail": "..."},
        "daily_risk": {"passed": true, "value": -0.02},
        "drawdown_risk": {"passed": true, "value": 0.05},
        "trade_risk": {"passed": true, "value": 0.018}
    },
    "execution_time_ms": 45
}
```

---

## Edge Cases (All → REJECT)

| Case | Handling |
|------|----------|
| Missing trade data | REJECT |
| Missing capital data | REJECT |
| Calculation overflow | REJECT |
| Negative values | REJECT |
| Division by zero | REJECT |
| Timeout on check | REJECT |

**There are no edge cases that result in APPROVE by default.**
