# Emergency Exit Worker - Trigger Conditions

## When panic_close is Published

### 1. Risk Kernel Triggers

| Condition | Threshold | Response Time |
|-----------|-----------|---------------|
| Account drawdown | > 15% in 1 hour | Immediate |
| Position leverage breach | > 3x max allowed | Immediate |
| Margin ratio critical | < 5% | Immediate |
| Exchange API failure | > 30s consecutive | Immediate |
| Multiple position losses | 3+ positions at -5% simultaneously | Immediate |

### 2. Exit Brain Triggers (Fatal Only)

| Condition | Description |
|-----------|-------------|
| Health check failure | heartbeat missing > 5 seconds WITH open positions |
| Decision loop stall | No decisions for > 30 seconds WITH active positions |
| Internal exception | Unrecoverable error in exit logic |

### 3. Manual Ops Triggers

| Scenario | Authorization |
|----------|---------------|
| Black swan event | Senior ops only |
| Exchange maintenance | Scheduled |
| Security incident | Requires 2FA confirm |

## Trigger Message Format

```json
{
  "event": "system.panic_close",
  "source": "risk_kernel|exit_brain|ops",
  "timestamp": 1707840000.123,
  "reason": "Human-readable reason",
  "severity": "CRITICAL",
  "context": {
    "open_positions": 5,
    "total_exposure_usd": 15000,
    "trigger_metric": "drawdown",
    "trigger_value": 0.18
  }
}
```

## What Does NOT Trigger panic_close

- ❌ Single position loss
- ❌ Normal market volatility
- ❌ AI model disagreement
- ❌ Low confidence signals
- ❌ High funding rates
- ❌ Maintenance windows (planned)

## Trigger Validation

EEW validates before executing:

```python
def validate_panic_trigger(event: dict) -> bool:
    # Must have required fields
    if not all(k in event for k in ['source', 'timestamp', 'reason']):
        return False
    
    # Source must be authorized
    if event['source'] not in ['risk_kernel', 'exit_brain', 'ops']:
        return False
    
    # Timestamp must be recent (prevent replay attacks)
    if time.time() - event['timestamp'] > 60:
        return False
    
    return True
```

## Cooldown

After panic_close.completed:
- 5-minute minimum cooldown before any trading resumes
- Manual acknowledgment required
- System enters FLAT state
